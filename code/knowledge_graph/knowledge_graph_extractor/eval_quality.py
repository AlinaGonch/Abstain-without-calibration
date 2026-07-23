"""
Quality evaluation for the KG-extraction adapter.

Layer 0 — format validity (deterministic, free): does the output parse, is the
          KG non-empty, are triples well-formed, how many per example.
Layer 1 — triple-set Precision / Recall / F1 vs GOLD (deterministic, free), at
          three match strictnesses:
            * exact      — strings identical (after strip)
            * normalized — lowercased, articles/punct removed, whitespace collapsed
            * fuzzy      — per-slot string similarity >= threshold
          Measures agreement with the teacher's triples (teacher-imitation).
Layer 2 — semantic quality (LLM-as-judge, judged against the SOURCE PASSAGE, not
          the gold triples):
            * faithfulness = fraction of predicted triples supported by the text
            * coverage     = fraction of salient facts in the text captured
          The judge is a LOCAL model (no API, no cost). See JUDGE_MODEL below.

Layer 1 and Layer 2 answer DIFFERENT questions: gold here is silver (distilled
from a teacher), so a triple can be correct (high Layer 2) yet match no gold
(low Layer 1). Report both; the gap is informative.

Run on a HELD-OUT test split the model and early-stopping never saw — not train.

  python eval_quality.py --test_file data/extractor_test_data.jsonl
  python eval_quality.py --test_file data/extractor_test_data.jsonl --skip_judge   # Layer 0+1 only
"""
import os
import re
import ast
import json
import time
import argparse
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "AlinaGonch/llama3_1_8b_kg_extractor"   # Hub id or local path
# Local judge (no API). Qwen2.5-7B-Instruct is ungated, strong at JSON, and is a
# DIFFERENT family from the Llama extractor (reduces self-preference bias).
# Alternatives: "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3".
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 2560                                 # high enough for very large KGs

# Must match the SYSTEM_PROMPT used in train.py exactly.
SYSTEM_PROMPT = """Extract a knowledge graph (KG) from the following text. Follow these steps:
1. **Entities**: Identify all entities in the text. Ensure each entity is precise and specific.
2. **Relations**: Extract relationships between entities as triples: ["entity1", "relation", "entity2"].
3. **Coreference Resolution**: Unify references to the same entity (e.g., "Apple Inc." and "Apple" should be the same entity).
**Important Requirements**:
- The KG must not be empty. Ensure at least one triple is extracted.
- All entities mentioned in the text must be included in the KG, either as part of a triple or as a standalone entity if no relation is found.
- If no explicit relation is found between entities, create a generic relation like "related to" or "associated with" to ensure all entities are connected.
- Each triple must have three non-empty elements: ["entity1", "relation", "entity2"]. None of these elements can be empty or null.
Please only return the KG as a Python list of triples. For example:
<python>
[
["Apple Inc.", "founded by", "Steve Jobs"],
["Apple Inc.", "headquartered in", "Cupertino, California"],
["Apple Inc.", "produces", "iPhone"],
["Steve Jobs", "associated with", "Cupertino, California"]
]
</python>
Text:"""

# --------------------------------------------------------------------------- #
# Generation (extractor)
# --------------------------------------------------------------------------- #
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = PeftModel.from_pretrained(base, ADAPTER_ID)   # comment out to eval the BASE model (baseline)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,            # greedy: deterministic, reproducible
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# --------------------------------------------------------------------------- #
# Shared parsing
# --------------------------------------------------------------------------- #
def parse_triples(raw):
    """Best-effort parse of model output (or a gold string) -> list, or None."""
    if raw is None:
        return None
    s = re.sub(r"</?python>", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"```(?:python|json)?", "", s).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    s = s[i:j + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(s)
            return parsed if isinstance(parsed, list) else None
        except Exception:
            continue
    return None


def clean_triples(triples):
    """Keep only well-formed [h, r, t] string triples."""
    if not triples:
        return []
    return [list(t) for t in triples
            if isinstance(t, (list, tuple)) and len(t) == 3
            and all(isinstance(e, str) and e.strip() for e in t)]


# --------------------------------------------------------------------------- #
# Layer 0 — format validity
# --------------------------------------------------------------------------- #
def layer0_flags(triples):
    if triples is None:
        return {"parses": False, "nonempty_kg": False, "all_well_formed": False,
                "n_triples": 0, "n_malformed": 0}
    malformed = sum(
        1 for t in triples
        if not (isinstance(t, (list, tuple)) and len(t) == 3
                and all(isinstance(e, str) and e.strip() for e in t))
    )
    return {"parses": True, "nonempty_kg": len(triples) > 0,
            "all_well_formed": malformed == 0,
            "n_triples": len(triples), "n_malformed": malformed}


# --------------------------------------------------------------------------- #
# Layer 1 — triple-set P / R / F1 vs gold
# --------------------------------------------------------------------------- #
def _norm_elem(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_triple(t):
    return tuple(_norm_elem(e) for e in t)


def _ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def make_match_fn(mode, thr=0.85):
    """Returns match(pred_triple, gold_triple) -> (is_match, score)."""
    if mode == "exact":
        return lambda p, g: (tuple(e.strip() for e in p) == tuple(e.strip() for e in g), 1.0)
    if mode == "normalized":
        return lambda p, g: (_norm_triple(p) == _norm_triple(g), 1.0)

    def fuzzy(p, g):
        rs = [_ratio(_norm_elem(p[i]), _norm_elem(g[i])) for i in range(3)]
        return (all(r >= thr for r in rs), sum(rs) / 3)
    return fuzzy


def _dedupe(triples):
    seen, out = set(), []
    for t in triples:
        k = tuple(e.strip() for e in t)
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def count_matches(pred, gold, match_fn):
    """One-to-one greedy matching; a gold triple is consumed at most once."""
    used = [False] * len(gold)
    matched = 0
    for p in pred:
        best_j, best_s = -1, -1.0
        for j, g in enumerate(gold):
            if used[j]:
                continue
            ok, s = match_fn(p, g)
            if ok and s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            used[best_j] = True
            matched += 1
    return matched


def prf(matched, n_pred, n_gold):
    p = matched / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = matched / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


MATCH_MODES = ["exact", "normalized", "fuzzy"]


# --------------------------------------------------------------------------- #
# Layer 2 — LLM-as-judge (LOCAL model; judged ONLY against the passage)
# --------------------------------------------------------------------------- #
_JUDGE = None  # lazily built (model, tokenizer); built ONCE, reused for all calls


def _get_judge():
    global _JUDGE
    if _JUDGE is None:
        jtok = AutoTokenizer.from_pretrained(JUDGE_MODEL, use_fast=True)
        jmodel = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        jmodel.eval()
        _JUDGE = (jmodel, jtok)
    return _JUDGE


@torch.no_grad()
def call_judge(system, user, max_tokens=2048):
    model, tok = _get_judge()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,            # greedy: deterministic
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _parse_judge_json(text):
    """Extract the first {...} JSON object from the judge's reply, or None."""
    if not text:
        return None
    s = re.sub(r"```(?:json)?", "", text).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


FAITH_SYS = (
    "You verify whether knowledge-graph triples are supported by a passage. "
    "Judge ONLY against the passage provided. Do not use outside knowledge. "
    "Label each triple: 'supported' (the passage states this relation), "
    "'partial' (entities are right but the relation is imprecise or overstated), "
    "or 'unsupported' (not stated in or inferable from the passage). "
    'Return ONLY JSON: {"labels": ["supported"|"partial"|"unsupported", ...]} '
    "with one label per triple, in order."
)

COVERAGE_SYS = (
    "You assess how completely a set of knowledge-graph triples captures a passage. "
    "First, independently list the salient atomic facts the passage asserts "
    "(ignore the triples while doing this). Then, for each fact, decide whether it "
    "is represented by at least one of the provided triples. "
    'Return ONLY JSON: {"facts": [{"fact": "...", "covered": true|false}, ...]}'
)


def judge_faithfulness(passage, triples):
    if not triples:
        return None, []
    numbered = "\n".join(f"{k}. {t}" for k, t in enumerate(triples))
    user = f"PASSAGE:\n{passage}\n\nTRIPLES:\n{numbered}"
    try:
        data = _parse_judge_json(call_judge(FAITH_SYS, user))
    except Exception:
        return None, []
    labels = (data or {}).get("labels", [])
    weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    scored = [weight.get(str(l).lower(), 0.0) for l in labels[:len(triples)]]
    return (sum(scored) / len(scored), labels) if scored else (None, [])


def judge_coverage(passage, triples):
    if not triples:
        return None, []
    user = f"PASSAGE:\n{passage}\n\nTRIPLES:\n{triples}"
    try:
        data = _parse_judge_json(call_judge(COVERAGE_SYS, user))
    except Exception:
        return None, []
    facts = (data or {}).get("facts", [])
    if not facts:
        return None, []
    covered = sum(1 for f in facts if isinstance(f, dict) and f.get("covered"))
    return covered / len(facts), facts


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="extractor_test_data.jsonl",
                    help="HELD-OUT file (not train, ideally not the early-stopping eval set)")
    ap.add_argument("--limit", type=int, default=200, help="cap examples (judge cost/time)")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
    ap.add_argument("--skip_judge", action="store_true", help="Layer 0 + 1 only (free, fast)")
    ap.add_argument("--out", default="eval_results.jsonl")
    args = ap.parse_args()

    examples = []
    with open(args.test_file) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.limit:
                break
    print(f"Evaluating {len(examples)} held-out examples\n")

    model, tok = load_model()
    match_fns = {m: make_match_fn(m, args.fuzzy_threshold) for m in MATCH_MODES}

    rows, n = [], len(examples)
    agg0 = {"parses": 0, "nonempty": 0, "wellformed": 0, "triples": 0, "malformed": 0}
    # Layer 1 micro accumulators + macro F1 lists, per strictness
    micro = {m: {"matched": 0, "pred": 0, "gold": 0} for m in MATCH_MODES}
    macro_f1 = {m: [] for m in MATCH_MODES}
    faith_pool, cov_list = [], []

    for idx, ex in enumerate(examples, 1):
        passage = ex["context"]
        raw = generate(model, tok, passage)
        parsed = parse_triples(raw)
        flags = layer0_flags(parsed)

        agg0["parses"] += flags["parses"]
        agg0["nonempty"] += flags["nonempty_kg"]
        agg0["wellformed"] += flags["all_well_formed"]
        agg0["triples"] += flags["n_triples"]
        agg0["malformed"] += flags["n_malformed"]

        pred = _dedupe(clean_triples(parsed))
        gold = _dedupe(clean_triples(parse_triples(ex.get("kg_triplets"))))
        # Store the actual triples so F1=0 rows can be audited (teacher-divergence
        # vs genuine error) and the judge labels can be hand-checked later.
        row = {"context": passage, "raw_output": raw,
               "n_pred": len(pred), "n_gold": len(gold),
               "pred_triples": pred, "gold_triples": gold, **flags}

        # Layer 1 (always; free)
        for m in MATCH_MODES:
            matched = count_matches(pred, gold, match_fns[m])
            p, r, f = prf(matched, len(pred), len(gold))
            micro[m]["matched"] += matched
            micro[m]["pred"] += len(pred)
            micro[m]["gold"] += len(gold)
            macro_f1[m].append(f)
            row[f"prf_{m}"] = [round(p, 3), round(r, 3), round(f, 3)]

        # Layer 2 (optional)
        if not args.skip_judge and flags["parses"]:
            f_score, f_labels = judge_faithfulness(passage, pred)
            c_score, c_facts = judge_coverage(passage, pred)
            row.update({"faithfulness": f_score, "faith_labels": f_labels,
                        "coverage": c_score, "coverage_facts": c_facts})
            if pred and f_score is not None:
                faith_pool.extend({"supported": 1.0, "partial": 0.5}.get(str(l).lower(), 0.0)
                                  for l in f_labels[:len(pred)])
            if c_score is not None:
                cov_list.append(c_score)

        rows.append(row)
        if idx % 10 == 0:
            print(f"  {idx}/{n}")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n================  LAYER 0 — format validity  ================")
    print(f"  parse rate:          {agg0['parses']/n*100:5.1f}%")
    print(f"  non-empty KG rate:   {agg0['nonempty']/n*100:5.1f}%")
    print(f"  well-formed rate:    {agg0['wellformed']/n*100:5.1f}%")
    print(f"  mean triples/ex:     {agg0['triples']/n:5.1f}")
    if agg0["triples"]:
        print(f"  malformed triple %:  {agg0['malformed']/agg0['triples']*100:5.1f}%")

    print("\n================  LAYER 1 — P/R/F1 vs gold  ================")
    print(f"  {'strictness':<12} {'micro-P':>8} {'micro-R':>8} {'micro-F1':>9} {'macro-F1':>9}")
    for m in MATCH_MODES:
        mm = micro[m]
        p, r, f = prf(mm["matched"], mm["pred"], mm["gold"])
        mac = sum(macro_f1[m]) / len(macro_f1[m]) if macro_f1[m] else 0.0
        print(f"  {m:<12} {p*100:7.1f}% {r*100:7.1f}% {f*100:8.1f}% {mac*100:8.1f}%")
    print("  (exact->normalized lift = relation-phrasing mismatch; "
          "normalized->fuzzy = spelling/boundary slack)")

    if not args.skip_judge:
        print("\n================  LAYER 2 — semantic (vs passage)  ================")
        if faith_pool:
            print(f"  faithfulness:        {sum(faith_pool)/len(faith_pool)*100:5.1f}%   "
                  f"(supported triples / all triples; partial=0.5)")
        if cov_list:
            print(f"  coverage:            {sum(cov_list)/len(cov_list)*100:5.1f}%   "
                  f"(salient facts captured, mean over examples)")
        if not faith_pool and not cov_list:
            print("  (no judge scores produced — check the judge model loaded and "
                  "returned parseable JSON)")
        print("\n  NOTE: hand-label ~50-100 triples from eval_results.jsonl and compare to the")
        print("  judge labels to calibrate before trusting Layer 2 at scale.")

    print(f"\nPer-example detail written to {args.out}")


if __name__ == "__main__":
    main()
