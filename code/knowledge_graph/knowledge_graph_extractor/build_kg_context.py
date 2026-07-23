#!/usr/bin/env python3
"""
build_kg_context.py — add a judge-filtered `kg_context` column to the SQuAD 2.0
multi-strategy eval set (8k rows) by extracting a KG once per UNIQUE context.

Three stages, each resumable (incremental JSONL append keyed on context hash):

  extract   extractor adapter over unique contexts        -> <cache>.extract.jsonl
  judge     faithfulness labels per triple                -> <cache>.judged.jsonl
  join      merge into eval file, add kg_context column   -> output .jsonl

Usage on the cluster (local paths + HF_HUB_OFFLINE=1 recommended on compute nodes):

  python build_kg_context.py extract --eval-file squad2_eval_8k.jsonl --cache kg/sq8k
  python build_kg_context.py judge   --cache kg/sq8k
  python build_kg_context.py join    --eval-file squad2_eval_8k.jsonl --cache kg/sq8k \
                                     --output squad2_eval_8k_kg.jsonl

By default `join` uses judge-filtered triples (supported only; --keep-partial to
include partials; --unfiltered to bypass the judge stage entirely).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import signal
import time
from pathlib import Path

# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "AlinaGonch/llama3_1_8b_kg_extractor"   # Hub id or local path
# Local judge (no API). Qwen2.5-7B-Instruct is ungated, strong at JSON, and is a
# DIFFERENT family from the Llama extractor (reduces self-preference bias).
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

# Rendering of the kg_context column (keep stable; prompt_template.py relies on it).
KG_HEADER = "Facts extracted from the passage:"
KG_TRIPLE_FMT = '("{s}", "{r}", "{o}")'

FAITH_SYS = (
    "You verify whether knowledge-graph triples are supported by a passage. "
    "Judge ONLY against the passage provided. Do not use outside knowledge. "
    "Label each triple: 'supported' (the passage states this relation), "
    "'partial' (entities are right but the relation is imprecise or overstated), "
    "or 'unsupported' (not stated in or inferable from the passage). "
    'Return ONLY JSON: {"labels": ["supported"|"partial"|"unsupported", ...]} '
    "with one label per triple, in order."
)


# ---------------------------------------------------------------------------
# Parsing / cleaning
# ---------------------------------------------------------------------------

def parse_triples(raw):
    """Best-effort parse of model output -> (list-or-None, truncated: bool)."""
    if raw is None:
        return None, False
    s = re.sub(r"</?python>", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"```(?:python|json)?", "", s).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None, True  # no closing bracket at all -> almost certainly truncated
    body = s[i:j + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(body)
            return (parsed, False) if isinstance(parsed, list) else (None, False)
        except Exception:
            continue
    # salvage: output was cut mid-triple or outer array left unclosed ->
    # try closing after each complete inner list, starting from the last "]"
    # (body always ends with "]" by construction, so this covers the common
    # case of a truncated outer array whose last inner triple is complete)
    last = len(body) - 1
    while last != -1:
        cand = body[:last + 1].rstrip().rstrip(",")
        if not cand.startswith("["):
            break
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(cand + "]")
                if isinstance(parsed, list) and parsed:
                    return parsed, True
            except Exception:
                continue
        last = body.rfind("]", 0, last)
    return None, True


def _dedupe(triples):
    seen, out = set(), []
    for t in triples:
        k = tuple(e.strip() for e in t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def clean_triples(triples):
    """Keep only well-formed [h, r, t] string triples."""
    if not triples:
        return []
    return [list(t) for t in triples
            if isinstance(t, (list, tuple)) and len(t) == 3
            and all(isinstance(e, str) and e.strip() for e in t)]


def render_kg_block(triples):
    if not triples:
        return ""
    lines = [KG_HEADER]
    for s, r, o in triples:
        lines.append(KG_TRIPLE_FMT.format(
            s=s.replace('"', "'"), r=r.replace('"', "'"), o=o.replace('"', "'")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval-file / cache helpers
# ---------------------------------------------------------------------------

def normalize_context(text):
    return re.sub(r"\s+", " ", text).strip()


def context_hash(text):
    return hashlib.sha1(normalize_context(text).encode("utf-8")).hexdigest()


def load_eval_rows(path, context_key):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    elif p.suffix == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported eval file type: {p.suffix}")
    if rows and context_key not in rows[0]:
        raise KeyError(f"Column '{context_key}' not found. "
                       f"Available: {sorted(rows[0].keys())}. Use --context-key.")
    return rows


def unique_contexts(rows, context_key):
    out = {}
    for r in rows:
        h = context_hash(r[context_key])
        out.setdefault(h, r[context_key])
    return out


def load_stage(path):
    """Read a stage JSONL -> {hash: record}, tolerating a torn final line."""
    recs = {}
    p = Path(path)
    if not p.exists():
        return recs
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            recs[r["hash"]] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return recs


def install_sigterm_flag():
    stop = {"flag": False}

    def _handler(signum, frame):
        print(f"[signal] caught {signum}; stopping after current batch.", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return stop


def load_4bit(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    import torch
    from peft import PeftModel

    rows = load_eval_rows(args.eval_file, args.context_key)
    ctxs = unique_contexts(rows, args.context_key)
    items = sorted(ctxs.items())
    if args.num_workers > 1:
        items = [(h, c) for h, c in items
                 if int(h, 16) % args.num_workers == args.worker_id]
    out_path = Path(f"{args.cache}.extract"
                    f"{'.w%d' % args.worker_id if args.num_workers > 1 else ''}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(load_stage(str(out_path)))
    if args.num_workers > 1:  # honor a previously merged single-file cache too
        done |= set(load_stage(f"{args.cache}.extract.jsonl"))
    todo = [(h, c) for h, c in items if h not in done]
    print(f"[extract] rows={len(rows)} unique={len(ctxs)} "
          f"shard={len(items)} cached={len(items)-len(todo)} todo={len(todo)}")
    if not todo:
        return

    model, tok = load_4bit(BASE_MODEL)
    model = PeftModel.from_pretrained(model, ADAPTER_ID)  # adapter name has no dots
    model.eval()
    tok.padding_side = "left"                    # decoder-only batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token            # Llama-3.1 ships without a pad token

    def build_prompt(context):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    stop = install_sigterm_flag()
    n_done, n_trunc, n_fail, t0 = 0, 0, 0, time.time()
    with out_path.open("a", encoding="utf-8") as fout:
        for i in range(0, len(todo), args.batch_size):
            if stop["flag"]:
                break
            batch = todo[i:i + args.batch_size]
            prompts = [build_prompt(c) for _, c in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_prompt_tokens
                      ).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for (h, ctx), raw in zip(batch, outs):
                parsed, truncated = parse_triples(raw)
                triples = _dedupe(clean_triples(parsed))
                n_trunc += int(truncated)
                n_fail += int(not triples)
                fout.write(json.dumps({
                    "hash": h, "context": ctx, "raw_output": raw,
                    "triples": triples, "n_triples": len(triples),
                    "parses": parsed is not None and not truncated,
                    "truncated": truncated,
                }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(batch)
            print(f"[extract] {n_done}/{len(todo)} "
                  f"({n_done / max(time.time() - t0, 1e-6):.2f} ctx/s, "
                  f"truncated={n_trunc}, empty={n_fail})", flush=True)
    if n_trunc > 0.05 * max(n_done, 1):
        print("[extract] WARNING: >5% truncated — raise MAX_NEW_TOKENS.")


# ---------------------------------------------------------------------------
# Stage 2: judge
# ---------------------------------------------------------------------------

def _parse_judge_json(text):
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


def cmd_judge(args):
    import torch

    # merge all extract shards
    extracted = {}
    prefix = Path(args.cache)
    for f in sorted(prefix.parent.glob(prefix.name + ".extract*.jsonl")):
        extracted.update(load_stage(str(f)))
    if not extracted:
        raise SystemExit("[judge] no extract cache found — run `extract` first.")
    out_path = Path(f"{args.cache}.judged.jsonl")
    done = set(load_stage(str(out_path)))
    todo = [r for h, r in sorted(extracted.items()) if h not in done]
    print(f"[judge] extracted={len(extracted)} cached={len(done)} todo={len(todo)}")
    if not todo:
        return

    jmodel, jtok = load_4bit(JUDGE_MODEL)

    @torch.no_grad()
    def call_judge(passage, triples):
        numbered = "\n".join(f"{k}. {t}" for k, t in enumerate(triples))
        messages = [
            {"role": "system", "content": FAITH_SYS},
            {"role": "user", "content": f"PASSAGE:\n{passage}\n\nTRIPLES:\n{numbered}"},
        ]
        inputs = jtok.apply_chat_template(messages, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True
                                          ).to(jmodel.device)
        out = jmodel.generate(**inputs, max_new_tokens=args.judge_max_tokens,
                              do_sample=False, pad_token_id=jtok.eos_token_id)
        return jtok.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)

    stop = install_sigterm_flag()
    n_done, n_judge_fail, t0 = 0, 0, time.time()
    with out_path.open("a", encoding="utf-8") as fout:
        for rec in todo:
            if stop["flag"]:
                break
            triples = rec["triples"]
            judge_failed = False
            if not triples:
                labels = []
            else:
                data = _parse_judge_json(call_judge(rec["context"], triples))
                raw_labels = [str(l).lower() for l in (data or {}).get("labels", [])]
                judge_failed = not raw_labels
                n_judge_fail += int(judge_failed)
                # length repair: pad missing with 'unsupported' (conservative), trim extras
                labels = (raw_labels + ["unsupported"] * len(triples))[:len(triples)]
            weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
            scored = [weight.get(l, 0.0) for l in labels]
            fout.write(json.dumps({
                "hash": rec["hash"],
                "faith_labels": labels,
                "faithfulness": (sum(scored) / len(scored)) if scored else None,
                "judge_failed": judge_failed,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"[judge] {n_done}/{len(todo)} "
                      f"({n_done / max(time.time() - t0, 1e-6):.2f} ctx/s, "
                      f"judge_parse_failures={n_judge_fail})", flush=True)
    if n_judge_fail:
        print(f"[judge] WARNING: {n_judge_fail} contexts got no parseable judge "
              f"labels; their triples were conservatively marked 'unsupported' "
              f"(judge_failed=true). Delete those lines from {out_path.name} and "
              f"re-run to retry.")


# ---------------------------------------------------------------------------
# Stage 3: join
# ---------------------------------------------------------------------------

def cmd_join(args):
    rows = load_eval_rows(args.eval_file, args.context_key)
    extracted = {}
    prefix = Path(args.cache)
    for f in sorted(prefix.parent.glob(prefix.name + ".extract*.jsonl")):
        extracted.update(load_stage(str(f)))
    judged = {} if args.unfiltered else load_stage(f"{args.cache}.judged.jsonl")
    keep = {"supported"} | ({"partial"} if args.keep_partial else set())

    n_miss, n_empty, kept_ratios = 0, 0, []
    out_rows = []
    for r in rows:
        h = context_hash(r[args.context_key])
        rec = extracted.get(h)
        row = dict(r)
        if rec is None:
            n_miss += 1
            row.update(kg_context=None, kg_triples=None, kg_meta=None)
            out_rows.append(row)
            continue
        triples = rec["triples"]
        if not args.unfiltered:
            j = judged.get(h)
            if j is None:
                n_miss += 1
                row.update(kg_context=None, kg_triples=None,
                           kg_meta={"hash": h, "missing": "judge"})
                out_rows.append(row)
                continue
            triples = [t for t, l in zip(triples, j["faith_labels"]) if l in keep]
            if rec["triples"]:
                kept_ratios.append(len(triples) / len(rec["triples"]))
        block = render_kg_block(triples)
        n_empty += int(not block)
        row.update(
            kg_context=block,
            kg_triples=triples,
            kg_meta={"hash": h, "n_raw": rec["n_triples"], "n_kept": len(triples),
                     "parses": rec["parses"], "truncated": rec["truncated"],
                     "filtered": not args.unfiltered},
        )
        out_rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[join] wrote {len(out_rows)} rows -> {out}")
    print(f"[join] missing={n_miss} emptyKG={n_empty}"
          + (f" mean-kept-ratio={sum(kept_ratios)/len(kept_ratios):.3f}"
             if kept_ratios else ""))
    if n_miss:
        print("[join] NOTE: rows with kg_context=None — finish extract/judge first.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", required=True,
                        help="cache path prefix, e.g. kg/sq8k")
    common.add_argument("--context-key", default="context")

    ex = sub.add_parser("extract", parents=[common])
    ex.add_argument("--eval-file", required=True)
    ex.add_argument("--batch-size", type=int, default=24)
    ex.add_argument("--max-prompt-tokens", type=int, default=3072)
    ex.add_argument("--num-workers", type=int, default=1)
    ex.add_argument("--worker-id", type=int, default=0)
    ex.set_defaults(func=cmd_extract)

    jd = sub.add_parser("judge", parents=[common])
    jd.add_argument("--judge-max-tokens", type=int, default=2048)
    jd.set_defaults(func=cmd_judge)

    jn = sub.add_parser("join", parents=[common])
    jn.add_argument("--eval-file", required=True)
    jn.add_argument("--output", required=True)
    jn.add_argument("--keep-partial", action="store_true",
                    help="also keep 'partial' triples (default: supported only)")
    jn.add_argument("--unfiltered", action="store_true",
                    help="skip judge filtering entirely (raw extractor output)")
    jn.set_defaults(func=cmd_join)

    args = ap.parse_args()
    if args.cmd == "extract" and not (0 <= args.worker_id < args.num_workers):
        ap.error("--worker-id must be in [0, --num-workers)")
    args.func(args)


if __name__ == "__main__":
    main()
