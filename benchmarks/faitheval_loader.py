"""
FaithEval — all 3 variants (counterfactual / inconsistent / unanswerable).
Single-context QA

Variant semantics:
  counterfactual : context is edited to a counterfactual answer; faithful model
                   must answer FROM CONTEXT -> is_answerable=True, gold=ctx answer (EM).
  unanswerable   : context lacks the answer -> is_answerable=False, gold=["I don't know"].
  inconsistent   : conflicting docs -> faithful response = flag the conflict.
                   is_answerable=False, ground_truths=["I don't know"]
"""
from typing import List, Optional
import argparse
from datasets import load_dataset

from ood_common import CanonicalExample, gold_for, write_jsonl, SCORE_EM, GOLD_ANSWER, GOLD_ABSTAIN, GOLD_CONFLICT
from prompt_template import get_chat_messages, PROMPT_STRATEGIES_LIST

VARIANTS = {
    "counterfactual": "Salesforce/FaithEval-counterfactual-v1.0",
    "inconsistent":   "Salesforce/FaithEval-inconsistent-v1.0",
    "unanswerable":   "Salesforce/FaithEval-unanswerable-v1.0",
}

DEFAULT_STRATEGIES = ["zero_shot_conf", "cot_conf", "self_reflect_conf"]


def _g(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _answers(row) -> List[str]:
    a = _g(row, "answers", "answer", "gold_answer", "answerKey")
    if a is None:
        return []
    if isinstance(a, dict):  # squad-style {"text":[...]}
        return list(a.get("text", []))
    if isinstance(a, list):
        return [str(x) for x in a]
    return [str(a)]


def load(variants: Optional[List[str]] = None, strategies: Optional[List[str]] = None,
         limit: Optional[int] = None) -> List[CanonicalExample]:
    variants = variants or list(VARIANTS)
    strategies = strategies or DEFAULT_STRATEGIES

    # checking if the provided strategies are valid
    for s in strategies:
        assert s in PROMPT_STRATEGIES_LIST, f"{s} not in your PROMPT_STRATEGIES_LIST"

    out = []
    for variant in variants:
        ds = load_dataset(VARIANTS[variant], split="test")
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            question = _g(row, "question", "query", default="")
            context = _g(row, "context", "passage", "text", default="")
            if not question or not context:
                continue
            if variant == "counterfactual":
                is_ans, grt, gtype, skind = True, _answers(row), GOLD_ANSWER, SCORE_EM
            elif variant == "unanswerable":
                is_ans, grt, gtype, skind = False, None, GOLD_ABSTAIN, "none"
            else:  # inconsistent
                is_ans, grt, gtype, skind = False, None, GOLD_CONFLICT, "none"
            for s in strategies:
                out.append(CanonicalExample(
                    id=f"faitheval_{variant}-{i}-{s}",
                    messages=get_chat_messages(s, context=context, question=question),
                    ground_truths=gold_for(is_ans, grt, has_gold_string=(skind == SCORE_EM)),
                    is_answerable=is_ans,
                    metadata={"dataset": "faitheval", "variant": variant, "strategy": s,
                              "score_kind": skind, "gold_response_type": gtype},
                ))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="faitheval.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--variants", nargs="*"); ap.add_argument("--strategies", nargs="*")
    a = ap.parse_args()
    print("wrote", write_jsonl(a.out, variants=a.variants, strategies=a.strategies, limit=a.limit), "->", a.out)
