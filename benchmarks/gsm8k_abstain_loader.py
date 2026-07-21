"""
GSM8K-Abstain (AbstentionBench). 
"""

from typing import List, Optional
from datasets import load_dataset
import argparse
from ood_common import CanonicalExample, extract_gsm8k_gold, make_abstain_variant, gold_for, write_jsonl, SCORE_NUMERIC, GOLD_ANSWER, GOLD_ABSTAIN
from ood_prompt_templates import get_ood_chat_messages, KIND_MATH_FREE, OOD_STRATEGIES

SOURCE_REPO, SOURCE_CONFIG, SOURCE_SPLIT = "openai/gsm8k", "main", "test"


def _ex(uid, text, strat, is_ans, gold):
    return CanonicalExample(
        id=uid,
        messages=get_ood_chat_messages(strat, text, kind=KIND_MATH_FREE),
        ground_truths=gold_for(is_ans, [gold] if gold else None, has_gold_string=is_ans),
        is_answerable=is_ans,
        metadata={"dataset": "gsm8k_abstain", "variant": None, "strategy": strat,
                  "score_kind": SCORE_NUMERIC if is_ans else "none",
                  "gold_response_type": GOLD_ANSWER if is_ans else GOLD_ABSTAIN,
                  "unanswerability_type": None if is_ans else "underspecified_context"},
    )


def load(strategies: Optional[List[str]] = None, limit: Optional[int] = None,
         include_unpaired_answerable: bool = True) -> List[CanonicalExample]:
    strategies = strategies or OOD_STRATEGIES
    ds = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)
    out: List[CanonicalExample] = []
    seen = 0
    for i, row in enumerate(ds):
        if limit is not None and seen >= limit:
            break
        seen += 1
        gold = extract_gsm8k_gold(row["answer"])
        ans_text, unans_text, has_ctx = make_abstain_variant(row["question"])
        for s in strategies:
            if has_ctx or include_unpaired_answerable:
                out.append(_ex(f"gsm8k_abstain-{i}-ans-{s}", ans_text, s, True, gold))
            if has_ctx and unans_text:
                out.append(_ex(f"gsm8k_abstain-{i}-unans-{s}", unans_text, s, False, None))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gsm8k_abstain.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--strategies", nargs="*")
    a = ap.parse_args()
    print("wrote", write_jsonl(a.out, strategies=a.strategies, limit=a.limit), "->", a.out)
