"""
GPQA-Abstain (AbstentionBench). 
Multiple choice QA science .
"""

import random
from typing import List, Optional
import argparse
from datasets import load_dataset
from ood_common import CanonicalExample, make_abstain_variant, gold_for, write_jsonl, SCORE_MC, GOLD_ANSWER, GOLD_ABSTAIN
from ood_prompt_templates import get_ood_chat_messages, KIND_MC, OOD_STRATEGIES

SOURCE_REPO, SOURCE_CONFIG, SOURCE_SPLIT, SEED = "Idavidrein/gpqa", "gpqa_main", "train", 42
_Q, _CORRECT = "Question", "Correct Answer"
_INCORRECT = ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def _ex(uid, stem, options, gold_letter, strat, is_ans, correct):
    return CanonicalExample(
        id=uid,
        messages=get_ood_chat_messages(strat, stem, choices=options, kind=KIND_MC),
        ground_truths=gold_for(is_ans, [correct], has_gold_string=is_ans),
        is_answerable=is_ans,
        metadata={"dataset": "gpqa_abstain", "variant": None, "strategy": strat,
                  "score_kind": SCORE_MC if is_ans else "none",
                  "options": options, "gold_letter": gold_letter if is_ans else None,
                  "gold_response_type": GOLD_ANSWER if is_ans else GOLD_ABSTAIN,
                  "unanswerability_type": None if is_ans else "underspecified_context"},
    )


def load(strategies: Optional[List[str]] = None, config: str = SOURCE_CONFIG,
         limit: Optional[int] = None, include_unpaired_answerable: bool = True) -> List[CanonicalExample]:
    
    strategies = strategies or OOD_STRATEGIES
    ds = load_dataset(SOURCE_REPO, config, split=SOURCE_SPLIT)
    rng = random.Random(SEED)
    out= []
    seen = 0
    for i, row in enumerate(ds):
        if limit is not None and seen >= limit:
            break
        seen += 1
        stem = (row.get(_Q) or "").strip()
        correct = (row.get(_CORRECT) or "").strip()
        options = [correct] + [(row.get(k) or "").strip() for k in _INCORRECT if (row.get(k) or "").strip()]
        if not stem or not correct or len(options) < 2:
            continue
        rng.shuffle(options)
        gl = chr(ord("A") + options.index(correct))
        ans_text, unans_text, has_ctx = make_abstain_variant(stem)
        for s in strategies:
            if has_ctx or include_unpaired_answerable:
                out.append(_ex(f"gpqa_abstain-{i}-ans-{s}", ans_text, options, gl, s, True, correct))
            if has_ctx and unans_text:
                out.append(_ex(f"gpqa_abstain-{i}-unans-{s}", unans_text, options, None, s, False, correct))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpqa_abstain.jsonl"); ap.add_argument("--config", default=SOURCE_CONFIG)
    ap.add_argument("--limit", type=int); ap.add_argument("--strategies", nargs="*")
    a = ap.parse_args()
    print("wrote", write_jsonl(a.out, config=a.config, strategies=a.strategies, limit=a.limit), "->", a.out)
