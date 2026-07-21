"""MMLU-Pro (TIGER-Lab/MMLU-Pro). Hard 10-option MC.

Unanswerable construction = REWRITE the correct option as an abstain channel
(default "Not given"): the genuinely correct answer is removed from the option
list and replaced by "Not given" at the same position. The faithful response is
then to select that option OR say "I don't know" — both count as a correct
abstention (see ood_metrics.annotate, which maps the abstain-option letter to
'abstain'). Selecting any real letter = hallucination.

Modes:
  unanswerable_mode="not_given"   -> replace correct option text with ABSTAIN_OPTION
  unanswerable_mode="drop_correct"-> replace correct option with a wrong distractor,
                                     NO abstain channel (forced-choice; abstain only
                                     via free-text "I don't know")
Each source item yields an answerable item (capability + over-abstention axis)
AND, when build_unanswerable, its unanswerable counterpart (balanced)."""

from typing import List, Optional
import argparse
from datasets import load_dataset
from ood_common import (CanonicalExample, gold_for, write_jsonl, SCORE_MC,
                        GOLD_ANSWER, GOLD_ABSTAIN)
from ood_prompt_templates import get_ood_chat_messages, KIND_MC, OOD_STRATEGIES

SOURCE_REPO, SOURCE_SPLIT = "TIGER-Lab/MMLU-Pro", "test"
ABSTAIN_OPTION = "Not given"


def _answerable_ex(uid, stem, options, gold_letter, strat, correct, category):
    return CanonicalExample(
        id=uid, messages=get_ood_chat_messages(strat, stem, choices=options, kind=KIND_MC),
        ground_truths=gold_for(True, [correct], has_gold_string=True), is_answerable=True,
        metadata={"dataset": "mmlu_pro", "variant": "answerable", "category": category,
                  "strategy": strat, "score_kind": SCORE_MC, "options": options,
                  "gold_letter": gold_letter, "gold_response_type": GOLD_ANSWER})


def _unanswerable_ex(uid, stem, options, strat, correct_idx, category, mode, distractor=None):
    opts = list(options)
    if mode == "not_given":
        opts[correct_idx] = ABSTAIN_OPTION
        abstain_letter = chr(ord("A") + correct_idx)
        unt = "correct_option_replaced_with_not_given"
    else:  # drop_correct: substitute a wrong value, no abstain channel
        opts[correct_idx] = distractor or "(removed)"
        abstain_letter = None
        unt = "correct_option_replaced_with_distractor"
    return CanonicalExample(
        id=uid, messages=get_ood_chat_messages(strat, stem, choices=opts, kind=KIND_MC),
        ground_truths=gold_for(False, None, has_gold_string=False), is_answerable=False,
        metadata={"dataset": "mmlu_pro", "variant": "unanswerable", "category": category,
                  "strategy": strat, "score_kind": "none", "options": opts,
                  "abstain_option_letter": abstain_letter,
                  "gold_response_type": GOLD_ABSTAIN, "unanswerability_type": unt})


def load(strategies: Optional[List[str]] = None, categories: Optional[List[str]] = None,
         limit: Optional[int] = None, build_unanswerable: bool = True,
         unanswerable_mode: str = "not_given") -> List[CanonicalExample]:
    
    strategies = strategies or OOD_STRATEGIES
    ds = load_dataset(SOURCE_REPO, split=SOURCE_SPLIT)
    out: List[CanonicalExample] = []
    seen = 0
    for i, row in enumerate(ds):
        if limit is not None and seen >= limit:
            break
        cat = row.get("category")
        if categories and cat not in categories:
            continue
        seen += 1
        stem = (row.get("question") or "").strip()
        options = list(row.get("options") or [])
        ans_letter = (row.get("answer") or "").strip()
        if not stem or len(options) < 2 or not ans_letter:
            continue
        correct_idx = (ord(ans_letter) - ord("A")) if len(ans_letter) == 1 else int(row.get("answer_index", 0))
        if not (0 <= correct_idx < len(options)):
            continue
        correct = options[correct_idx]
        gl = chr(ord("A") + correct_idx)
        # a wrong distractor for drop_correct mode = any other option
        distractor = next((o for j, o in enumerate(options) if j != correct_idx), "(removed)")
        for s in strategies:
            out.append(_answerable_ex(f"mmlu_pro-{i}-ans-{s}", stem, options, gl, s, correct, cat))
            if build_unanswerable:
                out.append(_unanswerable_ex(f"mmlu_pro-{i}-unans-{s}", stem, options, s,
                                            correct_idx, cat, unanswerable_mode, distractor))
    return out


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmlu_pro.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--strategies", nargs="*"); ap.add_argument("--categories", nargs="*")
    ap.add_argument("--no-unanswerable", action="store_true")
    ap.add_argument("--mode", default="not_given", choices=["not_given", "drop_correct"])
    a = ap.parse_args()
    print("wrote", write_jsonl(a.out, strategies=a.strategies, limit=a.limit, categories=a.categories,
                            build_unanswerable=not a.no_unanswerable, unanswerable_mode=a.mode), "->", a.out)
