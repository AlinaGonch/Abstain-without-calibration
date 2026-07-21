"""HotpotQA 
Multi-passage QA
Unanswerable construction modes (remove the evidence the answer depends on):
  "drop_paragraphs"   (default) delete the whole supporting paragraph(s)
  "drop_sentences"    keep paragraphs but delete ONLY the supporting sentences
  "replace_sentences" keep paragraphs but REPLACE supporting sentences with a
                      neutral filler (information removed, length preserved).
In all modes: is_answerable=False, gold=["I don't know"]."""

from typing import List, Optional, Tuple
from datasets import load_dataset
import argparse
from ood_common import CanonicalExample, gold_for, write_jsonl, SCORE_EM, GOLD_ANSWER, GOLD_ABSTAIN
from prompt_template import get_chat_messages, PROMPT_STRATEGIES_LIST

SOURCE_REPO, SOURCE_CONFIG, SOURCE_SPLIT = "hotpotqa/hotpot_qa", "distractor", "validation"
DEFAULT_STRATEGIES = ["zero_shot_passages", "cot_multihop", "self_reflect_passages"]
FILLER = "[information removed]"


def _parse_context(row):
    """Return (paragraphs, title_to_idx). paragraphs = list of (title, [sentences])."""
    ctx = row.get("context") or {}
    titles = ctx.get("title") or []
    sents = ctx.get("sentences") or []
    paragraphs = [(t, list(s)) for t, s in zip(titles, sents)]
    return paragraphs, {t: i for i, t in enumerate(titles)}


def _supporting(row) -> List[Tuple[str, int]]:
    sf = row.get("supporting_facts") or {}
    return list(zip(sf.get("title") or [], sf.get("sent_id") or []))


def _passages_from(paragraphs) -> List[dict]:
    return [{"title": t, "text": " ".join(s).strip()} for t, s in paragraphs if " ".join(s).strip()]


def _make_unanswerable(paragraphs, title_to_idx, supporting, mode):
    paras = [(t, list(s)) for t, s in paragraphs]  # deep-ish copy
    sup_titles = {t for t, _ in supporting}
    if mode == "drop_paragraphs":
        reduced = [(t, s) for t, s in paras if t not in sup_titles]
        return reduced if (reduced and len(reduced) < len(paras)) else None
    # sentence-level modes
    changed = False
    for title, sid in supporting:
        idx = title_to_idx.get(title)
        if idx is None or not (0 <= sid < len(paras[idx][1])):
            continue
        if mode == "drop_sentences":
            paras[idx][1][sid] = ""
        elif mode == "replace_sentences":
            paras[idx][1][sid] = FILLER
        changed = True
    if not changed:
        return None
    # prune emptied sentences for drop_sentences
    cleaned = [(t, [x for x in s if x != ""]) for t, s in paras]
    return cleaned


def load(strategies: Optional[List[str]] = None, limit: Optional[int] = None,
         make_unanswerable: bool = True, unanswerable_mode: str = "drop_paragraphs") -> List[CanonicalExample]:
    
    strategies = strategies or DEFAULT_STRATEGIES
    for s in strategies:
        assert s in PROMPT_STRATEGIES_LIST, f"{s} not in your PROMPT_STRATEGIES_LIST"
    ds = load_dataset(SOURCE_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)

    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        question = row.get("question", "")
        answer = row.get("answer", "")
        paragraphs, title_to_idx = _parse_context(row)
        if not question or not paragraphs:
            continue
        passages = _passages_from(paragraphs)
        for s in strategies:
            out.append(CanonicalExample(
                id=f"hotpot-{i}-ans-{s}",
                messages=get_chat_messages(s, question=question, passages=passages),
                ground_truths=gold_for(True, [answer], has_gold_string=True), is_answerable=True,
                metadata={"dataset": "hotpotqa", "variant": "answerable", "strategy": s,
                          "score_kind": SCORE_EM, "gold_response_type": GOLD_ANSWER}))
        if make_unanswerable:
            supporting = _supporting(row)
            reduced = _make_unanswerable(paragraphs, title_to_idx, supporting, unanswerable_mode) if supporting else None
            if reduced:
                red_pass = _passages_from(reduced)
                if red_pass:
                    for s in strategies:
                        out.append(CanonicalExample(
                            id=f"hotpot-{i}-unans-{s}",
                            messages=get_chat_messages(s, question=question, passages=red_pass),
                            ground_truths=gold_for(False, None, has_gold_string=False), is_answerable=False,
                            metadata={"dataset": "hotpotqa", "variant": f"unanswerable_{unanswerable_mode}",
                                      "strategy": s, "score_kind": "none",
                                      "gold_response_type": GOLD_ABSTAIN,
                                      "dropped_titles": sorted({t for t, _ in supporting})}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hotpotqa.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--strategies", nargs="*"); ap.add_argument("--no-unanswerable", action="store_true")
    ap.add_argument("--mode", default="drop_paragraphs",
                    choices=["drop_paragraphs", "drop_sentences", "replace_sentences"])
    a = ap.parse_args()
    print("wrote", write_jsonl(a.out, strategies=a.strategies, limit=a.limit,
                            make_unanswerable=not a.no_unanswerable, unanswerable_mode=a.mode), "->", a.out)
