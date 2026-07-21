"""NoMIRACL — multilingual relevance/abstention. Multi-passage -> YOUR *_passages.
Decision-level only (no gold answer strings):
  non_relevant subset -> is_answerable=False, gold=["I don't know"]
  relevant subset     -> is_answerable=True,  gold=[]  (no EM; decision-level)
Per-language abstain keywords are handled by your judge; the contract abstain
string is language-agnostic here (prompts are English-instruction)."""

from typing import List, Optional
from ood_common import CanonicalExample, gold_for, write_jsonl, GOLD_ANSWER, GOLD_ABSTAIN
from prompt_template import get_chat_messages, PROMPT_STRATEGIES_LIST

SOURCE_REPO = "miracl/nomiracl-instruct"
DEFAULT_LANGS = ["en"]
DEFAULT_STRATEGIES = ["zero_shot_passages", "cot_passages", "self_reflect_passages"]


def _g(row, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _passages(row) -> List[dict]:
    docs = _g(row, "positive_passages", "negative_passages", "docs", "passages", "candidates", default=[])
    out = []
    for d in (docs or []):
        if isinstance(d, str):
            out.append({"text": d})
        else:
            out.append({"title": _g(d, "title", default=""),
                        "text": _g(d, "text", "passage", "contents", default="")})
    return out

def extract_question(row):
    suffix = row.split('CONTEXTS:')[0]
    _, _, question = suffix.partition('QUESTION:')
    return question

def extract_context(row):
    return row.split('CONTEXTS:')[1]

def load(langs: Optional[List[str]] = None, subsets: Optional[List[str]] = None,
         strategies: Optional[List[str]] = None, limit: Optional[int] = None) -> List[CanonicalExample]:
    from datasets import load_dataset
    langs = langs or DEFAULT_LANGS
    subsets = subsets or ["relevant", "non_relevant"]
    strategies = strategies or DEFAULT_STRATEGIES
    for s in strategies:
        assert s in PROMPT_STRATEGIES_LIST, f"{s} not in your PROMPT_STRATEGIES_LIST"

    out = []
    for lang in langs:
        for subset in subsets:
            try:
                ds = load_dataset(SOURCE_REPO, split='test')
            except Exception:
                # some releases expose split names like "<lang>.<subset>"; adjust if needed
                ds = load_dataset(SOURCE_REPO, split='test')
            is_ans = (subset == "relevant")
            for i, row in enumerate(ds):
                if limit is not None and i >= limit:
                    break
                question = extract_question(row['messages'][0]['content'])
                passages = extract_context(row['messages'][0]['content'])
                if not question or not passages:
                    continue
                for s in strategies:
                    out.append(CanonicalExample(
                        id=f"nomiracl_{lang}_{subset}-{i}-{s}",
                        messages=get_chat_messages(s, question=question, passages=passages),
                        ground_truths=gold_for(is_ans, None, has_gold_string=False),
                        is_answerable=is_ans,
                        metadata={"dataset": "nomiracl", "variant": f"{lang}/{subset}",
                                  "language": lang, "strategy": s, "score_kind": "none",
                                  "gold_response_type": GOLD_ANSWER if is_ans else GOLD_ABSTAIN},
                    ))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nomiracl.jsonl"); ap.add_argument("--limit", type=int)
    ap.add_argument("--langs", nargs="*"); ap.add_argument("--subsets", nargs="*"); ap.add_argument("--strategies", nargs="*")
    a = ap.parse_args()
    print("wrote",write_jsonl(a.out, langs=a.langs, subsets=a.subsets, strategies=a.strategies, limit=a.limit), "->", a.out)
