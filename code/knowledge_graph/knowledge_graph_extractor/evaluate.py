"""
Evaluate KG extraction predictions against gold triples.

Computes:
  - Triple-level Precision/Recall/F1 (exact match on head + relation + tail)
  - Entity-level P/R/F1 (set of entity strings across triples)
  - Relation-only P/R/F1 (which relations got captured at all)
"""
import argparse
import json
import re
from pathlib import Path


def normalize_str(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_relation(r: str) -> str:
    r = r.strip().lower()
    r = re.sub(r"\s+", "_", r)
    return r


def triple_key(t: dict) -> tuple[str, str, str]:
    return (
        normalize_str(t["head"]),
        normalize_relation(t["relation"]),
        normalize_str(t["tail"]),
    )


def entity_set(triples: list[dict]) -> set[str]:
    return {normalize_str(t[k]) for t in triples for k in ("head", "tail")}


def relation_set(triples: list[dict]) -> set[str]:
    return {normalize_relation(t["relation"]) for t in triples}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path,
                        help="JSONL with both 'triples' (gold) and 'predicted_triples'")
    args = parser.parse_args()

    triple_tp = triple_fp = triple_fn = 0
    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    n = 0
    n_empty_pred = 0
    n_empty_gold = 0

    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            gold = ex.get("triples", []) or []
            pred = ex.get("predicted_triples", []) or []

            gold_t = {triple_key(t) for t in gold if all(k in t for k in ("head","relation","tail"))}
            pred_t = {triple_key(t) for t in pred if all(k in t for k in ("head","relation","tail"))}
            triple_tp += len(gold_t & pred_t)
            triple_fp += len(pred_t - gold_t)
            triple_fn += len(gold_t - pred_t)

            ge, pe = entity_set(gold), entity_set(pred)
            ent_tp += len(ge & pe)
            ent_fp += len(pe - ge)
            ent_fn += len(ge - pe)

            gr, pr = relation_set(gold), relation_set(pred)
            rel_tp += len(gr & pr)
            rel_fp += len(pr - gr)
            rel_fn += len(gr - pr)

            if not pred:
                n_empty_pred += 1
            if not gold:
                n_empty_gold += 1
            n += 1

    print(f"Examples evaluated: {n}")
    print(f"  empty gold:        {n_empty_gold}")
    print(f"  empty predictions: {n_empty_pred}")
    print()
    for label, (tp, fp, fn) in [
        ("Triple (head + relation + tail)", (triple_tp, triple_fp, triple_fn)),
        ("Entity                          ", (ent_tp, ent_fp, ent_fn)),
        ("Relation                        ", (rel_tp, rel_fp, rel_fn)),
    ]:
        p, r, f1 = prf(tp, fp, fn)
        print(f"{label}   P={p:.3f}  R={r:.3f}  F1={f1:.3f}   (tp={tp} fp={fp} fn={fn})")


if __name__ == "__main__":
    main()
