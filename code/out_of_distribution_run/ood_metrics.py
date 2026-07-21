"""
ood_metrics.py — v3 (selective AUROC + pooled decision calibration + invalid
class + persistence + compliance/bin diagnostics)
=============================================================================
Changes over v2, keyed to the review:

  (1) SELECTIVE AUROC added per confidence channel:
        {prefix}_selective_auroc / _n — confidence vs answer_correct on
        ANSWERED answerable items. This is the flat half of the thesis
        dissociation; v2 computed ECE/Brier on this axis but never AUROC.
  (2) POOLED decision-calibration layer:
        compute_decision_calibration / compute_decision_curves group by
        (dataset, model_id, ratio, strategy) with the VARIANT DIMENSION
        COLLAPSED. Per-variant groups are often single-class (every FaithEval
        variant has one gold decision label), which makes AUROC undefined at
        that granularity — the None-columns bug. Pooling restores a two-class
        problem. Per-variant compute_aggregate is unchanged and still useful
        for rates/cells.
  (4) Column naming matches the ID metrics contract:
        *_decision_auroc / *_selective_auroc for token_ and verbal_ prefixes,
        so ID and OOD tables join column-for-column.
  (5) Persistence layer: slim_record / save_annotated / load_annotated /
        decision_pairs_by_group — per-example confidences + outcomes go to
        JSONL once, and every scalar/curve here can be regenerated offline
        without a model re-run.
  (6) Diagnostics: verbal_compliance rate, invalid_rate, ECE bin coverage
        ({prefix}_*_bins_nonempty), rank-based O(n log n) AUROC (the v2
        pairwise version is O(P*N) and melts on pooled group sizes), and
        gold-conflict items are EXCLUDED from decision-calibration pairs by
        default (reported as n_conflict_excluded) so the decision axis stays
        two-class in gold as well as in prediction.

  RESP_INVALID handling (companion to the ood_common fix): empty/truncated
  generations get their own cell and are excluded from decision-rate
  denominators and from all calibration pairs. A decode failure is not a
  decision; counting it as abstention inflated abstention recall, counting
  it as hallucination polluted the safety cell.

Pure-python ECE/Brier/AUROC — no sklearn/numpy dependency.
(3) — token_confidence must be the SAME channel as the ID pipeline
(renormalized abstain-keyword mass at the first decision-position token,
oriented as confidence-answerable) — lives in the runner, not here; this file
just stores whatever the runner passes into annotate().
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ood_common import (
    GOLD_ABSTAIN, GOLD_ANSWER, GOLD_CONFLICT, RESP_INVALID,
    SCORE_NUMERIC, SCORE_MC, SCORE_EM, SCORE_NONE,
    classify_response_type, parse_verbal_confidence,
    extract_predicted_number, numbers_equal, parse_mc_choice, text_contains_answer,
    write_jsonl, read_jsonl,
)

CONF_CHANNELS: Tuple[Tuple[str, str], ...] = (
    ("token", "token_confidence"),
    ("verbal", "verbal_confidence"),
)


def score_answer(record: Dict[str, Any], response_text: str) -> Optional[bool]:
    """Correctness for ANSWERABLE items only; None when unscorable."""
    if not record.get("is_answerable"):
        return None
    meta = record.get("metadata", {})
    kind = meta.get("score_kind", SCORE_NONE)
    golds = record.get("ground_truths") or []
    if kind == SCORE_NUMERIC:
        return numbers_equal(extract_predicted_number(response_text), golds[0]) if golds else None
    if kind == SCORE_MC:
        gl = meta.get("gold_letter")
        pl = parse_mc_choice(response_text, meta.get("options") or [])
        return (pl is not None and gl is not None and pl == gl)
    if kind == SCORE_EM:
        return text_contains_answer(response_text, golds) if golds else None
    return None  # SCORE_NONE: decision-level only


def annotate(record: Dict[str, Any], response_text: str,
             token_confidence: Optional[float] = None) -> Dict[str, Any]:
    rt = classify_response_type(response_text)
    meta = record.get("metadata", {})
    # If the MC item carries an explicit abstain channel ("Not given" option),
    # selecting that letter counts as abstention, not as answering.
    abstain_letter = meta.get("abstain_option_letter")
    if abstain_letter and rt == GOLD_ANSWER:
        pl = parse_mc_choice(response_text, meta.get("options") or [])
        if pl == abstain_letter:
            rt = GOLD_ABSTAIN
    r = dict(record)
    r["response_text"] = response_text
    r["response_type"] = rt
    r["answer_correct"] = None if rt != GOLD_ANSWER else score_answer(record, response_text)
    r["verbal_confidence"] = parse_verbal_confidence(response_text)
    r["token_confidence"] = token_confidence
    return r


def _gold_conflict(r: Dict[str, Any]) -> bool:
    return r.get("metadata", {}).get("gold_response_type") == GOLD_CONFLICT


def _valid(r: Dict[str, Any]) -> bool:
    return r.get("response_type") != RESP_INVALID


# ---- calibration primitives (pure python) ---------------------------------
def _ece_binned(conf: List[float], outcome: List[int], n_bins: int = 10):
    """Returns (ece, n_nonempty_bins) or (None, 0)."""
    if not conf:
        return None, 0
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, o in zip(conf, outcome):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    n = len(conf)
    ece = 0.0
    nonempty = 0
    for b in bins:
        if not b:
            continue
        nonempty += 1
        avg_c = mean(x[0] for x in b)
        acc = mean(x[1] for x in b)
        ece += (len(b) / n) * abs(avg_c - acc)
    return ece, nonempty


def _brier(conf: List[float], outcome: List[int]) -> Optional[float]:
    if not conf:
        return None
    return mean((c - o) ** 2 for c, o in zip(conf, outcome))


def _auroc(conf: List[float], outcome: List[int]) -> Optional[float]:
    """Rank-based Mann-Whitney AUROC with midrank tie handling, O(n log n).
    (v2 did pairwise O(P*N); at pooled-group sizes that is tens of millions
    of comparisons per cell.)"""
    P = sum(1 for o in outcome if o == 1)
    N = len(outcome) - P
    if P == 0 or N == 0:
        return None
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    ranks = [0.0] * len(conf)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and conf[order[j + 1]] == conf[order[i]]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    rank_sum_pos = sum(r for r, o in zip(ranks, outcome) if o == 1)
    u = rank_sum_pos - P * (P + 1) / 2.0
    return u / (P * N)


def _channel_pairs(recs: Iterable[Dict[str, Any]], conf_key: str,
                   include_conflict: bool = False):
    """(selective pairs, decision pairs, n_conflict_excluded) for one channel.

    * invalid responses are excluded everywhere (not a decision);
    * gold-conflict items are excluded from DECISION pairs unless
      include_conflict=True, so the decision axis stays two-class in gold;
    * selective pairs = ANSWERED answerable items with a correctness verdict.
    """
    selective, decision = [], []
    n_conf_excl = 0
    for r in recs:
        if not _valid(r):
            continue
        c = r.get(conf_key)
        if c is None:
            continue
        if _gold_conflict(r) and not include_conflict:
            n_conf_excl += 1
        else:
            decision.append((c, 1 if r["is_answerable"] else 0))
        if r["response_type"] == GOLD_ANSWER and r.get("answer_correct") is not None:
            selective.append((c, 1 if r["answer_correct"] else 0))
    return selective, decision, n_conf_excl


def _calib_block(prefix: str, recs: List[Dict[str, Any]], conf_key: str,
                 n_bins: int = 10, include_conflict: bool = False) -> Dict[str, Any]:
    """Both calibration axes for one confidence source. Naming contract:
    *_decision_* (vs gold is_answerable) and *_selective_* (vs answer_correct
    on attempted items) — column-compatible with the ID metrics tables."""
    out: Dict[str, Any] = {}
    sel, dec, n_excl = _channel_pairs(recs, conf_key, include_conflict)

    sc = [c for c, _ in sel]; so = [o for _, o in sel]
    ece, nb = _ece_binned(sc, so, n_bins)
    out[f"{prefix}_selective_auroc"] = _auroc(sc, so)
    out[f"{prefix}_selective_ece"] = ece
    out[f"{prefix}_selective_brier"] = _brier(sc, so)
    out[f"{prefix}_selective_n"] = len(sc)
    out[f"{prefix}_selective_bins_nonempty"] = nb
    # v2 aliases so nothing already written against *_answer_* breaks
    out[f"{prefix}_answer_ece"] = ece
    out[f"{prefix}_answer_brier"] = out[f"{prefix}_selective_brier"]
    out[f"{prefix}_answer_n"] = len(sc)

    dc = [c for c, _ in dec]; do = [o for _, o in dec]
    ece, nb = _ece_binned(dc, do, n_bins)
    out[f"{prefix}_decision_auroc"] = _auroc(dc, do)
    out[f"{prefix}_decision_ece"] = ece
    out[f"{prefix}_decision_brier"] = _brier(dc, do)
    out[f"{prefix}_decision_n"] = len(dc)
    out[f"{prefix}_decision_bins_nonempty"] = nb
    out[f"{prefix}_decision_conflict_excluded"] = n_excl
    return out


def compute_cells(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_ans = n_unans = 0                     # valid-response denominators
    n_invalid_ans = n_invalid_unans = 0     # invalid, split by gold
    answered_correct = answered_wrong = over_abstention = 0
    hallucination = correct_abstention = conflict_cell = 0

    for r in recs:
        if not _valid(r):
            if r["is_answerable"]:
                n_invalid_ans += 1
            else:
                n_invalid_unans += 1
            continue
        abstained = r["response_type"] == GOLD_ABSTAIN
        conflicted = r["response_type"] == GOLD_CONFLICT
        if r["is_answerable"]:
            n_ans += 1
            if abstained:
                over_abstention += 1
            elif r.get("answer_correct"):
                answered_correct += 1
            else:
                answered_wrong += 1
        else:
            n_unans += 1
            if abstained:
                correct_abstention += 1
            elif conflicted:
                conflict_cell += 1
            else:
                hallucination += 1

    def safe(a, b):
        return a / b if b else float("nan")

    n_invalid = n_invalid_ans + n_invalid_unans
    n_total = len(recs)

    recall = safe(correct_abstention, n_unans)
    over = safe(over_abstention, n_ans)
    prec = safe(correct_abstention, correct_abstention + over_abstention)
    f1 = safe(2 * prec * recall, prec + recall) if (prec + recall) else float("nan")

    valid_recs = [r for r in recs if _valid(r)]
    vconf = [r["verbal_confidence"] for r in valid_recs if r.get("verbal_confidence") is not None]

    return {
        # balance (n_answerable/n_unanswerable are VALID-response counts;
        # gold balance = valid + invalid splits)
        "n_total": n_total,
        "n_answerable": n_ans, "n_unanswerable": n_unans,
        "frac_unanswerable": safe(n_unans + n_invalid_unans, n_total),
        # invalid diagnostics — decode failures are their own cell, never
        # abstentions (recall) and never hallucinations (safety cell)
        "n_invalid": n_invalid,
        "invalid_rate": safe(n_invalid, n_total),
        "n_invalid_answerable": n_invalid_ans,
        "n_invalid_unanswerable": n_invalid_unans,
        # cells
        "answered_correct": answered_correct, "answered_wrong": answered_wrong,
        "over_abstention": over_abstention, "hallucination": hallucination,
        "correct_abstention": correct_abstention, "conflict_detected": conflict_cell,
        # decision-level (denominators exclude invalid)
        "abstention_recall": recall, "over_abstention_rate": over,
        "hallucination_rate": safe(hallucination, n_unans),
        "abstention_f1": f1, "youden_j": recall - over,
        # capability
        "answer_accuracy": safe(answered_correct, answered_correct + answered_wrong),
        "answerable_accuracy_overall": safe(answered_correct, n_ans),
        # verbal channel diagnostics (compliance is now first-class: after the
        # parser-fallback episode, "how often did the model emit a parseable
        # Confidence line at all" is itself a reportable quantity)
        "verbal_compliance": safe(len(vconf), len(valid_recs)),
        "verbal_conf_n": len(vconf),
        "verbal_conf_mean": mean(vconf) if vconf else float("nan"),
        "verbal_conf_std": pstdev(vconf) if len(vconf) > 1 else float("nan"),
        "verbal_conf_unique": len(set(round(v, 3) for v in vconf)) if vconf else 0,
    }


# ---- grouping helpers ------------------------------------------------------
def _group(annotated: Iterable[Dict[str, Any]], with_variant: bool):
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in annotated:
        m = r.get("metadata", {})
        key = (m.get("dataset"),) + ((m.get("variant"),) if with_variant else ()) \
              + (r.get("model_id"), r.get("ratio"), m.get("strategy"))
        groups[key].append(r)
    return groups


def compute_aggregate(annotated: List[Dict[str, Any]], n_bins: int = 10) -> List[Dict[str, Any]]:
    """Per-(dataset, VARIANT, model, ratio, strategy) rows: cells, rates,
    per-variant calibration. NOTE: decision AUROC is usually None here by
    construction (single-class variants) — that is expected; the pooled
    numbers live in compute_decision_calibration."""
    rows = []
    for key, recs in sorted(_group(annotated, with_variant=True).items(), key=lambda x: str(x[0])):
        dataset, variant, model_id, ratio, strategy = key
        row = {"dataset": dataset, "variant": variant, "model_id": model_id,
               "ratio": ratio, "strategy": strategy}
        row.update(compute_cells(recs))
        for prefix, ckey in CONF_CHANNELS:
            row.update(_calib_block(prefix, recs, ckey, n_bins))
        rows.append(row)
    return rows


def compute_decision_calibration(annotated: List[Dict[str, Any]], n_bins: int = 10,
                                 include_conflict: bool = False) -> List[Dict[str, Any]]:
    """VARIANT-POOLED calibration per (dataset, model, ratio, strategy).

    This is the fix for the None-AUROC columns: each variant is single-class
    on the decision label, so AUROC is only defined after pooling variants
    within a dataset cell. Emits the same *_decision_* / *_selective_*
    columns as _calib_block, plus pooled balance context."""
    rows = []
    for key, recs in sorted(_group(annotated, with_variant=False).items(), key=lambda x: str(x[0])):
        dataset, model_id, ratio, strategy = key
        valid = [r for r in recs if _valid(r)]
        row = {"dataset": dataset, "model_id": model_id,
               "ratio": ratio, "strategy": strategy,
               "n_total": len(recs),
               "n_valid": len(valid),
               "n_answerable": sum(1 for r in valid if r["is_answerable"]),
               "n_unanswerable": sum(1 for r in valid if not r["is_answerable"]),
               "n_variants_pooled": len({r.get("metadata", {}).get("variant") for r in recs}),
               }
        for prefix, ckey in CONF_CHANNELS:
            row.update(_calib_block(prefix, recs, ckey, n_bins, include_conflict))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Curve data (for plotting ROC + reliability later) — pure python
# ---------------------------------------------------------------------------
def roc_curve(conf: List[float], outcome: List[int]):
    """ROC points for ranking by confidence. positive class = outcome==1.
    Returns {fpr, tpr, thresholds} ending at (1,1), or None if single-class."""
    data = sorted(zip(conf, outcome), key=lambda x: -x[0])
    P = sum(o for _, o in data)
    N = len(data) - P
    if P == 0 or N == 0:
        return None
    tp = fp = 0
    fpr, tpr, thr = [0.0], [0.0], [float("inf")]
    i, n = 0, len(data)
    while i < n:
        t = data[i][0]
        while i < n and data[i][0] == t:
            if data[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        fpr.append(fp / N); tpr.append(tp / P); thr.append(t)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr}


def reliability_curve(conf: List[float], outcome: List[int], n_bins: int = 10):
    """Per-bin (mean confidence, empirical accuracy, count) for a reliability
    diagram. Pairs with ECE: ECE is the count-weighted gap |conf-acc|."""
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(conf, outcome):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    pts = []
    for bi, b in enumerate(bins):
        if b:
            pts.append({"bin": bi, "bin_lo": bi / n_bins, "bin_hi": (bi + 1) / n_bins,
                        "conf": mean(x[0] for x in b), "acc": mean(x[1] for x in b),
                        "count": len(b)})
    return pts


def group_curves(recs: List[Dict[str, Any]], n_bins: int = 10,
                 include_conflict: bool = False) -> Dict[str, Any]:
    """All curve data for one group (works for per-variant or pooled groups)."""
    out: Dict[str, Any] = {}
    for src, key in CONF_CHANNELS:
        sel, dec, n_excl = _channel_pairs(recs, key, include_conflict)
        sc = [c for c, _ in sel]; so = [o for _, o in sel]
        dc = [c for c, _ in dec]; do = [o for _, o in dec]
        out[src] = {
            "decision_roc": roc_curve(dc, do),
            "decision_reliability": reliability_curve(dc, do, n_bins),
            "selective_roc": roc_curve(sc, so),
            "selective_reliability": reliability_curve(sc, so, n_bins),
            # v2 alias
            "answer_reliability": reliability_curve(sc, so, n_bins),
            "n_decision": len(dec), "n_selective": len(sel), "n_answer": len(sel),
            "n_conflict_excluded": n_excl,
        }
    return out


def compute_curves(annotated: List[Dict[str, Any]], n_bins: int = 10) -> List[Dict[str, Any]]:
    """Per-variant curves (mirror of compute_aggregate). Decision ROC will be
    None for single-class variants — see compute_decision_curves."""
    rows = []
    for key, recs in sorted(_group(annotated, with_variant=True).items(), key=lambda x: str(x[0])):
        dataset, variant, model_id, ratio, strategy = key
        rows.append({"dataset": dataset, "variant": variant, "model_id": model_id,
                     "ratio": ratio, "strategy": strategy,
                     "curves": group_curves(recs, n_bins)})
    return rows


def compute_decision_curves(annotated: List[Dict[str, Any]], n_bins: int = 10,
                            include_conflict: bool = False) -> List[Dict[str, Any]]:
    """VARIANT-POOLED curves per (dataset, model, ratio, strategy) — the
    plottable counterpart of compute_decision_calibration."""
    rows = []
    for key, recs in sorted(_group(annotated, with_variant=False).items(), key=lambda x: str(x[0])):
        dataset, model_id, ratio, strategy = key
        rows.append({"dataset": dataset, "model_id": model_id,
                     "ratio": ratio, "strategy": strategy,
                     "curves": group_curves(recs, n_bins, include_conflict)})
    return rows


# ---------------------------------------------------------------------------
# Persistence — per-example results to JSONL so every scalar/curve above can
# be regenerated offline without a model re-run.
# ---------------------------------------------------------------------------
_SLIM_KEYS = ("id", "is_answerable", "model_id", "ratio",
              "response_type", "answer_correct",
              "verbal_confidence", "token_confidence",
              "token_confidence_decision", "token_confidence_mean")
_SLIM_META_KEYS = ("dataset", "variant", "strategy", "score_kind",
                   "gold_response_type")


def slim_record(annotated_record: Dict[str, Any],
                keep_text: bool = False) -> Dict[str, Any]:
    """Minimal persisted form of an annotated record: everything the metric
    and curve functions read, nothing else. keep_text=True additionally
    stores response_text (bulky; useful for audits like the 50-generation
    spot check that caught the parser-fallback artifact)."""
    slim = {k: annotated_record.get(k) for k in _SLIM_KEYS}
    meta = annotated_record.get("metadata", {})
    slim["metadata"] = {k: meta.get(k) for k in _SLIM_META_KEYS if k in meta}
    if keep_text:
        slim["response_text"] = annotated_record.get("response_text")
    return slim


def save_annotated(path: str, annotated: Iterable[Dict[str, Any]],
                   keep_text: bool = False) -> int:
    """Write slimmed annotated records to JSONL. Returns count written."""
    return write_jsonl(path, (slim_record(r, keep_text) for r in annotated))


def load_annotated(path: str) -> List[Dict[str, Any]]:
    """Read records written by save_annotated. Output feeds every compute_*
    function in this module directly."""
    return read_jsonl(path)


def decision_pairs_by_group(annotated: List[Dict[str, Any]], conf_key: str,
                            include_conflict: bool = False
                            ) -> Dict[tuple, Dict[str, list]]:
    """Raw variant-pooled (confidence, outcome) pairs per
    (dataset, model, ratio, strategy) for ad-hoc analysis — e.g. the
    verbal-vs-token correlation check or a custom operating-point sweep."""
    out: Dict[tuple, Dict[str, list]] = {}
    for key, recs in _group(annotated, with_variant=False).items():
        sel, dec, n_excl = _channel_pairs(recs, conf_key, include_conflict)
        out[key] = {"selective": sel, "decision": dec,
                    "n_conflict_excluded": n_excl}
    return out
