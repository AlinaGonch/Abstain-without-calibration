"""
ood_metrics.py — v2 (four-cell + decision-level + CONFIDENCE CALIBRATION)
=========================================================================
Answers the "do we collect what's needed for confidence estimation + balances?"
question. We now collect and score, per example:

  detected response_type  (answer | abstain | conflict)
  answer_correct          (bool | None)
  verbal_confidence       (0-1 | None)   <- from self_reflect "Confidence:"
  token_confidence        (0-1 | None)   <- mean token prob, captured by runner

and compute, per (dataset, model, ratio, strategy):

  * four-cell taxonomy (answered_correct/wrong, hallucination, over_abstention,
    correct_abstention) + conflict cell
  * decision-level: abstention_recall (TPR), over_abstention_rate (FPR),
    hallucination_rate, abstention_f1, youden_j
  * capability: answer_accuracy, answerable_accuracy_overall
  * CALIBRATION (two distinct axes your thesis separates):
      - answer-correctness ECE/Brier  (token-level signal on ANSWERED items)
      - abstention-DECISION ECE/Brier/AUROC (confidence vs gold is_answerable)
    computed for verbal_ and token_ confidence independently.
  * verbal confidence COLLAPSE diagnostics: mean/std/unique of verbal_confidence
  * BALANCE: n_answerable/n_unanswerable, frac_unanswerable, calibration n + bin
    coverage (so you can see when ECE is computed on too few points).

Pure-python ECE/Brier/AUROC — no sklearn/numpy dependency.
"""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from ood_common import (
    GOLD_ABSTAIN, GOLD_ANSWER, GOLD_CONFLICT,
    SCORE_NUMERIC, SCORE_MC, SCORE_EM, SCORE_NONE,
    classify_response_type, parse_verbal_confidence,
    extract_predicted_number, numbers_equal, parse_mc_choice, text_contains_answer,
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
    # An empty emission is an explicit refusal to answer for this evaluation.
    rt = GOLD_ABSTAIN if not (response_text or "").strip() else classify_response_type(response_text)
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


# ---- calibration primitives (pure python) ---------------------------------
def _ece(conf: List[float], outcome: List[int], n_bins: int = 10) -> Optional[float]:
    if not conf:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(conf, outcome):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, o))
    n = len(conf)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = mean(x[0] for x in b)
        acc = mean(x[1] for x in b)
        ece += (len(b) / n) * abs(avg_c - acc)
    return ece


def _brier(conf: List[float], outcome: List[int]) -> Optional[float]:
    if not conf:
        return None
    return mean((c - o) ** 2 for c, o in zip(conf, outcome))


def _auroc(conf: List[float], outcome: List[int]) -> Optional[float]:
    pos = [c for c, o in zip(conf, outcome) if o == 1]
    neg = [c for c, o in zip(conf, outcome) if o == 0]
    if not pos or not neg:
        return None
    # Mann-Whitney U / (|pos||neg|)
    wins = 0.0
    for p in pos:
        for ng in neg:
            wins += 1.0 if p > ng else (0.5 if p == ng else 0.0)
    return wins / (len(pos) * len(neg))


def _auprc(scores: List[float], outcome: List[int]) -> Optional[float]:
    positives = sum(outcome)
    if not positives or positives == len(outcome):
        return None
    ordered = sorted(zip(scores, outcome), key=lambda pair: pair[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


_TOKEN_FIELDS = ("answer_mean_logprob", "answer_min_logprob", "answer_mean_entropy",
                 "answer_max_entropy", "answer_mean_top1_top2_margin",
                 "answer_min_top1_top2_margin", "answer_first_token_entropy",
                 "answer_first_token_margin", "decision_refusal_prefix_first_token_mass",
                 "decision_refusal_prefix_vs_best_other_logit_margin",
                 "decision_refusal_prefix_vs_other_logsumexp_margin")


def _values(recs, field):
    return [float(r[field]) for r in recs if r.get(field) is not None and math.isfinite(float(r[field]))]


def _distribution(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "median": None, "std": None, "p25": None, "p75": None}
    def quantile(q):
        pos = (n - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": n, "mean": mean(values), "median": quantile(.5),
            "std": pstdev(values) if n > 1 else 0.0,
            "p25": quantile(.25), "p75": quantile(.75)}


def _token_diagnostic_block(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """OOD analogue of ID diagnostics; raw uncertainty is rank-only, never ECE."""
    groups = {
        "correct_answer": [r for r in recs if r["response_type"] == GOLD_ANSWER and r.get("answer_correct") is True],
        "incorrect_answer": [r for r in recs if r["response_type"] == GOLD_ANSWER and r.get("answer_correct") is False],
        "correct_abstention": [r for r in recs if not r["is_answerable"] and r["response_type"] == GOLD_ABSTAIN],
        "false_abstention": [r for r in recs if r["is_answerable"] and r["response_type"] == GOLD_ABSTAIN],
    }
    out = {"token_diagnostic_groups": {name: {field: _distribution(_values(items, field))
                                                for field in _TOKEN_FIELDS}
                                       for name, items in groups.items()}}

    attempted = [r for r in recs if r["response_type"] == GOLD_ANSWER and r.get("answer_correct") is not None]
    for key, field, invert in (
        ("token_entropy_selective", "answer_mean_entropy", False),
        ("first_token_entropy_selective", "answer_first_token_entropy", False),
        ("first_token_margin_selective", "answer_first_token_margin", True),
        ("min_token_logprob_selective", "answer_min_logprob", True),
    ):
        valid = [r for r in attempted if r.get(field) is not None and math.isfinite(float(r[field]))]
        scores = [(-float(r[field]) if invert else float(r[field])) for r in valid]
        labels = [0 if r["answer_correct"] else 1 for r in valid]  # 1 = incorrect; higher = uncertain
        out[key + "_auroc"] = _auroc(scores, labels)
        out[key + "_auprc"] = _auprc(scores, labels)
        out[key + "_n_valid"] = len(valid)

    for key, field in (("decision_refusal_prefix_mass_answerability", "decision_refusal_prefix_first_token_mass"),
                       ("decision_refusal_prefix_margin_answerability",
                        "decision_refusal_prefix_vs_best_other_logit_margin"),
                       ("decision_refusal_prefix_logsumexp_margin_answerability",
                        "decision_refusal_prefix_vs_other_logsumexp_margin")):
        valid = [r for r in recs if r.get(field) is not None and math.isfinite(float(r[field]))]
        scores = [float(r[field]) for r in valid]
        labels = [0 if r["is_answerable"] else 1 for r in valid]  # 1 = gold unanswerable
        out[key + "_auroc"] = _auroc(scores, labels)
        out[key + "_auprc"] = _auprc(scores, labels)
        out[key + "_n_valid"] = len(valid)
    return out


def _calib_block(prefix: str, recs: List[Dict[str, Any]], conf_key: str) -> Dict[str, Any]:
    """Two calibration axes for one confidence source."""
    out: Dict[str, Any] = {}

    # axis 1: answer-correctness (token-level ECE), on ANSWERED answerable items
    ac, ao = [], []
    for r in recs:
        if r["response_type"] == GOLD_ANSWER and r.get("answer_correct") is not None \
                and r.get(conf_key) is not None:
            ac.append(r[conf_key]); ao.append(1 if r["answer_correct"] else 0)
    out[f"{prefix}_answer_ece"] = _ece(ac, ao)
    out[f"{prefix}_answer_brier"] = _brier(ac, ao)
    out[f"{prefix}_answer_n"] = len(ac)

    # axis 2: abstention-decision — confidence(can answer) vs gold is_answerable
    dc, do = [], []
    for r in recs:
        if r.get(conf_key) is not None:
            dc.append(r[conf_key]); do.append(1 if r["is_answerable"] else 0)
    out[f"{prefix}_decision_ece"] = _ece(dc, do)
    out[f"{prefix}_decision_brier"] = _brier(dc, do)
    out[f"{prefix}_decision_auroc"] = _auroc(dc, do)
    out[f"{prefix}_decision_n"] = len(dc)
    return out


def compute_cells(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_ans = n_unans = 0
    answered_correct = answered_wrong = answered_unscorable = over_abstention = 0
    hallucination = correct_abstention = conflict_cell = 0

    for r in recs:
        abstained = r["response_type"] == GOLD_ABSTAIN
        conflicted = r["response_type"] == GOLD_CONFLICT
        if r["is_answerable"]:
            n_ans += 1
            if abstained:
                over_abstention += 1
            elif r.get("answer_correct") is True:
                answered_correct += 1
            elif r.get("answer_correct") is False:
                answered_wrong += 1
            else:
                answered_unscorable += 1
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

    recall = safe(correct_abstention, n_unans)
    over = safe(over_abstention, n_ans)
    prec = safe(correct_abstention, correct_abstention + over_abstention)
    f1 = safe(2 * prec * recall, prec + recall) if (prec + recall) else float("nan")

    vconf = [r["verbal_confidence"] for r in recs if r.get("verbal_confidence") is not None]
    marker_values = [r.get("answer_span_marker_found") for r in recs
                     if r.get("answer_span_marker_found") is not None]
    empty_outputs = sum(not (r.get("response_text") or "").strip() for r in recs)
    # Only non-abstaining answer attempts are candidates for answer scoring.
    # `None` is unscorable, not incorrect (e.g. a decision-only benchmark).
    answer_attempts = [r for r in recs if r["response_type"] == GOLD_ANSWER]
    unscorable = sum(r.get("answer_correct") is None for r in answer_attempts)

    return {
        # balance
        "n_total": len(recs), "n_answerable": n_ans, "n_unanswerable": n_unans,
        "frac_unanswerable": safe(n_unans, n_ans + n_unans),
        "answer_span_marker_rate": mean(marker_values) if marker_values else float("nan"),
        "answer_span_marker_rate_n_valid": len(marker_values),
        "empty_output_rate": safe(empty_outputs, len(recs)),
        "unscorable_rate": safe(unscorable, len(answer_attempts)),
        "unscorable_rate_n_valid": len(answer_attempts),
        # cells
        "answered_correct": answered_correct, "answered_wrong": answered_wrong,
        "answered_unscorable": answered_unscorable,
        "over_abstention": over_abstention, "hallucination": hallucination,
        "correct_abstention": correct_abstention, "conflict_detected": conflict_cell,
        # decision-level
        "abstention_recall": recall, "over_abstention_rate": over,
        "hallucination_rate": safe(hallucination, n_unans),
        "abstention_f1": f1, "youden_j": recall - over,
        # capability
        "answer_accuracy": safe(answered_correct, answered_correct + answered_wrong),
        # Unscorable answers (answer_correct=None) are reported separately and
        # excluded from correctness denominators, never silently called wrong.
        "answerable_accuracy_overall": safe(answered_correct, answered_correct + answered_wrong),
        # verbal-confidence collapse diagnostics
        "verbal_conf_n": len(vconf),
        "verbal_conf_mean": mean(vconf) if vconf else float("nan"),
        "verbal_conf_std": pstdev(vconf) if len(vconf) > 1 else float("nan"),
        "verbal_conf_unique": len(set(round(v, 3) for v in vconf)) if vconf else 0,
    }


def compute_aggregate(annotated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in annotated:
        m = r.get("metadata", {})
        key = (m.get("dataset"), m.get("variant"), r.get("model_id"), r.get("ratio"), m.get("strategy"))
        groups[key].append(r)

    rows = []
    for (dataset, variant, model_id, ratio, strategy), recs in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {"dataset": dataset, "variant": variant, "model_id": model_id,
               "ratio": ratio, "strategy": strategy}
        row.update(compute_cells(recs))
        row.update(_calib_block("verbal", recs, "verbal_confidence"))
        row.update(_calib_block("token", recs, "token_confidence"))
        row.update(_token_diagnostic_block(recs))
        rows.append(row)
    return rows
