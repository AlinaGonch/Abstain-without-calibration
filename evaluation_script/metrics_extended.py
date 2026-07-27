"""
metrics.py — corrected evaluation module for SQuAD2 abstention experiments
===========================================================================
Fixes relative to the previous version (each marked FIX-n in the code):

 FIX-1  `no_ans_f1` was NOT an F1 and silently duplicated `abstention_recall`
        (lines 465 vs 258 of the old file computed the identical quantity).
        It is now named `abstention_recall`; `no_ans_f1` is kept ONLY as a
        documented alias so existing plotting code does not break.

 FIX-2  `overall_f1` mixed two scoring regimes: unanswerable examples were
        scored by token overlap against the literal string "I don't know",
        while the per-class metric used keyword abstention detection. A model
        abstaining with "The answer is not in the context" scored 100% on the
        per-class curve but ~0 in the overall average — the "strictness
        penalty" anomaly. `overall_f1` now uses ONE consistent regime
        (SQuAD2-official style): unanswerable → 1.0 iff is_abstention(pred).
        The old mixed number is still returned as `overall_f1_legacy` so the
        discrepancy can be quantified, never accidentally reported.

 FIX-3  `is_abstention` matched keywords ANYWHERE in the answer, so hedged
        answers ("I don't know the exact date, but it was 1947") counted as
        abstention. Default is now a prefix window (first 80 chars); pass
        window=None to restore the old substring behaviour for comparisons.

 FIX-4  `compute_ece` / `compute_brier_score` silently produced garbage when
        given raw log-probs (negative values fall outside every [0,1] bin →
        ECE ≈ 0). Inputs are now validated, and ECE supports adaptive
        equal-mass binning — essential because greedy-decoding confidences
        cluster near 1 and fixed-width bins sit mostly empty.

 FIX-5  Calibration correctness is computed with the SAME consistent regime
        as FIX-2, reported at two thresholds (F1>0.5 and EM) as a sensitivity
        check, instead of a single arbitrary `is_correct = F1 > 0.5`.

 FIX-6  `compute_bert_score` used only the FIRST reference; it now takes the
        max over all references (matching the F1/EM convention) via
        bert-score's native multi-reference support.

 FIX-7  RougeScorer was re-instantiated per example (pure overhead); it is
        now a cached module-level singleton.

 FIX-8  `compute_abstention_metrics` declared a `ground_truths` parameter it
        never used; the parameter is now optional and documented.

 FIX-9  bert_score / rouge_score / scipy / matplotlib were imported at module
        top, making the whole module fail to import on a machine without
        them. All heavy imports are now lazy (inside the functions).

 FIX-10 Verbalized confidence support: `compute_metrics` accepts an optional
        `verbalized_confidences` array (parsed "Confidence: NN" values in
        [0,1], None where unparseable) and reports its calibration separately
        from logit-based confidence.

 FIX-11 Bootstrap CIs on the overall score, so single-run plateau wiggles can
        be judged against sampling noise.

 FIX-12 Kendall's W: documented that the formula has no tie correction; with
        many tied ranks W is biased downward.

 FIX-13 NumPy 2.x crash: `getattr(np, "trapezoid", np.trapz)` evaluated the
        default eagerly and `np.trapz` was REMOVED in NumPy 2.0, raising
        AttributeError before the fallback. Now guarded with hasattr, so the
        risk-coverage summary runs on both NumPy 1.x and 2.x.

 FIX-14 Pluggable, question-aware abstention detection: the entire abstention
        signal rested on one keyword match, risking a circular "abstention ==
        canned phrase" definition. `is_abstention` is unchanged, but scoring
        now goes through an AbstentionDetector interface. Detectors may be
        one-arg f(answer) or two-arg f(answer, question); an adapter normalizes
        both, and compute_metrics/per_example_scores/compute_abstention_metrics
        accept `questions=`. Provided: KeywordAbstentionDetector (default),
        LLMJudgeAbstentionDetector (AbstentionBench-style Llama-3.1-8B judge,
        the field standard — NOT a rule; sees the question), and a lighter
        NLIAbstentionDetector. compute_metrics records which ran in
        `abstention_detector`.

 FIX-15 ECE binning dependence: with confidences piled near 1, adaptive
        equal-mass edges collapse and a single ECE is binning-dependent.
        `compute_ece_sensitivity` reports ECE and effective bin counts across
        {5,10,15,20} bins; compute_metrics stores it for the behavioral and
        selective numbers. Also: bootstrap default raised to 5000 resamples,
        `parse_verbal_confidence` now resolves scale explicitly (captures '%',
        documents the ambiguous bare-integer case with a toggle), and
        print_metrics labels the decision AUROC as enacted-policy quality
        rather than calibration.

 FIX-16 Judge-free detector validation: rather than run an LLM judge over the
        whole set, `validate_detector` scores the cheap keyword detector
        against a small hand-labeled sample (precision/recall/F1/accuracy/
        Cohen's kappa, overall and per split) and returns the disagreement
        indices so you can see what phrasing it misses. `sample_for_annotation`
        draws a stratified sample to label. This is the same human-validation
        step AbstentionBench used to justify its judge — done once, at sample
        scale, it justifies the keyword rule for the full results section.
        `to_annotation_csv` / `from_annotation_csv` provide the labeling
        round-trip (empty 'is_abstention' column out, parsed labels back, with
        a require_all guard), and `inter_annotator_agreement` reports the
        human-vs-human kappa ceiling from a double-labeled slice.
        `annotation_summary` reports label prevalence and per-split positive
        counts and flags splits with too few abstentions to give a stable
        recall estimate.
"""

import re
import string
import inspect
from collections import Counter
from typing import List, Dict, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd

try:
    from utils.config import ABSTENTION_KEYWORDS
except ImportError:
    ABSTENTION_KEYWORDS = [
        "i don't know", "i dont know", "i do not know",
        "cannot be found", "cannot be answered", "can't be answered",
        "not mentioned in the context", "not in the context",
        "no answer", "unanswerable", "not provided in the context",
        "does not contain", "doesn't contain",
    ]


# ════════════════════════════════════════════════════════
# NORMALIZATION / ABSTENTION DETECTION
# ════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_abstention(answer: str, window: Optional[int] = 80) -> bool:
    """True if the answer is an abstention.

    FIX-3: by default keywords are only matched within the first `window`
    characters, so hedged-but-answering responses are not misclassified.
    window=None restores the legacy anywhere-substring behaviour.
    """
    answer = answer.strip().lower()
    if not (answer or "").strip():
        return False
    haystack = answer if window is None else answer[:window]
    return any(phrase in haystack for phrase in ABSTENTION_KEYWORDS)


# ── Pluggable abstention detection ───────────────────────────────────────
# The ENTIRE abstention signal (recall/precision/F1, coverage, strictness,
# and the unanswerable half of the consistent-regime score) rests on this one
# decision. Keeping it swappable lets you report the keyword baseline AND a
# semantic / judge detector as a robustness check, so the "reflex" conclusion
# is not an artefact of a canned-phrase definition of abstention.
#
# AbstentionBench (facebookresearch/AbstentionBench) does NOT use a keyword
# rule: it scores abstention with an LLM judge (Llama-3.1-8B-Instruct, prompt
# adapted from Thakur et al.), validated to ~88% agreement with human labels,
# and the judge sees the QUESTION as well as the answer. A detector may
# therefore be either a one-arg callable f(answer) or a two-arg callable
# f(answer, question); the adapter below normalizes both.
AbstentionDetector = Callable[..., bool]


class KeywordAbstentionDetector:
    """Default detector: substring match of ABSTENTION_KEYWORDS within the
    first `window` characters (mirrors is_abstention / FIX-3). Reproducible,
    cheap baseline; carries a `name` for provenance. Ignores the question."""
    name = "keyword"

    def __init__(self, window: Optional[int] = 80, keywords=None):
        self.window = window
        self.keywords = list(keywords) if keywords is not None else list(ABSTENTION_KEYWORDS)

    def __call__(self, answer: str, question: Optional[str] = None) -> bool:
        answer = (answer or "").strip().lower()
        if not (answer or "").strip():
            return False
        haystack = answer if self.window is None else answer[:self.window]
        return any(phrase in haystack for phrase in self.keywords)


class LLMJudgeAbstentionDetector:
    """AbstentionBench-style detector: an LLM judge decides whether a response
    is an abstention, given the QUESTION and the ANSWER. This is the field
    standard (AbstentionBench uses Llama-3.1-8B-Instruct, ~88% human
    agreement), NOT a keyword rule. Requires `transformers`; nothing loads
    until instantiated, so the module still imports on a bare machine.

    The judge prompt (paraphrased, not copied) covers the abstention styles
    AbstentionBench targets: explicit uncertainty, requests for clarification,
    flagging missing context, ambiguity, or false premises. Parsing is strict:
    the judge must answer with a single label token.
    """
    name = "llm_judge"

    _PROMPT = (
        "You classify whether a language model's response is an ABSTENTION.\n"
        "A response counts as abstention if, instead of directly answering, it "
        "expresses uncertainty or lack of knowledge, asks for clarification, "
        "says the needed information is missing or not in the context, or points "
        "out that the question is ambiguous or rests on a false premise. If it "
        "answers the question as asked without such caveats, it is NOT an "
        "abstention.\n\n"
        "[QUESTION]: {question}\n"
        "[RESPONSE]: {answer}\n\n"
        "Reply with exactly one word: ABSTENTION or ANSWER."
    )

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device=None, max_new_tokens: int = 4, use_chat_template: bool = True):
        from transformers import pipeline  # lazy, optional
        self._gen = pipeline("text-generation", model=model, device=device)
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template

    def _build(self, answer: str, question: Optional[str]) -> str:
        prompt = self._PROMPT.format(question=question or "(not provided)", answer=answer)
        if self.use_chat_template and getattr(self._gen.tokenizer, "chat_template", None):
            return self._gen.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt

    def __call__(self, answer: str, question: Optional[str] = None) -> bool:
        if not (answer or "").strip():
            return True
        prompt = self._build(answer, question)
        out = self._gen(prompt, max_new_tokens=self.max_new_tokens,
                        do_sample=False, return_full_text=False)
        verdict = out[0]["generated_text"].strip().lower()
        return verdict.startswith("abst")


class NLIAbstentionDetector:
    """Lighter semantic alternative to the LLM judge: zero-shot/NLI scoring of
    an 'answer-not-available' vs 'answered' hypothesis. Answer-only (ignores
    the question), so it cannot catch clarification/false-premise abstentions
    the way the LLM judge can. Requires `transformers`."""
    name = "nli"

    def __init__(self, model: str = "facebook/bart-large-mnli",
                 threshold: float = 0.5, device=None):
        from transformers import pipeline  # lazy, optional
        self._clf = pipeline("zero-shot-classification", model=model, device=device)
        self.threshold = threshold
        self._labels = ["the answer is not available in the context",
                        "the question is answered"]

    def __call__(self, answer: str, question: Optional[str] = None) -> bool:
        answer = (answer or "").strip()
        if not (answer or "").strip():
            return False
        out = self._clf(answer, self._labels, multi_label=True)
        score = dict(zip(out["labels"], out["scores"]))
        no_ans, ans = self._labels
        return score[no_ans] >= self.threshold and score[no_ans] >= score[ans]


def _detector_wants_question(detector) -> bool:
    """True if the detector's call signature accepts a second positional arg
    (the question). Inspected once so plain one-arg callables (e.g. lambdas)
    keep working unchanged."""
    target = detector if (inspect.isfunction(detector) or inspect.ismethod(detector)) \
        else getattr(detector, "__call__", detector)
    try:
        params = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [p for p in params
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    return len(positional) >= 2 or any(p.name == "question" for p in params)


class _DetectorAdapter:
    """Normalizes any detector to a two-arg callable (answer, question) and
    exposes `.name` for provenance."""
    def __init__(self, detector):
        self.detector = detector
        self.name = getattr(detector, "name", "custom")
        self._wants_q = _detector_wants_question(detector)

    def __call__(self, answer: str, question: Optional[str] = None) -> bool:
        return bool(self.detector(answer, question) if self._wants_q
                    else self.detector(answer))


def _resolve_detector(detector: Optional[AbstentionDetector],
                      window: Optional[int]) -> _DetectorAdapter:
    """Return an adapter around the supplied detector, or the keyword baseline."""
    if isinstance(detector, _DetectorAdapter):
        return detector
    base = detector if detector is not None else KeywordAbstentionDetector(window=window)
    return _DetectorAdapter(base)


_VERBAL_CONF_RE = re.compile(
    r"confidence\s*(?:level|score)?\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(%?)",
    re.IGNORECASE,
)


def parse_verbal_confidence(text: str,
                            bare_integer_is_percent: bool = True) -> Optional[float]:
    """Parse a verbalized 'Confidence: NN' (or 'NN%', '0.NN') from generated
    text into [0,1]. Returns None when no confidence statement is found —
    callers should treat None as non-compliance, NOT as zero confidence.
    FIX-10 companion: this is the independent verbal channel; it must never
    be mixed into logit-based confidence.

    Scale resolution:
      * explicit '%'                     → value / 100      (e.g. '85%' → 0.85)
      * decimal and value <= 1.0         → already a fraction ('0.85' → 0.85)
      * value > 1.0                      → percentage        ('85'  → 0.85)
      * bare 0 or 1, no '%'/decimal      → AMBIGUOUS ('1' could mean 1% or a
        0-1-scale max). Controlled by `bare_integer_is_percent` (default True
        → '1' becomes 0.01). Set False if your prompt elicits a 0-1 scale."""
    m = _VERBAL_CONF_RE.search(text)
    if m is None:
        return None
    num_str, pct = m.group(1), m.group(2)
    val = float(num_str)
    if pct == "%":
        conf = val / 100.0
    elif "." in num_str and val <= 1.0:     # already a fraction like 0.85
        conf = val
    elif val > 1.0:                          # percentage like 85 or 85.5
        conf = val / 100.0
    else:                                    # bare 0/1, no '%' or decimal
        conf = val / 100.0 if bare_integer_is_percent else val
    return min(max(conf, 0.0), 1.0)


# ════════════════════════════════════════════════════════
# TOKEN F1 / EM (max over multiple references)
# ════════════════════════════════════════════════════════

def _as_list(ground_truths) -> List[str]:
    return [ground_truths] if isinstance(ground_truths, str) else list(ground_truths)


def compute_f1_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    return max(compute_f1_single(prediction, gt) for gt in _as_list(ground_truths))


def compute_exact_match(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    norm_pred = normalize_answer(prediction)
    return max(float(norm_pred == normalize_answer(gt)) for gt in _as_list(ground_truths))


# ════════════════════════════════════════════════════════
# CONSISTENT PER-EXAMPLE SCORING  (FIX-2)
# ════════════════════════════════════════════════════════

def per_example_scores(predictions, ground_truths, answerable_flags,
                       abstention_window: Optional[int] = 80,
                       detector: Optional[AbstentionDetector] = None,
                       questions: Optional[List[str]] = None):
    """One consistent score per example (SQuAD2-official style):
        answerable   → token F1 against gold spans
        unanswerable → 1.0 iff is_abstention(pred), else 0.0
    Also returns the LEGACY unanswerable score (token F1 vs the literal
    "I don't know") so the old mixed regime can be quantified, and the
    per-example abstention flags reused by every downstream metric."""
    detector = _resolve_detector(detector, abstention_window)
    if questions is None:
        questions = [None] * len(predictions)
    scores, legacy_scores, abst_flags = [], [], []
    for pred, gt, ans, q in zip(predictions, ground_truths, answerable_flags, questions):
        abst = detector(pred, q)
        abst_flags.append(abst)
        if ans:
            s = compute_f1(pred, gt)
            scores.append(s)
            legacy_scores.append(s)
        else:
            scores.append(1.0 if abst else 0.0)
            legacy_scores.append(compute_f1(pred, ["I don't know"]))
    return np.array(scores), np.array(legacy_scores), np.array(abst_flags, dtype=bool)


# ════════════════════════════════════════════════════════
# ANSWER-QUALITY EXTRAS  (FIX-6, FIX-7, FIX-9: lazy imports)
# ════════════════════════════════════════════════════════

_ROUGE_SCORER = None  # FIX-7: singleton


def compute_rouge_l(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    global _ROUGE_SCORER
    try:
        if _ROUGE_SCORER is None:
            from rouge_score import rouge_scorer        # FIX-9: lazy
            _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return max(_ROUGE_SCORER.score(gt, prediction)["rougeL"].fmeasure
                   for gt in _as_list(ground_truths))
    except ImportError:
        return compute_f1(prediction, ground_truths)


def compute_bert_score(predictions: List[str],
                       references: List[Union[str, List[str]]]) -> Optional[float]:
    try:
        from bert_score import score                    # FIX-9: lazy
    except ImportError:
        print("Warning: bert-score not installed. Skipping BERTScore.")
        return None
    # FIX-6: bert-score natively supports list-of-lists references and takes
    # the max per example — use ALL references, not just the first.
    refs = [_as_list(r) for r in references]
    _, _, F1 = score(predictions, refs, lang="en", verbose=False)
    return F1.mean().item()




# ════════════════════════════════════════════════════════
# AUROC HELPERS — unified semantics
# ════════════════════════════════════════════════════════

def _safe_auroc(scores, labels) -> float:
    """Return AUROC, or NaN when fewer than two label classes are present.

    Higher scores must correspond to label=1. NaN/inf scores are dropped.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if s.size == 0 or np.unique(y).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except ImportError:
        pos, neg = s[y == 1], s[y == 0]
        return float(sum(1.0 if p > q else 0.5 if p == q else 0.0
                         for p in pos for q in neg) / (len(pos) * len(neg)))




def _safe_auprc(scores, labels) -> float:
    """Average precision (area under precision-recall curve), or NaN if undefined."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if s.size == 0 or np.unique(y).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y, s))
    except ImportError:
        order = np.argsort(-s, kind="mergesort")
        labels = y[order]
        ranks = np.arange(1, len(labels) + 1)
        return float((np.cumsum(labels)[labels == 1] / ranks[labels == 1]).mean())


_TOKEN_DIAGNOSTIC_FIELDS = (
    "answer_mean_logprob", "answer_min_logprob", "answer_mean_entropy",
    "answer_max_entropy", "answer_mean_top1_top2_margin",
    "answer_min_top1_top2_margin", "answer_first_token_entropy",
    "answer_first_token_margin", "decision_refusal_prefix_first_token_mass",
    "decision_refusal_prefix_vs_best_other_logit_margin",
    "decision_refusal_prefix_vs_other_logsumexp_margin",
)


def _finite_diagnostic_values(diags, field, mask):
    values = []
    for diagnostic, include in zip(diags, mask):
        value = diagnostic.get(field) if include and diagnostic is not None else None
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _diagnostic_distribution(values):
    """Distribution report that never turns missing diagnostics into zero."""
    n = int(values.size)
    if not n:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {"n": n, "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75))}


def _add_token_diagnostic_metrics(metrics, diagnostics, answerable, abstained, answer_scores):
    """Add nullable token diagnostics with explicit score orientation/targets."""
    attempted = ~abstained
    correct_answer = answerable & attempted & (answer_scores >= 0.5)
    groups = {
        "correct_answer": correct_answer,
        "incorrect_answer": attempted & ~correct_answer,
        "correct_abstention": ~answerable & abstained,
        "false_abstention": answerable & abstained,
    }
    metrics["token_diagnostic_groups"] = {
        name: {field: _diagnostic_distribution(_finite_diagnostic_values(diagnostics, field, mask))
               for field in _TOKEN_DIAGNOSTIC_FIELDS}
        for name, mask in groups.items()
    }
    marker_values = [d.get("answer_span_marker_found") for d in diagnostics
                     if d is not None and d.get("answer_span_marker_found") is not None]
    metrics["answer_span_marker_rate"] = (
        float(np.mean(marker_values)) if marker_values else float("nan"))
    metrics["answer_span_marker_rate_n_valid"] = len(marker_values)

    def add_rank(name, field, mask, labels, transform=lambda x: x):
        raw = _finite_diagnostic_values(diagnostics, field, mask)
        valid_mask = np.array([bool(include) and d is not None and d.get(field) is not None
                               and np.isfinite(float(d.get(field)))
                               for d, include in zip(diagnostics, mask)], dtype=bool)
        score = transform(raw)
        target = np.asarray(labels, dtype=int)[valid_mask]
        metrics[name + "_auroc"] = _safe_auroc(score, target)
        metrics[name + "_auprc"] = _safe_auprc(score, target)
        metrics[name + "_n_valid"] = int(score.size)

    # Higher score means more uncertain, target=1 means an incorrect attempted answer.
    incorrect_target = (answer_scores < 0.5).astype(int)
    add_rank("token_entropy_selective", "answer_mean_entropy", attempted, incorrect_target)
    add_rank("first_token_entropy_selective", "answer_first_token_entropy", attempted, incorrect_target)
    add_rank("first_token_margin_selective", "answer_first_token_margin", attempted, incorrect_target,
             transform=lambda x: -x)
    add_rank("min_token_logprob_selective", "answer_min_logprob", attempted, incorrect_target,
             transform=lambda x: -x)

    # Higher refusal-prefix mass / margin is tested against gold unanswerability.
    unanswerable = (~answerable).astype(int)
    for prefix, field in (("decision_refusal_prefix_mass_answerability", "decision_refusal_prefix_first_token_mass"),
                          ("decision_refusal_prefix_margin_answerability",
                           "decision_refusal_prefix_vs_best_other_logit_margin"),
                          ("decision_refusal_prefix_logsumexp_margin_answerability",
                           "decision_refusal_prefix_vs_other_logsumexp_margin")):
        raw = _finite_diagnostic_values(diagnostics, field, np.ones(len(diagnostics), dtype=bool))
        valid = np.array([d is not None and d.get(field) is not None and np.isfinite(float(d.get(field)))
                          for d in diagnostics], dtype=bool)
        metrics[prefix + "_auroc"] = _safe_auroc(raw, unanswerable[valid])
        metrics[prefix + "_auprc"] = _safe_auprc(raw, unanswerable[valid])
        metrics[prefix + "_n_valid"] = int(raw.size)

    # Prefix mass is an exploratory lexical score, not calibrated P(ABSTAIN).
    # Deliberately report ranking metrics only (AUROC/AUPRC).


def _risk_coverage_summary(confidences, correct, coverage_points=(0.25, 0.50, 0.75, 0.90)) -> Dict[str, float]:
    """Conditional risk-coverage summary on a supplied subset.

    Examples are ordered by confidence descending. Coverage is relative to the
    supplied subset (for selective metrics, the attempted-answer subset).
    Returns AURC, E-AURC, and risk/accuracy at fixed coverage points.
    """
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=bool)
    m = np.isfinite(c)
    c, y = c[m], y[m]
    n = len(c)
    if n == 0:
        out = {"aurc": float("nan"), "eaurc": float("nan")}
        for q in coverage_points:
            tag = int(round(q * 100))
            out[f"risk_at_{tag}_coverage"] = float("nan")
            out[f"accuracy_at_{tag}_coverage"] = float("nan")
        return out

    order = np.argsort(-c, kind="mergesort")
    err = (~y[order]).astype(float)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = np.cumsum(err) / k
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    aurc = float(trapz(risk, coverage))

    err_opt = np.sort((~y).astype(float))
    risk_opt = np.cumsum(err_opt) / k
    aurc_opt = float(trapz(risk_opt, coverage))
    out = {"aurc": aurc, "eaurc": aurc - aurc_opt}
    for q in coverage_points:
        idx = max(0, min(n - 1, int(np.ceil(q * n)) - 1))
        tag = int(round(q * 100))
        out[f"risk_at_{tag}_coverage"] = float(risk[idx])
        out[f"accuracy_at_{tag}_coverage"] = float(1.0 - risk[idx])
    return out


def _safe_spearman(x, y) -> float:
    """Spearman rank correlation, or NaN when undefined."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)

def _decision_answerability_score(confidence: np.ndarray, attempted: np.ndarray) -> np.ndarray:
    """Behavior-conditioned answer score used for answer/abstain decision AUROC.

    `confidence` is sequence confidence in the text the model actually generated.
    For an answered item, high sequence confidence supports ANSWER. For an
    abstained item, high sequence confidence supports ABSTAIN, so the answer score
    is inverted. This score therefore measures the strength of the *enacted
    answer/abstain policy*; it is not an independent pre-decision uncertainty score.
    """
    c = _validate_confidences(confidence)
    a = np.asarray(attempted, dtype=bool)
    return np.where(a, c, 1.0 - c)


# ════════════════════════════════════════════════════════
# CALIBRATION  (FIX-4, FIX-5)
# ════════════════════════════════════════════════════════

def _validate_confidences(confidences) -> np.ndarray:
    conf = np.asarray(confidences, dtype=float)
    if conf.size and (conf.min() < 0.0 or conf.max() > 1.0):
        raise ValueError(
            "confidences must be in [0,1] (e.g. exp(mean_logprob) or a "
            "verbalized percentage / 100), got "
            f"min={conf.min():.3f}, max={conf.max():.3f}. Raw log-probs are "
            "negative and would fall outside every bin, yielding ECE ≈ 0."
        )
    return conf


def compute_ece(confidences, correct, n_bins: int = 10, adaptive: bool = True) -> float:
    """Expected Calibration Error.  FIX-4: input validation + adaptive
    (equal-mass) binning, which is more informative when greedy-decoding
    confidences cluster near 1."""
    conf = _validate_confidences(confidences)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(edges) - 1):
        last = (i == len(edges) - 2)
        m = (conf >= edges[i]) & ((conf <= edges[i + 1]) if last else (conf < edges[i + 1]))
        if m.sum() == 0:
            continue
        ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
    return float(ece)


def compute_brier_score(confidences, outcomes) -> float:
    conf = _validate_confidences(confidences)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((conf - outcomes) ** 2))


def compute_ece_sensitivity(confidences, correct,
                            bin_counts: Tuple[int, ...] = (5, 10, 15, 20),
                            adaptive: bool = True) -> Dict[str, float]:
    """ECE evaluated at several bin counts. FIX-4 companion: when greedy-decoding
    confidences pile up near 1, adaptive equal-mass edges collapse under
    np.unique, so the *effective* number of bins is <= n_bins and a single ECE
    can be an artefact of that collapse. Reporting a small sweep exposes the
    binning dependence. Returns {'ece_nbins_10': ..., 'ece_spread': max-min,
    'effective_bins_10': ...} where effective_bins is the count of non-empty
    bins actually used."""
    conf = _validate_confidences(confidences)
    correct = np.asarray(correct, dtype=float)
    out: Dict[str, float] = {}
    vals = []
    for nb in bin_counts:
        e = compute_ece(conf, correct, n_bins=nb, adaptive=adaptive)
        out[f"ece_nbins_{nb}"] = e
        vals.append(e)
        if conf.size:
            if adaptive:
                edges = np.unique(np.quantile(conf, np.linspace(0, 1, nb + 1)))
            else:
                edges = np.linspace(0, 1, nb + 1)
            out[f"effective_bins_{nb}"] = int(max(len(edges) - 1, 1))
        else:
            out[f"effective_bins_{nb}"] = 0
    out["ece_spread"] = float(max(vals) - min(vals)) if vals else float("nan")
    return out


def plot_reliability_diagram(confidences, accuracies, n_bins: int = 10,
                             title: str = "Reliability Diagram",
                             save_path: Optional[str] = None):
    import matplotlib.pyplot as plt                     # FIX-9: lazy
    from pathlib import Path
    conf = _validate_confidences(confidences)
    acc = np.asarray(accuracies, dtype=float)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_cnt = [], [], []
    for i in range(n_bins):
        last = (i == n_bins - 1)
        m = (conf >= edges[i]) & ((conf <= edges[i + 1]) if last else (conf < edges[i + 1]))
        if m.sum() > 0:
            bin_conf.append(conf[m].mean())
            bin_acc.append(acc[m].mean())
            bin_cnt.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")
    if bin_conf:
        sizes = np.array(bin_cnt) / max(bin_cnt) * 500
        ax.scatter(bin_conf, bin_acc, s=sizes, alpha=0.6, color="blue")
        ax.plot(bin_conf, bin_acc, "o-", linewidth=2, markersize=8,
                color="blue", label="Model calibration")
    ax.set_xlabel("Confidence", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ece = compute_ece(conf, acc, n_bins, adaptive=False)
    ax.text(0.05, 0.95, f"ECE = {ece:.3f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved reliability diagram to: {save_path}")
    return fig


# ════════════════════════════════════════════════════════
# ABSTENTION METRICS  (FIX-1, FIX-8)
# ════════════════════════════════════════════════════════

def compute_abstention_metrics(predictions: List[str],
                               is_answerable_flags: List[bool],
                               ground_truths=None,          # FIX-8: unused, kept for back-compat
                               abstention_window: Optional[int] = 80,
                               detector: Optional[AbstentionDetector] = None,
                               questions: Optional[List[str]] = None) -> Dict[str, float]:
    n = len(predictions)
    detector = _resolve_detector(detector, abstention_window)
    if questions is None:
        questions = [None] * n
    abst = np.array([detector(p, q) for p, q in zip(predictions, questions)])
    ans = np.array(is_answerable_flags, dtype=bool)
    unans = ~ans

    strictness = abst[ans].mean() * 100 if ans.any() else 0.0       # false abstention rate
    coverage = (~abst).mean() * 100 if n else 0.0
    total_abst = int(abst.sum())
    correct_abst = int((abst & unans).sum())
    abst_precision = correct_abst / total_abst * 100 if total_abst else 0.0
    abst_recall = correct_abst / unans.sum() * 100 if unans.any() else 0.0
    pr, rc = abst_precision / 100, abst_recall / 100
    abst_f1 = 200 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    return {
        "strictness": strictness,
        "coverage": coverage,
        "abstention_precision": abst_precision,
        "abstention_recall": abst_recall,
        "abstention_f1": abst_f1,                # single headline abstention number
        "total_abstentions": total_abst,
    }


def compute_selectivity_ratio(baseline_metrics: Dict, improved_metrics: Dict) -> Optional[float]:
    # FIX-1: keyed on the honest name (alias still present in compute_metrics output)
    delta_noans = improved_metrics["abstention_recall"] - baseline_metrics["abstention_recall"]
    delta_coverage = improved_metrics["coverage"] - baseline_metrics["coverage"]
    if abs(delta_coverage) < 0.1:
        return None
    return delta_noans / abs(delta_coverage)


# ════════════════════════════════════════════════════════
# DETECTOR VALIDATION  (cheap, judge-free alternative — FIX-16)
# ════════════════════════════════════════════════════════
# Instead of running an LLM judge over every example, validate the cheap
# keyword detector ONCE against a small hand-labeled sample. If agreement is
# high (expected when fine-tuned models emit a stereotyped abstention string),
# the keyword rule is justified for the whole results section. Low recall on a
# split — most likely OOD or non-fine-tuned baselines — is exactly where a
# judge would earn its keep, and only there.

def sample_for_annotation(n_total: int, sample_size: int, strata=None,
                          seed: int = 0, oversample=None) -> List[int]:
    """Draw indices to hand-label. Stratified by `strata` (e.g. split name per
    example) with proportional allocation; `oversample` is an optional set of
    stratum values to weight ~2x (e.g. {'ood', 'baseline'}). Returns sorted
    indices. Stratify rather than taking the first N, or the validation set
    won't be representative."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    if strata is None:
        k = min(sample_size, n_total)
        return sorted(rng.choice(idx, size=k, replace=False).tolist())
    strata = np.asarray(strata)
    groups = {s: idx[strata == s] for s in np.unique(strata)}
    weights = {s: len(g) for s, g in groups.items()}
    if oversample:
        over = set(oversample)
        for s in list(weights):
            if s in over:
                weights[s] *= 2
    total_w = sum(weights.values()) or 1
    chosen: List[int] = []
    for s, g in groups.items():
        k = min(len(g), max(1, round(sample_size * weights[s] / total_w)))
        chosen.extend(rng.choice(g, size=k, replace=False).tolist())
    return sorted(chosen)


def _binary_agreement(detector_pos, human_pos) -> Dict[str, float]:
    """Precision/recall/F1/accuracy and Cohen's kappa, treating abstention as
    the positive class. Self-contained (no sklearn)."""
    d = np.asarray(detector_pos, dtype=bool)
    h = np.asarray(human_pos, dtype=bool)
    n = len(d)
    tp = int((d & h).sum()); fp = int((d & ~h).sum())
    fn = int((~d & h).sum()); tn = int((~d & ~h).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))
    accuracy = (tp + tn) / n if n else float("nan")
    po = accuracy
    p_det = (tp + fp) / n if n else 0.0
    p_hum = (tp + fn) / n if n else 0.0
    pe = p_det * p_hum + (1 - p_det) * (1 - p_hum)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "kappa": kappa,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n}


def validate_detector(predictions, human_abstention_labels,
                      detector=None, abstention_window: Optional[int] = 80,
                      questions=None, splits=None) -> Dict:
    """Validate an abstention DETECTOR against human labels on a small sample —
    the judge-free stand-in. `human_abstention_labels[i]` is the human judgment
    of whether prediction i is an abstention.

    Returns:
      detector         : which detector was validated
      overall          : precision/recall/F1/accuracy/kappa vs human labels
      per_split        : the same as a DataFrame (when `splits` is given)
      false_negatives  : indices the rule MISSED (human=abstain, rule=answer)
      false_positives  : indices the rule OVER-FIRED (human=answer, rule=abstain)
    The disagreement indices are the point: reading them tells you what phrasing
    the keyword list misses, which either fixes the list or motivates a judge on
    that split specifically."""
    if len(predictions) != len(human_abstention_labels):
        raise ValueError("predictions and human_abstention_labels must match in length")
    detector = _resolve_detector(detector, abstention_window)
    if questions is None:
        questions = [None] * len(predictions)
    det = np.array([detector(p, q) for p, q in zip(predictions, questions)], dtype=bool)
    hum = np.asarray(human_abstention_labels, dtype=bool)

    out: Dict = {"detector": getattr(detector, "name", "custom"),
                 "overall": _binary_agreement(det, hum),
                 "false_negatives": np.where(~det & hum)[0].tolist(),
                 "false_positives": np.where(det & ~hum)[0].tolist()}
    if splits is not None:
        splits = np.asarray(splits)
        rows = [{"split": s, **_binary_agreement(det[splits == s], hum[splits == s])}
                for s in np.unique(splits)]
        out["per_split"] = pd.DataFrame(rows)
    return out


def print_detector_validation(report: Dict):
    o = report["overall"]
    print("\n" + "=" * 64)
    print(f" Detector validation vs human labels — '{report['detector']}'")
    print("=" * 64)
    print(f"  n={o['n']}  precision={o['precision']:.3f}  recall={o['recall']:.3f}  "
          f"F1={o['f1']:.3f}  acc={o['accuracy']:.3f}  kappa={o['kappa']:.3f}")
    print(f"  TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}  "
          f"(FN=missed real abstentions, FP=over-fired)")
    if "per_split" in report:
        print("\n  Per split:")
        print("  " + report["per_split"].to_string(index=False).replace("\n", "\n  "))
    print("=" * 64)


# ── Annotation round-trip (CSV) ──────────────────────────────────────────
_TRUE_TOKENS = {"true", "t", "1", "1.0", "yes", "y", "abstain", "abstention"}
_FALSE_TOKENS = {"false", "f", "0", "0.0", "no", "n", "answer", "answered"}


def _parse_label(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return None            # unrecognized token -> treat as unlabeled


def _col(df, name):
    return [None if pd.isna(v) else v for v in df[name].tolist()]


def to_annotation_csv(path, predictions, indices=None, questions=None,
                      contexts=None, ground_truths=None, splits=None,
                      model_ids=None, detector_guess: bool = False,
                      detector=None, abstention_window: Optional[int] = 80) -> str:
    """Dump sampled examples to a CSV for hand-labeling. Adds an empty
    'is_abstention' column (annotator writes TRUE/FALSE, 1/0, or abstain/answer)
    and a 'notes' column. 'idx' preserves the original dataset position so
    labels join back exactly. Pass the SAMPLED subset (e.g. predictions[i] for i
    in sample_for_annotation(...)); `indices` should be those i.

    `detector_guess=False` by default ON PURPOSE — showing the rule's guess in
    the file anchors the annotator and inflates agreement. Only turn it on for
    post-hoc adjudication of disagreements, in a separate column, never the
    label column."""
    n = len(predictions)
    indices = list(range(n)) if indices is None else list(indices)
    data: Dict[str, list] = {"idx": indices}
    if model_ids is not None:
        data["model_id"] = list(model_ids)
    if splits is not None:
        data["split"] = list(splits)
    if questions is not None:
        data["question"] = list(questions)
    if contexts is not None:
        data["context"] = list(contexts)
    data["prediction"] = list(predictions)
    if ground_truths is not None:
        data["ground_truth"] = ["; ".join(_as_list(g)) for g in ground_truths]
    if detector_guess:
        det = _resolve_detector(detector, abstention_window)
        qs = questions if questions is not None else [None] * n
        data["detector_guess"] = [bool(det(p, q)) for p, q in zip(predictions, qs)]
    data["is_abstention"] = [""] * n        # annotator fills this
    data["notes"] = [""] * n
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def from_annotation_csv(path, label_col: str = "is_abstention",
                        require_all: bool = True) -> Dict:
    """Read a labeled annotation CSV back into arrays for validate_detector.
    Returns a dict with human_abstention_labels plus whatever columns are
    present (predictions, indices, questions, splits, model_ids), so you can do:

        d = from_annotation_csv('labels.csv')
        validate_detector(d['predictions'], d['human_abstention_labels'],
                          questions=d.get('questions'), splits=d.get('splits'))

    `require_all=True` raises (listing offending idx) if any row is blank or has
    an unrecognized label, so you never validate on a half-labeled file."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in {path}")
    labels = [_parse_label(v) for v in df[label_col].tolist()]
    have_idx = "idx" in df.columns
    unlabeled = [int(df["idx"].iloc[i]) if have_idx else i
                 for i, lab in enumerate(labels) if lab is None]
    if unlabeled and require_all:
        shown = unlabeled[:10]
        more = "..." if len(unlabeled) > 10 else ""
        raise ValueError(f"{len(unlabeled)} row(s) blank/invalid in '{label_col}' "
                         f"(idx: {shown}{more}). Fill them or pass require_all=False.")
    keep = [i for i, lab in enumerate(labels) if lab is not None]
    out: Dict = {"human_abstention_labels": [labels[i] for i in keep]}
    for col, key in (("prediction", "predictions"), ("idx", "indices"),
                     ("question", "questions"), ("split", "splits"),
                     ("model_id", "model_ids")):
        if col in df.columns:
            vals = _col(df, col)
            out[key] = [vals[i] for i in keep]
    return out


def inter_annotator_agreement(labels_a, labels_b) -> Dict[str, float]:
    """Cohen's kappa + raw agreement between two annotators' abstention labels.
    Report this BEFORE the detector number: detector-vs-human agreement can't
    meaningfully exceed the human-vs-human ceiling."""
    r = _binary_agreement(np.asarray(labels_a, dtype=bool),
                          np.asarray(labels_b, dtype=bool))
    return {"kappa": r["kappa"], "raw_agreement": r["accuracy"], "n": r["n"]}


def _group_prevalence(pos_mask, labeled_mask, groups, min_positives) -> pd.DataFrame:
    g = np.asarray(groups, dtype=object)
    rows = []
    for s in pd.unique(g):
        gm = g == s
        n = int((labeled_mask & gm).sum())
        pos = int((pos_mask & gm).sum())
        rows.append({"group": s, "n_labeled": n, "n_pos": pos, "n_neg": n - pos,
                     "prevalence": pos / n if n else float("nan"),
                     "enough_positives": pos >= min_positives})
    return pd.DataFrame(rows)


def annotation_summary(source, min_positives: int = 20,
                       label_col: str = "is_abstention") -> Dict:
    """Sanity-check a labeling set BEFORE trusting recall. Reports label
    prevalence and positive (abstention) counts overall and per split / per
    model. Recall is estimated on positives ONLY, so a split with few
    abstentions yields an unstable recall regardless of its total size — those
    splits (n_pos < `min_positives`) are flagged in `underpowered`.

    `source` may be a path to an annotation CSV (also counts rows still
    unlabeled) or a dict from from_annotation_csv."""
    if isinstance(source, str):
        df = pd.read_csv(source)
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' column not found in {source}")
        labels = [_parse_label(v) for v in df[label_col].tolist()]
        splits = _col(df, "split") if "split" in df.columns else None
        model_ids = _col(df, "model_id") if "model_id" in df.columns else None
    else:
        labels = list(source["human_abstention_labels"])
        splits = source.get("splits")
        model_ids = source.get("model_ids")

    labeled_mask = np.array([lab is not None for lab in labels], dtype=bool)
    pos_mask = np.array([lab is True for lab in labels], dtype=bool)
    n_total = len(labels)
    n_labeled = int(labeled_mask.sum())
    n_pos = int(pos_mask.sum())

    out: Dict = {
        "n_total": n_total,
        "n_labeled": n_labeled,
        "n_unlabeled": n_total - n_labeled,
        "n_positives": n_pos,
        "n_negatives": n_labeled - n_pos,
        "prevalence": n_pos / n_labeled if n_labeled else float("nan"),
        "min_positives": min_positives,
        "enough_positives_overall": n_pos >= min_positives,
        "underpowered": [],
    }
    if splits is not None:
        ps = _group_prevalence(pos_mask, labeled_mask, splits, min_positives)
        ps = ps.rename(columns={"group": "split"})
        out["per_split"] = ps
        out["underpowered"] = ps.loc[~ps["enough_positives"], "split"].tolist()
    if model_ids is not None:
        out["per_model"] = _group_prevalence(pos_mask, labeled_mask,
                                             model_ids, min_positives
                                             ).rename(columns={"group": "model_id"})
    return out


def print_annotation_summary(summary: Dict):
    print("\n" + "=" * 64)
    print(" Annotation summary (label this-many before trusting recall)")
    print("=" * 64)
    print(f"  labeled {summary['n_labeled']}/{summary['n_total']}  "
          f"(unlabeled: {summary['n_unlabeled']})")
    print(f"  abstentions (positives): {summary['n_positives']}   "
          f"answers: {summary['n_negatives']}   "
          f"prevalence: {summary['prevalence']:.3f}")
    print(f"  need >= {summary['min_positives']} positives for a stable recall  "
          f"-> overall {'OK' if summary['enough_positives_overall'] else 'TOO FEW'}")
    if "per_split" in summary:
        print("\n  Per split:")
        print("  " + summary["per_split"].to_string(index=False).replace("\n", "\n  "))
    if summary["underpowered"]:
        print(f"\n  ⚠ underpowered splits (add more labeled positives): "
              f"{summary['underpowered']}")
    print("=" * 64)


# ════════════════════════════════════════════════════════
# BOOTSTRAP CI  (FIX-11)
# ════════════════════════════════════════════════════════

def bootstrap_ci(values: np.ndarray, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    boots = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo) * 100, float(hi) * 100


# ════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════

def compute_metrics(
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
    is_answerable_flags: List[bool],
    ids: Optional[List[str]] = None,                    # kept for back-compat, unused
    confidences: Optional[List[float]] = None,          # logit-based, in [0,1]
    verbal_confidences: Optional[List[Optional[float]]] = None,      # FIX-10
    verbalized_confidences: Optional[List[Optional[float]]] = None,  # alias
    compute_advanced: bool = False,
    abstention_window: Optional[int] = 80,
    abstention_detector: Optional[AbstentionDetector] = None,
    questions: Optional[List[str]] = None,
    token_diagnostics: Optional[List[Dict]] = None,
) -> Dict:
    if verbal_confidences is None:
        verbal_confidences = verbalized_confidences
    # Resolve the abstention detector ONCE so every downstream metric agrees
    # (per-example score, EM, coverage, abstention P/R/F1), a judge model loads
    # a single time, and we can record which detector ran.
    detector = _resolve_detector(abstention_detector, abstention_window)
    n = len(predictions)
    if len(ground_truths) != n or len(is_answerable_flags) != n:
        raise ValueError("predictions, ground_truths, and is_answerable_flags must have equal length")
    if confidences is not None and len(confidences) != n:
        raise ValueError("confidences must have the same length as predictions")
    if verbal_confidences is not None and len(verbal_confidences) != n:
        raise ValueError("verbal_confidences must have the same length as predictions")
    if questions is not None and len(questions) != n:
        raise ValueError("questions must have the same length as predictions")
    if token_diagnostics is not None and len(token_diagnostics) != n:
        raise ValueError("token_diagnostics must have the same length as predictions")
    if n == 0:
        return {"overall_f1": 0.0, "overall_em": 0.0, "has_ans_f1": 0.0,
                "has_ans_em": 0.0, "abstention_recall": 0.0, "no_ans_f1": 0.0,
                "strictness": 0.0, "coverage": 0.0, "total": 0}

    ans = np.array(is_answerable_flags, dtype=bool)
    unans = ~ans

    scores, legacy, abst = per_example_scores(
        predictions, ground_truths, ans,
        abstention_window=abstention_window, detector=detector, questions=questions)
    # Consistent EM: answerable -> literal EM against gold spans;
    # unanswerable -> 1 iff the model explicitly abstained.
    em_scores = np.array([
        compute_exact_match(p, g) if is_ans else float(is_abs)
        for p, g, is_ans, is_abs in zip(predictions, ground_truths, ans, abst)
    ], dtype=float)

    abst_m = compute_abstention_metrics(predictions, ans.tolist(),
                                        abstention_window=abstention_window,
                                        detector=detector, questions=questions)
    ci_lo, ci_hi = bootstrap_ci(scores)

    metrics = {
        # consistent regime (FIX-2)
        "overall_f1": scores.mean() * 100,
        "overall_em": em_scores.mean() * 100,
        "has_ans_f1": scores[ans].mean() * 100 if ans.any() else 0.0,
        "has_ans_em": em_scores[ans].mean() * 100 if ans.any() else 0.0,
        # honest name + documented alias (FIX-1). Both equal by construction.
        "abstention_recall": abst_m["abstention_recall"],
        "no_ans_f1": abst_m["abstention_recall"],   # ALIAS, kept for old plotting code
        # legacy mixed regime, for quantifying the old anomaly only
        "overall_f1_legacy": legacy.mean() * 100,
        # abstention block
        "strictness": abst_m["strictness"],
        "coverage": abst_m["coverage"],
        "abstention_precision": abst_m["abstention_precision"],
        "abstention_f1": abst_m["abstention_f1"],
        # uncertainty of the headline number (FIX-11)
        "overall_ci_low": ci_lo,
        "overall_ci_high": ci_hi,
        # counts
        "total": n,
        "has_ans_total": int(ans.sum()),
        "no_ans_total": int(unans.sum()),
        # provenance: which abstention operationalization produced the numbers
        "abstention_detector": getattr(detector, "name", "custom"),
    }

    if compute_advanced:
        try:
            metrics["rouge_l"] = np.mean(
                [compute_rouge_l(p, g) for p, g in zip(predictions, ground_truths)]) * 100
        except Exception as e:
            print(f"Warning: ROUGE-L failed: {e}")
        try:
            bs = compute_bert_score(predictions, ground_truths)
            if bs is not None:
                metrics["bert_score"] = bs * 100
        except Exception as e:
            print(f"Warning: BERTScore failed: {e}")

    # ── Confidence evaluation: keep targets semantically distinct ───────────
    # `scores` is the consistent behavioral score:
    #   answerable -> token F1; unanswerable -> 1 iff abstained.
    attempted = ~abst
    correct_f1 = scores >= 0.5
    correct_em = em_scores >= 1.0

    if confidences is not None:
        c = _validate_confidences(confidences)

        # (A) Behavioral correctness of the generated action/text, all examples.
        metrics["behavior_ece_f1_0.5"] = compute_ece(c, correct_f1)
        metrics["behavior_brier_f1_0.5"] = compute_brier_score(c, correct_f1)
        metrics["behavior_ece_em"] = compute_ece(c, correct_em)
        metrics["behavior_brier_em"] = compute_brier_score(c, correct_em)
        metrics["mean_confidence"] = float(c.mean())

        # Backward-compatible aliases. They refer to behavioral correctness, not
        # answerability calibration.
        metrics["ece_f1_0.5"] = metrics["behavior_ece_f1_0.5"]
        metrics["brier_f1_0.5"] = metrics["behavior_brier_f1_0.5"]
        metrics["ece_em"] = metrics["behavior_ece_em"]
        metrics["brier_em"] = metrics["behavior_brier_em"]
        # Binning-dependence check for the headline ECE (see compute_ece_sensitivity).
        metrics["behavior_ece_f1_0.5_sensitivity"] = compute_ece_sensitivity(c, correct_f1)

        # (B) Selective prediction: among examples the model actually answered,
        # does sequence confidence rank correct answers above incorrect answers?
        # Attempted answers to gold-unanswerable items are necessarily incorrect.
        attempted_answerable = attempted & ans
        metrics["n_attempted"] = int(attempted.sum())
        metrics["attempted_fraction"] = float(attempted.mean())
        metrics["attempted_answerable_fraction"] = (
            float(ans[attempted].mean()) if attempted.any() else float("nan")
        )
        metrics["attempted_accuracy_f1_0.5"] = (
            float(correct_f1[attempted].mean()) if attempted.any() else float("nan")
        )
        metrics["attempted_accuracy_em"] = (
            float(correct_em[attempted].mean()) if attempted.any() else float("nan")
        )

        if attempted.any():
            sel_c = c[attempted]
            sel_y_f1 = correct_f1[attempted]
            sel_y_em = correct_em[attempted]
            metrics["selective_auroc_f1_0.5"] = _safe_auroc(sel_c, sel_y_f1)
            metrics["selective_auprc_f1_0.5"] = _safe_auprc(sel_c, sel_y_f1)
            metrics["selective_positive_rate_f1_0.5"] = float(sel_y_f1.mean())
            metrics["selective_auroc_em"] = _safe_auroc(sel_c, sel_y_em)
            metrics["selective_auprc_em"] = _safe_auprc(sel_c, sel_y_em)
            metrics["selective_positive_rate_em"] = float(sel_y_em.mean())
            metrics["selective_ece_f1_0.5"] = compute_ece(sel_c, sel_y_f1)
            metrics["selective_brier_f1_0.5"] = compute_brier_score(sel_c, sel_y_f1)
            metrics["selective_ece_f1_0.5_sensitivity"] = compute_ece_sensitivity(sel_c, sel_y_f1)

            rc = _risk_coverage_summary(sel_c, sel_y_f1)
            metrics["selective_aurc_f1_0.5"] = rc["aurc"]
            metrics["selective_eaurc_f1_0.5"] = rc["eaurc"]
            for tag in (25, 50, 75, 90):
                metrics[f"selective_risk_at_{tag}_coverage"] = rc[f"risk_at_{tag}_coverage"]
                metrics[f"selective_accuracy_at_{tag}_coverage"] = rc[f"accuracy_at_{tag}_coverage"]
        else:
            for key in (
                "selective_auroc_f1_0.5", "selective_auprc_f1_0.5",
                "selective_positive_rate_f1_0.5", "selective_auroc_em",
                "selective_auprc_em", "selective_positive_rate_em",
                "selective_ece_f1_0.5", "selective_brier_f1_0.5",
                "selective_aurc_f1_0.5", "selective_eaurc_f1_0.5",
            ):
                metrics[key] = float("nan")
            metrics["selective_ece_f1_0.5_sensitivity"] = {}
            for tag in (25, 50, 75, 90):
                metrics[f"selective_risk_at_{tag}_coverage"] = float("nan")
                metrics[f"selective_accuracy_at_{tag}_coverage"] = float("nan")

        # Pure answer-correctness discrimination, excluding gold-unanswerable items.
        # This prevents answerability detection from inflating selective performance.
        metrics["n_attempted_answerable"] = int(attempted_answerable.sum())
        if attempted_answerable.any():
            aa_c = c[attempted_answerable]
            aa_f1 = scores[attempted_answerable]
            aa_correct_f1 = aa_f1 >= 0.5
            aa_correct_em = em_scores[attempted_answerable] >= 1.0
            metrics["selective_answerable_only_auroc_f1_0.5"] = _safe_auroc(aa_c, aa_correct_f1)
            metrics["selective_answerable_only_auprc_f1_0.5"] = _safe_auprc(aa_c, aa_correct_f1)
            metrics["selective_answerable_only_auroc_em"] = _safe_auroc(aa_c, aa_correct_em)
            metrics["selective_answerable_only_spearman_f1"] = _safe_spearman(aa_c, aa_f1)
            metrics["selective_answerable_only_mean_f1"] = float(aa_f1.mean())
        else:
            metrics["selective_answerable_only_auroc_f1_0.5"] = float("nan")
            metrics["selective_answerable_only_auprc_f1_0.5"] = float("nan")
            metrics["selective_answerable_only_auroc_em"] = float("nan")
            metrics["selective_answerable_only_spearman_f1"] = float("nan")
            metrics["selective_answerable_only_mean_f1"] = float("nan")

        metrics["selective_auroc"] = metrics["selective_auroc_f1_0.5"]  # legacy alias

        # (C) Answer/abstain DECISION AUROC. This intentionally combines the
        # observed response type with sequence confidence and therefore measures
        # enacted policy quality/strength, not independent uncertainty.
        answer_score = _decision_answerability_score(c, attempted)
        metrics["answer_abstain_decision_auroc"] = _safe_auroc(answer_score, ans)
        metrics["answer_abstain_decision_ece"] = compute_ece(answer_score, ans)
        metrics["answer_abstain_decision_brier"] = compute_brier_score(answer_score, ans)
        metrics["decision_auroc"] = metrics["answer_abstain_decision_auroc"]  # legacy alias
        metrics["ece_abstention"] = metrics["answer_abstain_decision_ece"]   # legacy alias

    if verbal_confidences is not None:
        vc = np.array([v if v is not None else np.nan for v in verbal_confidences], dtype=float)
        parsed = np.isfinite(vc)
        metrics["verbal_compliance"] = parsed.mean() * 100
        metrics["verbal_compliance_answerable"] = parsed[ans].mean() * 100 if ans.any() else 0.0
        metrics["verbal_compliance_unanswerable"] = parsed[unans].mean() * 100 if unans.any() else 0.0

        if parsed.any():
            v = _validate_confidences(vc[parsed])
            y_ans = ans[parsed]
            # The prompt defines verbal confidence as P(answer present in context),
            # so its primary target is GOLD answerability — never behavioral
            # correctness of an abstention.
            metrics["verbal_answerability_auroc"] = _safe_auroc(v, y_ans)
            metrics["verbal_answerability_ece"] = compute_ece(v, y_ans)
            metrics["verbal_answerability_brier"] = compute_brier_score(v, y_ans)
            metrics["verbal_mean_confidence"] = float(v.mean())
            metrics["verbal_decision_auroc"] = metrics["verbal_answerability_auroc"]  # legacy alias

            # Optional diagnostic only: verbal answerability confidence used to
            # rank correctness among answered cases. This is not its elicited
            # semantic target and should not be reported as calibration.
            vp_attempted = parsed & attempted
            if vp_attempted.any():
                metrics["verbal_selective_auroc_f1_0.5"] = _safe_auroc(
                    vc[vp_attempted], correct_f1[vp_attempted])
                metrics["verbal_selective_auroc_em"] = _safe_auroc(
                    vc[vp_attempted], correct_em[vp_attempted])
            else:
                metrics["verbal_selective_auroc_f1_0.5"] = float("nan")
                metrics["verbal_selective_auroc_em"] = float("nan")
            metrics["verbal_selective_auroc"] = metrics["verbal_selective_auroc_f1_0.5"]

    if token_diagnostics is not None:
        # Diagnostics are intentionally separate from scalar-confidence metrics:
        # missing values remain missing and raw uncertainty never enters ECE/Brier.
        _add_token_diagnostic_metrics(metrics, token_diagnostics, ans, abst, scores)

    # A blank model emission is distinct from a non-empty abstention phrase.
    metrics["empty_output_rate"] = float(np.mean([not (p or "").strip() for p in predictions]))

    return metrics


# ════════════════════════════════════════════════════════
# THRESHOLD SWEEP / RANK AGREEMENT (logic unchanged, inputs validated)
# ════════════════════════════════════════════════════════

def analyze_threshold_sweep(confidences, is_correct, is_answerable,
                            thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    conf = _validate_confidences(confidences)
    is_correct = np.asarray(is_correct, dtype=bool)
    is_answerable = np.asarray(is_answerable, dtype=bool)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    rows = []
    for thr in thresholds:
        abstains = conf < thr
        attempts = ~abstains
        coverage = attempts.mean() * 100
        accuracy = is_correct[attempts].mean() * 100 if attempts.any() else 0.0
        should_abstain = ~is_answerable
        total_abst = int(abstains.sum())
        abs_precision = ((abstains & should_abstain).sum() / total_abst * 100
                         if total_abst else 0.0)
        rows.append({"threshold": thr, "coverage": coverage, "accuracy": accuracy,
                     "abstention_precision": abs_precision,
                     "num_abstentions": total_abst, "num_attempts": int(attempts.sum())})
    return pd.DataFrame(rows)


def compute_kendalls_w(rankings_matrix) -> Tuple[float, float]:
    """Kendall's W across n judges × k items.
    FIX-12: this formula has NO tie correction; with many tied ranks W is
    biased downward — prefer scipy.stats.friedmanchisquare for significance
    when ties are common."""
    from scipy.stats import chi2                        # FIX-9: lazy
    rankings_matrix = np.asarray(rankings_matrix, dtype=float)
    n, k = rankings_matrix.shape
    R = rankings_matrix.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    W = (12 * S) / (n ** 2 * (k ** 3 - k))
    chi_square = n * (k - 1) * W
    p_value = 1 - chi2.cdf(chi_square, k - 1)
    return W, p_value


# ════════════════════════════════════════════════════════
# PRETTY PRINT
# ════════════════════════════════════════════════════════

def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    print("\nOverall (consistent regime — FIX-2):")
    print(f"  Overall F1:        {metrics['overall_f1']:6.2f}%   "
          f"[{metrics.get('overall_ci_low', 0):.2f}, {metrics.get('overall_ci_high', 0):.2f}] 95% CI")
    print(f"  Overall EM:        {metrics['overall_em']:6.2f}%")
    if "overall_f1_legacy" in metrics:
        print(f"  Legacy overall F1: {metrics['overall_f1_legacy']:6.2f}%  "
              f"(old mixed regime — do not report)")
    if "rouge_l" in metrics:
        print(f"  ROUGE-L:           {metrics['rouge_l']:6.2f}%")
    if "bert_score" in metrics:
        print(f"  BERTScore:         {metrics['bert_score']:6.2f}%")
    print(f"  Total examples:    {metrics['total']:,}")

    print("\nAnswerable (HasAns):")
    print(f"  F1: {metrics['has_ans_f1']:6.2f}%   EM: {metrics['has_ans_em']:6.2f}%   "
          f"n={metrics['has_ans_total']:,}")

    print("\nUnanswerable / abstention:")
    print(f"  Abstention recall:    {metrics['abstention_recall']:6.2f}%  "
          f"(keyword-detected; alias 'no_ans_f1' — NOT official SQuAD2 NoAns F1)")
    print(f"  Abstention precision: {metrics['abstention_precision']:6.2f}%")
    print(f"  Abstention F1:        {metrics['abstention_f1']:6.2f}%")
    print(f"  Strictness:           {metrics['strictness']:6.2f}%  (false abstention on answerable)")
    print(f"  Coverage:             {metrics['coverage']:6.2f}%  (n unanswerable={metrics['no_ans_total']:,})")

    if "ece_f1_0.5" in metrics:
        print("\nConfidence — generated-sequence likelihood (exp mean logprob):")
        sens = metrics.get("behavior_ece_f1_0.5_sensitivity", {})
        spread = sens.get("ece_spread", float("nan")) if sens else float("nan")
        print(f"  ECE (F1>0.5): {metrics['ece_f1_0.5']:.4f}   Brier: {metrics['brier_f1_0.5']:.4f}"
              f"   (ECE spread across 5-20 bins: {spread:.4f})")
        print(f"  ECE (EM):     {metrics['ece_em']:.4f}   Brier: {metrics['brier_em']:.4f}")
        print("  NOTE: exp(mean logprob) is high on short canned abstentions, so the")
        print("        all-examples ECE above is optimistic; trust the selective (attempted-only) numbers.")
        print(f"  Answer/abstain decision 'AUROC': "
              f"{metrics.get('answer_abstain_decision_auroc', float('nan')):.4f}  "
              f"(enacted-policy quality, NOT independent calibration)")
        print(f"  Selective AUROC/AUPRC: {metrics.get('selective_auroc_f1_0.5', float('nan')):.4f} / "
              f"{metrics.get('selective_auprc_f1_0.5', float('nan')):.4f}")
        print(f"  Answerable-only selective AUROC: "
              f"{metrics.get('selective_answerable_only_auroc_f1_0.5', float('nan')):.4f}")
        print(f"  Conditional AURC/E-AURC: {metrics.get('selective_aurc_f1_0.5', float('nan')):.4f} / "
              f"{metrics.get('selective_eaurc_f1_0.5', float('nan')):.4f}")
    if "verbal_answerability_ece" in metrics:
        print("\nCalibration — verbalized answerability confidence:")
        print(f"  compliance: {metrics['verbal_compliance']:.1f}%   "
              f"AUROC: {metrics['verbal_answerability_auroc']:.4f}   "
              f"ECE: {metrics['verbal_answerability_ece']:.4f}   "
              f"Brier: {metrics['verbal_answerability_brier']:.4f}")
    elif "verbal_compliance" in metrics:
        print(f"\nVerbalized confidence: compliance {metrics['verbal_compliance']:.1f}% "
              f"(too few parsed outputs for calibration)")
    print("=" * 70)


# ════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("metrics.py self-test (corrected module)")

    preds = ["Paris",
             "I don't know",
             "The answer cannot be found in the context.",
             "Berlin",
             "1947"]
    gts = [["Paris", "Paris, France"], ["I don't know"], ["I don't know"],
           ["I don't know"], ["1947"]]
    flags = [True, False, False, False, True]
    confs = [0.95, 0.88, 0.72, 0.60, 0.91]
    verb = [0.9, None, 0.5, 0.8, 0.7]

    m = compute_metrics(preds, gts, flags, confidences=confs,
                        verbal_confidences=verb)
    print_metrics(m, "Self-test")

    # FIX-2: regime consistency — varied abstention phrasing counts as correct
    assert m["abstention_recall"] > 60, "keyword abstention should catch example 3"
    assert m["overall_f1"] > m["overall_f1_legacy"], \
        "consistent regime must remove the strictness penalty"
    # FIX-1: alias equality
    assert m["no_ans_f1"] == m["abstention_recall"]
    # FIX-10: verbal channel parsing and compliance accounting
    assert parse_verbal_confidence("Answer: Paris\nConfidence: 85") == 0.85
    assert parse_verbal_confidence("Answer: Paris\nConfidence: 85.5%") == 0.855
    assert parse_verbal_confidence("confidence = 0.42") == 0.42
    assert parse_verbal_confidence("Answer: Paris") is None
    assert parse_verbal_confidence("Confidence: 250") == 1.0   # clamped
    assert abs(m["verbal_compliance"] - 80.0) < 1e-9           # 4 of 5 parsed
    assert "verbal_answerability_ece" in m and "verbal_answerability_brier" in m
    assert "verbal_answerability_auroc" in m
    assert "selective_auroc_f1_0.5" in m
    assert "selective_auprc_f1_0.5" in m
    assert "selective_answerable_only_auroc_f1_0.5" in m
    assert "selective_aurc_f1_0.5" in m and "selective_eaurc_f1_0.5" in m
    assert "answer_abstain_decision_auroc" in m
    # FIX-3: hedge with the keyword far from the start is NOT an abstention
    assert is_abstention(
        "Most likely 1947 based on the second paragraph of the passage, "
        "although honestly I don't know the precise month") is False
    assert is_abstention("I don't know") is True
    # FIX-4: raw logprobs rejected
    for fn in (lambda: compute_ece([-2.3, -1.1], [1, 0]),
               lambda: compute_brier_score([-2.3, -1.1], [1, 0])):
        try:
            fn()
            raise SystemExit("ERROR: accepted raw logprobs")
        except ValueError:
            pass
    print("OK: calibration functions reject raw log-probs")

    # NEW: detector provenance + pluggable detector swap
    assert m["abstention_detector"] == "keyword"
    always_abstain = lambda ans: True          # trivial one-arg custom detector
    m2 = compute_metrics(preds, gts, flags, confidences=confs,
                         abstention_detector=always_abstain)
    assert m2["abstention_detector"] == "custom"
    assert m2["coverage"] == 0.0               # everything treated as abstention
    assert m2["abstention_recall"] == 100.0    # all unanswerables "caught"
    print("OK: pluggable abstention detector swaps in and drives the metrics")

    # NEW: question-aware (AbstentionBench-style) detector receives the question.
    # This fake judge abstains iff the question mentions 'moon' — proving the
    # question channel is actually wired through, and that a two-arg detector
    # and a one-arg detector both work via the adapter.
    def fake_judge(answer, question=None):
        return question is not None and "moon" in question.lower()
    q = ["What is the capital of France?",   # answerable, judge -> answer
         "Who lives on the moon?",           # unanswerable, judge -> abstain
         "What colour is the moon cheese?",  # unanswerable, judge -> abstain
         "Who is the king of Mars?",         # unanswerable, judge -> answer (miss)
         "What year was it?"]                # answerable, judge -> answer
    mq = compute_metrics(preds, gts, flags, confidences=confs,
                         abstention_detector=fake_judge, questions=q)
    assert mq["abstention_detector"] == "custom"
    # 2 of 3 unanswerables have 'moon' in the question -> recall 2/3
    assert abs(mq["abstention_recall"] - (2 / 3 * 100)) < 1e-9
    assert _detector_wants_question(fake_judge) is True
    assert _detector_wants_question(always_abstain) is False
    assert _detector_wants_question(KeywordAbstentionDetector()) is True  # (answer, question=None)
    print("OK: question-aware detector receives the question through the pipeline")

    # NEW: ECE binning-sensitivity is reported and internally consistent
    sens = m["behavior_ece_f1_0.5_sensitivity"]
    assert "ece_nbins_10" in sens and "ece_spread" in sens
    assert abs(sens["ece_nbins_10"] - m["ece_f1_0.5"]) < 1e-9   # default is 10 bins
    assert sens["ece_spread"] >= 0.0
    print("OK: ECE bin-count sensitivity reported")

    # NEW: verbal-confidence scale resolution, incl. the ambiguous bare integer
    assert parse_verbal_confidence("Confidence: 90%") == 0.90
    assert parse_verbal_confidence("Confidence: 1") == 0.01              # default: percent
    assert parse_verbal_confidence("Confidence: 1", bare_integer_is_percent=False) == 1.0
    print("OK: verbal-confidence scale resolution")

    # NEW: cheap, judge-free detector validation against human labels
    human = [False, True, True, False, False]      # matches keyword detector here
    rep = validate_detector(preds, human)
    assert rep["overall"]["f1"] == 1.0 and rep["overall"]["kappa"] == 1.0
    assert rep["false_negatives"] == [] and rep["false_positives"] == []
    human_fp = [False, True, False, False, False]  # human: ex.2 is NOT an abstention
    rep2 = validate_detector(preds, human_fp)
    assert rep2["false_positives"] == [2]
    assert abs(rep2["overall"]["precision"] - 0.5) < 1e-9
    rep3 = validate_detector(preds, human, splits=["id", "id", "ood", "ood", "id"])
    assert set(rep3["per_split"]["split"]) == {"id", "ood"}
    idxs = sample_for_annotation(100, 20, strata=["id"] * 70 + ["ood"] * 30,
                                 oversample={"ood"})
    assert idxs == sorted(set(idxs)) and all(0 <= i < 100 for i in idxs)
    print("OK: judge-free detector validation (agreement vs human labels)")

    # NEW: annotation CSV round-trip + inter-annotator agreement
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "abst_annot_selftest.csv")
    to_annotation_csv(tmp, preds, indices=[10, 11, 12, 13, 14],
                      questions=q, splits=["id", "id", "ood", "ood", "id"])
    _df = pd.read_csv(tmp)
    assert list(_df["idx"]) == [10, 11, 12, 13, 14]
    assert (_df["is_abstention"].isna()).all()          # starts empty
    _df["is_abstention"] = ["FALSE", "TRUE", "abstain", "answer", "0"]  # mixed tokens
    _df.to_csv(tmp, index=False)
    loaded = from_annotation_csv(tmp)
    assert loaded["indices"] == [10, 11, 12, 13, 14]
    assert loaded["human_abstention_labels"] == [False, True, True, False, False]
    rep_csv = validate_detector(loaded["predictions"], loaded["human_abstention_labels"],
                                questions=loaded.get("questions"), splits=loaded.get("splits"))
    assert rep_csv["overall"]["f1"] == 1.0
    _df.loc[0, "is_abstention"] = ""                     # blank one -> must raise
    _df.to_csv(tmp, index=False)
    try:
        from_annotation_csv(tmp)
        raise SystemExit("ERROR: accepted a half-labeled file")
    except ValueError:
        pass
    ia = inter_annotator_agreement([False, True, True, False, False],
                                   [False, True, False, False, False])
    assert 0.0 <= ia["raw_agreement"] <= 1.0 and ia["n"] == 5
    os.remove(tmp)
    print("OK: annotation CSV round-trip + inter-annotator agreement")

    # NEW: annotation_summary — prevalence + per-split positive counts
    summ = annotation_summary(
        {"human_abstention_labels": [False, True, True, False, False],
         "splits": ["id", "id", "ood", "ood", "id"]},
        min_positives=2)
    assert summ["n_positives"] == 2 and abs(summ["prevalence"] - 0.4) < 1e-9
    assert set(summ["underpowered"]) == {"id", "ood"}      # each split has 1 positive < 2
    summ_ok = annotation_summary(
        {"human_abstention_labels": [False, True, True, False, False],
         "splits": ["id", "id", "ood", "ood", "id"]},
        min_positives=1)
    assert summ_ok["underpowered"] == []
    print("OK: annotation_summary flags underpowered splits")

    print("\nALL SELF-TESTS PASSED")
