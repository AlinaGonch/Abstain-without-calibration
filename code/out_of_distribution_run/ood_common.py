"""
ood_common.py  — v2 (reconciled with your real data_loader / prompt_template)
=============================================================================
Shared primitives for the OOD abstention loaders.

Reconciled to your uploaded code:
  * Canonical record = {id, messages, ground_truths, is_answerable} exactly
    (CanonicalExample emits those 4 keys, plus an ADDITIVE `metadata` dict that
    your training/generation ignores but the scorer/judge use). Drop metadata
    if you prefer strict parity — nothing in generation reads it.
  * Unanswerable items carry ground_truths == [ABSTAIN_STRING] (["I don't know"]),
    matching SquadDataLoader, so EM against ground_truths already rewards
    abstention. Items that have no gold string at all (e.g. NoMIRACL relevant)
    use ground_truths == [] to signal "decision-level only, no EM".
  * Contract markers match prompt_template.py: "Final response:" / "I don't know"
    and the self_reflect "Confidence:" line.

No train/val splitting anywhere (per request).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---- contract (must match prompt_template.py) -----------------------------
FINAL_RESPONSE_MARKER = "Final response:"
ABSTAIN_STRING = "I don't know"
CONFIDENCE_MARKER = "Confidence:"

# score_kind values stored in metadata, used by the scorer
SCORE_NUMERIC = "numeric"   # gsm8k-abstain
SCORE_MC = "mc"             # gpqa-abstain, mmlu-pro
SCORE_EM = "em"             # faitheval(counterfactual), hotpot answerable
SCORE_NONE = "none"         # no gold answer string (decision-level only)

# gold response classes (third class supports your conflict_metrics)
GOLD_ANSWER = "answer"
GOLD_ABSTAIN = "abstain"
GOLD_CONFLICT = "conflict"
# detected-only class (never a gold label): empty / truncated / format-dead
# generations. Kept OUT of the abstain class so decode failures and
# max_new_tokens truncations cannot inflate correct-abstention counts.
RESP_INVALID = "invalid"

_ABSTAIN_PATTERNS = [
    r"i\s+don['’`]?t\s+know",
    r"\bcannot\s+be\s+(answered|determined|solved)\b",
    r"\b(not\s+enough|insufficient|missing)\s+(information|context|data|evidence)\b",
    r"\bunanswerable\b",
    r"\bnot\s+given\b",
    r"\bnone\s+of\s+(the\s+)?(above|these|the\s+options)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)
_CONFLICT_RE = re.compile(
    r"\b(conflict|conflicting|inconsistent|contradict|disagree|two different)\b",
    re.IGNORECASE,
)


@dataclass
class CanonicalExample:
    id: str
    messages: List[Dict[str, str]]
    ground_truths: List[str]
    is_answerable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def gold_for(is_answerable: bool, answers: Optional[List[str]], has_gold_string: bool) -> List[str]:
    """Apply the ground_truths convention uniformly."""
    if not is_answerable:
        return [ABSTAIN_STRING]
    if has_gold_string and answers:
        return list(answers)
    return []  # answerable but no gold string -> decision-level only


# ---- sentence split + abstain construction (AbstentionBench method) --------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]


def find_final_question(text: str) -> Tuple[List[str], Optional[str]]:
    sents = split_sentences(text)
    if not sents:
        return [], None
    q_idx = next((i for i in range(len(sents) - 1, -1, -1)
                  if sents[i].rstrip().endswith("?")), len(sents) - 1)
    return sents[:q_idx], sents[q_idx]


def make_abstain_variant(problem_text: str) -> Tuple[str, Optional[str], bool]:
    context, question = find_final_question(problem_text)
    if question is None:
        return problem_text, None, False
    has_context = len(context) >= 1 and sum(len(c) for c in context) >= 15
    return (problem_text, question, True) if has_context else (problem_text, None, False)


# ---- numeric + MC parsing -------------------------------------------------
_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_MC_LETTER_RE = re.compile(r"\b([A-J])\b")
# Anchored forms: "B)", "B.", "(B)", "option B", "answer is B", "answer: B"
_MC_ANCHORED_RE = re.compile(
    r"(?:\(\s*([A-J])\s*\)"            # (B)
    r"|\b([A-J])\s*[).:]"              # B)  B.  B:
    r"|\b(?:option|choice|answer)\s*(?:is|:)?\s*\(?([A-J])\b)",  # option B / answer is B
    re.IGNORECASE,
)
# A standalone capital I is almost always the English pronoun ("I think ...",
# "I don't know"), not the tenth option letter. Only trust bare I when it is
# immediately anchored (handled above); otherwise drop it from bare matching.
_MC_BARE_NO_I_RE = re.compile(r"\b([A-HJ])\b")


def extract_gsm8k_gold(answer_field: str) -> Optional[str]:
    m = _GSM8K_FINAL_RE.search(answer_field or "")
    return m.group(1).replace(",", "") if m else None


_MARKER_LOW = FINAL_RESPONSE_MARKER.lower()
_CONF_CUT_RE = re.compile(r"\n?\s*confidence\s*[:=]", re.IGNORECASE)


def _final_segment(text: str) -> str:
    """Everything after the LAST 'Final response:' marker (case-insensitive),
    CUT BEFORE the following 'Confidence' line. Mirrors the ID pipeline's
    _extract_final_response: without the Confidence cut, the trailing
    'Confidence: NN' contaminates every downstream parse — most damagingly
    extract_predicted_number, which takes the LAST number in the segment and
    would score the confidence value instead of the answer on _conf
    strategies. No marker -> whole text (still Confidence-cut, matching ID's
    fallback)."""
    low = text.lower()
    fr = low.rfind(_MARKER_LOW)
    seg = text[fr + len(FINAL_RESPONSE_MARKER):] if fr != -1 else text
    m = _CONF_CUT_RE.search(seg)
    if m is not None:
        seg = seg[:m.start()]
    return seg


def extract_predicted_number(text: str) -> Optional[str]:
    nums = _NUMBER_RE.findall(_final_segment(text)) or _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _to_float(s) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def numbers_equal(pred, gold, rel_tol: float = 1e-4) -> bool:
    pf, gf = _to_float(pred), _to_float(gold)
    if pf is None or gf is None:
        return False
    return abs(pf) <= rel_tol if gf == 0 else abs(pf - gf) / max(abs(gf), 1e-9) <= rel_tol


def parse_mc_choice(text: str, options: List[str]) -> Optional[str]:
    """Parse the chosen option letter from the final segment.

    Order of preference:
      1. anchored letter forms — '(B)', 'B)', 'B.', 'option B', 'answer is B' —
         these are unambiguous and may legitimately be the letter I;
      2. bare letters EXCLUDING I — a standalone 'I' in prose is the pronoun
         ('I think the answer is C' must parse as C, and on 10-option
         MMLU-Pro a pronoun hit would otherwise be a *valid but wrong* parse;
      3. option-text containment fallback (unchanged).
    """
    seg = _final_segment(text)
    m = _MC_ANCHORED_RE.search(seg)
    if m:
        letter = next(g for g in m.groups() if g)
        return letter.upper()
    m = _MC_BARE_NO_I_RE.search(seg)
    if m:
        return m.group(1)
    seg_low = seg.lower()
    for i, opt in enumerate(options):
        if opt and str(opt).lower() in seg_low:
            return chr(ord("A") + i)
    return None


# ---- response-type + confidence -------------------------------------------
def classify_response_type(text: str) -> str:
    """Detected property (NOT gold): 'abstain' | 'conflict' | 'answer' | 'invalid'.

    Empty or whitespace-only text — including a 'Final response:' marker with
    nothing after it (truncation signature) — is 'invalid', not 'abstain':
    a decode failure is not a decision, and counting it as abstention would
    inflate abstention recall on unanswerable items. Ordering is abstain
    before conflict by design (an 'I don't know' verdict wins even when the
    reasoning mentions contradiction)."""
    if not text or not text.strip():
        return RESP_INVALID
    seg = _final_segment(text)
    if not seg.strip():
        return RESP_INVALID
    if _ABSTAIN_RE.search(seg):
        return GOLD_ABSTAIN
    if _CONFLICT_RE.search(seg):
        return GOLD_CONFLICT
    return GOLD_ANSWER


def parse_verbal_confidence(text: str) -> Optional[float]:
    """Pull the 'Confidence: NN' score (0-100) -> [0,1].

    Robustness rules (each guards against a previously observed artifact):
      * Key off the LAST 'Confidence:' marker — self_reflect traces can carry
        an initial-confidence line in the reasoning before the final one.
      * The number must appear immediately after the marker (only whitespace
        or a '~' allowed in between). This rejects placeholder regurgitation
        like 'Confidence: <an integer from 0 to 100 ...>', where a greedy
        digit search would silently extract 0 — a value the model never
        asserted.
      * No numeric fallback: any non-match returns None so compliance can be
        measured honestly downstream.
    """
    if CONFIDENCE_MARKER not in text:
        return None
    after = text.rsplit(CONFIDENCE_MARKER, 1)[1]
    m = re.match(r"[ \t~]*(\d{1,3})(?:\s*/\s*100)?\b", after)
    if not m:
        return None
    v = max(0.0, min(100.0, float(m.group(1))))
    return v / 100.0


def text_contains_answer(response: str, golds: List[str]) -> bool:
    """Containment-style EM used for free-text QA (faitheval cf / hotpot).

    Short golds (<= 4 chars, e.g. 'no', '3', 'red') are matched with word
    boundaries so they cannot fire inside longer words ('no' in 'north',
    '3' in '39'). Longer golds keep plain containment, which tolerates
    surrounding punctuation and inflection the way the ID scorer does."""
    seg = _final_segment(response).lower()
    for g in golds:
        g = (g or "").strip().lower()
        if not g or g == ABSTAIN_STRING.lower():
            continue
        if len(g) <= 4:
            if re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", seg):
                return True
        elif g in seg:
            return True
    return False


# ---- IO -------------------------------------------------------------------
def write_jsonl(path: str, examples) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            row = ex.to_dict() if isinstance(ex, CanonicalExample) else ex
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
