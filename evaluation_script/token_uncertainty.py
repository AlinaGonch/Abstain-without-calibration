"""Compact, online token-level uncertainty diagnostics for free generation.

This module deliberately never stores per-vocabulary logits.  Callers retain
only scalar statistics for answer-span steps, plus optional compact traces.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


FINAL_RESPONSE_MARKER = "final response:"
_MARKER_PUNCT = set(":;,.!?-–—•)]}\"'")
REFUSAL_TEMPLATES = [
    "I don't know",
    "I cannot answer this question",
    "There is no information related to the question",
    "The answer is not provided in the context",
]


def final_response_token_span(
    token_ids: Sequence[int], tokenizer: Any, n: Optional[int] = None,
) -> Tuple[int, int, bool]:
    """Return the existing answer span and whether its format marker was seen.

    The incremental decode and character-to-token mapping intentionally match
    the legacy runners.  No marker retains the old whole-generation fallback.
    """
    ids = [int(x) for x in token_ids[:n]] if n is not None else [int(x) for x in token_ids]
    pieces = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in ids]
    text = "".join(pieces)
    cumulative, total = [], 0
    for piece in pieces:
        total += len(piece)
        cumulative.append(total)
    marker_at = text.lower().rfind(FINAL_RESPONSE_MARKER)
    if marker_at < 0:
        return 0, len(ids), False
    answer_char = marker_at + len(FINAL_RESPONSE_MARKER)
    confidence_at = text.lower().find("confidence", answer_char)
    end_char = confidence_at if confidence_at >= 0 else len(text)
    start = 0
    while start < len(ids) and cumulative[start] <= answer_char:
        start += 1
    end = start
    # Include the token that ends exactly at the answer boundary.  The legacy
    # '<' condition dropped a one-token answer when no Confidence line followed.
    while end < len(ids) and cumulative[end] <= end_char:
        end += 1
    # Do not attribute punctuation/whitespace shared with the marker to answer.
    while start < end:
        stripped = pieces[start].strip()
        if stripped and not all(ch in _MARKER_PUNCT for ch in stripped):
            break
        start += 1
    while end > start and not pieces[end - 1].strip():
        end -= 1
    return start, max(start, end), True


def compute_step_uncertainty(logits: torch.Tensor, selected_token_id: int) -> Dict[str, float]:
    """Compute scalar uncertainty statistics from one raw-logit vector."""
    values = logits.detach().float()
    log_probs = torch.log_softmax(values, dim=-1)
    probs = torch.softmax(values, dim=-1)
    selected = int(selected_token_id)
    top2 = torch.topk(probs, k=min(2, probs.numel())).values
    margin = top2[0] - (top2[1] if top2.numel() == 2 else top2.new_zeros(()))
    entropy = -(probs * log_probs).sum()
    return {
        "selected_logprob": float(log_probs[selected]),
        "selected_probability": float(probs[selected]),
        "entropy": float(entropy),
        "top1_top2_margin": float(margin),
    }


def summarize_answer_span(
    step_stats: Sequence[Dict[str, float]],
    token_ids: Optional[Sequence[int]] = None,
    token_texts: Optional[Sequence[str]] = None,
    save_token_traces: bool = False,
) -> Dict[str, Any]:
    """Summarize answer steps; an empty span is explicitly nullable."""
    out: Dict[str, Any] = {"answer_token_count": len(step_stats)}
    if not step_stats:
        out.update({
            "answer_mean_logprob": None, "answer_min_logprob": None,
            "answer_logprob_std": None, "answer_mean_entropy": None,
            "answer_max_entropy": None, "answer_first_token_entropy": None,
            "answer_mean_top1_top2_margin": None,
            "answer_min_top1_top2_margin": None, "answer_first_token_margin": None,
            "answer_first_token_selected_probability": None,
        })
        if save_token_traces:
            out.update({"answer_token_ids": [], "answer_token_texts": [],
                        "answer_token_logprobs": [], "answer_token_entropies": [],
                        "answer_token_margins": []})
        return out
    lps = [s["selected_logprob"] for s in step_stats]
    entropies = [s["entropy"] for s in step_stats]
    margins = [s["top1_top2_margin"] for s in step_stats]
    out.update({
        "answer_mean_logprob": float(sum(lps) / len(lps)),
        "answer_min_logprob": float(min(lps)),
        "answer_logprob_std": float(math.sqrt(sum((x - sum(lps) / len(lps)) ** 2 for x in lps) / len(lps))),
        "answer_mean_entropy": float(sum(entropies) / len(entropies)),
        "answer_max_entropy": float(max(entropies)),
        "answer_first_token_entropy": float(entropies[0]),
        "answer_mean_top1_top2_margin": float(sum(margins) / len(margins)),
        "answer_min_top1_top2_margin": float(min(margins)),
        "answer_first_token_margin": float(margins[0]),
        "answer_first_token_selected_probability": float(step_stats[0]["selected_probability"]),
    })
    if save_token_traces:
        out.update({
            "answer_token_ids": [int(x) for x in token_ids or []],
            "answer_token_texts": list(token_texts or []),
            "answer_token_logprobs": lps, "answer_token_entropies": entropies,
            "answer_token_margins": margins,
        })
    return out


def compute_decision_abstention_stats(
    logits: torch.Tensor, abstention_token_ids: Iterable[int],
) -> Dict[str, Optional[float]]:
    """Exploratory first-token refusal-prefix score, not abstention probability."""
    ids = sorted({int(token_id) for token_id in abstention_token_ids
                  if 0 <= int(token_id) < logits.numel()})
    if not ids:
        return {"decision_refusal_prefix_first_token_mass": None,
                "decision_refusal_prefix_vs_best_other_logit_margin": None,
                "decision_refusal_prefix_vs_other_logsumexp_margin": None}
    values = logits.detach().float()
    probs = torch.softmax(values, dim=-1)
    index = torch.tensor(ids, device=values.device, dtype=torch.long)
    mass = probs.index_select(0, index).sum().clamp(0.0, 1.0)
    abstain_best = values.index_select(0, index).max()
    other = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    other[index] = False
    other_best = values[other].max() if other.any() else values.new_tensor(float("-inf"))
    # Set-level evidence: accounts for all candidate refusal-prefix tokens,
    # rather than retaining only the largest individual refusal-prefix logit.
    refusal_set_lse = torch.logsumexp(values.index_select(0, index), dim=0)
    other_set_lse = torch.logsumexp(values[other], dim=0) if other.any() else values.new_tensor(float("-inf"))
    return {
        "decision_refusal_prefix_first_token_mass": float(mass),
        "decision_refusal_prefix_vs_best_other_logit_margin": float(abstain_best - other_best),
        "decision_refusal_prefix_vs_other_logsumexp_margin": float(refusal_set_lse - other_set_lse),
    }


@torch.inference_mode()
def score_refusal_templates(model: Any, tokenizer: Any, prompt_token_ids: Sequence[int]) -> Dict[str, Any]:
    """Teacher-force complete refusal templates; values are ranking scores only.

    The four candidates are batched into one forward pass.  This is optional in
    runners because it is an additional probe, not part of free generation.
    """
    rows, lengths = [], []
    for template in REFUSAL_TEMPLATES:
        continuation = tokenizer(template, add_special_tokens=False)["input_ids"]
        rows.append(list(prompt_token_ids) + continuation)
        lengths.append(len(continuation))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(map(len, rows))
    input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=model.device)
    attention_mask = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = torch.tensor(row, device=model.device)
        attention_mask[i, :len(row)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    results, means = [], []
    prefix_len = len(prompt_token_ids)
    for i, (template, count) in enumerate(zip(REFUSAL_TEMPLATES, lengths)):
        if count == 0:
            result = {"template": template, "total_logprob": None, "mean_logprob": None,
                      "min_logprob": None, "token_count": 0}
        else:
            target = input_ids[i, prefix_len:prefix_len + count]
            # Causal logits at position p predict token p+1.
            lp = torch.log_softmax(logits[i, prefix_len - 1:prefix_len - 1 + count], dim=-1)
            selected = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            result = {"template": template, "total_logprob": float(selected.sum()),
                      "mean_logprob": float(selected.mean()), "min_logprob": float(selected.min()),
                      "token_count": int(count)}
            means.append(result["mean_logprob"])
        results.append(result)
    valid = [(r["template"], r["mean_logprob"]) for r in results if r["mean_logprob"] is not None]
    logsumexp_mean = (float(torch.logsumexp(torch.tensor([v for _, v in valid]), 0))
                      if valid else None)
    best = max(valid, key=lambda item: item[1])[0] if valid else None
    return {"refusal_template_scores": results,
            "refusal_template_max_mean_logprob": max((v for _, v in valid), default=None),
            "refusal_template_logsumexp_mean_score": logsumexp_mean,
            "best_refusal_template": best}
