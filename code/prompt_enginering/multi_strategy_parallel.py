"""
Multi-strategy evaluation: fine-tuned ratio adapters vs. pretrained base model
==============================================================================
Runs one or more PROMPTING STRATEGIES (zero_shot, cot, few_shot_balance,
self_reflect) through:
  1. the pretrained base model (no adapter)  -> dotted baselines in the plot
  2. every ratio adapter (0.0 ... 1.0)       -> solid curves in the plot

Each (strategy x job) pair writes its own predictions file under a
per-strategy sub-directory, so resume/aggregation stay independent across
strategies. Run a single strategy with --strategies zero_shot, or several at
once with --strategies zero_shot cot few_shot_balance self_reflect.

The only thing that changes across strategies is the PROMPT (and, for the
reasoning strategies, the generation length budget). Generation, the dual
logit/verbal confidence extraction, the metrics, the resume logic and the
GPU parallelism are all strategy-agnostic and shared verbatim, so the
zero_shot numbers this produces are bit-for-bit identical to the original
script.

Fixes relative to the original pipeline:
  * Adapter switching uses load_adapter()/set_adapter() on ONE PeftModel
    wrapper -- no repeated PeftModel.from_pretrained() on the same base
    (which injects/stacks LoRA modules and can corrupt later ratios).
  * Baseline is evaluated BEFORE wrapping, on the untouched base model.
  * Adapter repo names match what the training script pushes:
        AlinaGonch/{MODEL_SHORT}-squad-ratio-{ratio:.2f}
  * Eval set is a STRATIFIED, shuffled sample of SQuAD2 validation
    (the raw first-1000 slice is grouped by article and unbalanced).
  * Overall F1 uses ONE consistent scoring regime (SQuAD2-official style):
    unanswerable examples score 1.0 if the model abstained (same
    is_abstention predicate as the per-class metric), else 0.0.
    The legacy token-overlap-vs-"I don't know" overall is also reported
    so you can quantify the discrepancy ("strictness penalty").
  * Bootstrap 95% CIs per ratio so plateau wiggles can be judged
    against sampling noise.
  * HF token from environment only.

Usage:
    HF_TOKEN=hf_xxx python multi_strategy_parallel.py
    # default runs all four strategies; restrict with --strategies:
    HF_TOKEN=hf_xxx python multi_strategy_parallel.py --strategies cot self_reflect
    # optional: --n_eval 2000 --batch_size 24 --out_dir results/squad2/compare
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import string
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import shutil
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ════════════════════════════════════════════════════════
# CONFIG  (mirrors the training script's presets)
# ════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN")  # never hardcode -- rotate the leaked one!

# Overridable via --model_id / --lora_rank. Propagated through EVAL_* env vars
# because mp 'spawn' workers RE-IMPORT this module: a global mutated in main()
# would silently revert to the default inside every worker. main() sets both
# the global and the env var; workers pick the env var up here at import.
MODEL_ID = os.environ.get("EVAL_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
LORA_RANK = int(os.environ.get("EVAL_LORA_RANK", "16"))

MODEL_PRESETS = {
    # WARNING/verify: training script says short "llama32-3b" for Llama-3.2-3B,
    # this file says "llama3.2". One of them mismatches the actual Hub repos —
    # check https://huggingface.co/AlinaGonch and align. The startup preflight
    # below will catch it either way (missing-repo error before any GPU work).
    "meta-llama/Llama-3.2-3B-Instruct":  {"short": "llama3.2",  "no_system_role": False},
    "meta-llama/Llama-3.1-8B-Instruct":  {"short": "llama31-8b",  "no_system_role": False},
    "microsoft/Phi-3-mini-4k-instruct":  {"short": "phi3-mini",   "no_system_role": False},
    "microsoft/Phi-3-medium-4k-instruct":{"short": "phi3-medium", "no_system_role": False},
    "ibm-granite/granite-4.1-3b":        {"short": "granite41-3b","no_system_role": False},
    "ibm-granite/granite-4.1-8b":        {"short": "granite41-8b","no_system_role": False},
    # The ORIGINAL sweep's Qwen3-4B adapters were trained on the HYBRID
    # release and own the qwen3-4b-* Hub namespace:
    "Qwen/Qwen3-4B":                     {"short": "qwen3-4b",    "no_system_role": False},
    # The 2507 Instruct is a DIFFERENT base model with its OWN namespace.
    # (Previously this row said "qwen3-4b", which would have loaded the
    # hybrid model's adapters onto the 2507 base — cross-model contamination.)
    "Qwen/Qwen3-4B-Instruct-2507":       {"short": "qwen3-4b-instruct", "no_system_role": False},
    "Qwen/Qwen3-14B":                    {"short": "qwen3-14b",   "no_system_role": False},
    "google/gemma-3-4b-it":              {"short": "gemma3-4b",   "no_system_role": True},
    "google/gemma-3-12b-it":             {"short": "gemma3-12b",  "no_system_role": True},
}
MODEL_SHORT = MODEL_PRESETS[MODEL_ID]["short"]
NO_SYSTEM_ROLE = MODEL_PRESETS[MODEL_ID]["no_system_role"]

# MUST match what train_one_ratio() pushed to the Hub.
# Rank suffix mirrors the training script: '' for the default r=16 sweep,
# '-r{rank}' for rank-ablation adapters.
ADAPTER_REPO_TMPL = "AlinaGonch/{short}-squad-ratio-{ratio:.2f}"


def rank_suffix(sep: str = "-") -> str:
    return "" if LORA_RANK == 16 else f"{sep}r{LORA_RANK}"


def adapter_repo(ratio: float) -> str:
    return ADAPTER_REPO_TMPL.format(short=MODEL_SHORT, ratio=ratio) + rank_suffix("-")

# Single source of truth for the SQuAD2 source. MUST be the SAME dataset the
# training script used for VAL_SPLIT, otherwise the held-out reconstruction
# below selects a different first-n slice and the exclusion is meaningless.
SQUAD_V2 = "rajpurkar/squad_v2"

RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED = 42
MAX_SEQ_LENGTH = 1024
# Generation budget is strategy-dependent: zero_shot / few_shot only need the
# answer + the "Confidence: NN" line, but cot / self_reflect emit a reasoning
# trace BEFORE the "Final response:" line, so a tight cap would truncate them
# mid-thought and starve the parser of the Final response:/Confidence: block.
# per-strategy values live in STRATEGIES below; this is the fallback default.
MAX_NEW_TOKENS = 64          # answer + "Confidence: NN" line (zero_shot/few_shot)

# Training-time instruction PLUS a verbalized-confidence elicitation line.
# The verbal confidence is an INDEPENDENT channel: it is parsed from the text
# and calibrated separately; its tokens are EXCLUDED from the logit-based
# confidence so the two signals never contaminate each other.
#
# CAVEAT for the fine-tuned adapters: they were trained to emit ONLY the
# answer string, so they may ignore the Confidence line. That is measured,
# not hidden -- `verbal_compliance` reports the fraction of outputs where a
# confidence could be parsed, and verbal calibration is computed only over
# that subset. Low compliance after fine-tuning is itself a finding
# (format-following lost to narrow SFT).
# ──────────────────────────────────────────────────────────────────────────
# PROMPTING STRATEGIES  (prompts delegated to prompt_template.py)
# ──────────────────────────────────────────────────────────────────────────
# Prompts are NOT defined here. They live in prompt_template.py as the *_conf
# strategy builders (zero_shot_conf / few_shot_balance_conf / cot_conf /
# self_reflect_conf), so the ratio-sweep uses the SAME canonical prompt module
# as the rest of the pipeline (sft/dpo/OOD) rather than a divergent inline copy.
#
# Every *_conf builder emits the SAME output contract, which the extractor and
# token-span logic below key off:
#       Final response: <final answer or "I don't know">
#       Confidence: <0-100 answerability>
# Reasoning strategies (cot_conf, self_reflect_conf) place their trace BEFORE
# the Final response line; the extractor takes the LAST "Final response:" and
# the answer-span logit confidence excludes both the reasoning and the
# Confidence tokens.
#
# The public strategy KEYS stay canonical (zero_shot / few_shot_balance / cot /
# self_reflect) so output subdirs, summary columns and plots line up
# column-for-column with the OOD pipeline; each maps to its prompt_template
# builder name and a generation budget (reasoning strategies need more tokens
# to reach the Final response / Confidence block).
STRATEGIES = {
    "zero_shot":        {"template": "zero_shot_conf",        "max_new_tokens": 96},
    "few_shot_balance": {"template": "few_shot_balance_conf", "max_new_tokens": 96},
    "cot":              {"template": "cot_conf",              "max_new_tokens": 320},
    "self_reflect":     {"template": "self_reflect_conf",     "max_new_tokens": 320},
}
DEFAULT_STRATEGIES = ["zero_shot", "few_shot_balance", "cot", "self_reflect"]

# ════════════════════════════════════════════════════════
# METRICS -- single source of truth: the corrected metrics.py module
# (consistent scoring regime, validated [0,1] confidences, dual-channel
# calibration, deprecated-alias compatibility). Keep metrics.py next to
# this script or on PYTHONPATH.
# ════════════════════════════════════════════════════════
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.metrics_extended import (
    compute_metrics,
    parse_verbal_confidence,
    print_metrics,
)
from evaluation.token_uncertainty import (
    compute_decision_abstention_stats,
    compute_step_uncertainty,
    final_response_token_span,
    score_refusal_templates,
    summarize_answer_span,
)

# Canonical prompt module (shared with sft/dpo/OOD). The *_conf builders emit
# the Final response / Confidence contract this runner parses.
from utils.prompt_template import get_chat_messages


# ════════════════════════════════════════════════════════
# DATA -- stratified, shuffled eval subset
# ════════════════════════════════════════════════════════
def load_early_stopping_ids(n: int = 1000) -> set:
    """Reconstruct the EXACT set of example ids the training run saw for
    in-training evaluation / early stopping, so the test set holds them out.

    Mirrors the training selection bit-for-bit. Training did:

        val_ds = load_dataset(VAL_SPLIT, split="validation").select(range(n))

    i.e. the FIRST n rows of the squad_v2 validation split in natural (on-disk)
    order, with NO shuffle. We reproduce that exact slice here and take its
    canonical SQuAD `id`s. Matching downstream is by `id`, so it is robust to
    any later row reordering. Reproducibility depends only on the dataset
    version being the same one training pulled (pinned by SQUAD_V2)."""
    ds = load_dataset(SQUAD_V2, split="validation").select(range(n))
    return set(ds["id"])


def build_eval_set(n_eval: int, seed: int, exclude_ids: set | None = None):
    val = load_dataset(SQUAD_V2, split="validation")

    # Hold out the in-training (early-stopping) slice by canonical id BEFORE
    # stratifying -- checkpoint selection saw these, so scoring on them leaks.
    if exclude_ids:
        before = len(val)
        val = val.filter(lambda ex: ex["id"] not in exclude_ids)
        removed = before - len(val)
        print(f"Excluded {removed} early-stopping examples "
              f"({len(val)} of {before} remain).")
        if removed != len(exclude_ids):
            print(f"  WARNING: removed {removed} but expected {len(exclude_ids)} "
                  f"excluded ids. The held-out slice may not match training -- "
                  f"check that SQUAD_V2 ('{SQUAD_V2}') and the dataset version "
                  f"match the training run's VAL_SPLIT before reporting these "
                  f"as held-out numbers.")

    answerable = val.filter(lambda x: len(x["answers"]["text"]) > 0)
    unanswerable = val.filter(lambda x: len(x["answers"]["text"]) == 0)
    n_half = n_eval // 2
    subset = concatenate_datasets([
        answerable.shuffle(seed=seed).select(range(min(n_half, len(answerable)))),
        unanswerable.shuffle(seed=seed).select(range(min(n_eval - n_half, len(unanswerable)))),
    ]).shuffle(seed=seed)
    print(f"Eval set: {len(subset)} examples "
          f"({sum(1 for ex in subset if len(ex['answers']['text']) > 0)} answerable)")
    return subset


# ════════════════════════════════════════════════════════
# PROMPTING + GENERATION
# ════════════════════════════════════════════════════════
def build_messages(context, question, strategy_spec):
    """Build the chat-message list for one example under a given strategy.

    Prompts come from prompt_template.get_chat_messages via the strategy's
    `template` name (a *_conf builder), so this runner shares the canonical
    prompt module with the rest of the pipeline. The only runner-local concern
    is NO_SYSTEM_ROLE (Gemma 3), which rejects role="system": we fold the
    system turn into the first user turn, preserving any few-shot user/assistant
    exemplars in between.
    """
    msgs = get_chat_messages(strategy_spec["template"], context=context,
                             question=question)
    if NO_SYSTEM_ROLE and msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"]
        folded, out = False, []
        for m in msgs[1:]:
            if not folded and m["role"] == "user":
                out.append({"role": "user",
                            "content": f"{system}\n\n{m['content']}"})
                folded = True
            else:
                out.append(m)
        if not folded:   # no user turn (shouldn't happen) -> prepend as user
            out.insert(0, {"role": "user", "content": system})
        return out
    return msgs


def make_prompt(tokenizer, context, question, strategy_spec):
    return tokenizer.apply_chat_template(
        build_messages(context, question, strategy_spec),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # silently ignored by non-Qwen templates
    )


def get_terminators(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    added = tokenizer.get_added_vocab()
    for tok in ("<|eot_id|>", "<|end|>", "<|end_of_text|>", "<|im_end|>", "<end_of_turn>"):
        if tok in added:
            ids.add(added[tok])
    return list(ids)


_FINAL_MARKER = "final response:"   # matched case-insensitively


def _abstention_first_token_ids(tokenizer):
    """Exploratory refusal-prefix ids; not interpreted as abstention probability."""
    ids = set()
    for phrase in ("I don't know", "I cannot answer this question",
                   "There is no information related to the question",
                   "The answer is not provided in the context"):
        for prefix in ("", " ", "\n"):
            encoded = tokenizer(prefix + phrase, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.add(encoded[0])
    return sorted(ids)


def _final_response_token_span(gen_tokens, tokenizer, n):
    """Token-index span [start, end) of the FINAL-RESPONSE segment, used to
    restrict the log-prob confidence to the answer tokens only.

    The generation may contain a reasoning / reflection trace BEFORE the answer
    (cot_conf, self_reflect_conf), so we anchor on the LAST "Final response:"
    marker and end before the following "Confidence" line. Reasoning tokens and
    the verbalized-confidence tokens are therefore excluded from the answer-span
    confidence. Robust to markers straddling token boundaries: we decode
    incrementally, record the cumulative character length after each token, then
    map the marker character positions back to token indices. Falls back to the
    whole span when no "Final response:" marker is present.
    """
    # Compatibility wrapper: both ID and OOD now call the same implementation.
    return final_response_token_span(gen_tokens[:n].tolist(), tokenizer)
    running = ""
    cum = []                      # cum[t] = len(running) after decoding token t
    for t in range(n):
        running += tokenizer.decode([gen_tokens[t].item()], skip_special_tokens=True)
        cum.append(len(running))
    low = running.lower()
    fr = low.rfind(_FINAL_MARKER)             # LAST final-response marker
    if fr == -1:
        return 0, n                            # format ignored -> whole span
    ans_char = fr + len(_FINAL_MARKER)
    conf = low.find("confidence", ans_char)
    end_char = conf if conf != -1 else len(running)

    # first token whose cumulative length exceeds the answer-start char
    start = 0
    while start < n and cum[start] <= ans_char:
        start += 1
    # first token whose cumulative length reaches the answer-end char
    end = start
    while end < n and cum[end] < end_char:
        end += 1
    _PUNCT = set(":;,.!?-–—•)]}\"'")
    # trim LEADING tokens that are only the marker's trailing colon / whitespace
    # (e.g. a ": " token shared with the marker) so the span starts at the
    # actual answer content
    while end > start:
        piece = tokenizer.decode([gen_tokens[start].item()],
                                 skip_special_tokens=True)
        st = piece.strip()
        if st and not all(ch in _PUNCT for ch in st):
            break
        start += 1
    # trim trailing whitespace-only tokens from the span
    while end > start:
        piece = tokenizer.decode([gen_tokens[end - 1].item()],
                                 skip_special_tokens=True)
        if piece.strip():
            break
        end -= 1
    return start, max(start, end)


def _extract_final_response(full_text: str) -> str:
    """Scored answer string: the text after the LAST 'Final response:' marker,
    up to the 'Confidence' line. Graceful fallbacks when the model ignored the
    format (no marker -> strip a trailing Confidence line off the whole text).
    Anchoring on the LAST marker means a stray 'final response' inside the
    reasoning trace does not capture the reasoning instead of the answer -- and
    keeping the result to just the final answer is what lets is_abstention's
    first-80-char window see an 'I don't know' that would otherwise be buried
    behind a reasoning paragraph."""
    low = full_text.lower()
    fr = low.rfind(_FINAL_MARKER)
    text = full_text[fr + len(_FINAL_MARKER):] if fr != -1 else full_text
    c = re.search(r"\n?\s*confidence\s*[:=]", text, re.IGNORECASE)
    if c is not None:
        text = text[:c.start()]
    return text.strip()


@torch.inference_mode()
def run_inference(model, tokenizer, eval_ds, batch_size, desc, strategy_spec):
    terminators = get_terminators(tokenizer)
    terminator_set = set(terminators)
    max_new_tokens = strategy_spec.get("max_new_tokens", MAX_NEW_TOKENS)
    predictions, ground_truths, answerable_flags, response_types = [], [], [], []
    confidences, verbal_confidences, raw_responses, token_diagnostics = [], [], [], []
    refusal_scores = []
    save_token_traces = os.environ.get("SAVE_TOKEN_TRACES", "0") == "1"
    score_refusal = os.environ.get("SCORE_REFUSAL_TEMPLATES", "0") == "1"

    # length-sort for padding efficiency, but keep gold alignment
    order = sorted(range(len(eval_ds)),
                   key=lambda i: len(eval_ds[i]["context"]) + len(eval_ds[i]["question"]))
    for start in tqdm(range(0, len(order), batch_size), desc=desc):
        idxs = order[start:start + batch_size]
        batch = [eval_ds[i] for i in idxs]
        prompts = [make_prompt(tokenizer, ex["context"], ex["question"], strategy_spec)
                   for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,            # raw per-step logits -> confidences
        )
        sequences = gen.sequences
        # (gen_len, batch, vocab) -- gen_len <= MAX_NEW_TOKENS, short with EOS
        prompt_len = inputs["input_ids"].shape[1]
        for j, ex in enumerate(batch):
            gen_tokens = sequences[j][prompt_len:]
            # cut at first terminator (right-side pads after EOS are pad/eos)
            cut = len(gen_tokens)
            for t, tok in enumerate(gen_tokens.tolist()):
                if tok in terminator_set:
                    cut = t
                    break
            n = min(cut, len(gen.logits))

            full_text = tokenizer.decode(gen_tokens[:n], skip_special_tokens=True).strip()
            pred = _extract_final_response(full_text)         # scored text
            verbal = parse_verbal_confidence(full_text)      # independent channel
            empty_output = not pred.strip()

            # Logit confidence over the ANSWER tokens ONLY -- the verbalized
            # confidence line and the reasoning/marker scaffold are excluded so they
            # cannot add noise to the sequence-likelihood signal.
            a_start, a_end, marker_found = _final_response_token_span(gen_tokens, tokenizer, n)
            answer_stats = [
                compute_step_uncertainty(gen.logits[t][j], int(gen_tokens[t]))
                for t in range(a_start, a_end)
            ]
            answer_ids = gen_tokens[a_start:a_end].tolist()
            diag = summarize_answer_span(
                answer_stats, answer_ids,
                [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in answer_ids],
                save_token_traces=save_token_traces,
            )
            diag["answer_span_marker_found"] = marker_found
            # Legacy scalar remains exactly its old contract, including 0.0 for empty.
            conf = (float(torch.exp(torch.tensor(diag["answer_mean_logprob"])).clamp(0.0, 1.0))
                    if diag["answer_mean_logprob"] is not None else 0.0)
            if marker_found and a_end > a_start:
                prefix_ids = _abstention_first_token_ids(tokenizer)
                diag.update(compute_decision_abstention_stats(gen.logits[a_start][j], prefix_ids))
            else:
                diag.update(compute_decision_abstention_stats(torch.empty(0), []))
            token_diagnostics.append(diag)
            if score_refusal:
                prompt_ids = inputs["input_ids"][j][inputs["attention_mask"][j].bool()].tolist()
                refusal_scores.append(score_refusal_templates(model, tokenizer, prompt_ids))
            else:
                refusal_scores.append(None)
            answerable = len(ex["answers"]["text"]) > 0

            empty_output = not pred.strip()

            if empty_output:
                response_type = "empty"
            elif answerable==False:
                response_type = "abstain"
            else:
                response_type = "answer"

            
            gt = list(set(ex["answers"]["text"])) if answerable else ["I don't know"]
            predictions.append(pred)
            ground_truths.append(gt)
            answerable_flags.append(answerable)
            confidences.append(conf)
            verbal_confidences.append(verbal)
            raw_responses.append(full_text)
            response_types.append(response_type)
        del gen
    return (predictions, ground_truths, answerable_flags,
            confidences, verbal_confidences, raw_responses, response_types, token_diagnostics, refusal_scores)


# ════════════════════════════════════════════════════════
# VISUALIZATION -- same layout as the reference figure
# ════════════════════════════════════════════════════════
def plot_f1_vs_ratio(summary_df, baseline, save_path, model_short):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(summary_df["ratio"], summary_df["has_ans_f1"], "o-",
            color="tab:blue", label="Answerable F1")
    ax.plot(summary_df["ratio"], summary_df["abstention_recall"], "o-",
            color="tab:orange", label="Unanswerable F1")
    ax.plot(summary_df["ratio"], summary_df["overall_f1"], "o-",
            color="tab:green", label="Overall F1")

    # 95% bootstrap CI band on the overall curve
    ax.fill_between(summary_df["ratio"], summary_df["overall_ci_low"],
                    summary_df["overall_ci_high"], color="tab:green", alpha=0.15)

    # Pretrained zero-shot baselines (dotted), color-matched to their curves
    ax.axhline(baseline["has_ans_f1"], linestyle=":", color="tab:blue",
               label=f"Answerable Baseline ({baseline['has_ans_f1']:.1f})")
    ax.axhline(baseline["abstention_recall"], linestyle=":", color="tab:orange",
               label=f"Unanswerable Baseline ({baseline['abstention_recall']:.1f})")
    ax.axhline(baseline["overall_f1"], linestyle=":", color="tab:green",
               label=f"Overall Baseline ({baseline['overall_f1']:.1f})")

    ax.set_xlabel("Ratio")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 Score vs Ratio — {model_short} (zero-shot, fine-tuned vs pretrained)")
    ax.set_xticks(summary_df["ratio"])
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")
    plt.close(fig)


def plot_calibration_vs_ratio(summary_df, baseline, save_path, model_short):
    """ECE / Brier vs ratio, with the pretrained zero-shot model as dotted
    baselines -- same visual grammar as the F1 figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(summary_df["ratio"], summary_df["ece_f1_0.5"], "o-",
            color="tab:red", label="ECE (correct = F1 > 0.5)")
    ax.plot(summary_df["ratio"], summary_df["brier_f1_0.5"], "o-",
            color="tab:purple", label="Brier (correct = F1 > 0.5)")
    ax.plot(summary_df["ratio"], summary_df["ece_abstention"], "o-",
            color="tab:brown", label="ECE of abstention decision")
    # Verbal channel: plotted only where the format was actually followed
    # (compliance can collapse after narrow SFT — absence is the finding).
    if "verbal_ece_f1_0.5" in summary_df.columns:
        vmask = summary_df["verbal_ece_f1_0.5"].notna()
        if vmask.any():
            ax.plot(summary_df.loc[vmask, "ratio"],
                    summary_df.loc[vmask, "verbal_ece_f1_0.5"], "s--",
                    color="tab:olive", label="ECE verbalized confidence")
    if "verbal_ece_f1_0.5" in baseline:
        ax.axhline(baseline["verbal_ece_f1_0.5"], linestyle=":", color="tab:olive",
                   label=f"Verbal ECE Baseline ({baseline['verbal_ece_f1_0.5']:.3f})")
    ax.axhline(baseline["ece_f1_0.5"], linestyle=":", color="tab:red",
               label=f"ECE Baseline ({baseline['ece_f1_0.5']:.3f})")
    ax.axhline(baseline["brier_f1_0.5"], linestyle=":", color="tab:purple",
               label=f"Brier Baseline ({baseline['brier_f1_0.5']:.3f})")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Calibration error (lower = better)")
    ax.set_title(f"Calibration vs Ratio — {model_short} (zero-shot)")
    ax.set_xticks(summary_df["ratio"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════
# RESUME -- skip ratios whose inference already completed
# ════════════════════════════════════════════════════════
def _load_predictions(path: Path):
    """Reload a saved predictions file as the six aligned lists run_inference
    returns: (preds, gts, flags, confs, vconfs, raws). Returns None if the file
    is missing or unreadable/half-written (a crash mid-dump), so the caller
    falls back to re-running that ratio. Lets a re-run reuse the expensive GPU
    inference and rebuild metrics on CPU from the stored predictions."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        preds  = [d["prediction"]        for d in data]
        raws   = [d["raw_response"]       for d in data]
        gts    = [d["ground_truths"]      for d in data]
        flags  = [d["answerable"]         for d in data]
        confs  = [d["logit_confidence"]   for d in data]
        vconfs = [d["verbal_confidence"]  for d in data]
        diags = [{
            k: v for k, v in d.items()
            if (
                k.startswith("answer_")
                or k.startswith("decision_")
                or k.startswith("refusal_template_")
                or k == "best_refusal_template"
            )
        } for d in data]
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None
    return preds, gts, flags, confs, vconfs, raws, diags


# ════════════════════════════════════════════════════════
# JOBS -- one inference job = pretrained baseline OR one ratio adapter
# ════════════════════════════════════════════════════════
# Parallelism axis: the 12 jobs (baseline + 11 ratios) are independent once you
# accept one model REPLICA per worker instead of one shared model. Each worker
# owns a GPU (or a GPU group, for models too big for a single card), loads its
# own base model, runs its slice of the jobs, and writes the SAME per-job
# predictions_*.json files the sequential path produced. Metrics + plots are
# then rebuilt by the parent from those JSONs -- so the GPU work parallelizes
# and the cheap CPU aggregation stays single-source-of-truth.

def _all_jobs(strategies):
    """One job per (strategy x {pretrained, ratio}). The strategy is carried on
    the job so a single flat job list can be partitioned round-robin across
    workers while each job still knows which prompt to use and which
    per-strategy sub-directory to write into."""
    jobs = []
    for strategy in strategies:
        jobs.append({"kind": "pretrained", "strategy": strategy})
        jobs += [{"kind": "ratio", "ratio": r, "strategy": strategy} for r in RATIOS]
    return jobs


def _strategy_dir(out_dir: Path, strategy: str) -> Path:
    """Per-strategy sub-directory under out_dir. Keeps each strategy's
    predictions / summary / plots isolated so resume and aggregation never mix
    strategies, and so zero_shot artifacts from an earlier run are untouched."""
    return out_dir / strategy


def _job_outfile(out_dir: Path, job: dict) -> Path:
    if job["kind"] == "pretrained":
        return out_dir / "predictions_pretrained.json"
    return out_dir / f"predictions_ratio_{job['ratio']:.2f}.json"


def _job_label(job: dict) -> str:
    return "pretrained" if job["kind"] == "pretrained" else f"ratio {job['ratio']:.2f}"


def _dump_predictions(path: Path, preds, gts, flags, confs, vconfs, raws, rtypes,
                      token_diagnostics, refusal_scores):
    with open(path, "w") as f:
        json.dump([{"prediction": p, "raw_response": r, "ground_truths": g,
                    "answerable": a, 'respose_type': rt, "logit_confidence": c, "verbal_confidence": v,
                    **d, **({} if rs is None else rs)}
                   for p, r, g, a, c, v, rt, d, rs in zip(
                       preds, raws, gts, flags, confs, vconfs, rtypes, token_diagnostics, refusal_scores)],
                  f, indent=2)


def _build_gpu_groups(workers: int, gpus_per_worker: int):
    """Contiguous GPU id groups, one per worker:
        workers=4, gpus_per_worker=1 -> [[0],[1],[2],[3]]
        workers=4, gpus_per_worker=2 -> [[0,1],[2,3],[4,5],[6,7]]
    The caller is responsible for workers*gpus_per_worker <= visible GPUs."""
    groups = []
    for w in range(workers):
        base = w * gpus_per_worker
        groups.append(list(range(base, base + gpus_per_worker)))
    return groups


def _load_base_for_worker(gpu_group):
    """Pin this process to its GPU group (must happen BEFORE any CUDA init, so
    device_map='auto' only ever sees the group), then load tokenizer + base.
    gpu_group=None -> inherit all visible GPUs (single-process path: one model
    sharded across everything, identical to the original behaviour)."""
    if gpu_group is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # batched decoder-only generation

    print(f"[worker gpus={gpu_group}] loading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()
    return tokenizer, base_model


def run_jobs_on_replica(gpu_group, jobs, batch_size, out_dir: Path, eval_ds, tag=""):
    """Execute a slice of jobs on ONE model replica.

    Each job carries its own strategy; the prompt and the generation budget come
    from STRATEGIES[strategy], and predictions are written into the per-strategy
    sub-directory. A single replica may now run several strategies AND several
    ratios, so:

      * The pretrained baseline for EACH strategy is forced ahead of that
        strategy's adapter jobs (baseline must be measured on the unwrapped base
        before any PeftModel wrapping). We sort pretrained-first globally, which
        satisfies this for every strategy at once since the unwrapped base is
        reused across strategies.
      * Adapters are cached by ratio and loaded at most ONCE per replica. If two
        strategies both need ratio 0.30, the second reuses the already-loaded
        adapter via set_adapter() instead of calling load_adapter() again (which
        would raise on a duplicate adapter_name).

    Only GPU inference + JSON dump happen here; metrics/plots are the parent's.
    """
    tokenizer, base_model = _load_base_for_worker(gpu_group)

    # baseline (if assigned) strictly before any adapter wrapping
    jobs = sorted(jobs, key=lambda j: 0 if j["kind"] == "pretrained" else 1)

    peft_model = None
    loaded_adapters = set()   # ratio-name -> already on this replica
    for job in jobs:
        strategy = job["strategy"]
        strategy_spec = STRATEGIES[strategy]
        sdir = _strategy_dir(out_dir, strategy)
        sdir.mkdir(parents=True, exist_ok=True)
        outfile = _job_outfile(sdir, job)
        label = f"{strategy}/{_job_label(job)}"
        desc = f"{tag}{label}" if tag else label

        if job["kind"] == "pretrained":
            print(f"[worker gpus={gpu_group}] === {strategy} :: pretrained (no adapter) ===")
            model = base_model
        else:
            ratio = job["ratio"]
            name = f"r{ratio:.2f}".replace(".", "_")
            repo = adapter_repo(ratio)
            print(f"[worker gpus={gpu_group}] === {strategy} :: ratio={ratio:.2f} :: {repo} ===")
            try:
                if name not in loaded_adapters:
                    if peft_model is None:
                        peft_model = PeftModel.from_pretrained(
                            base_model, repo, adapter_name=name, token=HF_TOKEN)
                    else:
                        peft_model.load_adapter(repo, adapter_name=name, token=HF_TOKEN)
                    loaded_adapters.add(name)
                peft_model.set_adapter(name)   # exactly one active adapter
                peft_model.eval()
            except Exception as e:
                print(f"[worker gpus={gpu_group}] SKIP {repo}: could not load -> {e}")
                continue
            model = peft_model

        preds, gts, flags, confs, vconfs, raws, rtypes, diags, refusal_scores = run_inference(
            model, tokenizer, eval_ds, batch_size, desc, strategy_spec)
        _dump_predictions(outfile, preds, gts, flags, confs, vconfs, raws, rtypes, diags, refusal_scores)
        print(f"[worker gpus={gpu_group}] wrote {strategy}/{outfile.name}")


def _worker_entry(gpu_group, jobs, batch_size, out_dir_str, eval_path_str):
    """Spawned-process target. Loads the eval set the parent saved (no re-download
    race, identical seeded ordering across workers) and runs its job slice."""
    from datasets import load_from_disk
    eval_ds = load_from_disk(eval_path_str)
    tag = f"[gpu{','.join(map(str, gpu_group))}] "
    run_jobs_on_replica(gpu_group, jobs, batch_size,
                         Path(out_dir_str), eval_ds, tag=tag)


# ════════════════════════════════════════════════════════
# AGGREGATION -- rebuild metrics + summary + plots from cached JSONs
# ════════════════════════════════════════════════════════
def aggregate_strategy(out_dir: Path, strategy: str, n_expected: int):
    """Aggregate ONE strategy's cached predictions into metrics, a summary.csv,
    all_metrics.json and the two plots, all inside its per-strategy subdir.
    Returns (summary_df, baseline_metrics) so the caller can build a combined
    cross-strategy view. Returns (None, None) and prints a notice if the
    strategy's baseline is missing (its plots/summary are skipped, the rest of
    the run is unaffected)."""
    sdir = _strategy_dir(out_dir, strategy)
    all_metrics = {}

    base_cached = _load_predictions(sdir / "predictions_pretrained.json")
    if base_cached is None or len(base_cached[0]) != n_expected:
        print(f"\n[{strategy}] pretrained predictions missing/stale "
              f"(need {n_expected} rows) -> skipping this strategy's aggregation. "
              f"Re-run its baseline job to enable plots.")
        return None, None
    preds, gts, flags, confs, vconfs, raws, diags = base_cached
    baseline = compute_metrics(preds, gts, flags, confidences=confs,
                               verbal_confidences=vconfs, token_diagnostics=diags, compute_advanced=False)
    all_metrics["pretrained"] = baseline
    print(f"\n=== [{strategy}] Pretrained (no adapter) ===")
    print({k: round(v, 2) for k, v in baseline.items() if isinstance(v, float)})

    rows = []
    for ratio in RATIOS:
        cache = _load_predictions(sdir / f"predictions_ratio_{ratio:.2f}.json")
        if cache is None or len(cache[0]) != n_expected:
            print(f"=== [{strategy}] ratio={ratio:.2f}: no/stale predictions -> omitted ===")
            continue
        preds, gts, flags, confs, vconfs, raws, diags = cache
        m = compute_metrics(preds, gts, flags, confidences=confs,
                            verbal_confidences=vconfs, token_diagnostics=diags, compute_advanced=False)
        all_metrics[f"ratio_{ratio:.2f}"] = m
        rows.append({"ratio": ratio, **m})
        print(f"=== [{strategy}] ratio={ratio:.2f} ===  "
              f"overall={m['overall_f1']:.2f} "
              f"[{m['overall_ci_low']:.2f}, {m['overall_ci_high']:.2f}]  "
              f"hasAns={m['has_ans_f1']:.2f}  abstRecall={m['abstention_recall']:.2f}  "
              f"strictness={m['strictness']:.2f}  "
              f"ECE={m['ece_f1_0.5']:.3f}  Brier={m['brier_f1_0.5']:.3f}  "
              f"verbalCompliance={m.get('verbal_compliance', 0.0):.0f}%  "
              f"(legacy overall={m['overall_f1_legacy']:.2f})")

    summary_df = pd.DataFrame(rows).sort_values("ratio") if rows else pd.DataFrame()
    summary_df.to_csv(sdir / "summary.csv", index=False)
    with open(sdir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    if not summary_df.empty:
        print(f"\n[{strategy}]\n" + summary_df.to_string(index=False, float_format="%.2f"))
        title = f"{MODEL_SHORT} [{strategy}]"
        plot_f1_vs_ratio(summary_df, baseline,
                         sdir / f"f1_vs_ratio_{strategy}.png", title)
        plot_calibration_vs_ratio(summary_df, baseline,
                                  sdir / f"calibration_vs_ratio_{strategy}.png", title)
    print(f"[{strategy}] outputs in: {sdir}")
    return summary_df, baseline


def aggregate_and_plot(out_dir: Path, strategies, n_expected: int):
    """Aggregate every requested strategy, then write a single combined
    long-format CSV (strategy x ratio + the pretrained baseline rows) so SQuAD2
    results drop straight into the same cross-strategy table shape as the OOD
    metrics file."""
    combined_rows = []
    for strategy in strategies:
        summary_df, baseline = aggregate_strategy(out_dir, strategy, n_expected)
        if baseline is not None:
            combined_rows.append({"strategy": strategy, "ratio": "base", **baseline})
        if summary_df is not None and not summary_df.empty:
            for _, r in summary_df.iterrows():
                row = r.to_dict()
                row_ratio = row.pop("ratio")
                combined_rows.append({"strategy": strategy,
                                      "ratio": f"{row_ratio:.2f}", **row})

    if combined_rows:
        combined = pd.DataFrame(combined_rows)
        # put strategy, ratio first for readability
        lead = ["strategy", "ratio"]
        combined = combined[lead + [c for c in combined.columns if c not in lead]]
        combined.to_csv(out_dir / "summary_all_strategies.csv", index=False)
        print(f"\nCombined cross-strategy summary -> "
              f"{out_dir / 'summary_all_strategies.csv'}")
    print(f"\nAll outputs in: {out_dir}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=None,
                    help="HF base model ID (must be in MODEL_PRESETS). "
                         "Propagated to spawn workers via EVAL_MODEL_ID.")
    ap.add_argument("--lora_rank", type=int, default=None,
                    help="Which rank's adapters to evaluate (default 16 = "
                         "the unsuffixed original sweep; other ranks resolve "
                         "'-r{rank}'-suffixed Hub repos and a suffixed out_dir). "
                         "Propagated to spawn workers via EVAL_LORA_RANK.")
    ap.add_argument("--skip_preflight", action="store_true",
                    help="Skip the startup Hub existence check of all adapter "
                         "repos. Not recommended: the preflight catches "
                         "short-name/suffix mismatches before any GPU work.")
    ap.add_argument("--n_eval", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--save-token-traces", action="store_true",
                    help="save compact per-answer token traces (off by default)")
    ap.add_argument("--score-refusal-templates", action="store_true",
                    help="run optional teacher-forced refusal-template ranking probe (extra forward pass)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Default: results/squad2/strategy_compare_answerability/"
                         "{MODEL_SHORT}{_r<rank> if not 16} — resolved AFTER "
                         "--model_id/--lora_rank are applied.")
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                    choices=list(STRATEGIES.keys()),
                    help="Prompting strategies to evaluate. Default: all four "
                         "(zero_shot few_shot_balance cot self_reflect). Each "
                         "gets its own sub-directory under --out_dir, so "
                         "strategies resume independently.")
    ap.add_argument("--val_jsonl", type=str, default=None,
                    help="DEPRECATED / ignored. The early-stopping slice is "
                         "reconstructed directly from the squad_v2 validation "
                         "split (first --n_exclude rows), matching training.")
    ap.add_argument("--n_exclude", type=int, default=1000,
                    help="First-n rows of the squad_v2 validation split used "
                         "in-training for early stopping; held out of this test "
                         "set. Must match the training run's .select(range(n)).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run every job even if a completed predictions "
                         "file already exists. Default: resume (skip completed).")
    # ── parallelism ────────────────────────────────────────────────────────
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel worker processes. 1 = original "
                         "single-process path (one model sharded across all "
                         "visible GPUs via device_map='auto'). >1 = spawn that "
                         "many replicas and partition the jobs round-robin.")
    ap.add_argument("--gpus_per_worker", type=int, default=1,
                    help="GPUs each worker shards its model across (for models "
                         "too big for one card). Needs "
                         "workers*gpus_per_worker <= visible GPUs.")
    args = ap.parse_args()

    # Environment propagation keeps spawned workers on the same compact-output policy.
    os.environ["SAVE_TOKEN_TRACES"] = "1" if args.save_token_traces else "0"
    os.environ["SCORE_REFUSAL_TEMPLATES"] = "1" if args.score_refusal_templates else "0"

    # ── model / rank overrides (BEFORE out_dir + any Hub access) ────────────
    # Set both the module globals (this parent process) and EVAL_* env vars
    # (spawn workers re-import the module and read them at import time).
    global MODEL_ID, MODEL_SHORT, NO_SYSTEM_ROLE, LORA_RANK
    if args.model_id is not None:
        if args.model_id not in MODEL_PRESETS:
            sys.exit(f"ERROR: --model_id {args.model_id!r} not in MODEL_PRESETS. "
                     f"Known: {', '.join(sorted(MODEL_PRESETS))}")
        MODEL_ID = args.model_id
        MODEL_SHORT = MODEL_PRESETS[MODEL_ID]["short"]
        NO_SYSTEM_ROLE = MODEL_PRESETS[MODEL_ID]["no_system_role"]
        os.environ["EVAL_MODEL_ID"] = MODEL_ID
    if args.lora_rank is not None:
        LORA_RANK = args.lora_rank
        os.environ["EVAL_LORA_RANK"] = str(LORA_RANK)
    print(f">>> Eval target: base={MODEL_ID} (short={MODEL_SHORT}), "
          f"adapters rank={LORA_RANK}{rank_suffix(' suffix=-') or ' (unsuffixed sweep)'}")
    print(f">>> Adapter repo pattern: {adapter_repo(0.5)}  (example, ratio=0.50)")

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"results/squad2/strategy_compare_answerability/{MODEL_SHORT}{rank_suffix('_')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── preflight: verify every adapter repo exists BEFORE any GPU work ─────
    # Catches short-name mismatches (e.g. llama3.2 vs llama32-3b) and missing
    # rank-suffixed repos in seconds, instead of an hour into generation.
    if not args.skip_preflight:
        from huggingface_hub import repo_exists
        missing = []
        for r in RATIOS:
            repo = adapter_repo(r)
            try:
                ok = repo_exists(repo, token=HF_TOKEN)
            except Exception as e:
                print(f"  preflight: could not check {repo} ({e}); continuing")
                ok = True
            if not ok:
                missing.append(repo)
        if missing:
            sys.exit(
                "ERROR: adapter repo(s) not found on the Hub:\n  "
                + "\n  ".join(missing)
                + "\nLikely causes: (a) MODEL_PRESETS short name doesn't match "
                  "what training pushed (check your Hub page), (b) the rank-"
                  "suffixed adapters haven't been trained/pushed yet, or "
                  "(c) wrong --lora_rank. Use --skip_preflight only if you "
                  "know some ratios are intentionally absent.")
        print(f"  preflight OK: all {len(RATIOS)} adapter repos exist.")

    # ── eval set: built ONCE here, deterministically, then shared with workers ─
    if args.val_jsonl:
        print("Note: --val_jsonl is ignored; the held-out slice is rebuilt "
              "from the squad_v2 validation split to match training.")
    exclude_ids = load_early_stopping_ids(n=args.n_exclude)
    print(f"Reconstructed {len(exclude_ids)} early-stopping ids from the first "
          f"{args.n_exclude} rows of {SQUAD_V2} validation -> held out.")
    eval_ds = build_eval_set(args.n_eval, SEED, exclude_ids=exclude_ids)
    n_expected = len(eval_ds)

    strategies = args.strategies
    print(f"Strategies to evaluate: {', '.join(strategies)}")
    print("Confidence elicitation = ANSWERABILITY "
          "(0 = answer NOT in context, 100 = answer present).")
    print("  NOTE 1 (resume): the resume check compares only ROW COUNT, not the "
          "prompt wording. If this --out_dir holds predictions generated with a "
          "different confidence wording, they will NOT be re-run automatically -- "
          "pass --overwrite or use a clean --out_dir. (The default out_dir is "
          "already tagged '_answerability' to avoid colliding with legacy runs.)")
    print("  NOTE 2 (scoring): verbal confidence now tracks answerability, so it "
          "should be calibrated against is_answerable, not F1-correctness. A "
          "correct abstention is LOW-confidence by design; scoring it against "
          "correctness would misread that as underconfidence.")

    # ── resume plan: which jobs still need GPU inference ────────────────────
    # Resume is per-strategy: each job's predictions live in its strategy
    # sub-directory, so completed strategies (e.g. an earlier zero_shot run
    # migrated into out_dir/zero_shot/) are skipped while new strategies run.
    def needs_run(job) -> bool:
        outfile = _job_outfile(_strategy_dir(out_dir, job["strategy"]), job)
        if args.overwrite:
            return True
        cached = _load_predictions(outfile)
        if cached is None:
            return True
        if len(cached[0]) != n_expected:
            print(f"  {job['strategy']}/{outfile.name}: {len(cached[0])} rows "
                  f"!= current eval size {n_expected} -> stale, will re-run")
            return True
        return False

    jobs = _all_jobs(strategies)
    pending = [j for j in jobs if needs_run(j)]
    done = [f"{j['strategy']}/{_job_label(j)}" for j in jobs if j not in pending]
    if done:
        print(f"Resuming: {len(done)} job(s) already completed -> "
              f"{', '.join(done)} (skipping their inference).")

    if not pending:
        print("All jobs already completed -> rebuilding summary/plots only "
              "(no model load).")
        aggregate_and_plot(out_dir, strategies, n_expected)
        return

    # ── dispatch the pending jobs ───────────────────────────────────────────
    if args.workers <= 1:
        # Single-process: one model sharded across all visible GPUs (original).
        print(f"Running {len(pending)} job(s) single-process "
              f"(device_map='auto' across all visible GPUs).")
        run_jobs_on_replica(None, pending, args.batch_size, out_dir, eval_ds)
    else:
        # Multi-worker: persist the eval set once, then partition jobs.
        eval_path = out_dir / "_eval_set_hf"
        if eval_path.exists():
            shutil.rmtree(eval_path)
        eval_ds.save_to_disk(str(eval_path))

        groups = _build_gpu_groups(args.workers, args.gpus_per_worker)
        buckets = [[] for _ in groups]
        for i, job in enumerate(pending):          # round-robin partition
            buckets[i % len(groups)].append(job)

        plan = ", ".join(f"gpus{g}:{len(b)}job(s)" for g, b in zip(groups, buckets))
        print(f"Spawning {len(groups)} worker(s) over GPU groups -> {plan}")

        mp.set_start_method("spawn", force=True)
        procs = []
        for grp, bucket in zip(groups, buckets):
            if not bucket:
                continue
            p = mp.Process(target=_worker_entry,
                           args=(grp, bucket, args.batch_size,
                                 str(out_dir), str(eval_path)))
            p.start()
            procs.append((grp, p))
        failures = []
        for grp, p in procs:
            p.join()
            if p.exitcode != 0:
                failures.append((grp, p.exitcode))
        if failures:
            print(f"WARNING: worker(s) exited non-zero: {failures}. "
                  f"Their jobs may be missing; aggregation will skip stale/absent "
                  f"ratios and you can re-run to fill gaps (resume is automatic).")

        shutil.rmtree(eval_path, ignore_errors=True)

    # ── rebuild metrics + summary + plots from whatever completed ───────────
    aggregate_and_plot(out_dir, strategies, n_expected)


if __name__ == "__main__":
    main()
