#!/usr/bin/env python3
"""
run_ood_abstention_eval.py
==========================
Custom-generation OOD abstention eval for base + ratio adapters. Captures the
abstention DECISION and CONFIDENCE signals lm-eval cannot: verbal confidence
(self_reflect) AND token-level confidence (generation logprobs), then computes
the full calibration suite in collect().

Reuses your Hub-discovery contract verbatim: HUB_AUTHOR / HUB_NAME_FILTER /
HUB_RATIO_REGEX / _normalize_ratio.

Multi-GPU + global progress bar
-------------------------------
One worker process per GPU (override --workers / --gpus). Each model's ratio
adapters are round-robin distributed across workers, so several ratios of the
same model evaluate concurrently. A single example-level progress bar is shown
in the parent (shared counter, smooth ex/s + ETA). Oversubscribe (--workers >
#GPUs) to raise per-GPU utilisation, memory permitting.

Inner saving / resume (designed for a wall-clock limit, e.g. 4h/GPU)
--------------------------------------------------------------------
Predictions are written INCREMENTALLY, not just at unit boundaries:
    * each example is appended to predictions.jsonl.partial and flushed
      immediately, so a kill loses at most the in-flight example;
    * on completion the partial is atomically promoted to predictions.jsonl
      (os.replace), so collect()/_is_done only ever see COMPLETE units;
    * on restart an incomplete unit reloads its partial, drops a corrupt
      trailing line if any, skips already-done example ids, and continues.

Crash/preemption safety on a cluster:
    * locks are heartbeated (os.utime) every example; _claim reclaims any lock
      whose heartbeat is older than --lock-ttl (default 900s) on ANY host, so a
      restart on a different node is not blocked by an orphaned lock;
    * a SIGTERM/SIGINT handler (SLURM preemption) flushes the partial, releases
      the lock, and exits cleanly for resume.

Usage:
    python run_ood_abstention_eval.py run --models qwen3-4b --gpus 0 1 2 3
    python run_ood_abstention_eval.py run --models qwen3-4b --workers 8 --gpus 0 1 2 3
    # ...kill at 4h, then simply re-run the SAME command -> resumes mid-unit
    python run_ood_abstention_eval.py collect
    python run_ood_abstention_eval.py clean-locks       # drop orphaned locks

NOTE: resume assumes the same --datasets / --strategies / --limit between runs
(example ids must match). Changing --limit mid-sweep invalidates partials.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import re
import signal
import socket
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import gc
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib

from ood_metrics import (annotate, compute_aggregate, compute_curves,
                         compute_decision_calibration, compute_decision_curves)
from ood_common import FINAL_RESPONSE_MARKER, ABSTAIN_STRING

# ---- Hub discovery (verbatim contract from your general runner) ------------
HUB_AUTHOR = "AlinaGonch"
HUB_RATIO_REGEX = r"ratio[_\-]?([0-9]+(?:\.[0-9]+)?)"
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama-3.2-3b":  {"base_repo": "meta-llama/Llama-3.2-3B-Instruct", "hub_name_filter": "llama3.2", "family": "llama"},
    "llama-3.1-8b":  {"base_repo": "meta-llama/Llama-3.1-8B-Instruct", "hub_name_filter": "llama31-8b", "family": "llama"},
    "granite-4.1-3b":{"base_repo": "ibm-granite/granite-4.1-3b",       "hub_name_filter": "granite41-3b", "family": "granite"},
    "granite-4.1-8b":{"base_repo": "ibm-granite/granite-4.1-8b",       "hub_name_filter": "granite41-8b", "family": "granite"},
    "phi-3-mini":    {"base_repo": "microsoft/Phi-3-mini-4k-instruct", "hub_name_filter": "phi3-mini", "family": "phi"},
    "phi3-medium":    {"base_repo": "microsoft/Phi-3-medium-4k-instruct", "hub_name_filter": "phi3-medium", "family": "phi"},
    "qwen3-4b":      {"base_repo": "Qwen/Qwen3-4B-Instruct-2507",                    "hub_name_filter": "qwen3-4b", "family": "qwen3"},
    "qwen3-14b":      {"base_repo": "Qwen/Qwen3-14B",                    "hub_name_filter": "qwen3-4b", "family": "qwen3"},
    "gemma-3-4b":    {"base_repo": "google/gemma-3-4b-it",            "hub_name_filter": "gemma3-4b", "family": "gemma"},
}

DATASET_LOADERS = {
    "faitheval":      ("faitheval_loader", "load"),
    "nomiracl":       ("nomiracl_loader", "load"),
    "hotpotqa":       ("hotpotqa_loader", "load"),
    "gsm8k_abstain":  ("gsm8k_abstain_loader", "load"),
    "gpqa_abstain":   ("gpqa_abstain_loader", "load"),
    "mmlu_pro":       ("mmlu_pro_loader", "load"),
}

OUTPUT_DIR = "ood_eval_out"

# tracks the unit currently being written, so the signal handler can flush+release
_ACTIVE: Dict[str, Any] = {"file": None, "lock": None}


def _normalize_ratio(raw: str) -> str:
    if "." in raw:
        val = float(raw)
    else:
        n = int(raw)
        val = n / 100.0 if n > 1 else float(n)
    return f"{val:.2f}"


def discover_adapters(hub_name_filter: str) -> Dict[str, str]:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    found: Dict[str, str] = {}
    for m in api.list_models(author=HUB_AUTHOR, token=HF_TOKEN):
        rid = m.id
        if hub_name_filter and hub_name_filter.lower() not in rid.lower():
            continue
        mt = re.search(HUB_RATIO_REGEX, rid, re.IGNORECASE)
        if not mt:
            continue
        found[_normalize_ratio(mt.group(1))] = rid
    found = dict(sorted(found.items(), key=lambda kv: float(kv[0])))
    print(f"[hub] {hub_name_filter}: {len(found)} adapters -> {list(found)}", flush=True)
    return found


# ---- prompt render with family quirks --------------------------------------
def _safe_adapter_name(name: str) -> str:
    """PEFT registers each adapter as a submodule, and torch's add_module
    forbids '.' in module names, so ratio labels like '0.00' must be mapped to a
    dot-free internal key. This affects ONLY the in-memory PEFT registration;
    the original `name` is still used for file paths and the 'ratio' field."""
    return "ad_" + re.sub(r"[^0-9A-Za-z]+", "_", str(name))


def render_prompt(tokenizer, messages: List[Dict[str, str]], family: str) -> str:
    msgs = [dict(m) for m in messages]
    if family == "gemma":  # no system role: fold system into first user turn
        sys_txt = "".join(m["content"] + "\n\n" for m in msgs if m["role"] == "system")
        rest = [m for m in msgs if m["role"] != "system"]
        if sys_txt and rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": sys_txt + rest[0]["content"]}
        msgs = rest
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if family == "qwen3":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(msgs, **kwargs)


# ---- model load / generate -------------------------------------------------
def load_base(base_repo: str):
    
    tok = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # left padding is REQUIRED for correct batched decoder-only generation:
    # it keeps every prompt's last real token adjacent to the first generated
    # token, so generated positions align across the batch.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


# ---- ID-MATCHING primary channel: answer-span sequence likelihood ----------
#
# Port of the ID pipeline's _final_response_token_span + exp(mean logprob)
# (multi_strategy_parallel.run_inference). This is the channel behind the
# in-distribution headline AUROCs, so OOD must compute the identical quantity:
#   * span = tokens after the LAST "final response:" (CASE-INSENSITIVE),
#     ending before the following "confidence" line;
#   * leading punctuation-only tokens trimmed, trailing whitespace trimmed;
#   * NO marker -> whole generation (graceful fallback, same as ID);
#   * confidence = exp(mean log-prob of chosen tokens over the span),
#     clamped to [0,1]; EMPTY span -> 0.0 ("abstention with no evidence").
# Orientation: this is confidence-in-the-emitted-answer; the ID analysis uses
# abstention_score = 1 - confidence, and ood_metrics' decision AUROC
# (confidence vs is_answerable) is the same measurement from the other side.

_ID_FINAL_MARKER = "final response:"
_ID_PUNCT = set(":;,.!?-–—•)]}\"'")


def _id_answer_span(tokenizer, row_tokens: List[int]) -> tuple:
    """[start, end) token span of the final-response segment — verbatim port
    of the ID span logic (incremental single-token decode -> char cumsum ->
    map marker chars back to token indices)."""
    n = len(row_tokens)
    running = ""
    cum = []
    for t in range(n):
        running += tokenizer.decode([row_tokens[t]], skip_special_tokens=True)
        cum.append(len(running))
    low = running.lower()
    fr = low.rfind(_ID_FINAL_MARKER)
    if fr == -1:
        return 0, n                       # format ignored -> whole span
    ans_char = fr + len(_ID_FINAL_MARKER)
    confp = low.find("confidence", ans_char)
    end_char = confp if confp != -1 else len(running)
    start = 0
    while start < n and cum[start] <= ans_char:
        start += 1
    end = start
    while end < n and cum[end] < end_char:
        end += 1
    while end > start:                    # trim leading marker-colon/punct tokens
        piece = tokenizer.decode([row_tokens[start]], skip_special_tokens=True).strip()
        if piece and not all(ch in _ID_PUNCT for ch in piece):
            break
        start += 1
    while end > start:                    # trim trailing whitespace-only tokens
        piece = tokenizer.decode([row_tokens[end - 1]], skip_special_tokens=True)
        if piece.strip():
            break
        end -= 1
    return start, max(start, end)


def _id_span_confidence(tokenizer, row_tokens: List[int],
                        row_probs: List[float]) -> float:
    import math
    start, end = _id_answer_span(tokenizer, row_tokens)
    if end <= start:
        return 0.0                        # empty answer = abstention, no evidence
    lps = [math.log(max(p, 1e-12)) for p in row_probs[start:end]]
    return min(1.0, max(0.0, math.exp(sum(lps) / len(lps))))


# ---- PRIMARY channel: answer-span exp(mean logprob) — ID-PORTED -------------
#
# Ported from the ID pipeline (multi_strategy_parallel._final_response_token_span
# / run_inference): token confidence = exp(mean logprob) of the ANSWER-SPAN
# tokens only — the span after the LAST "Final response:" marker
# (case-insensitive), ending before the following "Confidence" line, with
# marker-punctuation and whitespace tokens trimmed. Reasoning traces and the
# verbalized-confidence tokens are excluded. This is the exact channel behind
# the ID decision-AUROC numbers, so ID and OOD tables join like-for-like.
# Fallback semantics match ID: no marker -> whole span; empty span -> 0.0.

_FINAL_MARKER_LOW = FINAL_RESPONSE_MARKER.lower()   # matched case-insensitively
_SPAN_PUNCT = set(":;,.!?-–—•)]}\"'")


def _final_response_token_span(tokenizer, row_tokens: List[int]):
    """Token-index span [start, end) of the final-response segment, plus
    whether a marker was found. Verbatim port of the ID logic: incremental
    per-token decode -> cumulative char lengths -> map marker char positions
    back to token indices; trim leading marker-colon/whitespace tokens and
    trailing whitespace tokens."""
    n = len(row_tokens)
    pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in row_tokens]
    running = "".join(pieces)
    cum, total = [], 0
    for p in pieces:
        total += len(p)
        cum.append(total)
    low = running.lower()
    fr = low.rfind(_FINAL_MARKER_LOW)             # LAST final-response marker
    if fr == -1:
        return 0, n, False                        # format ignored -> whole span
    ans_char = fr + len(_FINAL_MARKER_LOW)
    conf = low.find("confidence", ans_char)
    end_char = conf if conf != -1 else len(running)

    start = 0
    while start < n and cum[start] <= ans_char:
        start += 1
    end = start
    while end < n and cum[end] < end_char:
        end += 1
    # trim LEADING tokens that are only the marker's trailing colon/whitespace
    while end > start:
        st = pieces[start].strip()
        if st and not all(ch in _SPAN_PUNCT for ch in st):
            break
        start += 1
    # trim trailing whitespace-only tokens
    while end > start:
        if pieces[end - 1].strip():
            break
        end -= 1
    return start, max(start, end), True


# ---- SECONDARY channel: decision-position abstain-keyword mass --------------
#
# Exploratory column (token_confidence_decision): softmax mass on
# abstain-opening tokens at the first answer-span token, reported as
# 1 - mass (confidence-answerable). NOT the join channel with ID tables —
# kept because it is free from the same forward pass and probes the decision
# more directly than sequence likelihood; may become a thesis footnote.

_ABSTAIN_PHRASES = [
    ABSTAIN_STRING,          # "I don't know"
    "I don’t know",          # curly apostrophe variant
    "Not given",
    "Unanswerable",
    "Cannot be determined",
    "None of the above",
]
_ABSTAIN_ID_CACHE: Dict[int, List[int]] = {}


def _abstain_first_token_ids(tokenizer) -> List[int]:
    """First-token ids of every abstain phrase under the prefixes a decoder
    might emit at the decision position ('', ' ', '\\n'). Cached per tokenizer."""
    key = id(tokenizer)
    ids = _ABSTAIN_ID_CACHE.get(key)
    if ids is not None:
        return ids
    found = set()
    for phrase in _ABSTAIN_PHRASES:
        for prefix in ("", " ", "\n"):
            toks = tokenizer(prefix + phrase, add_special_tokens=False)["input_ids"]
            if toks:
                found.add(toks[0])
    ids = sorted(found)
    _ABSTAIN_ID_CACHE[key] = ids
    return ids


def generate_batch(model, tokenizer, input_ids_list: List[List[int]],
                   max_new_tokens: int):
    """Greedy-generate a whole batch at once.

    Returns (texts, span_confs, decision_confs, mean_confs):
        texts          -- list[str], decoded continuations (specials stripped)
        span_confs     -- list[float], the ID-MATCHING PRIMARY channel:
                          exp(mean logprob) over the ANSWER-SPAN tokens only
                          (after last 'Final response:', before 'Confidence',
                          trimmed). No marker -> whole span; empty span -> 0.0.
                          Ported from multi_strategy_parallel.run_inference.
        decision_confs -- list[Optional[float]], SECONDARY exploratory channel:
                          1 - abstain-keyword softmax mass at the first
                          answer-span token (confidence-answerable). None when
                          no marker was generated.
        mean_confs     -- list[float], arithmetic mean token prob over all
                          generated tokens (old proxy, diagnostic only).
    """
    

    # left-pad the (pre-tokenized, cached) prompts -> [B, L_in]
    enc = tokenizer.pad({"input_ids": input_ids_list},
                        padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    in_len = input_ids.shape[1]

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_logits=True,
            pad_token_id=tokenizer.pad_token_id)

    # raw per-step logits, matching the ID pipeline's output_logits=True
    # (gen.scores would be processor-modified); fall back defensively.
    step_scores = getattr(gen, "logits", None) or gen.scores

    gen_tokens = gen.sequences[:, in_len:]                 # [B, T]
    texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    B, T = gen_tokens.shape
    if T == 0:
        return texts, [0.0] * B, [None] * B, [None] * B

    # resolve eos id(s); some families (e.g. llama3) use several
    try:
        raw_eos = model.generation_config.eos_token_id
    except Exception:
        raw_eos = None
    if raw_eos is None:
        raw_eos = tokenizer.eos_token_id
    if raw_eos is None:
        eos_list: List[int] = []
    elif isinstance(raw_eos, int):
        eos_list = [raw_eos]
    else:
        eos_list = list(raw_eos)

    # keep mask: True up to and INCLUDING the first eos per row, False after
    # (so trailing padding -- and pad==eos repeats -- are excluded).
    is_eos = torch.zeros_like(gen_tokens, dtype=torch.bool)
    for e in eos_list:
        is_eos |= (gen_tokens == e)
    eos_cum = is_eos.cumsum(dim=1)
    keep = (eos_cum - is_eos.long()) == 0                  # [B, T] bool

    # logprob of each chosen token, per step (avoids the [B,T,V] materialise;
    # log_softmax of RAW logits == the ID pipeline's computation).
    chosen_lp = torch.full((B, T), float("-inf"),
                           device=gen_tokens.device, dtype=torch.float32)
    n_steps = min(T, len(step_scores))
    for t in range(n_steps):
        lp = torch.log_softmax(step_scores[t].float(), dim=-1)   # [B, V]
        chosen_lp[:, t] = lp.gather(-1, gen_tokens[:, t:t + 1]).squeeze(-1)
    if n_steps < T:                       # defensive; normally n_steps == T
        keep[:, n_steps:] = False

    # old diagnostic: arithmetic mean prob over kept tokens
    probs = chosen_lp.exp() * keep.float()
    counts = keep.sum(dim=1).clamp(min=1)
    mean_confs = (probs.sum(dim=1) / counts).tolist()

    # ---- per-row channels off ONE span computation --------------------------
    abstain_idx = torch.tensor(_abstain_first_token_ids(tokenizer),
                               device=gen_tokens.device)
    token_lists = gen_tokens.tolist()
    span_confs: List[float] = []
    decision_confs: List[Optional[float]] = []
    for i in range(B):
        # the row's real content: exclude EOS itself from the span (ID cuts
        # BEFORE the first terminator), hence keep-count minus trailing eos.
        row_len = int(keep[i].sum().item())
        if row_len and bool(is_eos[i, row_len - 1].item()):
            row_len -= 1
        row = token_lists[i][:row_len]

        a_start, a_end, marker_found = _final_response_token_span(tokenizer, row)

        # PRIMARY: answer-span exp(mean logprob), ID fallbacks preserved
        if a_end > a_start:
            span_lp = chosen_lp[i, a_start:a_end]
            span_confs.append(float(span_lp.mean().exp().clamp(0.0, 1.0)))
        else:
            span_confs.append(0.0)   # empty answer = abstention with no evidence

        # SECONDARY: abstain-mass at the first answer-span token
        if not marker_found or a_end <= a_start or a_start >= len(step_scores):
            decision_confs.append(None)
            continue
        p = torch.softmax(step_scores[a_start][i].float(), dim=-1)
        abstain_mass = p.index_select(0, abstain_idx).sum().item()
        decision_confs.append(1.0 - min(1.0, max(0.0, abstain_mass)))

    return texts, span_confs, decision_confs, mean_confs


def _free_cuda():
    try:
        
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---- IO helpers ------------------------------------------------------------
def _pred_path(name: str, dataset: str) -> str:
    return os.path.join(OUTPUT_DIR, name, dataset, "predictions.jsonl")


def _partial_path(name: str, dataset: str) -> str:
    return _pred_path(name, dataset) + ".partial"


def _lock_path(name: str, dataset: str) -> str:
    return _pred_path(name, dataset) + ".lock"


def _unlink(p: str) -> None:
    try:
        os.remove(p)
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _is_done(mk: str, name: str, dataset: str) -> bool:
    """A unit is done iff its predictions.jsonl exists (atomically promoted)."""
    return os.path.exists(_pred_path(f"{mk}:{name}", dataset))


def _heartbeat(lock: str) -> None:
    try:
        os.utime(lock, None)
    except OSError:
        pass


def _claim(mk: str, name: str, dataset: str, lock_ttl: float = 900.0) -> bool:
    """Atomically claim a unit. True if this process now owns it; False if it is
    already done or owned by a LIVE worker. A lock is reclaimed if its owning pid
    is dead on this host OR if its heartbeat (mtime) is older than lock_ttl
    (cross-node safe, because live workers heartbeat their lock every example)."""
    full = f"{mk}:{name}"
    pth = _pred_path(full, dataset)
    if os.path.exists(pth):
        return False
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    lock = _lock_path(full, dataset)
    host = socket.gethostname()
    for attempt in (0, 1):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time():.0f} {host}".encode())
            os.close(fd)
            if os.path.exists(pth):
                _unlink(lock)
                return False
            return True
        except FileExistsError:
            prev_pid, prev_host = None, None
            try:
                parts = open(lock).read().split()
                prev_pid = int(parts[0])
                prev_host = parts[2] if len(parts) > 2 else None
            except Exception:
                pass
            try:
                age = time.time() - os.path.getmtime(lock)
            except OSError:
                age = None
            dead_here = (prev_host == host and prev_pid is not None
                         and not _pid_alive(prev_pid))
            timed_out = (lock_ttl and age is not None and age > lock_ttl)
            if attempt == 0 and (dead_here or timed_out):
                why = "dead pid" if dead_here else f"heartbeat {age:.0f}s > ttl"
                print(f"[reclaim] stale lock {lock} ({why})", flush=True)
                _unlink(lock)
                continue
            return False
    return False


def _load_partial(partial: str):
    """Return (done_ids, valid_records) from an existing partial file, dropping
    any corrupt (e.g. half-written) trailing line."""
    done_ids, valid = set(), []
    if not os.path.exists(partial):
        return done_ids, valid
    with open(partial, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # corrupt tail -> drop, the example will be redone
            valid.append(rec)
            done_ids.add(rec.get("id"))
    return done_ids, valid


def _rewrite_partial(partial: str, records: List[dict]) -> None:
    """Rewrite the partial file with only valid records (atomic)."""
    tmp = f"{partial}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, partial)


def _partial_count(mk: str, name: str, dataset: str) -> int:
    p = _partial_path(f"{mk}:{name}", dataset)
    if not os.path.exists(p):
        return 0
    c = 0
    try:
        with open(p, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    c += 1
    except Exception:
        return 0
    return c


def load_examples(dataset: str, limit: Optional[int], strategies: Optional[List[str]]):
    
    mod_name, fn = DATASET_LOADERS[dataset]
    mod = importlib.import_module(mod_name)
    kwargs: Dict[str, Any] = {"strategies": strategies, "limit": limit}
    return [ex.to_dict() for ex in getattr(mod, fn)(**kwargs)]


# ---- signal handling (clean preemption) ------------------------------------
def _on_term(signum, frame):
    f = _ACTIVE.get("file")
    if f is not None:
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass
    lk = _ACTIVE.get("lock")
    if lk:
        _unlink(lk)
    print(f"[signal {signum}] partial flushed, lock released -- safe to resume",
          flush=True)
    os._exit(143)


def _install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_term)
        except Exception:
            pass


# ---- core generation unit (incremental save + resume) ----------------------
def _process_unit(model, tok, fam, mk, name, dataset, examples, max_new_tokens,
                  worker_id=0, gpu_id=0, progress=None, lock_ttl=900.0,
                  save_every=30, batch_size=32, prompt_cache=None) -> None:
    """Claim -> resume partial -> generate in batches (each batch appended +
    flushed) -> atomically promote to predictions.jsonl -> release. No-op if
    already done or owned by a live worker.

    Prompts are rendered+tokenized once and cached in ``prompt_cache`` (shared
    across every ratio/base of the same model, since the chat template depends
    only on tokenizer + family + messages), so the per-dataset tokenisation cost
    is paid once instead of once per adapter.
    """
    if not _claim(mk, name, dataset, lock_ttl=lock_ttl):
        return
    full = f"{mk}:{name}"
    pth = _pred_path(full, dataset)
    partial = _partial_path(full, dataset)
    lock = _lock_path(full, dataset)

    done_ids, valid = _load_partial(partial)
    if os.path.exists(partial):
        _rewrite_partial(partial, valid)  # drop any corrupt trailing line

    f = open(partial, "a", encoding="utf-8")
    _ACTIVE["file"], _ACTIVE["lock"] = f, lock
    n_new = 0
    since_flush = 0
    try:
        # outstanding examples only (skip ids already in the partial)
        pending = []
        for i, ex in enumerate(examples):
            ex_id = ex.get("id", f"{dataset}#{i}")
            if ex_id in done_ids:
                continue
            pending.append((ex_id, ex))

        for b in range(0, len(pending), batch_size):
            chunk = pending[b:b + batch_size]

            # cached render + tokenize for the whole chunk
            batch_ids = []
            for ex_id, ex in chunk:
                key = (dataset, ex_id)
                ids = prompt_cache.get(key) if prompt_cache is not None else None
                if ids is None:
                    prompt = render_prompt(tok, ex["messages"], fam)
                    ids = tok(prompt)["input_ids"]
                    if prompt_cache is not None:
                        prompt_cache[key] = ids
                batch_ids.append(ids)

            # one batched forward pass: texts + all three confidence channels
            texts, sconfs, dconfs, mconfs = generate_batch(model, tok, batch_ids, max_new_tokens)

            for (ex_id, ex), text, sconf, dconf, mconf in zip(chunk, texts, sconfs, dconfs, mconfs):
                # PRIMARY channel = answer-span exp(mean logprob), ported from
                # the ID pipeline so ID and OOD tables join like-for-like.
                # decision (abstain-mass) and mean-prob kept as extra columns.
                r = annotate(ex, text, token_confidence=sconf)
                r["token_confidence_decision"] = dconf
                r["token_confidence_mean"] = mconf
                r["id"] = ex_id
                r["model_id"] = mk
                r["ratio"] = name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_new += 1
                if progress is not None:
                    with progress.get_lock():
                        progress.value += 1

            since_flush += len(chunk)
            if since_flush >= save_every:      # crash loses at most one batch
                f.flush()
                since_flush = 0
            _heartbeat(lock)

        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        f.close()
        _ACTIVE["file"] = None
        os.replace(partial, pth)  # atomic promotion -> unit is now "done"
        print(f"[done w{worker_id}|gpu{gpu_id}] {mk}:{name}/{dataset}: "
              f"new={n_new} resumed={len(done_ids)} -> {pth}", flush=True)
    finally:
        try:
            if not f.closed:
                f.close()
        except Exception:
            pass
        _unlink(lock)
        _ACTIVE["file"], _ACTIVE["lock"] = None, None


# ---- worker: process an assignment on a pinned GPU -------------------------
def _worker(worker_id, gpu_id, assignment, adapters_by_model, presets,
            datasets, strategies, limit, max_new_tokens, output_dir,
            progress=None, total_ex=None, ready=None, lock_ttl=900.0,
            save_every=30, batch_size=32):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    _install_signal_handlers()

    data = {d: load_examples(d, limit, strategies) for d in datasets}

    # Report REMAINING examples (exclude done units AND already-saved partials)
    # so the global bar sizes to true outstanding work and reaches 100%.
    remaining = 0
    for model_key, names in assignment.items():
        for name in names:
            for d in datasets:
                if _is_done(model_key, name, d):
                    continue
                remaining += max(0, len(data[d]) - _partial_count(model_key, name, d))
    if total_ex is not None:
        with total_ex.get_lock():
            total_ex.value += remaining
    if ready is not None:
        with ready.get_lock():
            ready.value += 1

    for model_key, names in assignment.items():
        preset = presets[model_key]
        fam = preset["family"]
        adapters = adapters_by_model[model_key]

        # names with outstanding work; skip the model entirely if none remain
        pending_names = [n for n in names
                         if not all(_is_done(model_key, n, d) for d in datasets)]
        if not pending_names:
            print(f"[w{worker_id}] {model_key}: all assigned units done", flush=True)
            continue

        print(f"[w{worker_id} gpu{gpu_id}] loading base {preset['base_repo']} "
              f"for {model_key} (names={pending_names})", flush=True)
        tok, base = load_base(preset["base_repo"])

        # ---- load each needed adapter ONCE, keep them all resident on the GPU,
        #      and switch the active one with set_adapter(). 'base' runs through
        #      the same wrapper with adapters disabled. No repeated PEFT loading.
        #      Adapters are registered under dot-free keys (PEFT/torch reject '.'
        #      in module names, so the '0.00'-style ratio labels can't be used
        #      directly); `name` itself is untouched for paths + the 'ratio' field.
        peft_model = None
        ratio_names = [n for n in pending_names if n != "base"]
        adapter_keys = {n: _safe_adapter_name(n) for n in ratio_names}
        for name in ratio_names:
            repo = adapters[name]
            akey = adapter_keys[name]
            if peft_model is None:
                from peft import PeftModel
                peft_model = PeftModel.from_pretrained(base, repo, adapter_name=akey)
            else:
                peft_model.load_adapter(repo, adapter_name=akey)
            print(f"[w{worker_id} gpu{gpu_id}]   + adapter {name} (key={akey}) <- {repo}", flush=True)

        # tokenised-prompt cache shared across all names of this model
        prompt_cache: Dict[Any, Any] = {}

        for name in pending_names:
            if name == "base":
                if peft_model is not None:
                    # run the underlying base weights (LoRA deltas disabled)
                    with peft_model.disable_adapter():
                        for d in datasets:
                            _process_unit(peft_model, tok, fam, model_key, name, d,
                                          data[d], max_new_tokens, worker_id, gpu_id,
                                          progress, lock_ttl, save_every,
                                          batch_size, prompt_cache)
                else:
                    for d in datasets:
                        _process_unit(base, tok, fam, model_key, name, d,
                                      data[d], max_new_tokens, worker_id, gpu_id,
                                      progress, lock_ttl, save_every,
                                      batch_size, prompt_cache)
            else:
                peft_model.set_adapter(adapter_keys[name])   # cheap GPU-resident switch
                for d in datasets:
                    _process_unit(peft_model, tok, fam, model_key, name, d,
                                  data[d], max_new_tokens, worker_id, gpu_id,
                                  progress, lock_ttl, save_every,
                                  batch_size, prompt_cache)

        del peft_model, base, tok
        _free_cuda()
    print(f"[w{worker_id} gpu{gpu_id}] finished", flush=True)


def _partition(plan, num_workers, include_base):
    """Round-robin each model's names (base + ratios) across workers so different
    ratios of the same model run concurrently. 'base' lands on worker 0 only."""
    assignments = [dict() for _ in range(num_workers)]
    for model_key, _preset, adapters in plan:
        names = (["base"] if include_base else []) + list(adapters)
        for i, name in enumerate(names):
            w = i % num_workers
            assignments[w].setdefault(model_key, []).append(name)
    return assignments


def _detect_gpus() -> List[int]:
    try:
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _monitor(procs, progress, total_ex, ready, n_workers, poll=0.5):
    """One global example-level progress bar in the parent, fed by shared counters."""
    bar = tqdm(total=None, unit="ex", desc="overall", dynamic_ncols=True, smoothing=0.1)
    have_total = False
    while any(p.is_alive() for p in procs):
        if not have_total and ready.value >= n_workers:
            bar.total = total_ex.value or None
            have_total = True
        bar.n = progress.value
        bar.refresh()
        time.sleep(poll)
    if not have_total:
        bar.total = total_ex.value or progress.value or None
    bar.n = progress.value
    bar.refresh()
    bar.close()
    for p in procs:
        p.join()


# ---- run (parallel orchestrator) -------------------------------------------
def run(models, datasets, strategies, limit, max_new_tokens, ratios_filter,
        include_base, num_workers, gpu_ids, lock_ttl=900.0, save_every=30,
        batch_size=32):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plan = []
    adapters_by_model: Dict[str, Dict[str, str]] = {}
    for mk in models:
        preset = MODEL_PRESETS[mk]
        adapters = discover_adapters(preset["hub_name_filter"])
        if ratios_filter:
            adapters = {r: v for r, v in adapters.items() if r in ratios_filter}
        if not adapters:
            print(f"  !! {mk}: NO adapters matched '{preset['hub_name_filter']}'", flush=True)
        adapters_by_model[mk] = adapters
        plan.append((mk, preset, adapters))

    presets = {mk: MODEL_PRESETS[mk] for mk in models}
    assignments = _partition(plan, num_workers, include_base)

    ctx = mp.get_context("spawn")
    progress = ctx.Value("L", 0)
    total_ex = ctx.Value("L", 0)
    ready    = ctx.Value("i", 0)

    procs = []
    for w in range(num_workers):
        nonempty = sum(len(v) for v in assignments[w].values())
        if nonempty == 0:
            continue
        gpu = gpu_ids[w % len(gpu_ids)]
        print(f"[launch] worker {w} -> gpu {gpu} | {nonempty} (model,name) tasks", flush=True)
        p = ctx.Process(
            target=_worker,
            args=(w, gpu, assignments[w], adapters_by_model, presets,
                  datasets, strategies, limit, max_new_tokens, OUTPUT_DIR,
                  progress, total_ex, ready, lock_ttl, save_every, batch_size),
            name=f"ood-worker-{w}",
        )
        p.start()
        procs.append(p)

    if not procs:
        print("[run] nothing to do -- all units already complete", flush=True)
        return

    _monitor(procs, progress, total_ex, ready, n_workers=len(procs))

    bad = [(p.name, p.exitcode) for p in procs if p.exitcode not in (0, None)]
    if bad:
        print(f"[warn] worker(s) exited non-zero: {bad} -- re-run to resume "
              f"(partials + claim/done checker skip finished work)", flush=True)


# ---- dry run ---------------------------------------------------------------
def dry_run(models, datasets, strategies, limit):
    print("DRY RUN -- no model weights are loaded\n" + "=" * 64)
    print("\n[1] Hub adapter discovery (HUB_AUTHOR=%s)" % HUB_AUTHOR)
    for mk in models:
        preset = MODEL_PRESETS[mk]
        try:
            adapters = discover_adapters(preset["hub_name_filter"])
            if not adapters:
                print(f"  !! {mk}: NO adapters matched filter "
                      f"'{preset['hub_name_filter']}' -- fix hub_name_filter / HF_TOKEN")
        except Exception as e:
            print(f"  !! {mk}: discovery FAILED: {e!r}")

    print("\n[2] Dataset example counts" + (f" (limit={limit})" if limit else " (full)"))
    total = 0
    for d in datasets:
        try:
            exs = load_examples(d, limit, strategies)
            n = len(exs); total += n
            ans = sum(1 for e in exs if e["is_answerable"])
            strats = sorted({e["metadata"].get("strategy") for e in exs})
            variants = sorted({str(e["metadata"].get("variant")) for e in exs})
            print(f"  {d:<15} total={n:<6} answerable={ans:<6} unanswerable={n-ans:<6} "
                  f"strategies={strats} variants={variants}")
        except Exception as e:
            print(f"  !! {d:<15} LOAD FAILED: {e!r}")

    print("\n[3] GPU / worker plan")
    gpus = _detect_gpus()
    print(f"  visible GPUs: {gpus or 'none (CPU)'}")
    print(f"  examples/unit (sum over datasets): {total}")
    print("\n[dry-run] done -- confirm each adapter map above is non-empty before a real run")


# ---- collect ---------------------------------------------------------------
def collect():
    files = glob.glob(os.path.join(OUTPUT_DIR, "**", "predictions.jsonl"), recursive=True)
    if not files:
        print("[collect] no predictions found"); return
    annotated = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            annotated.extend(json.loads(l) for l in f if l.strip())
    rows = compute_aggregate(annotated)

    with open(os.path.join(OUTPUT_DIR, "ood_metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        cols = list(rows[0].keys())
        with open(os.path.join(OUTPUT_DIR, "ood_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    curves = compute_curves(annotated)
    with open(os.path.join(OUTPUT_DIR, "ood_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)

    # variant-POOLED decision calibration — where decision AUROC is actually
    # defined (per-variant groups are single-class by construction).
    pooled = compute_decision_calibration(annotated)
    with open(os.path.join(OUTPUT_DIR, "ood_decision_calibration.json"), "w") as f:
        json.dump(pooled, f, indent=2)
    if pooled:
        cols = list(pooled[0].keys())
        with open(os.path.join(OUTPUT_DIR, "ood_decision_calibration.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(pooled)

    pooled_curves = compute_decision_curves(annotated)
    with open(os.path.join(OUTPUT_DIR, "ood_decision_curves.json"), "w") as f:
        json.dump(pooled_curves, f, indent=2)

    # smoke-run tell: the ID pipeline shows decision AUROC well above 0.5 on
    # this channel, so a pooled value below 0.5 most likely means a channel
    # orientation / extraction bug — investigate before the full sweep.
    for row in pooled:
        a = row.get("token_decision_auroc")
        if a is not None and a < 0.5:
            print(f"[collect][WARN] token_decision_auroc={a:.3f} < 0.5 for "
                  f"{row['dataset']}/{row['model_id']}/{row['ratio']}/{row['strategy']} "
                  f"— check the answer-span confidence extraction.")

    _print_comparison(rows)
    print(f"\n[collect] {len(annotated)} predictions over {len(files)} files -> "
          f"{OUTPUT_DIR}/ood_metrics.* (+ ood_curves.json)")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) and x == x else ("-" if x != x else str(x))


def _print_comparison(rows):
    by_ds: Dict[str, list] = {}
    for r in rows:
        by_ds.setdefault(f"{r['dataset']}/{r.get('variant')}", []).append(r)
    for ds, rs in by_ds.items():
        print(f"\n----- {ds} -----")
        hdr = ("model", "ratio", "strat", "abst_rec", "over_abs", "ans_acc",
               "tok_dec_auroc", "vrb_dec_auroc", "vrb_mean", "vrb_std")
        print("".join(f"{h:<13}" for h in hdr))
        for r in sorted(rs, key=lambda x: (str(x["model_id"]), str(x["ratio"]), str(x["strategy"]))):
            cells = (r["model_id"], r["ratio"], r["strategy"],
                     _fmt(r["abstention_recall"]), _fmt(r["over_abstention_rate"]),
                     _fmt(r["answer_accuracy"]), _fmt(r["token_decision_auroc"]),
                     _fmt(r["verbal_decision_auroc"]), _fmt(r["verbal_conf_mean"]),
                     _fmt(r["verbal_conf_std"]))
            print("".join(f"{str(c):<13}" for c in cells))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "collect"])
    ap.add_argument("--models", nargs="*", default=list(MODEL_PRESETS))
    ap.add_argument("--datasets", nargs="*", default=list(DATASET_LOADERS))
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--ratios", nargs="*", default=None, help="restrict to these ratio labels, e.g. 0.00 0.50 1.00")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="examples generated together per forward pass. Higher "
                         "raises GPU utilisation/throughput at the cost of memory; "
                         "a kill loses at most one in-flight batch.")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: #GPUs). Set > #GPUs to "
                         "oversubscribe and raise per-GPU utilisation, memory permitting.")
    ap.add_argument("--gpus", nargs="*", type=int, default=None,
                    help="GPU ids to use (default: all visible)")
    ap.add_argument("--lock-ttl", type=float, default=900.0,
                    help="seconds before a non-heartbeating lock is reclaimed (resume safety)")
    ap.add_argument("--save-every", type=int, default=1,
                    help="flush the partial file once at least N new examples have "
                         "been written (flushing is checked per batch)")
    ap.add_argument("--no-collect", action="store_true",
                    help="skip the automatic collect() after run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print discovered adapter map + example counts, load no weights")
    a = ap.parse_args()

    if a.mode == "collect":
        collect(); return
    if a.dry_run:
        dry_run(a.models, a.datasets, a.strategies, a.limit); return

    gpus = a.gpus if a.gpus is not None else _detect_gpus()
    workers = max(1, a.workers) if a.workers is not None else max(1, len(gpus))
    if workers > 1 and not gpus:
        print("[warn] >1 worker requested but no GPUs detected; forcing --workers 1")
        workers = 1
    if not gpus:
        gpus = [0]
    print(f"[plan] workers={workers} gpus={gpus} batch_size={a.batch_size} "
          f"lock_ttl={a.lock_ttl}s save_every={a.save_every}", flush=True)

    run(a.models, a.datasets, a.strategies, a.limit, a.max_new_tokens,
        set(a.ratios) if a.ratios else None, include_base=not a.no_base,
        num_workers=workers, gpu_ids=gpus, lock_ttl=a.lock_ttl, save_every=a.save_every,
        batch_size=a.batch_size)

    if not a.no_collect:
        collect()


if __name__ == "__main__":
    main()
