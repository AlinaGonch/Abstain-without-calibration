"""
ood_prompt_templates.py — derived (single source of truth = prompt_template.py)
==============================================================================
For the NO-CONTEXT reasoning datasets (GSM8K-Abstain, GPQA-Abstain, MMLU-Pro).
FaithEval/NoMIRACL/HotpotQA keep calling prompt_template.get_chat_messages
directly.

Instead of hand-copying your rule text, this module DERIVES each OOD prompt from
your real builders at runtime:

  1. call pt.get_chat_messages(strategy, context=<sentinel>, question=<sentinel>)
  2. take the system text + user template your builder produced
  3. drop the "Context:" line (there is none here)
  4. apply a small ordered set of phrase substitutions to retarget context->problem
     and remove the "Do not use outside knowledge." rule (wrong for math/science)

So if you edit SYSTEM_RULES, the CoT rules, or the self_reflect bullets/format in
prompt_template.py, those edits propagate here automatically. Only two things are
necessarily local: the noun mapping (problem/question) and the few-shot examples
(your SQuAD example bank is context-based and can't be reused for math/MC).

If a future edit changes the exact phrases below, the targeted substitutions
simply won't fire (the generic "the context"->noun fallback still runs); re-check
this file's SUBS list if you rename those rules.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import prompt_template as pt

OOD_STRATEGIES = ["zero_shot_conf", "few_shot_balance_conf", "cot_conf", "self_reflect_conf"]

KIND_MATH_FREE = "math_free"   # gsm8k-abstain
KIND_MC = "mc"                 # gpqa-abstain, mmlu-pro
_NOUN = {KIND_MATH_FREE: "problem", KIND_MC: "question"}

# OOD strategy -> the prompt_template.py strategy whose wording we inherit.
# few_shot_balance shares SYSTEM_RULES with zero_shot; we inherit that head and
# inject OOD few-shot ourselves.
_HER_STRATEGY = {
    "zero_shot_conf": "zero_shot_conf",
    "few_shot_balance_conf": "zero_shot_conf",
    "cot_conf": "cot_conf",
    "self_reflect_conf": "self_reflect_conf",
}

_SENT_CTX = "[[OOD_CTX]]"
_SENT_BODY = "[[OOD_BODY]]"

# Domain-neutral few-shot (your SQuAD bank is context-based; can't transfer).
# Stored as (body, answer); rendered with the same "Question:" label your builder
# uses for the live item, so the few-shot format matches.
_FEWSHOT = {
    KIND_MATH_FREE: [
        ("A baker has 12 muffins and sells 5. How many are left?", "Final response: 7"),
        ("A train leaves the station. How far has it travelled?", "Final response: I don't know"),
    ],
    KIND_MC: [
        ("What is 2 + 2?\nOptions:\nA) 3\nB) 4\nC) 5\nD) 6", "Final response: B"),
        ("Which value is larger?\nOptions:\nA) x\nB) y\nC) z\nD) w", "Final response: I don't know"),
    ],
}


def _subs(noun: str) -> List[Tuple[str, str]]:
    # ordered; earlier targeted phrases consume the generic ones
    return [
        ("based only on the given context", "using only the information given"),
        ("fully supported by the context", f"fully supported by the {noun}"),
        ("the given context", f"the {noun}"),
        ("the context", f"the {noun}"),
    ]


def _retarget(text: str, noun: str, drop_outside_knowledge: bool = True) -> str:
    lines = text.split("\n")
    if drop_outside_knowledge:
        lines = [ln for ln in lines if ln.strip() != "- Do not use outside knowledge."]
    text = "\n".join(lines)
    for find, repl in _subs(noun):
        text = text.replace(find, repl)
    return text


def _derive(her_strategy: str, kind: str) -> Tuple[str, str]:
    """Return (system_text, user_template) inherited from your builder, with the
    context line removed and wording retargeted. user_template still contains the
    _SENT_BODY placeholder."""
    pair = pt.get_chat_messages(her_strategy, context=_SENT_CTX, question=_SENT_BODY)
    noun = _NOUN[kind]
    system = _retarget(pair[0]["content"], noun)
    # user message: drop the line carrying the context sentinel, retarget prose
    user_lines = [ln for ln in pair[-1]["content"].split("\n") if _SENT_CTX not in ln]
    user = _retarget("\n".join(user_lines), noun, drop_outside_knowledge=False)
    return system, user


def _format_body(problem: str, choices: Optional[List[str]], kind: str) -> str:
    if kind == KIND_MC and choices:
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        opts = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
        return f"{problem}\nOptions:\n{opts}"
    return problem


def get_ood_chat_messages(
    strategy: str,
    problem_text: str,
    choices: Optional[List[str]] = None,
    kind: str = KIND_MATH_FREE,
) -> List[Dict[str, str]]:
    if strategy not in OOD_STRATEGIES:
        raise ValueError(f"Unknown OOD strategy {strategy!r}; expected {OOD_STRATEGIES}")

    system, user_tmpl = _derive(_HER_STRATEGY[strategy], kind)
    body = _format_body(problem_text, choices, kind)
    user = user_tmpl.replace(_SENT_BODY, body)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if strategy == "few_shot_balance":
        for ex_body, ex_ans in _FEWSHOT[kind]:
            msgs.append({"role": "user", "content": f"Question: {ex_body}"})
            msgs.append({"role": "assistant", "content": ex_ans})
    msgs.append({"role": "user", "content": user})
    return msgs
