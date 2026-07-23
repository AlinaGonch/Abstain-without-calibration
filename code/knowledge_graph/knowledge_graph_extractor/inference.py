"""
Run the fine-tuned KG extractor on a JSONL of contexts.

Each input line:  {"context": "...", "triples": [...optional gold...]}
Each output line: same fields + "predicted_triples": [...]
"""
import argparse
import json
import re
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

SYSTEM_PROMPT = (
    "You are a knowledge graph extraction system. Extract all factual "
    "(subject, relation, object) triples from the given text. Return only valid "
    "JSON in the format: {\"triples\": [{\"head\": \"...\", \"relation\": \"...\", "
    "\"tail\": \"...\"}]}. Only extract facts explicitly stated in the text. "
    "Do not infer, paraphrase, or hallucinate."
)


def parse_output(text: str) -> list[dict]:
    """Best-effort JSON extraction. Strips markdown fences, finds first {...}."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find the first JSON object in the output
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data.get("triples", []) if isinstance(data, dict) else []


def extract(model, tokenizer, context: str, max_new_tokens: int = 1024) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract knowledge graph triples from this text:\n\n{context}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_output(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path,
                        help="Path to the LoRA adapter (the `final/` dir from train.py)")
    parser.add_argument("--input", required=True, type=Path,
                        help="JSONL file; each line must have a 'context' field")
    parser.add_argument("--output", required=True, type=Path,
                        help="Where to write predictions JSONL")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading adapter from {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # 2x faster generation
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    n_total = 0
    n_parse_fail = 0
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            triples = extract(model, tokenizer, ex["context"],
                              max_new_tokens=args.max_new_tokens)
            if not triples:
                n_parse_fail += 1
            n_total += 1
            ex["predicted_triples"] = triples
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if n_total % 25 == 0:
                print(f"  {n_total} processed  ({n_parse_fail} empty)")

    print(f"\nDone. {n_total} processed, {n_parse_fail} with empty/unparseable output.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
