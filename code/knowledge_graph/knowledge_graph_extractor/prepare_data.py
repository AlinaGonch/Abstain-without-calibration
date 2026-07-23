"""
Prepare Claude-generated KG extractions for Llama 3.1 8B fine-tuning.

Uses the SAME prompt that was used to generate the data with Claude, so the
fine-tuned model learns the exact task formulation at inference time.

Input JSONL — each line is a JSON object with:
  - "context" (or "text"): the input text
  - "triples" (or "kg" / "knowledge_graph"): a list of [head, relation, tail] triples
    Triples may be lists [["e1","rel","e2"], ...] OR dicts [{"head":..., "relation":..., "tail":...}, ...]
"""
import argparse
import json
import random
import re
from pathlib import Path
from datasets import Dataset

# Exact prompt used to generate the data with Claude.
EXTRACTION_PROMPT = """Extract a knowledge graph (KG) from the following text. Follow these steps:

1. **Entities**: Identify all entities in the text. Ensure each entity is precise and specific.
2. **Relations**: Extract relationships between entities as triples: ["entity1", "relation", "entity2"].
3. **Coreference Resolution**: Unify references to the same entity (e.g., "Apple Inc." and "Apple" should be the same entity).

**Important Requirements**:
- The KG must not be empty. Ensure at least one triple is extracted.
- All entities mentioned in the text must be included in the KG, either as part of a triple or as a standalone entity if no relation is found.
- If no explicit relation is found between entities, create a generic relation like "related to" or "associated with" to ensure all entities are connected.
- Each triple must have three non-empty elements: ["entity1", "relation", "entity2"]. None of these elements can be empty or null.

Please only return the KG as a Python list of triples. For example:
[
    ["Apple Inc.", "founded by", "Steve Jobs"],
    ["Apple Inc.", "headquartered in", "Cupertino, California"],
    ["Apple Inc.", "produces", "iPhone"],
    ["Steve Jobs", "associated with", "Cupertino, California"]
]

Text: {TEXT}"""


# ---- Input parsing (handle several plausible storage shapes) ----

CONTEXT_KEYS = ("context", "text", "passage", "input")
TRIPLES_KEYS = ("triples", "kg", "knowledge_graph", "output", "graph", 'kg_triplets')


def get_context(ex: dict) -> str:
    for k in CONTEXT_KEYS:
        if k in ex and isinstance(ex[k], str):
            return ex[k].strip()
    return ""


def get_triples_raw(ex: dict):
    for k in TRIPLES_KEYS:
        if k in ex:
            return ex[k]
    return None


def coerce_triple(t):
    """Accept either ['h','r','t'] or {'head':..,'relation':..,'tail':..} -> tuple or None."""
    if isinstance(t, list) and len(t) == 3:
        h, r, tl = t
    elif isinstance(t, dict):
        h = t.get("head") or t.get("subject") or t.get("h")
        r = t.get("relation") or t.get("predicate") or t.get("r")
        tl = t.get("tail") or t.get("object") or t.get("t")
    else:
        return None
    if not (isinstance(h, str) and isinstance(r, str) and isinstance(tl, str)):
        return None
    h, r, tl = h.strip(), r.strip(), tl.strip()
    if not (h and r and tl):
        return None
    return [h, r, tl]


def parse_triples_field(raw):
    """raw may already be a list, or it may be a string containing a Python/JSON list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        # Strip markdown fences if present
        s = raw.strip()
        s = re.sub(r"^```(?:python|json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    triples = [coerce_triple(t) for t in raw]
    return [t for t in triples if t is not None]


# ---- Loading + filtering ----

def load_and_clean(input_path: str, ground_in_context: bool) -> list[dict]:
    examples = []
    n_lines = 0
    n_no_context = 0
    n_no_triples = 0
    n_no_grounded = 0

 
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            context = get_context(ex)
            if not context:
                n_no_context += 1
                continue
            triples_row = get_triples_raw(ex)
            triples = parse_triples_field(triples_row)
            if not triples:
                n_no_triples += 1
                continue
            if ground_in_context:
                grounded_triples = []
                for h, r, t in triples:
                    if h in context and t in context:
                        grounded_triples.append([h, r, t])
                if not grounded_triples:
                    n_no_grounded += 1
                    continue
                triples = grounded_triples
            examples.append({"context": context, "triples": triples})

    print(f"  Lines read:               {n_lines}")
    print(f"  Loaded:                   {len(examples)}")
    print(f"  Skipped (no context):     {n_no_context}")
    print(f"  Skipped (no triples):     {n_no_triples}")
    if ground_in_context:
        print(f"  Skipped (no grounded triples): {n_no_grounded}")
    return examples


# ---- Format for training ----

def format_triples(triples: list[list[str]]) -> str:
    """Format like the prompt example: one triple per line, triple itself inline."""
    if not triples:
        return "[]"
    lines = [json.dumps(t, ensure_ascii=False) for t in triples]
    return "[\n    " + ",\n    ".join(lines) + "\n]"


def format_for_training(ex: dict) -> dict:
    """Single user message with the full extraction prompt; assistant returns the list."""
    user_content = EXTRACTION_PROMPT.replace("{TEXT}", ex["context"])
    assistant_content = format_triples(ex["triples"])
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="JSONL with Claude-generated extractions")
    parser.add_argument("--output_dir", type=Path, default=Path("./data"))
    parser.add_argument("--eval_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground_in_context", action="store_true",
                        help="Drop triples whose head/tail substring isn't in the text. "
                             "OFF by default because coref-resolved entities often won't "
                             "appear verbatim.")
    args = parser.parse_args()

    print(f"Loading from {args.input}")
    examples = load_and_clean(args.input, ground_in_context=args.ground_in_context)

    if len(examples) < args.eval_size + 100:
        raise SystemExit(
            f"Only {len(examples)} valid examples — need at least "
            f"{args.eval_size + 100} for a meaningful train/eval split."
        )

    random.seed(args.seed)
    random.shuffle(examples)
    eval_set = examples[: args.eval_size]
    train_set = examples[args.eval_size :]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Dataset.from_list([format_for_training(e) for e in train_set])
    eval_ds = Dataset.from_list([format_for_training(e) for e in eval_set])
    train_ds.save_to_disk(args.output_dir / "train")
    eval_ds.save_to_disk(args.output_dir / "eval")

    # Raw eval set for the evaluation script
    with open(args.output_dir / "eval_raw.jsonl", "w", encoding='utf-8') as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    avg_triples = sum(len(e["triples"]) for e in train_set) / len(train_set)
    relations = {}
    for e in train_set:
        for h, r, t in e["triples"]:
            relations[r] = relations.get(r, 0) + 1
    top = sorted(relations.items(), key=lambda x: -x[1])[:10]

    print()
    print(f"Train: {len(train_ds):>5}    Eval: {len(eval_ds):>5}")
    print(f"Avg triples/example (train): {avg_triples:.1f}")
    print(f"Unique relations (train):    {len(relations)}")
    print(f"Top relations: {', '.join(f'{r}({c})' for r, c in top)}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()