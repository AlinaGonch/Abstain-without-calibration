"""
QLoRA fine-tuning of Llama 3.1 8B for text -> knowledge-graph-triple extraction.

Loads the base model in 4-bit (NF4 + double quant) and trains a LoRA adapter on
top, so an 8B fits on a single 24GB GPU. Each dataset row becomes a
prompt/completion chat pair, so loss is computed only on the triples, not on the
source passage. Early stopping halts the run if eval loss stops improving.

~6k examples x 3 epochs yields a small LoRA adapter and takes ~1-2h on one A100
(longer on a 4090).

Single GPU:   python train.py
Multi-GPU:    accelerate launch train.py   # then remove device_map="auto" below
"""
import os
import torch
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HUB_MODEL_ID = "AlinaGonch/llama3.1-8b-kg-extraction"

SYSTEM_PROMPT = """Extract a knowledge graph (KG) from the following text. Follow these steps:
1. **Entities**: Identify all entities in the text. Ensure each entity is precise and specific.
2. **Relations**: Extract relationships between entities as triples: ["entity1", "relation", "entity2"].
3. **Coreference Resolution**: Unify references to the same entity (e.g., "Apple Inc." and "Apple" should be the same entity).
**Important Requirements**:
- The KG must not be empty. Ensure at least one triple is extracted.
- All entities mentioned in the text must be included in the KG, either as part of a triple or as a standalone entity if no relation is found.
- If no explicit relation is found between entities, create a generic relation like "related to" or "associated with" to ensure all entities are connected.
- Each triple must have three non-empty elements: ["entity1", "relation", "entity2"]. None of these elements can be empty or null.
Please only return the KG as a Python list of triples. For example:
<python>
[
["Apple Inc.", "founded by", "Steve Jobs"],
["Apple Inc.", "headquartered in", "Cupertino, California"],
["Apple Inc.", "produces", "iPhone"],
["Steve Jobs", "associated with", "Cupertino, California"]
]
</python>
Text:"""

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
dataset = load_dataset(
    "json",
    data_files={
        "train": "extractor_train_data.jsonl",
        "eval": "extractor_eval_data.jsonl",
    },
)
# No separate eval file? Split from train instead:
# full = load_dataset("json", data_files="extractor_train_data.jsonl", split="train")
# parts = full.train_test_split(test_size=0.05, seed=42)
# dataset = {"train": parts["train"], "eval": parts["test"]}


def to_prompt_completion(example):
    """context/kg_triplets row -> prompt/completion chat pair.

    Using prompt + completion (rather than a single messages list) means
    completion_only_loss applies by default, so the passage is masked out and the
    model is trained only to produce the triples.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["context"]},
        ],
        "completion": [
            {"role": "assistant", "content": example["kg_triplets"]},
        ],
    }


train_set = dataset["train"].map(
    to_prompt_completion, remove_columns=dataset["train"].column_names
)
eval_set = dataset["eval"].map(
    to_prompt_completion, remove_columns=dataset["eval"].column_names
)
print(f"Train set: {len(train_set)} examples, Eval set: {len(eval_set)} examples")
print(f"Example prompt: {train_set[0]['prompt']}")

# --------------------------------------------------------------------------- #
# 4-bit base model (the "Q" in QLoRA)
# --------------------------------------------------------------------------- #
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",  # drop this when launching multi-GPU via accelerate/torchrun
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model.config.use_cache = False  # required when gradient checkpointing is enabled
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
training_args = SFTConfig(
    output_dir="final",
    per_device_train_batch_size=8,        # lower to 4/2 if you OOM on a 24GB card
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,        # effective batch = 8 * 4 = 32
    learning_rate=2e-4,
    num_train_epochs=3,
    max_length=2048,                      # older TRL versions call this max_seq_length
    bf16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    # eval and save MUST share strategy + interval for load_best_model_at_end.
    # ~576 total steps here, so eval every 50 gives ~11 checkpoints for early
    # stopping to work with. Bump to 100 if the eval pass is slow.
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,                   # best checkpoint is kept even if outside the last 3
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,              # lower eval loss is better
    hub_model_id=HUB_MODEL_ID,            # final push target; no push during training
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    peft_config=peft_config,
    processing_class=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,     # stop after 3 evals with no improvement
            early_stopping_threshold=0.0,  # any decrease in eval loss counts
        )
    ],
)

trainer.train()

# Best adapter is loaded in memory (load_best_model_at_end=True); push it once.
trainer.push_to_hub()
