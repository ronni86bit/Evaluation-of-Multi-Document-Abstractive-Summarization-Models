
from datasets import load_dataset
from typing import Dict, List, Tuple
import random
from .utils import join_docs, START_DOC

SPECIAL_SEP = "\n\n[DOC]\n\n"

def load_cnn_dm(split="train", version="3.0.0"):
    ds = load_dataset("cnn_dailymail", version, split=split)

    def _map(ex):
        return {"id": ex.get("id", ""), "document": ex["article"], "summary": ex["highlights"]}
    return ds.map(_map, remove_columns=ds.column_names)

def make_mds_example_from_group(examples: List[Dict], max_docs: int = 3) -> Dict:
    k = min(len(examples), max_docs)
    picked = random.sample(examples, k)
    multi = SPECIAL_SEP.join([p["document"] for p in picked])
 
    ref = max([p["summary"] for p in picked], key=len)
    return {"document": multi, "summary": ref}

def chunk_text(text: str, n_chunks: int = 3) -> List[str]:

    parts = [p for p in text.split("\n") if p.strip()]
    if n_chunks >= len(parts):
        return parts
    size = max(1, len(parts)//n_chunks)
    chunks = []
    for i in range(0, len(parts), size):
        chunks.append("\n".join(parts[i:i+size]))
        if len(chunks) == n_chunks:
            break
    return chunks

def preprocess_for_tokenizer(batch, tokenizer, max_source_length, max_target_length, model_type="t5", task_prefix="summarize: "):
    inputs = batch["document"]
    targets = batch["summary"]

    if model_type.lower().startswith("t5"):
        inputs = [f"{task_prefix}{inp}" for inp in inputs]

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
