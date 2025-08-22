
import os, math, json, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_rouge(preds: List[str], labels: List[str]) -> Dict[str, float]:
    scores = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    # return standard keys
    return {
        "rouge1": scores.get("rouge1", 0.0) * 100,
        "rouge2": scores.get("rouge2", 0.0) * 100,
        "rougeL": scores.get("rougeL", 0.0) * 100,
        "rougeLsum": scores.get("rougeLsum", 0.0) * 100,
    }

DOC_SEP = " [DOC] "
START_DOC = "[DOC] "

def join_docs(docs: List[str]) -> str:
    docs = [d.strip() for d in docs if d and d.strip()]
    return DOC_SEP.join(docs)

def add_prefix_for_t5(text: str, task_prefix: str = "summarize: ") -> str:
    return f"{task_prefix}{text}"
