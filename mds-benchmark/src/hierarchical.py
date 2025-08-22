
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HierarchicalSummarizer:
    def __init__(self, base_model="t5-base"):
        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    def _summ(self, text: str, max_len=96):
        enc = self.tok("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024)
        out = self.model.generate(**enc, max_length=max_len)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def summarize(self, documents: List[str], chunk_size=800):

        chunks = []
        for d in documents:
            for i in range(0, len(d), chunk_size):
                chunks.append(d[i:i+chunk_size])
     
        l1 = [self._summ(c) for c in chunks]
    
        fused = self._summ(" ".join(l1), max_len=160)
        return fused
