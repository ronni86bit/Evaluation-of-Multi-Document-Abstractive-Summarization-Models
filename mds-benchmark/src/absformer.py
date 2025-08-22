
"""
Absformer (unsupervised MDS baseline):
- Build a sentence graph with TextRank (via NetworkX over TF-IDF cosine).
- Select top-K sentences across the collection (multi-doc).
- Compress with a small abstractive model (e.g., t5-base) to produce a fluent final summary.
"""
from typing import List
import numpy as np, networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def textrank_select(sentences: List[str], top_k: int = 10) -> List[str]:
    vect = TfidfVectorizer().fit_transform(sentences)
    sim = (vect * vect.T).A
    np.fill_diagonal(sim, 0.0)
    g = nx.from_numpy_array(sim)
    scores = nx.pagerank_numpy(g)
    ranked = sorted(range(len(sentences)), key=lambda i: scores.get(i,0.0), reverse=True)
    return [sentences[i] for i in ranked[:top_k]]

class Absformer:
    def __init__(self, compressor_name="t5-base", device=None):
        self.tok = AutoTokenizer.from_pretrained(compressor_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(compressor_name)

    def summarize(self, documents: List[str], max_len=160) -> str:
        # naive sentence split
        sents = []
        for d in documents:
            sents.extend([s.strip() for s in d.split(". ") if s.strip()])
        picked = textrank_select(sents, top_k=min(15, max(5, len(sents)//10)))
        prompt = "summarize: " + " ".join(picked)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        out = self.model.generate(**enc, max_length=max_len)
        return self.tok.decode(out[0], skip_special_tokens=True)
