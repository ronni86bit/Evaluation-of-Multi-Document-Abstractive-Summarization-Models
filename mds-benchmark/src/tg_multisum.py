
"""
Graph-based MDS (TG-MultiSum-style) baseline:
- Build a TF-IDF cosine graph across sentences, aggregate by topics (spectral clustering).
- Generate per-topic mini-summaries with T5, then fuse.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def cluster_sentences(sentences: List[str], n_clusters: int = 3):
    tfidf = TfidfVectorizer().fit_transform(sentences)
    aff = (tfidf * tfidf.T).A
    np.fill_diagonal(aff, 1.0)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    labels = sc.fit_predict(aff)
    groups = [[] for _ in range(n_clusters)]
    for s, l in zip(sentences, labels):
        groups[l].append(s)
    return groups

class TGMDS:
    def __init__(self, generator="t5-base"):
        self.tok = AutoTokenizer.from_pretrained(generator, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator)

    def summarize(self, documents: List[str], topics=3, max_len=64) -> str:
        sents = []
        for d in documents:
            sents.extend([s.strip() for s in d.split(". ") if s.strip()])
        groups = cluster_sentences(sents, n_clusters=min(topics, max(1, len(sents)//10)))
        minis = []
        for g in groups:
            prompt = "summarize: " + " ".join(g)
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=768)
            out = self.model.generate(**enc, max_length=max_len)
            minis.append(self.tok.decode(out[0], skip_special_tokens=True))
        # fuse
        prompt = "summarize: " + " ".join(minis)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=768)
        out = self.model.generate(**enc, max_length=128)
        return self.tok.decode(out[0], skip_special_tokens=True)
