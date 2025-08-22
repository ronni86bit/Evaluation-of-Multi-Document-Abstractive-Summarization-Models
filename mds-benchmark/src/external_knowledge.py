
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

def top_keywords(texts: List[str], k=15):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vect.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    idx = scores.argsort()[::-1][:k]
    terms = np.array(vect.get_feature_names_out())[idx]
    return terms.tolist()

class KeywordGuidedSummarizer:
    def __init__(self, base_model="t5-base"):
        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    def summarize(self, documents: List[str], max_len=160):
        kws = top_keywords(documents, k=20)
        prefix = "summarize with keywords: " + ", ".join(kws) + " || "
        text = " ".join(documents)
        enc = self.tok(prefix + text, return_tensors="pt", truncation=True, max_length=1024)
        out = self.model.generate(**enc, max_length=max_len)
        return self.tok.decode(out[0], skip_special_tokens=True)
