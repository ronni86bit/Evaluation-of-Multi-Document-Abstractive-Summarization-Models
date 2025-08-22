
"""
Deep Communicating Agents (DCA) inspired baseline:
- Split into K agents, each creates a partial summary.
- Agents "communicate" by sharing their partial and refining once.
- Coordinator fuses into final.
"""
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DCAgentSummarizer:
    def __init__(self, agent_model="t5-small", coordinator_model="t5-base"):
        self.atok = AutoTokenizer.from_pretrained(agent_model, use_fast=True)
        self.amodel = AutoModelForSeq2SeqLM.from_pretrained(agent_model)
        self.ctok = AutoTokenizer.from_pretrained(coordinator_model, use_fast=True)
        self.cmodel = AutoModelForSeq2SeqLM.from_pretrained(coordinator_model)

    def _gen(self, tok, model, text, max_len=96):
        enc = tok("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024)
        out = model.generate(**enc, max_length=max_len)
        return tok.decode(out[0], skip_special_tokens=True)

    def summarize(self, documents: List[str], agents=3):
        text = " ".join(documents)
        # split by length into K parts
        L = len(text)
        parts = [text[i*L//agents:(i+1)*L//agents] for i in range(agents)]
        partials = [self._gen(self.atok, self.amodel, p) for p in parts]
        # communicate: each agent refines with others' partials context
        context = " ".join(partials)
        refined = [self._gen(self.atok, self.amodel, p + " " + context) for p in parts]
        # coordinator fuses
        final = self._gen(self.ctok, self.cmodel, " ".join(refined), max_len=160)
        return final
