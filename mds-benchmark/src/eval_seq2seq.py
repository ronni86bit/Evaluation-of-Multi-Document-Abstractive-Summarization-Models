
import os, json, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from .data import load_cnn_dm, preprocess_for_tokenizer
from .utils import compute_rouge

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path)

    ds = load_cnn_dm(args.split)

    inputs = ds["document"]
    refs = ds["summary"]

    preds = []
    for i in range(len(inputs)):
        inp = inputs[i]
        enc = tok(inp, truncation=True, max_length=cfg.get("max_source_length", 1024), return_tensors="pt")
        out = model.generate(**enc, max_length=cfg.get("generation_max_length", 142), min_length=cfg.get("generation_min_length", 56))
        txt = tok.decode(out[0], skip_special_tokens=True)
        preds.append(txt)

    scores = compute_rouge(preds, refs)
    print("ROUGE:", scores)
    os.makedirs("runs", exist_ok=True)
    df = pd.DataFrame({"pred": preds, "ref": refs})
    out_dir = os.path.join("runs", os.path.basename(args.checkpoint_path))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    with open(os.path.join(out_dir, "rouge.json"), "w") as f:
        json.dump(scores, f, indent=2)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
