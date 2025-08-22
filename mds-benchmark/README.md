# Multi-Document Abstractive Summarization Benchmark (CNN/DailyMail)

This repo provides **runnable training & evaluation code** for a suite of abstractive summarization models, plus simple multi-document (MDS) pipelines.

## Models covered

1. PRIMERA (`allenai/PRIMERA`)
2. BART (`facebook/bart-large-cnn` or `facebook/bart-large` for fine-tuning)
3. PEGASUS (`google/pegasus-large`)
4. T5 (`t5-base`, `t5-large`)
5. Long models: LED (`allenai/led-base-16384`), BigBird-Pegasus (`google/bigbird-pegasus-large-arxiv`), LongT5 (`google/long-t5-tglobal-base`)
6. Absformer (unsupervised MDS baseline with TextRank + compressor)
7. Graph-based MDS (TG-MultiSum-style baseline)
8. Hierarchical Transformer (sentence→doc→collection chunking with fusion)
9. Deep Communicating Agents (multi-agent chunk summarization & fusion)
10. External Knowledge (topic/keyword-guided prompting to the generator)

> Note: CNN/DailyMail is single-document by default. The included dataloader can **synthetically form multi-document sets** by grouping multiple articles or by chunking long articles into pseudo-documents. See `--mds_group_size` and `--mds_from_chunks` flags.

## Quickstart

```bash
python -m pip install -r requirements.txt

# Hugging Face login (optional, for private models or pushing)
# huggingface-cli login

# Example: fine-tune BART on CNN/DailyMail
python src/train_seq2seq.py --config configs/bart_large.json

# Evaluate a saved checkpoint
python src/eval_seq2seq.py --checkpoint_path runs/bart-large/checkpoint-best --config configs/bart_large.json
```

### Multi-document flags (synthetic)

- `--mds_group_size N` : pack N *different* articles into one example (concatenated with special separators).
- `--mds_from_chunks`  : split a single long article into N chunks and treat as documents.

These produce inputs like: `[DOC1] ... [DOC2] ... [DOC3] ...`

## Results

Use `src/eval_seq2seq.py` to compute **ROUGE-1/2/L**. The script also saves a CSV and prints a comparison table. 

## Repo layout

```
src/
  data.py                # dataset loading & MDS grouping
  utils.py               # metrics & helpers
  train_seq2seq.py       # generic HF Seq2Seq training
  eval_seq2seq.py        # generation + ROUGE
  absformer.py           # unsupervised MDS baseline
  tg_multisum.py         # graph-based MDS baseline
  hierarchical.py        # hierarchical chunking + fusion
  dca.py                 # multi-agent communicating baseline
  external_knowledge.py  # keyword/topic guided summarization

configs/
  *.json                 # per-model hyperparameters

scripts/
  run_*.sh               # ready-to-run commands
```

## Repro tips

- Use GPUs (A100/V100); set `per_device_train_batch_size` according to memory.
- For long models (LED/BigBird/LongT5), **increase `max_source_length`** and **enable gradient checkpointing**.
- Mixed precision (`fp16`/`bf16`) and `gradient_accumulation_steps` help fit bigger batches.
- For PRIMERA, set `global_attention` on document boundaries (implemented in `data.py`).

---

**Academic integrity**: The advanced baselines here are faithful *simplifications* of the research ideas so you can run them end-to-end on CNN/DailyMail. For official SOTA numbers, refer to the original papers.
