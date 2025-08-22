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

