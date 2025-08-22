
import os, json, math
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers import set_seed
from .data import load_cnn_dm, preprocess_for_tokenizer
from .utils import compute_rouge

@dataclass
class Config:
    model_name_or_path: str
    output_dir: str
    max_source_length: int = 1024
    max_target_length: int = 128
    lr: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    predict_with_generate: bool = True
    generation_max_length: int = 142
    generation_min_length: int = 56
    model_type: str = "auto"   # "t5" to add prefix, else "auto"
    seed: int = 42

def load_config(path: str) -> Config:
    with open(path) as f:
        d = json.load(f)
    return Config(**d)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)

    train_ds = load_cnn_dm("train")
    val_ds = load_cnn_dm("validation")

    def _proc(batch):
        return preprocess_for_tokenizer(
            batch, tokenizer, cfg.max_source_length, cfg.max_target_length, cfg.model_type
        )

    train_ds = train_ds.map(_proc, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_proc, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        predict_with_generate=cfg.predict_with_generate,
        report_to=["none"]
    )

    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_rouge(preds, refs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)

if __name__ == "__main__":
    main()
