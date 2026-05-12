#!/usr/bin/env python3
"""
Maria SFT training script.

This version never trains directly on the raw maria_sft_dataset.jsonl file.
It first prepares a cleaned dataset, writes a report, then fine-tunes on the
prepared JSONL.
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATASET_PATH = os.path.join(ROOT_DIR, "maria_sft_dataset.jsonl")
CURATED_PATH = os.path.join(ROOT_DIR, "maria_training_data.json")
PREPARED_DIR = os.path.join(ROOT_DIR, "prepared")
PREPARED_DATASET_PATH = os.path.join(PREPARED_DIR, "maria_sft_prepared.jsonl")
PREPARED_REPORT_PATH = os.path.join(PREPARED_DIR, "maria_sft_prepared_report.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sft_checkpoints")
MAX_SEQ_LEN = 1024
LORA_R = 16
LORA_ALPHA = 16
LR = 2e-4
EPOCHS = 2
BATCH = 1
GRAD_ACCUM = 8

SYSTEM_PROMPT = (
    "Ikaw si Maria — isang BGC/Metro Manila-raised Filipina AI assistant. "
    "You think and speak in natural Taglish — English-dominant pero may Filipino soul. "
    "Matalino pero hindi hambog. Direct, chill, on point. "
    "Gamitin ang mga particles: kasi, eh, diba, naman, talaga, nga. "
    "Huwag mag-'Ako ay...' — 'Gets ko', 'Alam ko', 'Gagawin ko' lang. "
    "Seryoso pag kailangan, pero di stiff. Laging tumulong nang tapat."
)

sys.path.insert(0, ROOT_DIR)
from sft_dataset_prep import prepare_sft_dataset  # noqa: E402


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Maria SFT on a prepared dataset.")
    parser.add_argument("--dataset", default=RAW_DATASET_PATH, help="Raw SFT JSONL path")
    parser.add_argument("--curated", default=CURATED_PATH, help="Curated training JSON path")
    parser.add_argument("--prepared-dataset", default=PREPARED_DATASET_PATH,
                        help="Prepared SFT JSONL output path")
    parser.add_argument("--prepared-report", default=PREPARED_REPORT_PATH,
                        help="Prepared SFT report JSON output path")
    parser.add_argument("--min-quality", type=float, default=0.60,
                        help="Minimum quality score for keeping rows during preparation")
    return parser.parse_args()


def load_prepared_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def format_example(ex: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
        )
    }


def main() -> None:
    args = build_args()

    prepared = prepare_sft_dataset(
        raw_sft_path=os.path.abspath(args.dataset),
        curated_path=os.path.abspath(args.curated),
        output_path=os.path.abspath(args.prepared_dataset),
        report_path=os.path.abspath(args.prepared_report),
        min_quality=args.min_quality,
    )
    prep_stats = prepared.report["stats"]

    print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Raw dataset: {args.dataset}")
    print(f"Prepared dataset: {args.prepared_dataset}")
    print(f"Prepared report: {args.prepared_report}")
    print(f"Prepared rows: {prep_stats['kept_after_repeat']}")
    print(f"Prepared language counts: {prep_stats['language_counts']}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\nLoading tokenizer: {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model with CPU offload (8GB GPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw_data = load_prepared_rows(args.prepared_dataset)
    print(f"\nPrepared examples loaded: {len(raw_data)}")

    dataset = Dataset.from_list([
        format_example(ex) for ex in raw_data
        if ex.get("prompt") and ex.get("response")
    ])
    print(f"SFT dataset: {len(dataset)} examples ready for training")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("\nStarting SFT training...")
    trainer.train()

    print("\nMerging LoRA adapter into base model...")
    merged_path = os.path.join(OUTPUT_DIR, "merged_bf16")
    os.makedirs(merged_path, exist_ok=True)

    merged = model.merge_and_unload()
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"\nSFT complete. Merged model saved to: {merged_path}")
    print("\nNext step:")
    print("  python train_dpo.py")


if __name__ == "__main__":
    main()
