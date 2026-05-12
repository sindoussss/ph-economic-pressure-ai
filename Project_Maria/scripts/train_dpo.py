#!/usr/bin/env python3
"""
Maria DPO Training Script — TRL + PEFT (no Unsloth)
Model  : ./sft_checkpoints/merged_bf16
Dataset: C:\Users\user\PycharmProjects\PythonProject\Project_Maria\maria_dpo_dataset.jsonl
Generated: 2026-02-27 04:54

Requirements:
    pip install trl transformers datasets bitsandbytes peft accelerate
Usage: python train_dpo.py
"""
import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

BASE_MODEL   = './sft_checkpoints/merged_bf16'
DATASET_PATH = 'C:\\Users\\user\\PycharmProjects\\PythonProject\\Project_Maria\\maria_dpo_dataset.jsonl'
OUTPUT_DIR   = './dpo_checkpoints'
MAX_SEQ_LEN  = 1024
LORA_R       = 8
LORA_ALPHA   = 16
BETA         = 0.1
LR           = 2e-05
EPOCHS       = 3
BATCH        = 1
GRAD_ACCUM   = 16

print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model.config.use_cache = False

lora_cfg = LoraConfig(r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

raw_data = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try: raw_data.append(json.loads(line))
            except: pass

dataset = Dataset.from_list([
    {'prompt': ex['prompt'], 'chosen': ex['chosen'], 'rejected': ex['rejected']}
    for ex in raw_data if ex.get('chosen') and ex.get('rejected')
])
print(f'DPO Dataset: {len(dataset)} examples')

dpo_config = DPOConfig(
    beta=0.1, loss_type="sigmoid",
    max_length=1024, max_prompt_length=256,
    output_dir=OUTPUT_DIR, num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-05, lr_scheduler_type="cosine",
    warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
    optim="paged_adamw_8bit", bf16=True,
    logging_steps=10, save_steps=50, save_total_limit=3,
    report_to="none", precompute_ref_log_probs=True,
)
DPOTrainer(model=model, ref_model=None, args=dpo_config,
           train_dataset=dataset, tokenizer=tokenizer).train()

print('Merging LoRA...')
merged = model.merge_and_unload()
merged_path = OUTPUT_DIR + '/merged_bf16'
os.makedirs(merged_path, exist_ok=True)
merged.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f'DPO complete. Saved to: {merged_path}')
print('Next: convert to GGUF and load into Ollama')
