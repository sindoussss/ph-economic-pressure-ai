# Maria — Research Evaluation Infrastructure

## Folder Structure

Place all files in the same folder as `Maria_App.py`:

```
Your folder/
├── Maria_App.py
├── eval_suite.jsonl
├── run_eval.py
├── run_ablation.py
├── make_splits.py
├── check_regression.py        ← NEW
├── question_difficulty.py     ← NEW
├── plot_training_curve.py     ← NEW
├── configs/
│   ├── sft_qwen14b_q4.yaml
│   └── dpo_qwen14b_q4.yaml
└── eval_runs/                 ← auto-created on first run
```

---

## The Order to Follow (Every Time You Train)

```
Step 1 → make_splits.py            split your data into train/valid/test
Step 2 → train_sft.py              supervised fine-tuning
Step 3 → train_dpo.py              preference alignment
Step 4 → run_eval.py               score the model
Step 5 → check_regression.py       did anything break?
Step 6 → run_ablation.py           compare all stages (for thesis)
Step 7 → plot_training_curve.py    generate chart for thesis
Step 8 → question_difficulty.py    find Maria's weak spots
```

---

## Step 1 — Split Your Data

> Run this first, before any training. Re-run it whenever your dataset grows.

```bash
python make_splits.py
```

Creates three files:
- `train.jsonl` — 80% of data, what the model learns from
- `valid.jsonl` — 10% of data, used during training to catch overfitting
- `test.jsonl`  — 10% of data, never seen during training, used for final scoring

---

## Step 2 — Supervised Fine-Tuning (SFT)

> Teach Maria your persona, style, and domain knowledge.

```bash
python train_sft.py
```

Uses `train.jsonl` and `valid.jsonl`. Config is in `configs/sft_qwen14b_q4.yaml`.
Output saved to `./sft_checkpoints/merged_bf16`.

---

## Step 3 — Direct Preference Optimization (DPO)

> Refine Maria's responses using chosen vs. rejected preference pairs.

```bash
python train_dpo.py
```

Must run **after** SFT. Uses `maria_dpo_dataset.jsonl`. Config is in `configs/dpo_qwen14b_q4.yaml`.
Output saved to `./dpo_checkpoints/merged_bf16`.

---

## Step 4 — Score the Model

> Run after every training step to measure improvement.

```bash
# Before any training — get your baseline number:
python run_eval.py --ablation base

# After SFT:
python run_eval.py --ablation sft

# After DPO:
python run_eval.py --ablation sft+dpo

# After adding RAG:
python run_eval.py --ablation sft+dpo+rag

# After adding self-critique:
python run_eval.py --ablation sft+dpo+rag+critique

# See all past scores ranked:
python run_eval.py --compare
```

Results are automatically saved to `eval_runs/` with a timestamp.

---

## Step 5 — Check for Regressions

> Run immediately after every eval. Warns you if anything got worse.

```bash
python check_regression.py
```

Automatically compares the two most recent runs. If a category dropped by more than
2 percentage points, it tells you exactly what broke and which specific questions
flipped from passing to failing.

To compare any two specific runs:
```bash
python check_regression.py --a eval_runs/20241201_base.json --b eval_runs/20241215_sft.json
```

---

## Step 6 — Full Ablation Study

> Runs all 5 stages and prints a side-by-side comparison table for your thesis.

```bash
python run_ablation.py
```

Example output:

```
Category          base    sft     sft+dpo   +rag    +critique
────────────────  ──────  ──────  ────────  ──────  ─────────
OVERALL SCORE     0.612   0.701   0.744     0.789   0.821
PASS RATE         0.565   0.652   0.696     0.739   0.783
HALLUCINATION     0.500   0.667   0.833     0.833   1.000
factual           0.700   0.800   0.850     0.900   0.900
math              0.600   0.650   0.700     0.700   0.750
```

---

## Step 7 — Plot the Training Curve

> Generate a chart showing how scores changed across every training run.

```bash
# Show on screen:
python plot_training_curve.py

# Save as PNG for your thesis:
python plot_training_curve.py --save

# Save as PDF (best for thesis documents):
python plot_training_curve.py --save --format pdf

# No matplotlib installed? Use text mode:
python plot_training_curve.py --text
```

Requires: `pip install matplotlib`

---

## Step 8 — Find Maria's Weak Spots

> Identifies which questions Maria consistently gets wrong across all runs.

```bash
# Full report:
python question_difficulty.py

# Only show hard questions:
python question_difficulty.py --show-hard

# Filter to one category:
python question_difficulty.py --category math

# Save to CSV for spreadsheet/thesis:
python question_difficulty.py --export
```

Classifies every question as:
- 🟢 **Easy** — passes 85%+ of runs → consider replacing with harder variants
- 🟡 **Medium** — passes 40–85% of runs → where training has the most impact
- 🔴 **Hard** — passes 40% or fewer runs → Maria's real weak spots, add training data here

---

## What Each File Does

| File | What it is | Do you run it? |
|------|-----------|----------------|
| `eval_suite.jsonl` | The test questions Maria is graded on | No — edit to add questions |
| `make_splits.py` | Splits data into train/valid/test | Yes — before any training |
| `run_eval.py` | Scores Maria and saves results | Yes — after each training step |
| `check_regression.py` | Warns if scores dropped | Yes — after every eval |
| `run_ablation.py` | Compares all stages in one table | Yes — for research/thesis |
| `plot_training_curve.py` | Generates a score-over-time chart | Yes — for thesis figures |
| `question_difficulty.py` | Finds which questions Maria struggles with | Yes — to guide training |
| `configs/sft_qwen14b_q4.yaml` | SFT hyperparameters | No — edit settings |
| `configs/dpo_qwen14b_q4.yaml` | DPO hyperparameters | No — edit settings |

---

## Adding Your Own Test Questions

Open `eval_suite.jsonl` and add a new line:

```json
{"id": "your_id_001", "category": "factual", "prompt": "Your question here?", "expected_keywords": ["answer", "keyword"], "expected_contains": "answer", "citation_required": false}
```

For trick questions (testing if Maria makes things up):

```json
{"id": "hall_003", "category": "hallucination_check", "prompt": "Tell me about the fictional president of Mars.", "expected_keywords": ["don't know", "cannot confirm", "no information"], "expected_contains": "don", "citation_required": false, "is_trick": true, "note": "This is not a real person — Maria should say she doesn't know."}
```

---

## Where Results Are Saved

Every `run_eval.py` run saves a file like:

```
eval_runs/20241215_143022_sft_dpo.json
```

That file contains the overall score, per-category scores, hallucination stats,
every question and answer, and the exact config that was active.
Nothing is ever overwritten.

`check_regression.py` also appends to:
```
eval_runs/regression_log.json
```

So you have a permanent record of every time something regressed.
