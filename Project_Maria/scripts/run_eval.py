"""
run_eval.py — Maria Research Evaluation Harness
================================================
Runs a fixed suite of prompts against Maria (via Ollama), scores the results,
and saves a timestamped JSON report to /eval_runs/.

Usage:
    python run_eval.py                          # baseline (no ablation flag)
    python run_eval.py --ablation base
    python run_eval.py --ablation sft
    python run_eval.py --ablation sft+dpo
    python run_eval.py --ablation sft+dpo+rag
    python run_eval.py --ablation sft+dpo+rag+critique

The "ablation" flag is just a label — it tags the result file so you can
compare runs side-by-side later. It does NOT automatically apply training;
you do that yourself (train SFT, DPO, etc.) and then run this script to
measure the improvement.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime

# ── Ollama import ──────────────────────────────────────────────────────────────
try:
    import ollama
except ImportError:
    print("❌ ollama not installed. Run: pip install ollama")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE   = os.path.join(BASE_DIR, "eval_suite.jsonl")
RUNS_DIR    = os.path.join(BASE_DIR, "eval_runs")
CONFIG_DIR  = os.path.join(BASE_DIR, "configs")

os.makedirs(RUNS_DIR, exist_ok=True)

# ── Model selection (same logic as Maria_App.py) ───────────────────────────────
def select_model() -> str:
    """Pick the best available Ollama model, preferring qwen2.5-14b."""
    try:
        models = ollama.list()
        names = [m.get("name", m.get("model", "")) for m in (models.get("models") or [])]
        priority = ["qwen2.5:14b", "qwen2.5:7b", "llama3.2:3b", "llama3:8b"]
        for p in priority:
            if any(p in n for n in names):
                return p
        return names[0] if names else "qwen2.5:14b"
    except Exception:
        return "qwen2.5:14b"

MODEL = select_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def check_response(response_text: str, item: dict) -> dict:
    """
    Score a single model response against the expected fields in the eval item.

    Returns a dict with:
        passed      — True/False overall
        keyword_hit — True/False (any expected keyword found)
        contains_hit — True/False (expected_contains substring found)
        hallucinated — True/False (only relevant for hallucination_check category)
        score       — float 0.0 – 1.0
        notes       — human-readable explanation
    """
    r = response_text.lower()
    notes = []

    # 1. Keyword check — at least ONE expected keyword must appear
    keywords = item.get("expected_keywords", [])
    keyword_hit = any(kw.lower() in r for kw in keywords)

    # 2. Contains check — the expected_contains substring must be present
    expected_contains = item.get("expected_contains", "")
    contains_hit = expected_contains.lower() in r if expected_contains else True

    # 3. Hallucination check (trick questions)
    is_trick = item.get("is_trick", False)
    hallucinated = False
    if is_trick:
        # For trick questions, the model SHOULD say it doesn't know.
        # If keyword_hit is True, it said the right uncertain thing → not hallucinated.
        hallucinated = not keyword_hit
        if hallucinated:
            notes.append("⚠️  HALLUCINATION: Model gave a confident answer to a trick question.")
        else:
            notes.append("✅  Correctly declined to fabricate an answer.")

    # 4. Overall pass/fail
    if is_trick:
        passed = not hallucinated
    else:
        passed = keyword_hit and contains_hit

    # 5. Score
    if is_trick:
        score = 1.0 if passed else 0.0
    else:
        score = (0.5 if keyword_hit else 0.0) + (0.5 if contains_hit else 0.0)

    if not keyword_hit:
        notes.append(f"Missing keywords: {keywords}")
    if not contains_hit:
        notes.append(f"Missing substring: '{expected_contains}'")

    return {
        "passed": passed,
        "keyword_hit": keyword_hit,
        "contains_hit": contains_hit,
        "hallucinated": hallucinated,
        "score": score,
        "notes": "; ".join(notes) if notes else "OK",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ABLATION CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_SYSTEM_PROMPTS = {
    "base": (
        "You are Maria, a helpful AI assistant."
    ),
    "sft": (
        "You are Maria, a helpful AI assistant trained with supervised fine-tuning "
        "on curated Filipino and general-knowledge conversations. "
        "Be precise, warm, and thorough."
    ),
    "sft+dpo": (
        "You are Maria, a helpful AI assistant trained with SFT and refined via DPO "
        "(Direct Preference Optimization). Prefer responses that are accurate, concise, "
        "and honest about uncertainty."
    ),
    "sft+dpo+rag": (
        "You are Maria, a helpful AI assistant using SFT, DPO, and Retrieval-Augmented "
        "Generation. When answering factual questions, cite your source. "
        "If you cannot find a source, say so."
    ),
    "sft+dpo+rag+critique": (
        "You are Maria, a helpful AI assistant using SFT, DPO, RAG, and self-critique. "
        "After giving your answer, briefly review it: check for errors, hallucinations, "
        "or missing citations, and correct them inline."
    ),
}


def get_system_prompt(ablation: str) -> str:
    ablation_key = ablation.lower().strip()
    return ABLATION_SYSTEM_PROMPTS.get(ablation_key, ABLATION_SYSTEM_PROMPTS["base"])


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET HASH (for B — Data Versioning)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dataset_hash(path: str) -> str:
    """MD5 hash of the eval file — changes whenever you edit eval_suite.jsonl."""
    if not os.path.exists(path):
        return "file_not_found"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def load_data_split_hashes() -> dict:
    """
    Load train/valid/test.jsonl and return their hashes for reproducibility.
    If a file doesn't exist yet, returns 'missing'.
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(BASE_DIR, f"{split}.jsonl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                splits[split] = hashlib.md5(f.read()).hexdigest()[:12]
        else:
            splits[split] = "missing"
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOGGER (for C — Experiment Config)
# ═══════════════════════════════════════════════════════════════════════════════

def load_active_config(ablation: str) -> dict:
    """
    Load the YAML config that corresponds to the ablation stage.
    Falls back gracefully if PyYAML is not installed or file is missing.
    """
    config_map = {
        "base":                  None,
        "sft":                   "sft_qwen14b_q4.yaml",
        "sft+dpo":               "dpo_qwen14b_q4.yaml",
        "sft+dpo+rag":           "dpo_qwen14b_q4.yaml",
        "sft+dpo+rag+critique":  "dpo_qwen14b_q4.yaml",
    }
    config_file = config_map.get(ablation.lower())
    if not config_file:
        return {"note": "No config file for base ablation."}

    config_path = os.path.join(CONFIG_DIR, config_file)
    if not os.path.exists(config_path):
        return {"note": f"Config file not found: {config_file}"}

    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # yaml not available — just record the filename
        return {"config_file": config_file, "note": "Install PyYAML to parse config details."}
    except Exception as e:
        return {"config_file": config_file, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EVAL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(ablation: str = "base", verbose: bool = True) -> dict:
    """
    Run the full evaluation suite and return the results dict.

    ablation: one of base | sft | sft+dpo | sft+dpo+rag | sft+dpo+rag+critique
    """

    # ── Load eval suite ────────────────────────────────────────────────────────
    if not os.path.exists(EVAL_FILE):
        print(f"❌ eval_suite.jsonl not found at {EVAL_FILE}")
        sys.exit(1)

    items = []
    with open(EVAL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    eval_hash    = compute_dataset_hash(EVAL_FILE)
    data_hashes  = load_data_split_hashes()
    active_config = load_active_config(ablation)
    system_prompt = get_system_prompt(ablation)

    print(f"\n{'='*60}")
    print(f"  Maria Evaluation Harness")
    print(f"{'='*60}")
    print(f"  Model      : {MODEL}")
    print(f"  Ablation   : {ablation}")
    print(f"  Eval hash  : {eval_hash}")
    print(f"  Train hash : {data_hashes.get('train', 'missing')}")
    print(f"  Valid hash : {data_hashes.get('valid', 'missing')}")
    print(f"  Test  hash : {data_hashes.get('test',  'missing')}")
    print(f"  Items      : {len(items)}")
    print(f"{'='*60}\n")

    # ── Run each item ──────────────────────────────────────────────────────────
    results     = []
    categories  = {}   # category → list of scores
    hall_total  = 0
    hall_fail   = 0

    for i, item in enumerate(items):
        prompt   = item["prompt"]
        category = item.get("category", "general")
        item_id  = item.get("id", f"item_{i:03d}")

        if verbose:
            print(f"[{i+1:02d}/{len(items)}] {item_id}  ({category})")
            print(f"  Q: {prompt[:80]}{'...' if len(prompt)>80 else ''}")

        # Call Ollama
        t0 = time.time()
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.1}   # low temp for reproducibility
            )
            response_text = response["message"]["content"]
            latency_ms    = int((time.time() - t0) * 1000)
            error         = None
        except Exception as e:
            response_text = ""
            latency_ms    = int((time.time() - t0) * 1000)
            error         = str(e)
            print(f"  ❌ Ollama error: {e}")

        # Score
        score_info = check_response(response_text, item)

        # Track hallucination stats
        if category == "hallucination_check":
            hall_total += 1
            if score_info["hallucinated"]:
                hall_fail += 1

        # Track per-category
        categories.setdefault(category, []).append(score_info["score"])

        # Print result
        status = "✅ PASS" if score_info["passed"] else "❌ FAIL"
        if verbose:
            print(f"  A: {response_text[:120].strip().replace(chr(10), ' ')}{'...' if len(response_text)>120 else ''}")
            print(f"  {status}  score={score_info['score']:.1f}  latency={latency_ms}ms")
            if score_info["notes"] != "OK":
                print(f"  ℹ️  {score_info['notes']}")
            print()

        results.append({
            "id":            item_id,
            "category":      category,
            "prompt":        prompt,
            "response":      response_text,
            "latency_ms":    latency_ms,
            "error":         error,
            **score_info,
        })

    # ── Aggregate scores ───────────────────────────────────────────────────────
    all_scores      = [r["score"] for r in results]
    overall_score   = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    pass_rate       = round(sum(1 for r in results if r["passed"]) / len(results), 4)

    per_category = {
        cat: {
            "count":    len(scores),
            "avg":      round(sum(scores) / len(scores), 4),
            "pass_rate": round(sum(1 for s in scores if s >= 1.0) / len(scores), 4),
        }
        for cat, scores in categories.items()
    }

    hall_accuracy = round(1 - (hall_fail / hall_total), 4) if hall_total else None

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — ablation: {ablation}")
    print(f"{'='*60}")
    print(f"  Overall score  : {overall_score:.4f}  ({overall_score*100:.1f}%)")
    print(f"  Pass rate      : {pass_rate:.4f}  ({pass_rate*100:.1f}%)")
    print(f"\n  Per-category scores:")
    for cat, info in per_category.items():
        bar = "█" * int(info["avg"] * 20)
        print(f"    {cat:<22} avg={info['avg']:.3f}  pass={info['pass_rate']:.3f}  [{bar:<20}]")
    if hall_accuracy is not None:
        print(f"\n  Hallucination accuracy : {hall_accuracy:.4f}  ({hall_accuracy*100:.1f}% correct refusals)")
        print(f"  Hallucination failures : {hall_fail}/{hall_total}")
    print(f"{'='*60}\n")

    # ── Build full report ──────────────────────────────────────────────────────
    report = {
        "run_timestamp":       datetime.now().isoformat(),
        "model":               MODEL,
        "ablation":            ablation,
        "system_prompt_used":  system_prompt,
        "eval_suite_hash":     eval_hash,
        "data_split_hashes":   data_hashes,
        "active_config":       active_config,
        "summary": {
            "total_items":            len(results),
            "overall_score":          overall_score,
            "pass_rate":              pass_rate,
            "hallucination_accuracy": hall_accuracy,
            "hallucination_failures": hall_fail,
            "hallucination_total":    hall_total,
        },
        "per_category": per_category,
        "results":      results,
    }

    # ── Save to /eval_runs/YYYYMMDD_HHMMSS_<ablation>.json ────────────────────
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_abl  = ablation.replace("+", "_")
    out_file  = os.path.join(RUNS_DIR, f"{stamp}_{safe_abl}.json")

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Results saved → {out_file}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARE RUNS (bonus utility)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_runs():
    """
    Print a quick leaderboard of all saved eval runs, sorted by overall score.
    Run with: python run_eval.py --compare
    """
    if not os.path.exists(RUNS_DIR):
        print("No eval_runs directory found yet.")
        return

    runs = []
    for fname in sorted(os.listdir(RUNS_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(RUNS_DIR, fname)) as f:
                    data = json.load(f)
                runs.append({
                    "file":      fname,
                    "ablation":  data.get("ablation", "?"),
                    "model":     data.get("model", "?"),
                    "score":     data["summary"]["overall_score"],
                    "pass_rate": data["summary"]["pass_rate"],
                    "hall_acc":  data["summary"].get("hallucination_accuracy"),
                    "timestamp": data.get("run_timestamp", "?"),
                })
            except Exception:
                pass

    runs.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  Maria Eval Run Leaderboard  ({len(runs)} runs)")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Ablation':<28} {'Score':>6} {'Pass%':>6} {'HallAcc':>8}  Timestamp")
    print(f"  {'-'*75}")
    for rank, r in enumerate(runs, 1):
        hall = f"{r['hall_acc']*100:.1f}%" if r['hall_acc'] is not None else "  N/A"
        print(f"  {rank:<3} {r['ablation']:<28} {r['score']*100:>5.1f}% {r['pass_rate']*100:>5.1f}%  {hall:>7}  {r['timestamp'][:19]}")
    print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maria Evaluation Harness")
    parser.add_argument(
        "--ablation",
        default="base",
        choices=["base", "sft", "sft+dpo", "sft+dpo+rag", "sft+dpo+rag+critique"],
        help="Which training stage to evaluate (default: base)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show leaderboard of all saved runs and exit"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not individual item results"
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs()
    else:
        run_eval(ablation=args.ablation, verbose=not args.quiet)