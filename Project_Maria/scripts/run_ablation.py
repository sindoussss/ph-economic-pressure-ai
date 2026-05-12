"""
run_ablation.py — Maria Ablation Study
=======================================
Runs the full evaluation suite across ALL training stages back-to-back
and prints a comparison table, so you can see exactly how much each
improvement (SFT, DPO, RAG, critique) contributes.

This is what makes the research "ablation-complete" — you're not just
saying "DPO helped", you're showing the numbers that prove it.

Usage:
    python run_ablation.py              # Run all 5 stages
    python run_ablation.py --stages base sft sft+dpo   # Run specific stages
    python run_ablation.py --compare-only               # Just show saved results

What each stage means:
    base                — Raw Ollama model, no training, no special prompt
    sft                 — After Supervised Fine-Tuning on Maria's conversation data
    sft+dpo             — After DPO (preference alignment on top of SFT)
    sft+dpo+rag         — DPO model + Retrieval-Augmented Generation (Wikipedia, docs)
    sft+dpo+rag+critique— Everything + self-critique (model checks its own answer)

Note on what "stage" means here:
    The --ablation flag changes the *system prompt* sent to the model.
    This simulates the behaviour you'd expect from each training stage,
    letting you run the eval even before training is complete.
    For a real ablation study in your thesis, swap in the actual trained
    model checkpoint for each stage.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Import run_eval from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import run_eval, compare_runs, RUNS_DIR

ALL_STAGES = [
    "base",
    "sft",
    "sft+dpo",
    "sft+dpo+rag",
    "sft+dpo+rag+critique",
]

STAGE_DESCRIPTIONS = {
    "base":                  "Raw base model — no fine-tuning, minimal prompt",
    "sft":                   "+ Supervised Fine-Tuning on Maria's conversations",
    "sft+dpo":               "+ DPO preference alignment (chosen > rejected)",
    "sft+dpo+rag":           "+ RAG (Wikipedia + document retrieval)",
    "sft+dpo+rag+critique":  "+ Self-critique (model reviews its own answer)",
}


def run_ablation_study(stages: list, verbose: bool = False):
    """Run eval for each stage and collect results."""
    print(f"\n{'='*65}")
    print(f"  Maria Ablation Study  —  {len(stages)} stages")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    for i, s in enumerate(stages, 1):
        print(f"  {i}. {s:<30} {STAGE_DESCRIPTIONS.get(s, '')}")
    print(f"{'='*65}\n")

    all_results = {}

    for stage in stages:
        print(f"\n{'─'*65}")
        print(f"  Running stage: {stage}")
        print(f"{'─'*65}")
        try:
            report = run_eval(ablation=stage, verbose=verbose)
            all_results[stage] = report
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted at stage '{stage}'. Saving partial results...")
            break
        except Exception as e:
            print(f"❌ Error during stage '{stage}': {e}")
            all_results[stage] = {"error": str(e)}

    # ── Print comparison table ─────────────────────────────────────────────────
    print_comparison_table(all_results, stages)

    # ── Save ablation summary ──────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RUNS_DIR, f"{stamp}_ablation_summary.json")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "stages_run":    stages,
        "comparison": {
            stage: {
                "overall_score":          r.get("summary", {}).get("overall_score"),
                "pass_rate":              r.get("summary", {}).get("pass_rate"),
                "hallucination_accuracy": r.get("summary", {}).get("hallucination_accuracy"),
                "per_category":           r.get("per_category", {}),
                "error":                  r.get("error"),
            }
            for stage, r in all_results.items()
        }
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Ablation summary saved → {summary_path}")
    return summary


def print_comparison_table(all_results: dict, stages: list):
    """Pretty-print the side-by-side comparison."""

    # Collect all categories across all runs
    all_cats = set()
    for r in all_results.values():
        all_cats.update(r.get("per_category", {}).keys())
    all_cats = sorted(all_cats)

    print(f"\n{'='*75}")
    print(f"  ABLATION COMPARISON TABLE")
    print(f"{'='*75}")

    # Header
    col_w = 12
    header = f"  {'Category':<22}"
    for stage in stages:
        short = stage.replace("sft+dpo+rag+critique", "+critique")[:col_w]
        header += f"  {short:>{col_w}}"
    print(header)
    print(f"  {'─'*22}" + f"  {'─'*col_w}" * len(stages))

    # Overall row
    row = f"  {'OVERALL SCORE':<22}"
    prev_score = None
    for stage in stages:
        r = all_results.get(stage, {})
        score = r.get("summary", {}).get("overall_score")
        if score is not None:
            delta = f" ({score - prev_score:+.3f})" if prev_score is not None else ""
            cell = f"{score:.3f}{delta}"
            prev_score = score
        else:
            cell = "error"
        row += f"  {cell:>{col_w}}"
    print(row)

    # Pass rate row
    row = f"  {'PASS RATE':<22}"
    for stage in stages:
        r = all_results.get(stage, {})
        val = r.get("summary", {}).get("pass_rate")
        cell = f"{val:.3f}" if val is not None else "error"
        row += f"  {cell:>{col_w}}"
    print(row)

    # Hallucination row
    row = f"  {'HALLUCINATION ACC':<22}"
    for stage in stages:
        r = all_results.get(stage, {})
        val = r.get("summary", {}).get("hallucination_accuracy")
        cell = f"{val:.3f}" if val is not None else "N/A"
        row += f"  {cell:>{col_w}}"
    print(row)

    print(f"  {'─'*22}" + f"  {'─'*col_w}" * len(stages))

    # Per-category rows
    for cat in all_cats:
        row = f"  {cat:<22}"
        for stage in stages:
            r = all_results.get(stage, {})
            cat_data = r.get("per_category", {}).get(cat, {})
            val = cat_data.get("avg")
            cell = f"{val:.3f}" if val is not None else "—"
            row += f"  {cell:>{col_w}}"
        print(row)

    print(f"{'='*75}")

    # ── Interpret deltas ───────────────────────────────────────────────────────
    print(f"\n  📊 What the numbers mean:")
    scores = {}
    for stage in stages:
        r = all_results.get(stage, {})
        s = r.get("summary", {}).get("overall_score")
        if s is not None:
            scores[stage] = s

    for i in range(1, len(stages)):
        prev_stage = stages[i-1]
        curr_stage = stages[i]
        if prev_stage in scores and curr_stage in scores:
            delta = scores[curr_stage] - scores[prev_stage]
            direction = "improved" if delta > 0 else "degraded"
            print(f"    {prev_stage} → {curr_stage}: {direction} by {abs(delta)*100:.1f} percentage points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maria Ablation Study Runner")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="Which stages to run (default: all 5)"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Show leaderboard from saved runs without running new evals"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each individual question result (default: summary only)"
    )
    args = parser.parse_args()

    if args.compare_only:
        compare_runs()
    else:
        run_ablation_study(stages=args.stages, verbose=args.verbose)