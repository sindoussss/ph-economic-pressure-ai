

import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "eval_runs")

# How much of a drop counts as a real regression (not just noise)?
# 0.02 = 2 percentage points. Below this is ignored as noise.
REGRESSION_THRESHOLD = 0.02


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_all_runs() -> list:
    """Return all non-ablation-summary run files, sorted oldest → newest."""
    if not os.path.exists(RUNS_DIR):
        return []
    files = []
    for fname in sorted(os.listdir(RUNS_DIR)):
        if fname.endswith(".json") and "ablation_summary" not in fname:
            full = os.path.join(RUNS_DIR, fname)
            files.append((fname, full))
    return files


def summarise_run(run: dict) -> dict:
    """Pull the key numbers out of a run JSON."""
    s = run.get("summary", {})
    return {
        "ablation":       run.get("ablation", "?"),
        "timestamp":      run.get("run_timestamp", "?")[:19],
        "overall":        s.get("overall_score"),
        "pass_rate":      s.get("pass_rate"),
        "hall_accuracy":  s.get("hallucination_accuracy"),
        "per_category":   run.get("per_category", {}),
        "results":        run.get("results", []),
    }


def compare_runs(old: dict, new: dict) -> dict:
    """
    Compare two summarised runs.
    Returns a dict describing what improved, regressed, or stayed the same.
    """
    regressions  = []
    improvements = []
    unchanged    = []

    # Overall score
    if old["overall"] is not None and new["overall"] is not None:
        delta = new["overall"] - old["overall"]
        entry = {"metric": "overall_score", "old": old["overall"], "new": new["overall"], "delta": round(delta, 4)}
        if delta <= -REGRESSION_THRESHOLD:
            regressions.append(entry)
        elif delta >= REGRESSION_THRESHOLD:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Hallucination accuracy
    if old["hall_accuracy"] is not None and new["hall_accuracy"] is not None:
        delta = new["hall_accuracy"] - old["hall_accuracy"]
        entry = {"metric": "hallucination_accuracy", "old": old["hall_accuracy"], "new": new["hall_accuracy"], "delta": round(delta, 4)}
        if delta <= -REGRESSION_THRESHOLD:
            regressions.append(entry)
        elif delta >= REGRESSION_THRESHOLD:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Per-category
    all_cats = set(old["per_category"].keys()) | set(new["per_category"].keys())
    for cat in sorted(all_cats):
        old_avg = old["per_category"].get(cat, {}).get("avg")
        new_avg = new["per_category"].get(cat, {}).get("avg")
        if old_avg is None or new_avg is None:
            continue
        delta = new_avg - old_avg
        entry = {"metric": f"category:{cat}", "old": old_avg, "new": new_avg, "delta": round(delta, 4)}
        if delta <= -REGRESSION_THRESHOLD:
            regressions.append(entry)
        elif delta >= REGRESSION_THRESHOLD:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Find questions that flipped pass→fail (the most actionable info)
    old_results  = {r["id"]: r for r in old.get("results", [])}
    new_results  = {r["id"]: r for r in new.get("results", [])}
    flipped_fail = []   # was passing, now failing
    flipped_pass = []   # was failing, now passing

    for qid, new_r in new_results.items():
        old_r = old_results.get(qid)
        if old_r is None:
            continue
        if old_r["passed"] and not new_r["passed"]:
            flipped_fail.append({
                "id":       qid,
                "category": new_r.get("category", "?"),
                "prompt":   new_r.get("prompt", "")[:80],
            })
        elif not old_r["passed"] and new_r["passed"]:
            flipped_pass.append({
                "id":       qid,
                "category": new_r.get("category", "?"),
                "prompt":   new_r.get("prompt", "")[:80],
            })

    return {
        "regressions":  regressions,
        "improvements": improvements,
        "unchanged":    unchanged,
        "flipped_fail": flipped_fail,
        "flipped_pass": flipped_pass,
    }


def print_report(old_meta: dict, new_meta: dict, comparison: dict):
    regressions  = comparison["regressions"]
    improvements = comparison["improvements"]
    flipped_fail = comparison["flipped_fail"]
    flipped_pass = comparison["flipped_pass"]

    print(f"\n{'='*62}")
    print(f"  Maria Regression Check")
    print(f"{'='*62}")
    print(f"  OLD : {old_meta['ablation']:<25} {old_meta['timestamp']}")
    print(f"  NEW : {new_meta['ablation']:<25} {new_meta['timestamp']}")
    print(f"{'='*62}")

    # ── Overall verdict ────────────────────────────────────────────────────────
    old_score = old_meta["overall"]
    new_score = new_meta["overall"]
    if old_score is not None and new_score is not None:
        delta = new_score - old_score
        if regressions:
            verdict = "⚠️  REGRESSION DETECTED"
        elif improvements:
            verdict = "✅  IMPROVEMENT"
        else:
            verdict = "➡️  NO SIGNIFICANT CHANGE"
        print(f"\n  {verdict}")
        arrow = "▲" if delta >= 0 else "▼"
        print(f"  Overall: {old_score:.4f} → {new_score:.4f}  ({arrow} {abs(delta)*100:.1f} pp)")

    # ── Regressions ────────────────────────────────────────────────────────────
    if regressions:
        print(f"\n  ❌ Regressions (dropped > {REGRESSION_THRESHOLD*100:.0f}pp):")
        for r in regressions:
            print(f"     {r['metric']:<30} {r['old']:.3f} → {r['new']:.3f}  (▼ {abs(r['delta'])*100:.1f}pp)")

    # ── Improvements ───────────────────────────────────────────────────────────
    if improvements:
        print(f"\n  ✅ Improvements (gained > {REGRESSION_THRESHOLD*100:.0f}pp):")
        for r in improvements:
            print(f"     {r['metric']:<30} {r['old']:.3f} → {r['new']:.3f}  (▲ {r['delta']*100:.1f}pp)")

    # ── Questions that newly broke ─────────────────────────────────────────────
    if flipped_fail:
        print(f"\n  ⚠️  Questions that BROKE (was passing, now failing):")
        for q in flipped_fail:
            print(f"     [{q['category']}] {q['id']}: {q['prompt']}...")

    # ── Questions that newly pass ──────────────────────────────────────────────
    if flipped_pass:
        print(f"\n  🎉 Questions now FIXED (was failing, now passing):")
        for q in flipped_pass:
            print(f"     [{q['category']}] {q['id']}: {q['prompt']}...")

    # ── Recommendation ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    if regressions or flipped_fail:
        print(f"  💡 Recommendation: Review the broken questions above.")
        print(f"     Consider rolling back or adjusting your training config.")
        print(f"     Key config file: configs/dpo_qwen14b_q4.yaml (lower beta)")
    elif improvements:
        print(f"  💡 Recommendation: Good progress! Run the full ablation study:")
        print(f"     python run_ablation.py")
    else:
        print(f"  💡 No meaningful change. Try more training steps or check")
        print(f"     that the correct model checkpoint is loaded in Ollama.")
    print(f"{'='*62}\n")


def save_regression_report(old_meta: dict, new_meta: dict, comparison: dict):
    """Append a regression check record to eval_runs/regression_log.json."""
    log_path = os.path.join(RUNS_DIR, "regression_log.json")
    history  = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append({
        "checked_at":   datetime.now().isoformat(),
        "old_run":      {"ablation": old_meta["ablation"], "timestamp": old_meta["timestamp"], "overall": old_meta["overall"]},
        "new_run":      {"ablation": new_meta["ablation"], "timestamp": new_meta["timestamp"], "overall": new_meta["overall"]},
        "regressions":  comparison["regressions"],
        "improvements": comparison["improvements"],
        "flipped_fail": comparison["flipped_fail"],
        "flipped_pass": comparison["flipped_pass"],
    })

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  📝 Logged to eval_runs/regression_log.json")


def main():
    parser = argparse.ArgumentParser(description="Maria Regression Detection")
    parser.add_argument("--a", default=None, help="Path to the OLD run JSON (default: second-latest)")
    parser.add_argument("--b", default=None, help="Path to the NEW run JSON (default: latest)")
    parser.add_argument(
        "--threshold", type=float, default=REGRESSION_THRESHOLD,
        help=f"Min drop to count as regression (default: {REGRESSION_THRESHOLD})"
    )
    args = parser.parse_args()

    global REGRESSION_THRESHOLD
    REGRESSION_THRESHOLD = args.threshold

    # ── Resolve which two runs to compare ─────────────────────────────────────
    if args.a and args.b:
        path_old, path_new = args.a, args.b
    else:
        all_runs = get_all_runs()
        if len(all_runs) < 2:
            print("❌ Need at least 2 eval runs to compare.")
            print("   Run: python run_eval.py --ablation base")
            print("   Then train something and run: python run_eval.py --ablation sft")
            sys.exit(1)
        _, path_old = all_runs[-2]
        _, path_new = all_runs[-1]

    print(f"\n  Loading runs...")
    print(f"  OLD: {os.path.basename(path_old)}")
    print(f"  NEW: {os.path.basename(path_new)}")

    old_run  = load_run(path_old)
    new_run  = load_run(path_new)
    old_meta = summarise_run(old_run)
    new_meta = summarise_run(new_run)

    comparison = compare_runs(old_meta, new_meta)
    print_report(old_meta, new_meta, comparison)
    save_regression_report(old_meta, new_meta, comparison)


if __name__ == "__main__":
    main()