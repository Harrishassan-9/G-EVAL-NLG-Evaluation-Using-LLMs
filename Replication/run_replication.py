"""
Orchestrator: runs G-EVAL on all four SummEval dimensions sequentially,
then prints a final results table and compares against the paper's values.

Usage:
    python run_replication.py

Scale-down replication (this config):
    50 instances per dimension × 4 dimensions = 200 API calls total.
    Dataset is 100 docs × 16 systems = 1600 rows; first 50 rows cover
    3 complete documents (48 rows, each doc has 16 systems) plus 2 rows
    of a 4th doc.  Summary-level correlations are computed over the 3
    complete docs — valid replication of the paper's methodology at
    reduced cost.

Estimated runtime (Groq free tier, llama-3.3-70b-versatile, sleep=15s):
    50 instances  → ~18 min/dim + 2 min cooldown →  ~1.2 hr total (MAX_INSTANCES=50)
    400 instances → ~105 min/dim + cooldowns      →  ~7 hrs total  (MAX_INSTANCES=400)
    1600 instances → ~7 hrs/dim + cooldowns       → ~28 hrs total  (MAX_INSTANCES=None)
    Run can be interrupted and resumed – each dimension checkpoints progress.

Paper targets (G-EVAL-4, Table 1 – Spearman):
    Coherence 0.582  |  Consistency 0.507  |  Fluency 0.506  |  Relevance 0.547
"""

import subprocess
import sys
import os
import json
import time
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau

# ── Configuration ─────────────────────────────────────────────────────────────

DIMENSIONS = [
    {
        "name":        "coherence",
        "prompt_fp":   "prompts/summeval/coh_detailed.txt",
        "save_fp":     "results/groq_coh.json",
        "score_range": [1, 2, 3, 4, 5],
    },
    {
        "name":        "consistency",
        "prompt_fp":   "prompts/summeval/con_detailed.txt",
        "save_fp":     "results/groq_con.json",
        "score_range": [1, 2, 3, 4, 5],
    },
    {
        "name":        "fluency",
        "prompt_fp":   "prompts/summeval/flu_detailed.txt",
        "save_fp":     "results/groq_flu.json",
        "score_range": [1, 2, 3],
    },
    {
        "name":        "relevance",
        "prompt_fp":   "prompts/summeval/rel_detailed.txt",
        "save_fp":     "results/groq_rel.json",
        "score_range": [1, 2, 3, 4, 5],
    },
]

# Paper's reported G-EVAL-4 Spearman correlations (Table 1)
PAPER_SPEARMAN = {
    "coherence":   0.582,
    "consistency": 0.507,
    "fluency":     0.506,
    "relevance":   0.547,
}

SLEEP              = 20   # seconds between API calls (~3 RPM, ~3600 TPM well inside 12K limit)
MAX_INSTANCES      = 50   # 50 per dimension × 4 dims = 200 calls total; set to None for full 1600
BETWEEN_DIM_SLEEP  = 120  # seconds to pause between dimensions to let the TPM window fully reset


# ── Correlation helpers (same logic as meta_eval_groq.py) ─────────────────────

def compute_correlations(filepath, dimension):
    """Return (pearson, spearman, kendalltau) for a completed results file."""
    with open(filepath, encoding="utf-8") as f:
        jobj = json.load(f)

    pred_by_doc  = {}
    human_by_doc = {}
    for item in jobj:
        did = item["doc_id"]
        pred_by_doc.setdefault(did,  []).append(item["g_eval_score"])
        human_by_doc.setdefault(did, []).append(item["scores"][dimension])

    p_sum = s_sum = k_sum = 0.0
    n = 0
    for did in pred_by_doc:
        p = pred_by_doc[did]
        h = human_by_doc[did]
        if len(set(p)) <= 1 or len(set(h)) <= 1:
            continue
        p_sum += pearsonr(p, h)[0]
        s_sum += spearmanr(p, h)[0]
        k_sum += kendalltau(p, h)[0]
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    return p_sum / n, s_sum / n, k_sum / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    final_results = {}

    for dim in DIMENSIONS:
        name      = dim["name"]
        save_fp   = dim["save_fp"]
        prompt_fp = dim["prompt_fp"]

        print(f"\n{'='*65}")
        print(f"  DIMENSION: {name.upper()}")
        print(f"{'='*65}")

        # ── Run evaluation ────────────────────────────────────────────────
        cmd = [
            sys.executable, "groq_eval.py",
            "--prompt_fp",   prompt_fp,
            "--save_fp",     save_fp,
            "--score_range", *[str(s) for s in dim["score_range"]],
            "--sleep",       str(SLEEP),
        ]
        if MAX_INSTANCES:
            cmd += ["--max_instances", str(MAX_INSTANCES)]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [ERROR] groq_eval.py exited with code {result.returncode}")
            continue

        # ── Compute & display correlations ────────────────────────────────
        if not os.path.exists(save_fp):
            print(f"  [WARN] {save_fp} not found – skipping correlation.")
            continue

        r, s, k = compute_correlations(save_fp, name)
        final_results[name] = {"pearson": r, "spearman": s, "kendalltau": k}

        print(f"\n  Correlations ({name}):")
        t = PrettyTable(["Dimension", "Pearson", "Spearman", "Kendall-Tau"])
        t.add_row([name, round(r, 4), round(s, 4), round(k, 4)])
        print(t)

        # Cool down between dimensions so the Groq TPM window fully resets
        if dim != DIMENSIONS[-1]:
            print(f"\n  [COOLDOWN] Waiting {BETWEEN_DIM_SLEEP}s before next dimension …")
            time.sleep(BETWEEN_DIM_SLEEP)

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  FINAL RESULTS vs. PAPER (G-EVAL-4, Table 1 – Spearman)")
    print(f"{'='*65}")

    summary = PrettyTable(["Dimension", "Ours (Spearman)", "Paper (Spearman)", "Diff"])
    avg_ours  = 0.0
    avg_paper = 0.0
    n_dim     = 0

    for dim in DIMENSIONS:
        name = dim["name"]
        if name not in final_results:
            summary.add_row([name, "N/A", PAPER_SPEARMAN[name], "N/A"])
            continue
        ours  = final_results[name]["spearman"]
        paper = PAPER_SPEARMAN[name]
        diff  = round(ours - paper, 4)
        summary.add_row([name, round(ours, 4), paper, diff])
        avg_ours  += ours
        avg_paper += paper
        n_dim += 1

    if n_dim:
        summary.add_row([
            "AVERAGE",
            round(avg_ours  / n_dim, 4),
            round(avg_paper / n_dim, 4),
            round((avg_ours - avg_paper) / n_dim, 4),
        ])

    print(summary)
    print("\nNote: differences expected – we use Llama-3.3-70B via Groq, not GPT-4.")


if __name__ == "__main__":
    main()
