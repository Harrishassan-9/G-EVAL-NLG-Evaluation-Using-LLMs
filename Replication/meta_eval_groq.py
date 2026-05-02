"""
Meta-evaluation script for G-EVAL results produced by groq_eval.py.

Computes summary-level Pearson, Spearman, and Kendall-Tau correlations
between G-EVAL scores and human judgments, following the methodology
of the original paper (Zhong et al., 2022 / G-EVAL paper Section 3).

Usage:
    python meta_eval_groq.py --input_fp results/groq_coh.json --dimension coherence
"""

import json
import argparse
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau


# ── Correlation helpers ───────────────────────────────────────────────────────

def accumulate_correlation(pred, human, acc):
    """Add one document's correlation into the accumulator dict."""
    assert len(pred) == len(human)
    if not acc:
        acc = {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    acc["pearson"]    += pearsonr(pred, human)[0]
    acc["spearman"]   += spearmanr(pred, human)[0]
    acc["kendalltau"] += kendalltau(pred, human)[0]
    return acc


def print_table(acc, n, dimension):
    """Pretty-print averaged correlations."""
    n = max(n, 1)
    table = PrettyTable(["Dimension", "Pearson", "Spearman", "Kendall-Tau"])
    table.add_row([
        dimension,
        round(acc["pearson"]    / n, 4),
        round(acc["spearman"]   / n, 4),
        round(acc["kendalltau"] / n, 4),
    ])
    print(table)
    return {
        "pearson":    round(acc["pearson"]    / n, 4),
        "spearman":   round(acc["spearman"]   / n, 4),
        "kendalltau": round(acc["kendalltau"] / n, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute correlation of G-EVAL scores with human judgments."
    )
    parser.add_argument("--input_fp",  type=str, required=True,
                        help="Path to results JSON produced by groq_eval.py.")
    parser.add_argument("--dimension", type=str, required=True,
                        choices=["coherence", "consistency", "fluency", "relevance"],
                        help="Which dimension to evaluate.")
    args = parser.parse_args()

    with open(args.input_fp, encoding="utf-8") as f:
        jobj = json.load(f)

    # ── Group by document ────────────────────────────────────────────────────
    pred_by_doc  = {}
    human_by_doc = {}

    for item in jobj:
        doc_id = item["doc_id"]
        if doc_id not in pred_by_doc:
            pred_by_doc[doc_id]  = []
            human_by_doc[doc_id] = []

        pred_by_doc[doc_id].append(item["g_eval_score"])
        human_by_doc[doc_id].append(item["scores"][args.dimension])

    total_docs     = len(pred_by_doc)
    total_instances = len(jobj)
    print(f"\nDimension  : {args.dimension}")
    print(f"Documents  : {total_docs}  |  Instances : {total_instances}")

    # ── Per-document correlation → average ───────────────────────────────────
    acc   = {}
    valid = 0
    skipped = 0

    for doc_id in pred_by_doc:
        p = pred_by_doc[doc_id]
        h = human_by_doc[doc_id]

        # Skip degenerate cases (no variance → undefined correlation)
        if len(set(h)) <= 1 or len(set(p)) <= 1:
            skipped += 1
            continue

        acc   = accumulate_correlation(p, h, acc)
        valid += 1

    print(f"Valid docs : {valid}  |  Skipped (no variance): {skipped}\n")

    if valid == 0:
        print("No valid documents – cannot compute correlations.")
    else:
        print_table(acc, valid, args.dimension)
