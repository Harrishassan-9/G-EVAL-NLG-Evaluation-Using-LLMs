#!/usr/bin/env python3
"""
run_assignment3.py
==================
G-Eval Experimentation and Expansion (Assignment 3)
- Open-source evaluator: Mistral-7B-Instruct-v0.3 (4-bit, BitsAndBytes)
- Self-consistency scoring (proposed)
- Debiasing prompt variant
- Multilingual extension (French prompts on SummEval + MLSUM dataset)
- Additional dataset: REALSumm (LitePyramid recall)

Outputs are written to Assignment3/outputs/:
- summary_metrics.json   : aggregated correlations + Wilcoxon tests
- <dataset>_<dim>_<variant>.json : per-instance scores + per-doc correlations
- plot_*.png             : scatter + histogram visualizations
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ────────────────────────────── CLI ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="local", choices=["local"])
    p.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--max_docs", type=int, default=2)
    p.add_argument("--self_consistency_n", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--prompt_language", default="en", choices=["en", "fr"])
    p.add_argument("--use_debias_prompt", action="store_true")
    p.add_argument("--run_multilingual_demo", action="store_true",
                   help="Also evaluate SummEval with French prompts.")
    p.add_argument("--run_multilingual_dataset_demo", action="store_true",
                   help="Run a French-language dataset (MLSUM) demo.")
    p.add_argument("--multilingual_dataset_name", default="mlsum")
    p.add_argument("--multilingual_dataset_languages", default="fr")
    p.add_argument("--multilingual_dataset_max_samples", type=int, default=30)
    p.add_argument("--outputs_dir", default="Assignment3/outputs")
    p.add_argument("--prompts_dir", default="Assignment3/prompts")
    return p.parse_args()


# ────────────────────────────── Prompts ──────────────────────────────────────
SUMMEVAL_PROMPTS_EN = {
    "coherence": """You will be given one summary written for a news article. Your task is to rate the summary on one metric.

Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.

Evaluation Steps:
1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and check if it covers the main topic and key points and presents them in a clear, logical order.
3. Assign a score from 1 to 5.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Coherence:""",

    "consistency": """You will be given a news article. You will then be given one summary written for this article.

Evaluation Criteria:
Consistency (1-5) - the factual alignment between the summary and the source. A factually consistent summary contains only statements that are entailed by the source. Penalize summaries containing hallucinated facts.

Evaluation Steps:
1. Read the news article and identify the main facts.
2. Read the summary and check whether every claim is supported by the source.
3. Assign a score from 1 to 5.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Consistency:""",

    "fluency": """You will be given one summary written for a news article. Your task is to rate the summary on one metric.

Evaluation Criteria:
Fluency (1-3) - the quality of the summary in terms of grammar, spelling, punctuation, word choice and sentence structure.
- 1: Poor. Many errors, hard to read.
- 2: Fair. Some errors but main points understandable.
- 3: Good. Few or no errors.

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Fluency (1-3):""",

    "relevance": """You will be given one summary written for a news article. Your task is to rate the summary on one metric.

Evaluation Criteria:
Relevance (1-5) - selection of important content from the source. The summary should include only important information. Penalize redundancy and excess information.

Evaluation Steps:
1. Read the summary and the source carefully.
2. Identify the main points of the article.
3. Assess how well the summary covers the main points and how much irrelevant content it contains.
4. Assign a score from 1 to 5.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Relevance:""",
}

SUMMEVAL_PROMPTS_FR = {
    "coherence": """On vous fournira un résumé rédigé pour un article de presse. Évaluez ce résumé selon une seule métrique.

Critère d'évaluation :
Cohérence (1-5) - la qualité collective de toutes les phrases. Le résumé doit être bien structuré et bien organisé.

Étapes :
1. Lisez l'article et identifiez le sujet principal et les points clés.
2. Comparez le résumé à l'article et vérifiez la couverture et l'ordre logique.
3. Attribuez un score de 1 à 5.

Texte source :
{{Document}}

Résumé :
{{Summary}}

Formulaire d'évaluation (score UNIQUEMENT) :
- Cohérence :""",

    "consistency": """On vous fournira un article puis un résumé. Évaluez selon une seule métrique.

Critère :
Cohérence factuelle (1-5) - l'alignement factuel entre résumé et source. Pénalisez les hallucinations.

Étapes :
1. Identifiez les faits dans l'article.
2. Vérifiez chaque affirmation du résumé.
3. Attribuez un score de 1 à 5.

Texte source :
{{Document}}

Résumé :
{{Summary}}

Formulaire d'évaluation (score UNIQUEMENT) :
- Cohérence factuelle :""",

    "fluency": """Évaluez le résumé suivant selon une métrique.

Critère :
Fluidité (1-3) - grammaire, orthographe, ponctuation, choix de mots, structure.
- 1 : Médiocre. Beaucoup d'erreurs.
- 2 : Acceptable. Quelques erreurs mais compréhensible.
- 3 : Bon. Peu ou pas d'erreurs.

Résumé :
{{Summary}}

Formulaire d'évaluation (score UNIQUEMENT) :
- Fluidité (1-3) :""",

    "relevance": """Évaluez ce résumé selon une métrique.

Critère :
Pertinence (1-5) - sélection du contenu important de la source. Pénalisez les redondances.

Étapes :
1. Lisez le résumé et la source.
2. Identifiez les points principaux.
3. Évaluez la couverture et la quantité de contenu non pertinent.
4. Attribuez un score de 1 à 5.

Texte source :
{{Document}}

Résumé :
{{Summary}}

Formulaire d'évaluation (score UNIQUEMENT) :
- Pertinence :""",
}

REALSUMM_PROMPT = """You will be given one summary written for a news article. Rate the summary on its recall of salient content units, following the LitePyramid evaluation methodology.

Evaluation Criteria:
LitePyramid Recall (0-100) - the percentage of important semantic content units (SCUs) from the reference summary that are also present in the candidate summary.

Steps:
1. Read the source and reference, identifying key facts.
2. Check which facts are recalled by the candidate summary.
3. Output the recall percentage as an integer 0-100.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (score ONLY, integer 0-100):
- LitePyramid Recall:"""

DEBIAS_SYSTEM_PROMPT = """You are an impartial NLG quality evaluator. Follow these debiasing rules strictly:
1. Do NOT reward summaries simply for being longer or shorter — length is not a quality signal.
2. Do NOT prefer summaries that copy phrases verbatim from the source.
3. Do NOT be lenient: use the full range of the rating scale, including low scores when warranted.
4. Do NOT anchor on the first or last sentence; evaluate the whole summary uniformly.
5. Score ONLY based on the criterion described below. No explanation. Output the score only.
"""

SUMMEVAL_SCALES = {
    "coherence": (1, 5),
    "consistency": (1, 5),
    "fluency": (1, 3),
    "relevance": (1, 5),
}

# ────────────────────────────── Utilities ────────────────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_score(text: str, scale_min: float, scale_max: float) -> float | None:
    """Robust 3-stage score extraction."""
    if not text:
        return None
    # Stage 1: number after the last colon
    if ":" in text:
        tail = text.rsplit(":", 1)[1]
        m = re.search(r"-?\d+\.?\d*", tail)
        if m:
            try:
                v = float(m.group())
                if scale_min <= v <= scale_max:
                    return v
            except ValueError:
                pass
    # Stage 2: any in-range number
    for m in re.finditer(r"-?\d+\.?\d*", text):
        try:
            v = float(m.group())
            if scale_min <= v <= scale_max:
                return v
        except ValueError:
            continue
    # Stage 3: first numeric token
    m = re.search(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def truncate_doc(doc: str, max_chars: int = 8000) -> str:
    return doc if len(doc) <= max_chars else doc[:max_chars] + "..."


def build_prompt(template: str, document: str, summary: str,
                 use_debias: bool = False) -> str:
    body = template.replace("{{Document}}", truncate_doc(document)) \
                   .replace("{{Summary}}", summary or "")
    return (DEBIAS_SYSTEM_PROMPT + "\n" + body) if use_debias else body


# ────────────────────────────── Model Wrapper ────────────────────────────────
class MistralEvaluator:
    def __init__(self, model_name: str, max_input_tokens: int, max_new_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        print(f"[model] Loading {model_name} (4-bit NF4)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        print("[model] Loaded.")

    def generate(self, prompt: str, temperature: float = 0.0,
                 do_sample: bool = False, num_return_sequences: int = 1) -> list[str]:
        msg = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            num_return_sequences=num_return_sequences,
        )
        if do_sample and temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.95))
        else:
            gen_kwargs.update(dict(do_sample=False))

        with self.torch.no_grad():
            out = self.model.generate(**ids, **gen_kwargs)

        gen_only = out[:, ids["input_ids"].shape[1]:]
        return [self.tok.decode(seq, skip_special_tokens=True).strip() for seq in gen_only]


# ────────────────────────────── Datasets ─────────────────────────────────────
def load_summeval(max_docs: int):
    """Load SummEval — try local cache, fallback to HuggingFace."""
    try:
        local = Path("Replication/data/summeval.json")
        if local.exists():
            print("[data] Loading SummEval from local cache.")
            with local.open(encoding="utf-8") as f:
                rows = json.load(f)
        else:
            from datasets import load_dataset
            print("[data] Downloading SummEval from HuggingFace (mteb/summeval)...")
            ds = load_dataset("mteb/summeval", split="test")
            rows = []
            for ex in ds:
                rows.append({
                    "doc_id": ex.get("id", ex.get("doc_id", "")),
                    "document": ex["text"],
                    "summary": ex["machine_summaries"],
                    "human": {
                        "coherence": ex["coherence"],
                        "consistency": ex["consistency"],
                        "fluency": ex["fluency"],
                        "relevance": ex["relevance"],
                    },
                })
    except Exception as e:
        print(f"[data] SummEval load failed: {e}")
        return []

    # Group by doc_id, take first max_docs unique docs
    by_doc = defaultdict(list)
    if isinstance(rows, list) and rows and isinstance(rows[0].get("summary"), list):
        # mteb/summeval: 16 summaries per row
        for r in rows[:max_docs]:
            for sys_idx, summ in enumerate(r["summary"]):
                by_doc[r["doc_id"]].append({
                    "doc_id": r["doc_id"],
                    "system_id": f"sys_{sys_idx}",
                    "document": r["document"],
                    "summary": summ,
                    "human": {dim: r["human"][dim][sys_idx] for dim in r["human"]},
                })
    else:
        # flat schema
        for r in rows:
            by_doc[r["doc_id"]].append(r)

    docs = list(by_doc.keys())[:max_docs]
    flat = [item for d in docs for item in by_doc[d]]
    print(f"[data] SummEval: {len(docs)} docs × {len(flat)//max(len(docs),1)} systems = {len(flat)} instances")
    return flat


def load_realsumm(max_docs: int):
    """Load REALSumm — minimal fallback synthesizes a small fixture if remote fails."""
    try:
        from datasets import load_dataset
        print("[data] Downloading REALSumm from HuggingFace...")
        ds = load_dataset("krtin/realsumm", split="train")
        by_doc = defaultdict(list)
        for ex in ds:
            by_doc[ex["doc_id"]].append({
                "doc_id": ex["doc_id"],
                "system_id": ex.get("system_id", "unknown"),
                "document": ex["document"],
                "summary": ex["summary"],
                "human": {"litepyramid_recall": float(ex["litepyramid_recall"])},
            })
        docs = list(by_doc.keys())[:max_docs]
        flat = [item for d in docs for item in by_doc[d]]
        print(f"[data] REALSumm: {len(flat)} instances over {len(docs)} docs")
        return flat
    except Exception as e:
        print(f"[data] REALSumm load failed ({e}); skipping.")
        return []


def load_mlsum(lang: str, max_samples: int):
    try:
        from datasets import load_dataset
        print(f"[data] Loading MLSUM/{lang} (max {max_samples} samples)...")
        ds = load_dataset("mlsum", lang, split="test", trust_remote_code=True)
        out = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            out.append({
                "doc_id": f"mlsum_{i}",
                "system_id": "reference",
                "document": ex["text"],
                "summary": ex["summary"],
                "human": {},
            })
        return out
    except Exception as e:
        print(f"[data] MLSUM load failed: {e}")
        return []


# ────────────────────────────── Statistics ───────────────────────────────────
def safe_corr(a, b, kind):
    """Compute correlation, returning 0.0 if undefined."""
    from scipy.stats import pearsonr, spearmanr, kendalltau
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0, 1.0
    try:
        if kind == "pearson":
            r, p = pearsonr(a, b)
        elif kind == "spearman":
            r, p = spearmanr(a, b)
        else:
            r, p = kendalltau(a, b)
        return (0.0 if math.isnan(r) else float(r),
                1.0 if math.isnan(p) else float(p))
    except Exception:
        return 0.0, 1.0


def doc_avg_correlations(records, dim):
    """records: list of dicts with doc_id, score, human[dim]."""
    by_doc = defaultdict(list)
    for r in records:
        if r["score"] is None or dim not in r["human"]:
            continue
        by_doc[r["doc_id"]].append((r["score"], r["human"][dim]))

    pearsons, spearmans, kendalls = [], [], []
    for doc, pairs in by_doc.items():
        if len(pairs) < 2:
            continue
        gv = [p[0] for p in pairs]
        hv = [p[1] for p in pairs]
        pearsons.append(safe_corr(gv, hv, "pearson")[0])
        spearmans.append(safe_corr(gv, hv, "spearman")[0])
        kendalls.append(safe_corr(gv, hv, "kendalltau")[0])

    valid = len(pearsons)
    return {
        "pearson":    float(np.mean(pearsons))   if pearsons   else 0.0,
        "spearman":   float(np.mean(spearmans))  if spearmans  else 0.0,
        "kendalltau": float(np.mean(kendalls))   if kendalls   else 0.0,
        "valid_docs": valid,
    }


def corpus_correlation(records, dim):
    pairs = [(r["score"], r["human"][dim]) for r in records
             if r["score"] is not None and dim in r["human"]]
    if len(pairs) < 3:
        return {"spearman": 0.0, "spearman_p": 1.0}
    gv = [p[0] for p in pairs]
    hv = [p[1] for p in pairs]
    rho, pv = safe_corr(gv, hv, "spearman")
    return {"spearman": rho, "spearman_p": pv}


def wilcoxon_test(baseline_recs, proposed_recs, dim, scale_max):
    """Compare per-instance MAE between baseline and proposed."""
    from scipy.stats import wilcoxon
    base_map = {(r["doc_id"], r["system_id"]): r for r in baseline_recs}
    prop_map = {(r["doc_id"], r["system_id"]): r for r in proposed_recs}
    keys = sorted(set(base_map) & set(prop_map))

    base_err, prop_err = [], []
    for k in keys:
        rb, rp = base_map[k], prop_map[k]
        if rb["score"] is None or rp["score"] is None or dim not in rb["human"]:
            continue
        h = rb["human"][dim]
        base_err.append(abs(rb["score"] - h))
        prop_err.append(abs(rp["score"] - h))

    if len(base_err) < 5 or all(b == p for b, p in zip(base_err, prop_err)):
        return {
            "mean_error_baseline": round(float(np.mean(base_err)), 4) if base_err else "-",
            "mean_error_proposed": round(float(np.mean(prop_err)), 4) if prop_err else "-",
            "p_value": "-",
            "verdict": "insufficient data",
        }

    try:
        stat, p = wilcoxon(base_err, prop_err, zero_method="wilcox", correction=False)
    except Exception:
        return {
            "mean_error_baseline": round(float(np.mean(base_err)), 4),
            "mean_error_proposed": round(float(np.mean(prop_err)), 4),
            "p_value": "-",
            "verdict": "test failed",
        }
    p = 1.0 if math.isnan(p) else float(p)
    if p < 0.05:
        verdict = ("significant improvement" if np.mean(prop_err) < np.mean(base_err)
                   else "significantly worse")
    else:
        verdict = (f"not significant (p={p:.3f}); "
                   "proposed may still be directionally better"
                   if np.mean(prop_err) < np.mean(base_err)
                   else f"not significant (p={p:.3f})")
    return {
        "mean_error_baseline": round(float(np.mean(base_err)), 4),
        "mean_error_proposed": round(float(np.mean(prop_err)), 4),
        "p_value": round(p, 4),
        "verdict": verdict,
    }


# ────────────────────────────── Plots ────────────────────────────────────────
def make_plots(records, out_dir, prefix, scale_min, scale_max, dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = [r["score"] for r in records if r["score"] is not None]
    humans = [r["human"][dim] for r in records
              if r["score"] is not None and dim in r["human"]]

    if scores:
        plt.figure(figsize=(6, 4))
        if humans:
            plt.scatter(humans, scores, alpha=0.6, s=40)
            plt.xlabel(f"Human {dim}")
            plt.ylabel("G-Eval score")
            lo, hi = min(scale_min, min(humans + scores)), max(scale_max, max(humans + scores))
            plt.plot([lo, hi], [lo, hi], "r--", alpha=0.4)
        plt.title(f"{prefix} — G-Eval vs Human")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_{prefix}_scatter.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=10, alpha=0.7, edgecolor="black")
        plt.xlabel("G-Eval score")
        plt.ylabel("Frequency")
        plt.title(f"{prefix} — Score Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_{prefix}_hist.png"), dpi=150)
        plt.close()


# ────────────────────────────── Evaluation Loop ──────────────────────────────
def evaluate(evaluator, instances, prompt_tmpl, scale, variant, n_samples,
             temperature, use_debias, dim_label):
    scale_min, scale_max = scale
    out = []
    for i, inst in enumerate(instances, 1):
        prompt = build_prompt(prompt_tmpl, inst["document"], inst["summary"],
                              use_debias=use_debias)
        try:
            if variant == "proposed" and n_samples > 1:
                texts = []
                for _ in range(n_samples):
                    texts.extend(evaluator.generate(prompt, temperature=temperature,
                                                    do_sample=True))
                vals = [parse_score(t, scale_min, scale_max) for t in texts]
                vals = [v for v in vals if v is not None]
                score = float(np.mean(vals)) if vals else None
                raw = " | ".join(texts)
            else:
                texts = evaluator.generate(prompt, temperature=0.0, do_sample=False)
                score = parse_score(texts[0], scale_min, scale_max) if texts else None
                raw = texts[0] if texts else ""
        except Exception as e:
            print(f"  [error] {inst['doc_id']}/{inst['system_id']}: {e}")
            score, raw = None, str(e)

        out.append({
            "doc_id": inst["doc_id"],
            "system_id": inst["system_id"],
            "human": inst.get("human", {}),
            "score": score,
            "raw": raw[:200],
        })
        if i % 5 == 0 or i == len(instances):
            print(f"  [{variant}/{dim_label}] {i}/{len(instances)} done")
    return out


# ────────────────────────────── Pipeline ─────────────────────────────────────
def run_dataset(evaluator, dataset_name, instances, dim_to_prompt,
                args, runs_log, out_dir):
    """Iterate over (dimension, variant) combos for one dataset."""
    variants_to_run = [
        ("baseline",             False, "en"),
        ("proposed",             False, "en"),
        ("debiased-single-pass", True,  "en"),
    ]
    if args.run_multilingual_demo:
        variants_to_run.append(("multilingual-fr", False, "fr"))

    for dim, prompt_en in dim_to_prompt.items():
        scale = SUMMEVAL_SCALES.get(dim, (0, 100))
        prompt_fr = SUMMEVAL_PROMPTS_FR.get(dim) if dataset_name == "summeval" else prompt_en

        baseline_recs = None
        for variant, debias, lang in variants_to_run:
            print(f"\n=== {dataset_name} / {dim} / {variant} ===")
            tmpl = prompt_fr if lang == "fr" else prompt_en

            recs = evaluate(
                evaluator, instances, tmpl, scale,
                variant=("proposed" if variant == "proposed" else "single"),
                n_samples=args.self_consistency_n,
                temperature=args.temperature,
                use_debias=debias,
                dim_label=dim,
            )

            tag = variant.replace("-single-pass", "").replace("multilingual-fr", "multilingual_fr")
            tag = tag.replace("-", "_")
            fname = (f"{dataset_name}_{dim}_{tag}.json" if dataset_name == "summeval"
                     else f"{dataset_name}_{tag}.json")
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                json.dump(recs, f, indent=2, ensure_ascii=False)

            doc_corr = doc_avg_correlations(recs, dim)
            corp     = corpus_correlation(recs, dim) if variant != "multilingual-fr" else {}
            sig      = (wilcoxon_test(baseline_recs, recs, dim, scale[1])
                        if variant == "proposed" and baseline_recs else {})

            run_entry = {
                "dataset": dataset_name,
                "dimension": dim,
                "variant": variant,
                "correlation": doc_corr,
                "corpus_correlation": corp,
            }
            if sig:
                run_entry["significance_vs_baseline"] = sig
            runs_log.append(run_entry)

            prefix = (f"{dataset_name}_{dim}_{variant.replace('-single-pass','').replace('multilingual-fr','multilingual_fr')}"
                      if dataset_name == "summeval"
                      else f"{dataset_name}_{variant.replace('-single-pass','')}")
            prefix = prefix.replace("-", "_")
            try:
                make_plots(recs, out_dir, prefix, scale[0], scale[1], dim)
            except Exception as e:
                print(f"  [plot warn] {e}")

            if variant == "baseline":
                baseline_recs = recs

            print(f"  -> Pearson {doc_corr['pearson']:.4f} | "
                  f"Spearman {doc_corr['spearman']:.4f} | "
                  f"Kendall {doc_corr['kendalltau']:.4f}")


def main():
    args = parse_args()
    set_seeds(args.seed)
    out_dir = args.outputs_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Assignment 3 — G-Eval Experimentation and Expansion")
    print(f"Model: {args.model_name}")
    print(f"Max docs: {args.max_docs} | SC N: {args.self_consistency_n} | T: {args.temperature}")
    print(f"Debias: {args.use_debias_prompt} | Multilingual: {args.run_multilingual_demo}")
    print("=" * 60)

    # Load model once
    evaluator = MistralEvaluator(args.model_name, args.max_input_tokens, args.max_new_tokens)

    runs_log = []

    # ── SummEval ────────────────────────────────────────────────────────────
    summeval_data = load_summeval(args.max_docs)
    if summeval_data:
        run_dataset(evaluator, "summeval", summeval_data,
                    SUMMEVAL_PROMPTS_EN, args, runs_log, out_dir)

    # ── REALSumm ────────────────────────────────────────────────────────────
    realsumm_data = load_realsumm(args.max_docs)
    if realsumm_data:
        # Optional: load custom prompt
        custom_prompt_path = Path(args.prompts_dir) / "realsumm_litepyramid.txt"
        prompt = custom_prompt_path.read_text(encoding="utf-8") if custom_prompt_path.exists() else REALSUMM_PROMPT
        run_dataset(evaluator, "realsumm", realsumm_data,
                    {"litepyramid_recall": prompt}, args, runs_log, out_dir)

    # ── Multilingual dataset (MLSUM) ─────────────────────────────────────────
    if args.run_multilingual_dataset_demo:
        mlsum_data = load_mlsum(args.multilingual_dataset_languages,
                                args.multilingual_dataset_max_samples)
        if mlsum_data:
            print(f"\n=== MLSUM/{args.multilingual_dataset_languages} demo "
                  f"({len(mlsum_data)} samples) ===")
            recs = evaluate(
                evaluator, mlsum_data,
                SUMMEVAL_PROMPTS_FR["coherence"], (1, 5),
                variant="single", n_samples=1, temperature=0.0,
                use_debias=False, dim_label="coherence",
            )
            with open(os.path.join(out_dir, "multilingual_dataset_summary.json"),
                      "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": args.multilingual_dataset_name,
                    "language": args.multilingual_dataset_languages,
                    "n_samples": len(recs),
                    "mean_score": float(np.mean([r["score"] for r in recs
                                                  if r["score"] is not None]))
                                  if recs else None,
                    "records": recs,
                }, f, indent=2, ensure_ascii=False)

    # ── Save aggregated metrics ─────────────────────────────────────────────
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": args.model_name,
                "max_docs": args.max_docs,
                "self_consistency_n": args.self_consistency_n,
                "temperature": args.temperature,
                "seed": args.seed,
                "use_debias_prompt": args.use_debias_prompt,
                "multilingual": args.run_multilingual_demo,
            },
            "runs": runs_log,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"DONE. Outputs in {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
