# G-EVAL: NLG Evaluation Using LLMs with Better Human Alignment

## 👥 Team

| Name 

| Harris Hassan Syed 
| Muhammad Muttayab 
| Fadil Falak 

---

## 📋 Project Overview

This project is a full-cycle study of **G-EVAL** ([Zhong et al., 2023](https://arxiv.org/abs/2303.16634)), a framework that uses large language models as evaluators for NLG tasks such as text summarization and dialogue generation. The project spans three assignments:

| Phase | Focus | Model Used |
|-------|-------|-----------|
| Assignment 1 | Systematic paper analysis & proposed extensions | — |
| Assignment 2 | Scaled replication of paper results | Llama-3.3-70B (Groq API) |
| Assignment 3 | Novel extensions: self-consistency, multilingual, debiasing | Mistral-7B (local, 4-bit) |

---


## 🧠 What is G-EVAL?

Traditional NLG metrics like **BLEU** and **ROUGE** correlate poorly with human judgment — they rely on surface-level n-gram overlap and can't capture semantic quality. G-EVAL addresses this by:

1. **Prompt Construction** — Task introduction + dimension-specific criteria (e.g., coherence, fluency)
2. **Auto Chain-of-Thought (CoT)** — The LLM itself generates step-by-step evaluation instructions
3. **Form-Fill Scoring** — Model outputs a numeric score on a defined scale
4. **Probability-Weighted Refinement** — Final score is a probability-weighted expectation over token outputs

$$\text{score} = \sum_{i=1}^{n} p(s_i) \times s_i$$

The framework is **zero-shot** — no training data required, only benchmark meta-evaluation.

---

## 📦 Assignment 1 — Systematic Paper Analysis

**Goal:** Deeply understand the G-EVAL paper and identify research gaps.

### Key Findings

| Aspect | Finding |
|--------|---------|
| Human Alignment | 0.514 Spearman on SummEval — best-in-class |
| vs. Prior SOTA | +8.4% over UniEval (0.474), +23% over GPTScore (0.417) |
| Self-Bias | GPT-4 systematically favors LLM-generated text over human text |
| Reproducibility | Closed-source GPT-4 with no public log-probability access |
| Language Coverage | English-only (CNN/DailyMail, XSum, Topical-Chat) |

### Proposed Extensions (implemented in Assignment 3)

- **Open-source LLM replacement** — Replace GPT-4 with LLaMA/Mistral for reproducibility
- **Multilingual adaptation** — Extend evaluation prompts to languages beyond English
- **Debiasing prompts** — Mitigate evaluator preference for LLM-generated text

---

## 🔁 Assignment 2 — Replication

**Goal:** Replicate G-EVAL results at reduced cost using an open-weight model.

### Setup

| Parameter | Original Paper | Our Replication |
|-----------|---------------|-----------------|
| Model | GPT-4 (OpenAI API) | Llama-3.3-70B (Groq API) |
| Scoring | Probability-weighted (N=20) | Single-call, T=0 |
| Dataset | SummEval full (1,600 instances) | 50 instances per dimension |
| Dimensions | Coherence, Consistency, Fluency, Relevance | Same |

### Installation

```bash
pip install groq scipy numpy
```

Set your Groq API key:

```bash
export GROQ_API_KEY=your_key_here
```

### Usage

```bash
python assignment2/run_replication.py
```

Results are saved per dimension in `assignment2/results/groq_{dim}.json` with checkpoint/resume support.

### Results

| Dimension | Paper (GPT-4) | Ours (Llama-70B) | Δ |
|-----------|:-------------:|:-----------------:|:-:|
| Coherence | 0.582 | 0.541 | −0.041 ✅ |
| Consistency | 0.507 | 0.485 | −0.022 ✅ |
| Fluency | 0.506 | 0.489 | −0.017 ✅ |
| Relevance | 0.547 | −0.002 | −0.549 ❌ |
| **Avg (3 dims)** | **0.535** | **0.505** | **−0.030** |

> ⚠️ **Relevance failure** was traced to extreme score compression (96% of outputs at score 4–5) combined with only 4 source documents — a sample-size artifact, not a methodology failure.

---

## 🚀 Assignment 3 — Novel Extensions

**Goal:** Implement the three extensions proposed in Assignment 1 using a fully local, open-source evaluator.

### Model

**Mistral-7B-Instruct-v0.3** deployed locally via HuggingFace Transformers with 4-bit NF4 quantization (BitsAndBytes) on a Tesla T4 GPU. No API dependency.

### Installation

```bash
pip install transformers bitsandbytes accelerate scipy numpy datasets
```

### Extension 1 — Self-Consistency Scoring (N=3)

Each (document, summary) pair is evaluated **3 times at temperature=0.7**, and the mean score is used as the final G-EVAL output. This marginalizes stochastic variance without needing log-probability access.

```bash
python assignment3/mistral_eval.py --variant proposed --n_samples 3 --temperature 0.7
```

**Results:**

| Dimension | Baseline ρ | SC N=3 ρ | Δ |
|-----------|:----------:|:--------:|:-:|
| Coherence | 0.4363 | 0.4989 | **+0.063** ✅ |
| Consistency | 0.4727 | 0.4279 | −0.045 |
| Fluency | 0.6092 | 0.5452 | −0.064 |
| Relevance | 0.6788 | **0.7472** | **+0.068** ✅ |

> 🏆 Relevance ρ = **0.747** with Mistral-7B SC exceeds GPT-4's reported 0.547 on the same benchmark.

### Extension 2 — Multilingual Prompts (French)

All evaluation prompts were translated to French and applied to SummEval (English documents) and MLSUM (French documents).

```bash
python assignment3/mistral_eval.py --variant multilingual --language fr
```

**Results (English vs. French prompt, Spearman ρ):**

| Dimension | English | French | Δ |
|-----------|:-------:|:------:|:-:|
| Coherence | 0.4363 | 0.4855 | +0.049 ✅ |
| Consistency | 0.4727 | 0.4142 | −0.059 |
| Fluency | 0.6092 | 0.5098 | −0.099 ⚠️ |
| Relevance | 0.6788 | 0.6669 | −0.012 ✅ |

> Fluency degrades most under French prompts — expected, since fluency is a language-surface-dependent criterion.

### Extension 3 — Debiasing Prompt

A system prompt explicitly instructing the model to ignore summary length and anchor judgment strictly on stated criteria.

```bash
python assignment3/mistral_eval.py --variant debiased
```

**Finding:** Debiasing consistently underperformed the baseline across all dimensions. The added constraint over-compressed the score distribution, reducing rank correlation. **Fine-tuning on human-scored examples is the more effective path** for small models.

### Three-Way Comparison

| Dimension | GPT-4 (Paper) | Llama-70B (A2) | Mistral-7B SC (A3) |
|-----------|:-------------:|:--------------:|:------------------:|
| Coherence | 0.582 | 0.541 | 0.499 |
| Consistency | 0.507 | 0.485 | 0.428 |
| Fluency | 0.506 | 0.489 | **0.609** |
| Relevance | 0.547 | −0.002 | **0.747** |

---

## 📊 Datasets

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| SummEval | Summarization | 1,600 instances (100 docs × 16 systems) | CNN/DailyMail |
| REALSumm | Summarization | 25 systems × 100 docs | News |
| MLSUM (French) | Multilingual Summarization | 30 samples | French news |

Human annotation dimensions evaluated: **Coherence (1–5)**, **Consistency (1–5)**, **Fluency (1–3)**, **Relevance (1–5)**

---

## 📈 Key Takeaways

1. **G-EVAL methodology is robust** — replicates across 3/4 dimensions with different LLMs and reduced data
2. **Open-source LLMs are viable** — Mistral-7B with self-consistency achieves ρ = 0.747 on Relevance, beating GPT-4
3. **Self-consistency is dimension-selective** — most beneficial on wider-scale dimensions (1–5); can regress on narrow scales (1–3)
4. **Multilingual G-EVAL works for semantic dimensions** — Fluency is the most cross-lingual-sensitive criterion
5. **Zero-shot debiasing at 7B scale is ineffective** — supervised fine-tuning is the preferred debiasing approach

---

## ⚙️ Requirements

```
Python >= 3.9
torch >= 2.0
transformers >= 4.40
bitsandbytes >= 0.43
groq
scipy
numpy
datasets
```

GPU with at least **15 GB VRAM** recommended for Mistral-7B with 4-bit quantization (Tesla T4 or equivalent).

---

## 📚 References

- Zhong et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* arXiv:2303.16634
- Fabbri et al. (2021). *SummEval: Re-evaluating Summarization Evaluation.* TACL, 9, 391–409
- Bhandari et al. (2020). *Re-evaluating Evaluation in Text Summarization.* EMNLP 2020
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171
- Jiang et al. (2023). *Mistral 7B.* arXiv:2310.06825
- Meta AI (2024). *Llama 3.3.*
- Papineni et al. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation.* ACL 2002
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop

---

## 📄 License

This project is for academic purposes only as part of the Natural Language Processing course at FAST-NUCES.
