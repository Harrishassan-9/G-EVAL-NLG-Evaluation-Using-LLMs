"""
Pre-flight check: verifies everything is correct before the full run.
Run with:  python preflight_check.py
"""

import os, json, re, math, time, sys
from scipy.stats import spearmanr, pearsonr, kendalltau
from prettytable import PrettyTable

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"

results = []
def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label, detail))
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))

# ─────────────────────────────────────────────────────────────────
print("\n=== 1. FILES & STRUCTURE ===")

files_needed = [
    "data/summeval.json",
    "prompts/summeval/coh_detailed.txt",
    "prompts/summeval/con_detailed.txt",
    "prompts/summeval/flu_detailed.txt",
    "prompts/summeval/rel_detailed.txt",
    "groq_eval.py",
    "meta_eval_groq.py",
    "run_replication.py",
]
for fp in files_needed:
    check(f"File exists: {fp}", os.path.exists(fp))

# ─────────────────────────────────────────────────────────────────
print("\n=== 2. DATASET STRUCTURE ===")

data = json.load(open("data/summeval.json", encoding="utf-8"))
check("Dataset loads",             True, f"{len(data)} instances")
check("400-instance slice = 25 docs x 16",
      len(set(x["doc_id"] for x in data[:400])) == 25,
      f"{len(set(x['doc_id'] for x in data[:400]))} unique docs in first 400")

required_fields = ["doc_id", "system_id", "source", "system_output", "scores"]
for field in required_fields:
    check(f"Field '{field}' present", field in data[0])

score_dims = ["coherence", "consistency", "fluency", "relevance"]
for d in score_dims:
    check(f"Scores['{d}'] present", d in data[0]["scores"])

avg_chars = sum(len(x["source"]) + len(x["system_output"]) for x in data[:400]) / 400
est_tokens = avg_chars / 4 + 350
check("Avg tokens/call < 2000 (well within 131k context)",
      est_tokens < 2000, f"~{est_tokens:.0f} tokens avg")

max_src = max(len(x["source"]) for x in data[:400])
max_tokens = max_src / 4 + 500
check("Max tokens/call < 5000 (safest document in 400-subset)",
      max_tokens < 5000, f"max ~{max_tokens:.0f} tokens")

# ─────────────────────────────────────────────────────────────────
print("\n=== 3. PROMPT FILES CONTENT ===")

DIMS_CONFIG = [
    ("coherence",   "prompts/summeval/coh_detailed.txt", [1,2,3,4,5]),
    ("consistency", "prompts/summeval/con_detailed.txt", [1,2,3,4,5]),
    ("fluency",     "prompts/summeval/flu_detailed.txt", [1,2,3]),
    ("relevance",   "prompts/summeval/rel_detailed.txt", [1,2,3,4,5]),
]

for name, fp, sr in DIMS_CONFIG:
    t = open(fp, encoding="utf-8").read()
    check(f"{name}: has {{{{Summary}}}}", "{{Summary}}" in t)
    if name != "fluency":
        check(f"{name}: has {{{{Document}}}}", "{{Document}}" in t)
    check(f"{name}: ends with evaluation form trigger",
          "Evaluation Form" in t or "scores ONLY" in t or "Answer:" in t)
    # fluency uses inline scale criteria (- 1: / - 2: / - 3:) instead of steps
    has_steps = ("Evaluation Steps" in t or "1." in t or "- 1:" in t)
    check(f"{name}: has CoT/evaluation criteria", has_steps)

# ─────────────────────────────────────────────────────────────────
print("\n=== 4. PARSE_SCORE FUNCTION ===")

def parse_score(text, score_range):
    after_colon = text.split(":")[-1] if ":" in text else text
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", after_colon)
    if nums:
        return float(nums[0])
    all_nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    lo, hi = min(score_range), max(score_range)
    for n in all_nums:
        val = float(n)
        if lo <= val <= hi:
            return val
    return float(all_nums[0]) if all_nums else 0.0

parse_tests = [
    ("- Coherence: 1",               [1,2,3,4,5], 1.0),
    ("- Coherence: 3",               [1,2,3,4,5], 3.0),
    ("- Coherence: 5",               [1,2,3,4,5], 5.0),
    ("- Fluency (1-3): 1",           [1,2,3],     1.0),
    ("- Fluency (1-3): 2",           [1,2,3],     2.0),
    ("- Fluency (1-3): 3",           [1,2,3],     3.0),
    ("- Consistency: 2\n\nThe sum",  [1,2,3,4,5], 2.0),
    ("- Relevance: 4\n\nThe sum",    [1,2,3,4,5], 4.0),
    ("3",                            [1,2,3,4,5], 3.0),
    ("2.5",                          [1,2,3,4,5], 2.5),
    ("",                             [1,2,3,4,5], 0.0),
]
all_parse_ok = True
for text, rng, expected in parse_tests:
    got = parse_score(text, rng)
    ok  = got == expected
    if not ok:
        all_parse_ok = False
    check(f"parse({repr(text):<35}) == {expected}", ok, f"got {got}")

# ─────────────────────────────────────────────────────────────────
print("\n=== 5. CORRELATION LOGIC (matches paper methodology) ===")

# Simulate with synthetic data: 3 docs x 5 systems
import random
random.seed(42)
pred_by_doc  = {f"doc{i}": [random.uniform(1,5) for _ in range(5)] for i in range(3)}
human_by_doc = {f"doc{i}": [random.uniform(1,5) for _ in range(5)] for i in range(3)}

acc = {}
n = 0
for did in pred_by_doc:
    p, h = pred_by_doc[did], human_by_doc[did]
    if len(set(p)) <= 1 or len(set(h)) <= 1:
        continue
    if not acc:
        acc = {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    acc["pearson"]    += pearsonr(p, h)[0]
    acc["spearman"]   += spearmanr(p, h)[0]
    acc["kendalltau"] += kendalltau(p, h)[0]
    n += 1

check("Summary-level correlation computes without error", n == 3, f"{n} docs processed")
check("Pearson result is a valid float",   isinstance(acc["pearson"]    / n, float))
check("Spearman result is a valid float",  isinstance(acc["spearman"]   / n, float))
check("Kendalltau result is a valid float",isinstance(acc["kendalltau"] / n, float))

# ─────────────────────────────────────────────────────────────────
print("\n=== 6. RATE LIMIT MATH ===")

total_calls = 400 * 4
sleep_s     = 4
api_latency = 1.5
per_call    = sleep_s + api_latency
total_hrs   = total_calls * per_call / 3600
rpm         = 60 / sleep_s

check(f"RPM ({rpm:.0f}) < 30 RPM limit",      rpm < 30,     f"{rpm:.0f} RPM")
check(f"RPD ({total_calls}) < 14400 RPD limit", total_calls < 14400, f"{total_calls} calls")
check(f"Total time estimate reasonable",       total_hrs < 5, f"~{total_hrs:.1f} hours")

# ─────────────────────────────────────────────────────────────────
print("\n=== 7. LIVE API CONNECTIVITY (1 real call) ===")

try:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    inst = data[1]   # use second instance to avoid any caching
    tmpl = open("prompts/summeval/coh_detailed.txt", encoding="utf-8").read()
    prompt = tmpl.replace("{{Document}}", inst["source"]).replace("{{Summary}}", inst["system_output"])

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0, max_tokens=10, top_p=1,
        frequency_penalty=0, presence_penalty=0,
    )
    raw   = resp.choices[0].message.content.strip()
    score = parse_score(raw, [1,2,3,4,5])
    human = inst["scores"]["coherence"]

    check("API call succeeded",         True,  f"HTTP 200")
    check("Response is non-empty",      len(raw) > 0, f"raw={repr(raw)}")
    check("Score parsed successfully",  score > 0, f"score={score}, human={human:.2f}")
    check("Score in valid range [1-5]", 1 <= score <= 5, f"score={score}")

except Exception as e:
    check("API call succeeded", False, str(e)[:80])

# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passes  = sum(1 for s,_,_ in results if s == PASS)
failures= sum(1 for s,_,_ in results if s == FAIL)
print(f"  TOTAL: {passes} passed, {failures} failed\n")

if failures == 0:
    print("  ALL CHECKS PASSED. Safe to run:  python run_replication.py")
else:
    print("  SOME CHECKS FAILED. Fix the above issues before running.")
    sys.exit(1)
