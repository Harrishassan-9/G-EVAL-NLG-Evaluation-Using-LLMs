"""
G-EVAL replication using Groq API (free tier) — llama-3.3-70b-versatile.

Methodology:
  Replicates the paper's "G-EVAL -Probs" variant (Table 1):
    • temperature = 0  (deterministic, one call per instance)
    • Prompt delivered as system message (identical to original gpt4_eval.py)
    • Score extracted from the model's text output

  Why not probability-weighted scoring?
    Groq's Llama-3.3-70B does not expose token logprobs.  The paper showed
    "G-EVAL-4 -Probs" still achieves Spearman ρ ≈ 0.502 avg vs 0.514 with
    probs (Table 1), so this is a valid replication target.

  One call per instance → total 200 calls for all 4 SummEval dimensions
  (50 per dimension), well within Groq free-tier limits (14 400 RPD, 30 RPM).

Usage:
    python groq_eval.py --prompt_fp prompts/summeval/coh_detailed.txt \
                        --save_fp   results/groq_coh.json

    # Reduced run (50 instances instead of 1600):
    python groq_eval.py --prompt_fp prompts/summeval/coh_detailed.txt \
                        --save_fp   results/groq_coh.json --max_instances 50

Checkpoints every 50 instances — safe to interrupt and resume.
"""

import os
import json
import re
import argparse
import time

import tqdm
from groq import Groq

# ── API key ───────────────────────────────────────────────────────────────────
# Load from environment. Set GROQ_API_KEY before running:
#   PowerShell : $env:GROQ_API_KEY = (Get-Content .env.local | Select-String 'GROQ_API_KEY').ToString().Split('=')[1]
#   bash/zsh   : export $(grep -v '^#' .env.local | xargs)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Copy .env.local.example to .env.local, add your key, then load it into your shell."
    )


# ── Score parsing ─────────────────────────────────────────────────────────────

def parse_score(text: str, score_range: list) -> float:
    """
    Extract the numeric score from the model's text response.

    Model outputs the form-fill line, e.g.:
        "- Coherence: 3"
        "- Fluency (1-3): 2"    ← the (1-3) must NOT be picked as the score
        "3"

    Strategy:
      1. Split on ':' and look in the portion after the LAST colon — this
         skips the range hints like "(1-3)" that appear before the colon.
      2. Fallback to any in-range number in the full string.
      3. Last resort: first number anywhere.
    """
    # Step 1: look after the last colon
    after_colon = text.split(":")[-1] if ":" in text else text
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", after_colon)
    if nums:
        return float(nums[0])

    # Step 2: any in-range number in full text
    all_nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    lo, hi   = min(score_range), max(score_range)
    for n in all_nums:
        val = float(n)
        if lo <= val <= hi:
            return val

    # Step 3: first number
    if all_nums:
        return float(all_nums[0])
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G-EVAL via Groq (llama-3.3-70b-versatile), text-based scoring"
    )
    parser.add_argument("--prompt_fp",   type=str, required=True,
                        help="Evaluation prompt template file.")
    parser.add_argument("--save_fp",     type=str, required=True,
                        help="Output JSON path for results.")
    parser.add_argument("--summeval_fp", type=str, default="data/summeval.json")
    parser.add_argument("--model",       type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--score_range", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Valid score integers, e.g. 1 2 3 4 5  or  1 2 3.")
    parser.add_argument("--sleep",         type=float, default=4.0,
                        help="Seconds to pause between calls (default 4.0 → ~15 RPM).")
    parser.add_argument("--max_instances", type=int,   default=50,
                        help="Limit dataset to first N instances (default 50; 50×4 dims = 200 total).")
    args = parser.parse_args()

    client = Groq(api_key=GROQ_API_KEY)

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.summeval_fp, encoding="utf-8") as f:
        summeval = json.load(f)

    with open(args.prompt_fp, encoding="utf-8") as f:
        prompt_template = f.read()

    # Optionally cap to first N instances (keeps per-document structure intact)
    if args.max_instances:
        summeval = summeval[:args.max_instances]

    print(f"Loaded   : {len(summeval)} instances from {args.summeval_fp}")
    print(f"Model    : {args.model}")
    print(f"Scores   : {args.score_range}")
    print(f"Sleep    : {args.sleep}s  (~{60/args.sleep:.0f} RPM)")
    est_mins = len(summeval) * (args.sleep + 1.5) / 60
    print(f"Est.time : ~{est_mins:.0f} min for this dimension\n")

    # ── Resume support ────────────────────────────────────────────────────────
    results   = []
    done_keys = set()
    if os.path.exists(args.save_fp):
        try:
            with open(args.save_fp, encoding="utf-8") as f:
                results = json.load(f)
            done_keys = {(r["doc_id"], r["system_id"]) for r in results}
            print(f"Resuming : {len(results)} instances already saved.\n")
        except Exception:
            results = []

    errors        = 0
    current_sleep = args.sleep

    for instance in tqdm.tqdm(summeval, desc="G-EVAL"):
        key = (instance["doc_id"], instance["system_id"])
        if key in done_keys:
            continue

        source        = instance["source"]
        system_output = instance["system_output"]
        cur_prompt    = (
            prompt_template
            .replace("{{Document}}", source)
            .replace("{{Summary}}",  system_output)
        )

        while True:  # retry loop (rate-limit back-off)
            try:
                # ── API call (matches original gpt4_eval.py message format) ──
                response = client.chat.completions.create(
                    model       = args.model,
                    messages    = [{"role": "system", "content": cur_prompt}],
                    temperature = 0,
                    max_tokens  = 10,      # enough for "- Coherence: 3\n"
                    top_p       = 1,
                    frequency_penalty = 0,
                    presence_penalty  = 0,
                )

                raw_text = response.choices[0].message.content.strip()
                score    = parse_score(raw_text, args.score_range)

                # ── Warn if nothing parsed ────────────────────────────────
                if score == 0.0:
                    tqdm.tqdm.write(
                        f"  [WARN] Unparseable response for "
                        f"{instance['doc_id']}/{instance['system_id']}: "
                        f"{repr(raw_text)}"
                    )

                record = dict(instance)
                record["g_eval_score"] = score
                record["raw_response"] = raw_text
                results.append(record)
                done_keys.add(key)

                # Checkpoint every 50 instances
                if len(results) % 50 == 0:
                    with open(args.save_fp, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                    tqdm.tqdm.write(f"  [CKPT] {len(results)} saved.")

                current_sleep = args.sleep   # reset after success
                time.sleep(current_sleep)
                break

            except Exception as e:
                err = str(e).lower()

                if "429" in err or "rate_limit" in err or "rate limit" in err:
                    wait = 65
                    tqdm.tqdm.write(f"\n  [RATE LIMIT] Sleeping {wait}s …")
                    time.sleep(wait)
                    current_sleep = min(current_sleep * 1.5, 20)

                elif "413" in err or "too large" in err or "context" in err:
                    # Truncate source and retry
                    if len(source) > 3000:
                        tqdm.tqdm.write(
                            f"\n  [TRUNC] Source too long "
                            f"({len(source)} chars) – truncating."
                        )
                        source = source[:3000]
                        cur_prompt = (
                            prompt_template
                            .replace("{{Document}}", source)
                            .replace("{{Summary}}",  system_output)
                        )
                    else:
                        tqdm.tqdm.write(f"\n  [ERROR] {e} – skipping.")
                        errors += 1
                        record = dict(instance)
                        record["g_eval_score"] = 0.0
                        record["raw_response"] = "ERROR"
                        results.append(record)
                        done_keys.add(key)
                        time.sleep(current_sleep)
                        break

                else:
                    tqdm.tqdm.write(f"\n  [ERROR] {e} – skipping.")
                    errors += 1
                    record = dict(instance)
                    record["g_eval_score"] = 0.0
                    record["raw_response"] = "ERROR"
                    results.append(record)
                    done_keys.add(key)
                    time.sleep(current_sleep)
                    break

    # ── Final save ────────────────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(args.save_fp) if os.path.dirname(args.save_fp) else ".",
        exist_ok=True,
    )
    with open(args.save_fp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone!")
    print(f"  Saved  : {len(results)} results -> {args.save_fp}")
    print(f"  Errors : {errors}")
