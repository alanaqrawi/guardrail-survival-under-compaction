"""Analyze the Llama between-items replication run (first-pass judge labels).

Reports per marking x content_type: W/G/D/X counts and survival, over all
digests and over valid summaries only (non-summary classification from
_classify_nonsummaries). Attempts the canonical GEE where estimable.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260610_150139"
BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"

man = json.loads((BI / f"verdicts_{RUN_ID}.json").read_text(encoding="utf-8"))
cls = json.loads((BI / f"nonsummary_classification_{RUN_ID}.json").read_text(encoding="utf-8"))
nonsum = {d["digest_file"] for d in cls["digests"] if d["nonsummary"]}

obs = man["observations"]
print(f"run {RUN_ID}: {len(obs)} observations, "
      f"{len(nonsum)}/{len(cls['digests'])} non-summary digests")

# tracer stats over valid digests only
per_digest = {}
for o in obs:
    per_digest[o["digest_file"]] = o["tracers"]["rate"]
valid_rates = [r for f, r in per_digest.items() if f not in nonsum]
print(f"tracer survival, valid digests only: mean "
      f"{sum(valid_rates)/len(valid_rates):.1%}, max {max(valid_rates):.0%} "
      f"(n={len(valid_rates)})")

def bucket(o):
    j = o.get("judge") or {}
    return j.get("bucket")

for scope_name, keep in [("ALL DIGESTS", lambda o: True),
                         ("VALID ONLY", lambda o: o["digest_file"] not in nonsum)]:
    print(f"\n=== {scope_name} ===")
    for marking in ("unmarked", "marked"):
        for ct in ("deontic", "epistemic"):
            sub = [o for o in obs if o["marking"] == marking
                   and o["content_type"] == ct and keep(o)
                   and bucket(o) in ("W", "G", "D", "X")]
            c = Counter(bucket(o) for o in sub)
            surv = c["W"] + c["G"]
            n = len(sub)
            pct = f"{surv/n:.0%}" if n else "-"
            print(f"  {marking:<9}{ct:<10} W={c['W']:<3} G={c['G']:<3} "
                  f"D={c['D']:<3} X={c['X']:<4} n={n:<4} surv={surv} ({pct})")

# survivors + D cases for the blind-read packet
flagged = [o for o in obs if bucket(o) in ("W", "G", "D")]
print(f"\nJudge-flagged W/G/D cases (blind-read candidates): {len(flagged)}")
for o in flagged:
    print(f"  {o['obs_id']:<38} {o['item_id']:<35} {o['marking']:<9} "
          f"judge={bucket(o)} target_present={o['deterministic_target_present']} "
          f"nonsummary={o['digest_file'] in nonsum}")

# GEE on valid-only (canonical spec), guarded
try:
    import pandas as pd
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    SL = BI.parent / "salience_ratings"
    matched = {}
    for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
        s = json.loads((SL / run).read_text(encoding="utf-8"))
        for d in s["details"]:
            if d["variant"] != "unmarked" or not d.get("admitted"):
                continue
            matched[(d["pair_id"], "deontic")] = d["d_mean"]
            matched[(d["pair_id"], "epistemic")] = d["e_mean"]
    for marking in ("unmarked", "marked"):
        rows = []
        for o in obs:
            b = bucket(o)
            if b not in ("W", "G", "D", "X") or o["marking"] != marking:
                continue
            if o["digest_file"] in nonsum:
                continue
            sal = matched.get((o["pair_id"], o["content_type"])) or 3.0
            rows.append(dict(y=1 if b in ("W", "G") else 0,
                             is_deontic=1 if o["content_type"] == "deontic" else 0,
                             sal=sal, pos=o["position"], conv=o["conversation_id"]))
        df = pd.DataFrame(rows)
        n_surv = int(df.y.sum())
        if n_surv < 3 or df[df.is_deontic == 0].y.sum() == 0:
            print(f"\nGEE {marking} (valid-only): NOT ESTIMABLE — "
                  f"{n_surv} total survivors, "
                  f"{int(df[df.is_deontic==0].y.sum())} epistemic survivors "
                  f"(separation); report counts only")
            continue
        m = GEE.from_formula("y ~ is_deontic + sal + pos", groups=df["conv"],
                             data=df, family=Binomial()).fit()
        print(f"\nGEE {marking} (valid-only): beta={m.params['is_deontic']:.3f} "
              f"SE={m.bse['is_deontic']:.3f} p={m.pvalues['is_deontic']:.2e}")
except Exception as e:  # noqa: BLE001
    print(f"\nGEE skipped: {type(e).__name__}: {e}")
