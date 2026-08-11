"""Apply the blind-read adjudication to the Llama replication run.

Produces verdicts_20260610_150139_final.json with a judge_final field:
- the 35 blind-read cases take the rater's label (the read covered every
  non-X first-pass label, i.e. all result-driving cases, plus 20 sampled X);
- all other observations keep the first-pass judge bucket (all X).

Rater: independent second model (Claude Opus 4.8), blind to all labels,
frozen v4 rubric. Read returned 2026-06-10.
"""
import json
from pathlib import Path

BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"
RUN = "20260610_150139"

RATER = {1: "X", 2: "X", 3: "X", 4: "X", 5: "X", 6: "X", 7: "X", 8: "X",
         9: "X", 10: "X", 11: "X", 12: "X", 13: "X", 14: "X", 15: "X",
         16: "X", 17: "X", 18: "X", 19: "X", 20: "X", 21: "G", 22: "X",
         23: "X", 24: "G", 25: "X", 26: "X", 27: "X", 28: "X", 29: "X",
         30: "X", 31: "X", 32: "X", 33: "X", 34: "X", 35: "X"}

key = json.loads((BI / f"blind_read_llama_KEY_{RUN}.json").read_text(encoding="utf-8"))
by_obs = {k["obs_id"]: RATER[k["case"]] for k in key}

man = json.loads((BI / f"verdicts_{RUN}.json").read_text(encoding="utf-8"))
n_override = 0
for o in man["observations"]:
    j = (o.get("judge") or {}).get("bucket")
    if o["obs_id"] in by_obs:
        rater = by_obs[o["obs_id"]]
        o["judge_final"] = rater
        o["adjudicated"] = True
        if rater != j:
            o["adjudication_note"] = f"blind read overrode {j} -> {rater}"
            n_override += 1
    else:
        o["judge_final"] = j
        o["adjudicated"] = False

man["adjudication"] = {
    "method": "blind read of all 15 first-pass W/G/D cases + 20 sampled X "
              "(35 cases), frozen v4 rubric, labels hidden",
    "rater": "independent second model (Claude Opus 4.8), blind to all labels",
    "human_review": "labels reviewed and adopted by a human author (final "
                    "adjudication authority), 2026-06-10",
    "date": "2026-06-10",
    "survival_level_agreement_vs_first_pass": "33/35",
    "survival_level_agreement_vs_v4": "33/35",
    "overrides": n_override,
}
out = BI / f"verdicts_{RUN}_final.json"
out.write_text(json.dumps(man, indent=2), encoding="utf-8")

from collections import Counter
for mk in ("unmarked", "marked"):
    for ct in ("deontic", "epistemic"):
        c = Counter(o["judge_final"] for o in man["observations"]
                    if o["marking"] == mk and o["content_type"] == ct
                    and o["judge_final"] in ("W", "G", "D", "X"))
        print(f"{mk:<9}{ct:<10} W={c['W']:<3} G={c['G']:<3} D={c['D']:<3} "
              f"X={c['X']:<4} surv={c['W'] + c['G']}")
print(f"\noverrides applied: {n_override};  wrote {out}")
