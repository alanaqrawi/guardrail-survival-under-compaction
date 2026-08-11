"""Analyze the Claude hard-cap (150-token) between-items cell (Exp 3, v10 experiments pass).

Uses adjudicated labels: the first-pass judge labels for run 20260804_223216, with a targeted
author/model read of every epistemic survivor (presence-is-not-inference). Only four epistemic
"survivals" were over-credited, all the Nexovia (P03) rule/fact-overlap confound where the fact
was reframed as a rule; these are conservatively corrected to X below. Every other survivor was
verbatim-present in the digest.

Reports, per marking, deontic vs epistemic survival and the GEE marginal coefficient (clustered on
conversation), plus a target-fixed-effects + conversation-clustered logit and a conversation
cluster bootstrap for the unmarked (headline) condition, matching how the Qwen cell is reported.
Run: .venv/Scripts/python.exe tracer/_analyze_claude_cell.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
import statsmodels.formula.api as smf

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SL = R / "results/salience_ratings"
RID = "20260804_223216"

# Adjudication: epistemic survivors over-credited by the judge (Nexovia P03 rule/fact overlap);
# the surviving text is the rule's, not the standalone fact. Conservatively corrected to X.
FLIPS = {
    "claude__unmarked__conv029__item4", "claude__unmarked__conv035__item7",
    "claude__marked__conv005__item0", "claude__marked__conv040__item2",
}

matched = {}
for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
    for d in json.loads((SL / run).read_text(encoding="utf-8"))["details"]:
        if d["variant"] == "unmarked" and d.get("admitted"):
            matched[(d["pair_id"], "deontic")] = d["d_mean"]
            matched[(d["pair_id"], "epistemic")] = d["e_mean"]

obs = json.loads((BI / f"verdicts_{RID}.json").read_text(encoding="utf-8"))["observations"]


def bucket(o):
    b = (o.get("judge") or {}).get("bucket")
    if o["obs_id"] in FLIPS:
        return "X"
    return b


rows = []
for o in obs:
    b = bucket(o)
    if b not in ("W", "G", "D", "X"):
        continue
    sal = matched.get((o["pair_id"], o["content_type"])) or o.get("salience_score") or 3.0
    rows.append({
        "y": 1 if b in ("W", "G") else 0,
        "is_deontic": 1 if o["content_type"] == "deontic" else 0,
        "sal": sal, "pos": o.get("position", 0.5),
        "conv": o["conversation_id"], "pair": o["pair_id"],
        "marking": o["marking"], "ct": o["content_type"],
    })
df = pd.DataFrame(rows)

print(f"Claude @150tok cell, adjudicated ({len(FLIPS)} epistemic flips -> X)\n")
for mk in ("unmarked", "marked"):
    d = df[df.marking == mk]
    dd = d[d.is_deontic == 1]; ee = d[d.is_deontic == 0]
    print(f"=== {mk} ===")
    print(f"  deontic survival {dd.y.sum()}/{len(dd)} ({100*dd.y.mean():.0f}%)  "
          f"epistemic {ee.y.sum()}/{len(ee)} ({100*ee.y.mean():.0f}%)")
    g = GEE.from_formula("y ~ is_deontic + sal + pos", groups="conv",
                         data=d, family=Binomial()).fit()
    ci = g.conf_int().loc["is_deontic"]
    print(f"  GEE (conv-clustered): beta={g.params['is_deontic']:.3f} "
          f"95% [{ci[0]:.2f}, {ci[1]:.2f}] p={g.pvalues['is_deontic']:.2e}")

# dependence-aware (headline unmarked)
d = df[df.marking == "unmarked"]
try:
    fe = smf.logit("y ~ is_deontic + sal + pos + C(pair)", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["conv"]})
    ci = fe.conf_int().loc["is_deontic"]
    print(f"\nunmarked target-FE + conv-cluster logit: beta={fe.params['is_deontic']:.3f} "
          f"95% [{ci[0]:.2f}, {ci[1]:.2f}] p={fe.pvalues['is_deontic']:.2e}")
except Exception as e:
    print("target-FE issue:", repr(e))

rng = np.random.default_rng(7)
ids = d["conv"].unique()
betas = []
for _ in range(1000):
    samp = rng.choice(ids, size=len(ids), replace=True)
    parts = [d[d.conv == i] for i in samp]
    bb = pd.concat(parts, ignore_index=True)
    bb["gid"] = np.repeat(np.arange(len(samp)), [len(p) for p in parts])
    try:
        r = GEE.from_formula("y ~ is_deontic + sal + pos", groups="gid",
                             data=bb, family=Binomial()).fit()
        betas.append(r.params["is_deontic"])
    except Exception:
        pass
print(f"unmarked conv-cluster bootstrap ({len(betas)}/1000): "
      f"beta 95% {np.round(np.percentile(betas, [2.5, 97.5]), 2)}")
