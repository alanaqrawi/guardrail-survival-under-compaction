"""Sensitivity check: re-fit the canonical between-items GEE excluding the one
non-summary digest (qwen__unmarked__conv023, flagged by _classify_nonsummaries).

Mirrors _refit_matched_salience.py exactly (matched covariate, GEE logistic,
robust SEs clustered on conversation) with a single-digest exclusion.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SL = R / "results/salience_ratings"

EXCLUDE = {"qwen__unmarked__conv023.txt"}


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def getb(o, f):
    r = o.get(f)
    return r.get("bucket") if isinstance(r, dict) else r


matched = {}
for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
    s = jload(SL / run)
    for d in s["details"]:
        if d["variant"] != "unmarked" or not d.get("admitted"):
            continue
        matched[(d["pair_id"], "deontic")] = d["d_mean"]
        matched[(d["pair_id"], "epistemic")] = d["e_mean"]


def fit(manfile, field, label):
    man = jload(BI / manfile)
    for excl in (False, True):
        for marking in ("unmarked", "marked"):
            rows = []
            for o in man["observations"]:
                b = getb(o, field)
                if b not in ("W", "G", "D", "X") or o["marking"] != marking:
                    continue
                if excl and o["digest_file"] in EXCLUDE:
                    continue
                sal = matched.get((o["pair_id"], o["content_type"]))
                if sal is None:
                    sal = o.get("salience_score") or 3.0
                rows.append(dict(
                    y=1 if b in ("W", "G") else 0,
                    is_deontic=1 if o["content_type"] == "deontic" else 0,
                    sal=sal, pos=o["position"], conv=o["conversation_id"],
                ))
            df = pd.DataFrame(rows)
            n_d = int(df[df.is_deontic == 1].shape[0])
            s_d = int(df[df.is_deontic == 1].y.sum())
            n_e = int(df[df.is_deontic == 0].shape[0])
            s_e = int(df[df.is_deontic == 0].y.sum())
            m = GEE.from_formula("y ~ is_deontic + sal + pos", groups=df["conv"],
                                 data=df, family=Binomial()).fit()
            beta = m.params["is_deontic"]
            se = m.bse["is_deontic"]
            p = m.pvalues["is_deontic"]
            tag = "EXCL conv023" if excl else "all digests "
            print(f"  {label}/{marking:<8} [{tag}] deontic {s_d}/{n_d}  "
                  f"epistemic {s_e}/{n_e}  beta={beta:.3f} SE={se:.3f} p={p:.2e}")


print("=== HUMAN-CANONICAL (judge_final) ===")
fit("verdicts_20260602_214506_final.json", "judge_final", "human")
print("=== v4 (judge_v2) ===")
fit("verdicts_20260602_214506_v2.json", "judge_v2", "v4")
