"""Dependence-aware re-analysis (Sol review items 9 / N1 / N2 / N5), existing data only.

For the unmarked deontic-vs-epistemic contrast we fit, on the SAME labels already in the paper:
  - the canonical GEE (logistic, clustered on conversation),
  - a target (pair) fixed-effects logistic with conversation-clustered robust SEs,
  - a nonparametric cluster bootstrap resampling whole conversations, and (separately) whole
    target pairs, for the coefficient and the deontic / epistemic / W-only survival rates.

v8 additions (Sol v4 review):
  - N5: the whole battery is run under BOTH label sets, the primary set (`judge_final`,
    targeted author adjudication) and the conservative frozen v4 rubric (`judge_v2.bucket`),
    so dependence robustness and label robustness are shown jointly.
  - N2: the conversation bootstrap and the target-pair bootstrap are reported separately (they
    are not a single joint multiway resample), and the pair bootstrap rests on only 10 pairs,
    so it is reported as unstable / indicative.
Bootstrap settings: B = 1000 resamples, seed = 7, percentile 95% intervals; resamples whose
GEE fit fails to converge are skipped and the surviving count is reported.

N1 target x predicate decomposition uses target retention = deterministic_target_present and
predicate (restriction) recoverable = bucket in {W, G}.

No new data are generated; counts are unchanged, only the inferential intervals are estimated
in a way that respects the repeated conversation/target structure.
Run: .venv/Scripts/python.exe tracer/_reanalysis_v7_clustered.py
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

B_REPS = 1000
SEED = 7


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# matched (re-authored) salience covariate, unmarked variant = canonical
matched = {}
for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
    for d in jload(SL / run)["details"]:
        if d["variant"] == "unmarked" and d.get("admitted"):
            matched[(d["pair_id"], "deontic")] = d["d_mean"]
            matched[(d["pair_id"], "epistemic")] = d["e_mean"]

man_final = jload(BI / "verdicts_20260602_214506_final.json")   # primary labels (judge_final)
man_v2 = jload(BI / "verdicts_20260602_214506_v2.json")          # conservative frozen v4 rubric


def primary_label(o):
    return o.get("judge_final")


def conservative_label(o):
    return (o.get("judge_v2") or {}).get("bucket")


def build_df(man, label_fn):
    rows = []
    for o in man["observations"]:
        if o["marking"] != "unmarked":
            continue
        b = label_fn(o)
        if b not in ("W", "G", "D", "X"):
            continue
        sal = matched.get((o["pair_id"], o["content_type"])) or o.get("salience_score") or 3.0
        rows.append({
            "y": 1 if b in ("W", "G") else 0,
            "w": 1 if b == "W" else 0,
            "is_deontic": 1 if o["content_type"] == "deontic" else 0,
            "sal": sal, "pos": o.get("position", 0.5),
            "conv": o["conversation_id"], "pair": o["pair_id"],
            "ct": o["content_type"], "bucket": b,
            "tgt": bool(o["deterministic_target_present"]),
        })
    return pd.DataFrame(rows)


def boot(df, cluster_col, B=B_REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = df[cluster_col].unique()
    betas, dds, ees, ws = [], [], [], []
    for _ in range(B):
        samp = rng.choice(ids, size=len(ids), replace=True)
        parts = [df[df[cluster_col] == i] for i in samp]
        bb = pd.concat(parts, ignore_index=True)
        bb["gid"] = np.repeat(np.arange(len(samp)), [len(p) for p in parts])
        try:
            r = GEE.from_formula("y ~ is_deontic + sal + pos", groups="gid",
                                 data=bb, family=Binomial()).fit()
            betas.append(r.params["is_deontic"])
        except Exception:
            pass
        d1 = bb[bb.is_deontic == 1]; e1 = bb[bb.is_deontic == 0]
        if len(d1):
            dds.append(100 * d1.y.mean()); ws.append(100 * d1.w.mean())
        if len(e1):
            ees.append(100 * e1.y.mean())
    q = lambda a: np.round(np.percentile(a, [2.5, 97.5]), 1)
    print(f"  [{cluster_col} bootstrap: {len(betas)}/{B} fits converged, seed={seed}] "
          f"beta 95% {q(betas)} | deontic% {q(dds)} | epistemic% {q(ees)} | W-only% {q(ws)}")


def run_all(df, name):
    print(f"\n===== {name} labels | n unmarked = {len(df)} =====")
    print(f"deontic survival {100*df[df.is_deontic==1].y.mean():.1f}% "
          f"| epistemic {100*df[df.is_deontic==0].y.mean():.1f}% "
          f"| W-only deontic {100*df[df.is_deontic==1].w.mean():.1f}%")
    g = GEE.from_formula("y ~ is_deontic + sal + pos", groups="conv",
                         data=df, family=Binomial()).fit()
    print(f"canonical GEE (conv clusters): beta={g.params['is_deontic']:.3f} "
          f"SE={g.bse['is_deontic']:.3f} p={g.pvalues['is_deontic']:.2e}")
    try:
        fe = smf.logit("y ~ is_deontic + sal + pos + C(pair)", data=df).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": df["conv"]})
        ci = fe.conf_int().loc["is_deontic"]
        print(f"target-FE + conv-cluster logit: beta={fe.params['is_deontic']:.3f} "
              f"95% [{ci[0]:.2f}, {ci[1]:.2f}] p={fe.pvalues['is_deontic']:.2e}")
    except Exception as e:
        print("target-FE fit issue:", repr(e))
    boot(df, "conv")
    boot(df, "pair")


df_primary = build_df(man_final, primary_label)
df_cons = build_df(man_v2, conservative_label)
run_all(df_primary, "PRIMARY (judge_final, targeted author adjudication)")
run_all(df_cons, "CONSERVATIVE (frozen v4 rubric, _v2.json judge_v2)")

# --- N1: target x predicate 2x2 from primary labels ---
print("\nN1 target x predicate (primary labels; target = deterministic_target_present; "
      "predicate recoverable = bucket in {W,G}):")
for ct in ("deontic", "epistemic"):
    s = df_primary[df_primary.ct == ct]
    tp_rp = int(((s.tgt) & (s.bucket.isin(["W", "G"]))).sum())
    tp_rm = int(((s.tgt) & (~s.bucket.isin(["W", "G"]))).sum())
    tm_rp = int(((~s.tgt) & (s.bucket.isin(["W", "G"]))).sum())
    tm_rm = int(((~s.tgt) & (~s.bucket.isin(["W", "G"]))).sum())
    print(f"  {ct} (n={len(s)}): T+/P+ {tp_rp} | T+/P- {tp_rm} | "
          f"T-/P+ {tm_rp} | T-/P- {tm_rm}  (buckets W={int((s.bucket=='W').sum())} "
          f"G={int((s.bucket=='G').sum())} D={int((s.bucket=='D').sum())} X={int((s.bucket=='X').sum())})")
