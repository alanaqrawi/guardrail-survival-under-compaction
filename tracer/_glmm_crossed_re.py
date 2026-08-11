"""Crossed-RE binomial GLMM (the spec model):
   survival ~ is_deontic + salience + position + (1|conversation) + (1|target)

Pre-committed: ONE specification, ONE solver (variational Bayes), report whatever it gives —
including non-significant or non-converged. No solver-hunting.
Uses the MATCHED (re-authored) salience covariate (now canonical). Anchor = unmarked.
Reports posterior mean (beta), posterior SD, 95% credible interval, normal-approx p,
and the random-effect SDs (how much conversation- and target-clustering there actually is).
GEE-on-conversation printed alongside for comparison.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SL = R / "results/salience_ratings"
def jload(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def getb(o, f):
    r = o.get(f); return r.get("bucket") if isinstance(r, dict) else r

# matched (re-authored) salience covariate
matched = {}
for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
    for d in jload(SL/run)["details"]:
        if d["variant"] == "unmarked" and d.get("admitted"):
            matched[(d["pair_id"], "deontic")] = d["d_mean"]
            matched[(d["pair_id"], "epistemic")] = d["e_mean"]

def build_df(manfile, field, marking):
    man = jload(BI/manfile); rows=[]
    for o in man["observations"]:
        b=getb(o,field)
        if b not in ("W","G","D","X") or o["marking"]!=marking: continue
        rows.append({"y":1 if b in ("W","G") else 0,
                     "is_deontic":1 if o["content_type"]=="deontic" else 0,
                     "salience":matched.get((o["pair_id"],o["content_type"]), o.get("salience_score") or 3.0),
                     "position":o.get("position",0.5),
                     "conv_id":str(o["conversation_id"]), "target":str(o["target"])})
    return pd.DataFrame(rows)

def gee_beta(df):
    res=GEE.from_formula("y ~ is_deontic + salience + position", groups="conv_id",
                         data=df, family=Binomial()).fit()
    b=res.params["is_deontic"]; se=res.bse["is_deontic"]; p=res.pvalues["is_deontic"]
    return b,se,p

def glmm_fit(df):
    # crossed random intercepts for conversation and target
    vc = {"conv": "0 + C(conv_id)", "target": "0 + C(target)"}
    model = BinomialBayesMixedGLM.from_formula("y ~ is_deontic + salience + position", vc, df)
    res = model.fit_vb()
    names = list(res.model.exog_names)            # fixed-effect names
    i = names.index("is_deontic")
    bmean = res.fe_mean[i]; bsd = res.fe_sd[i]
    z = bmean/bsd; p = 2*(1-norm.cdf(abs(z)))
    ci = (bmean-1.96*bsd, bmean+1.96*bsd)
    # random-effect posterior SDs (vcp_mean are log-sd params); expose raw
    vcp = dict(zip(res.model.vcp_names, res.vcp_mean))
    return bmean, bsd, p, ci, vcp

print("Crossed-RE binomial GLMM  vs  GEE-on-conversation   (matched salience covariate)")
print("Spec: survival ~ is_deontic + salience + position + (1|conversation) + (1|target)")
print("="*82)
for manfile, field, label in [("verdicts_20260602_214506_final.json","judge_final","HUMAN-CANONICAL"),
                              ("verdicts_20260602_214506_v2.json","judge_v2","v4-REPRODUCIBLE")]:
    for marking in ("unmarked","marked"):
        df = build_df(manfile, field, marking)
        n=len(df); nd=int(df.is_deontic.sum())
        print(f"\n--- {label} / {marking}  (n={n}, deontic={nd}, epistemic={n-nd}, "
              f"targets={df.target.nunique()}, convs={df.conv_id.nunique()}) ---")
        try:
            gb,gse,gp = gee_beta(df)
            print(f"  GEE  (conv only): beta={gb:.3f} SE={gse:.3f} p={gp:.2e}")
        except Exception as e:
            print(f"  GEE failed: {e}")
        try:
            bmean,bsd,p,ci,vcp = glmm_fit(df)
            anchor = "  <<< ANCHOR" if (marking=="unmarked" and label=="HUMAN-CANONICAL") else ""
            print(f"  GLMM (conv+target): beta={bmean:.3f} postSD={bsd:.3f} p~={p:.2e} "
                  f"95%CrI=[{ci[0]:.3f},{ci[1]:.3f}]{anchor}")
            print(f"       RE variance params (log-sd): {', '.join(f'{k}={v:.3f}' for k,v in vcp.items())}")
        except Exception as e:
            print(f"  GLMM DID NOT CONVERGE / failed: {type(e).__name__}: {e}")
