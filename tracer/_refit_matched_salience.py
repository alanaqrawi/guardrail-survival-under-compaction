"""Action 1: re-fit the between-items GEE with the MATCHED (re-authored) salience covariate.

The original GEE loaded salience from run 20260601_234619 = the ORIGINAL pairs, before the
epistemic items were re-authored. The text actually summarized was the re-authored items, whose
salience lives in runs 20260602_213530 (8 pairs) + 20260602_214234 (P09/P10 re-rate).
Re-fit with the matched covariate and confirm beta is stable.
"""
import json
from pathlib import Path
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SL = R / "results/salience_ratings"

def jload(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def getb(o, f):
    r = o.get(f); return r.get("bucket") if isinstance(r, dict) else r

# --- build matched salience per item from re-authored runs (unmarked variant) ---
matched = {}   # (pair_id, content_type) -> salience
for run in ["salience_20260602_213530.json", "salience_20260602_214234.json"]:
    s = jload(SL/run)
    for d in s["details"]:
        if d["variant"] != "unmarked": continue
        if not d.get("admitted"):  # only admitted pairs carry a usable rating
            continue
        matched[(d["pair_id"], "deontic")]   = d["d_mean"]
        matched[(d["pair_id"], "epistemic")] = d["e_mean"]

# original covariate (run 234619) for the diff display
orig = {}
s0 = jload(SL/"salience_20260601_234619.json")
for d in s0["details"]:
    if d["variant"] != "unmarked": continue
    orig[(d["pair_id"], "deontic")]   = d["d_mean"]
    orig[(d["pair_id"], "epistemic")] = d["e_mean"]

print("=== salience covariate: ORIGINAL (234619, used in GEE) vs MATCHED (re-authored) ===")
print(f"  {'item':<32} {'orig':>5} {'matched':>8} {'diff':>6}")
for pid in sorted(set(k[0] for k in matched)):
    for ct in ("deontic","epistemic"):
        o = orig.get((pid,ct)); m = matched.get((pid,ct))
        if o is None or m is None: continue
        flag = "" if abs(o-m) < 1e-9 else "  <-- changed"
        print(f"  {pid+'/'+ct:<32} {o:>5.2f} {m:>8.2f} {m-o:>6.2f}{flag}")

def refit(manfile, field, label):
    man = jload(BI/manfile)
    print(f"\n=== {label} ({manfile}, {field}) ===")
    for marking in ("unmarked","marked"):
        rows=[]
        for o in man["observations"]:
            b=getb(o,field)
            if b not in ("W","G","D","X") or o["marking"]!=marking: continue
            sal_matched = matched.get((o["pair_id"], o["content_type"]))
            if sal_matched is None:  # safety
                sal_matched = o.get("salience_score") or 3.0
            rows.append({"y":1 if b in ("W","G") else 0,
                         "is_deontic":1 if o["content_type"]=="deontic" else 0,
                         "sal_orig":o.get("salience_score") or 3.0,
                         "sal_matched":sal_matched,
                         "pos":o.get("position",0.5), "g":o["conversation_id"]})
        df=pd.DataFrame(rows)
        out={}
        for tag,salcol in [("orig-covariate","sal_orig"),("matched-covariate","sal_matched")]:
            d2 = df.rename(columns={salcol:"sal"})
            res=GEE.from_formula("y ~ is_deontic + sal + pos", groups="g", data=d2, family=Binomial()).fit()
            b=res.params["is_deontic"]; se=res.bse["is_deontic"]; p=res.pvalues["is_deontic"]
            out[tag]=(b,se,p)
        bo=out["orig-covariate"]; bm=out["matched-covariate"]
        print(f"  {marking}:")
        print(f"     orig-covariate    beta={bo[0]:.3f} SE={bo[1]:.3f} p={bo[2]:.2e}")
        print(f"     matched-covariate beta={bm[0]:.3f} SE={bm[1]:.3f} p={bm[2]:.2e}   (delta beta={bm[0]-bo[0]:+.3f})")

refit("verdicts_20260602_214506_final.json", "judge_final", "HUMAN-CANONICAL")
refit("verdicts_20260602_214506_v2.json", "judge_v2", "v4-REPRODUCIBLE")
