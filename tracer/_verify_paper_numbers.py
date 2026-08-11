"""INDEPENDENT verification of the paper numbers.
Different counting method than _extract_paper_numbers.py (raw Counter loops, separate GEE fit,
denominator audits). Prints MATCH/MISMATCH against the reported values.
"""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SP = R / "results/stress_probe"

def jload(p): return json.loads(Path(p).read_text(encoding="utf-8"))

final = jload(BI/"verdicts_20260602_214506_final.json")
v4    = jload(BI/"verdicts_20260602_214506_v2.json")
v3    = jload(BI/"verdicts_20260602_214506_v3labels.json")
v1    = jload(BI/"verdicts_20260602_214506.json")

def getb(o, field):
    r = o.get(field)
    return r.get("bucket") if isinstance(r, dict) else r

# ---- denominator audit (this is where 172 vs 173 lives) ----
print("=== DENOMINATOR AUDIT (total obs per marking/content_type, and how many lack a valid bucket) ===")
for tag, man, field in [("human/judge_final", final, "judge_final"),
                         ("v4/judge_v2", v4, "judge_v2")]:
    print(f"  {tag}:")
    for marking in ("unmarked","marked"):
        for ct in ("deontic","epistemic"):
            allo = [o for o in man["observations"] if o["marking"]==marking and o["content_type"]==ct]
            valid = [o for o in allo if getb(o,field) in ("W","G","D","X")]
            bad = len(allo)-len(valid)
            print(f"    {marking}/{ct}: total={len(allo)}  valid={len(valid)}  no-bucket={bad}")

# ---- survival counts via raw Counter ----
print("\n=== SURVIVAL (raw Counter) ===")
def surv_count(man, field, marking, ct):
    c = Counter(getb(o,field) for o in man["observations"]
                if o["marking"]==marking and o["content_type"]==ct and getb(o,field) in ("W","G","D","X"))
    n = sum(c.values()); s = c["W"]+c["G"]
    return s, n, c
for tag, man, field in [("human", final, "judge_final"), ("v4", v4, "judge_v2")]:
    for marking in ("unmarked","marked"):
        for ct in ("deontic","epistemic"):
            s,n,c = surv_count(man,field,marking,ct)
            print(f"  {tag} {marking}/{ct}: {s}/{n}  ({c['W']}W {c['G']}G {c['D']}D {c['X']}X)")

# ---- GEE independent re-fit ----
print("\n=== GEE (independent re-fit) ===")
def fit(man, field, marking):
    rows=[]
    for o in man["observations"]:
        b=getb(o,field)
        if b not in ("W","G","D","X") or o["marking"]!=marking: continue
        rows.append({"y":1 if b in ("W","G") else 0,
                     "is_deontic":1 if o["content_type"]=="deontic" else 0,
                     "sal":o.get("salience_score") or 3.0, "pos":o.get("position",0.5),
                     "g":o["conversation_id"]})
    df=pd.DataFrame(rows)
    res=GEE.from_formula("y ~ is_deontic + sal + pos", groups="g", data=df, family=Binomial()).fit()
    return res.params["is_deontic"], res.bse["is_deontic"], res.pvalues["is_deontic"]
for tag, man, field in [("human", final, "judge_final"), ("v4", v4, "judge_v2")]:
    for marking in ("unmarked","marked"):
        b,se,p = fit(man,field,marking)
        print(f"  {tag} {marking}: beta={b:.3f} SE={se:.3f} p={p:.2e} CI=[{b-1.96*se:.3f},{b+1.96*se:.3f}]")

# ---- progression of marked epistemic survival ----
print("\n=== MARKED EPISTEMIC survival progression ===")
for tag, man, field in [("v1", v1, "judge"), ("v3", v3, "judge_v2"), ("v4", v4, "judge_v2"), ("human", final, "judge_final")]:
    s,n,c = surv_count(man,field,"marked","epistemic")
    print(f"  {tag}: {s}/{n}  ({c['W']}W {c['G']}G {c['D']}D {c['X']}X)")

# ---- agreements (independent recompute on the 20 adjudicated cases) ----
print("\n=== AGREEMENTS (recompute) ===")
import random
def adjudicated_cases(man):
    bg=[o for o in man["observations"] if getb(o,"judge")=="G" and o.get("deterministic_target_present") is True and o.get("digest_file")]
    ed=[o for o in man["observations"] if getb(o,"judge")=="D" and o["content_type"]=="epistemic" and o.get("deterministic_target_present") is False and o.get("digest_file")]
    rng=random.Random(42); eds=rng.sample(ed,min(5,len(ed)))
    return bg+eds
HUMAN = {1:1,2:1,3:1,4:1,5:1,6:1,7:0,8:1,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:0,17:0,18:0,19:0,20:0}
def agree(man, field):
    cases = adjudicated_cases(man)[:20]
    a=0
    for i,o in enumerate(cases,1):
        b=getb(o,field)
        sv = 1 if b in ("W","G") else 0
        if sv == HUMAN[i]: a+=1
    return a
print(f"  v3 vs human: {agree(v3,'judge_v2')}/20")
print(f"  v4 vs human: {agree(v4,'judge_v2')}/20")

# epistemic-G bucket agreement (the 10 marked epistemic-G from v3)
v3_marked_eg = [o for o in v3["observations"] if o["content_type"]=="epistemic" and o["marking"]=="marked" and getb(o,"judge_v2")=="G"]
print(f"  v3 marked epistemic-G bucket size: {len(v3_marked_eg)} (judge said all G=survived)")

# ---- deontic moved under inference fix ----
moved = [o for o in final["observations"] if o["content_type"]=="deontic" and getb(o,"judge_v2")!=o.get("judge_final")]
print(f"\n=== deontic moved under presence-not-inference fix: {len(moved)} ===")

# ---- behavioral replay ----
rep = jload(BI/"behavioral_replay_20260602_214506.json")
vc = Counter("DISARMED" if "DISARMED" in r["verdict"] else "ARMED" if "STILL ARMED" in r["verdict"] else "INCONCLUSIVE" for r in rep)
print(f"\n=== replay: {dict(vc)}  (total {len(rep)}) ===")

# ---- stress probe 45k ----
print("\n=== STRESS 45k (independent) ===")
sp = jload(SP/"verdicts_20260531_231543.json")
for cid, blob in sp["cells"].items():
    recs=[r for r in blob["records"] if r.get("judge",{}).get("bucket") in ("W","G","D","X")]
    c=Counter(r["judge"]["bucket"] for r in recs)
    tr=sum(r["tracers"]["rate"] for r in recs)/len(recs)
    ref=sum(1 for r in recs if r.get("deterministic",{}).get("target_present"))/len(recs)
    print(f"  {cid:<24} {c['W']}W {c['G']}G {c['D']}D {c['X']}X  tracer={tr*100:.0f}% ref={ref*100:.0f}%")
