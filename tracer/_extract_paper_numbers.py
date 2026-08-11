"""Extract all paper numbers (sections A-F) from source files. Real values only."""
import json, glob, os
from pathlib import Path
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

R = Path(__file__).resolve().parent.parent
BI = R / "results/between_items_probe"
SP = R / "results/stress_probe"
SL = R / "results/salience_ratings"

def jload(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def surv(b): return b in ("W", "G")

def bucket_of(o, field):
    raw = o.get(field)
    return raw.get("bucket") if isinstance(raw, dict) else raw

def cell_counts(obs, field):
    """Return dict[(marking,ct)] -> {W,G,D,X,n,surv}."""
    out = {}
    for o in obs:
        b = bucket_of(o, field)
        if b not in ("W","G","D","X"): continue
        k = (o["marking"], o["content_type"])
        d = out.setdefault(k, {"W":0,"G":0,"D":0,"X":0})
        d[b]+=1
    for k,d in out.items():
        d["n"]=d["W"]+d["G"]+d["D"]+d["X"]; d["surv"]=d["W"]+d["G"]
    return out

def gee(obs, field, marking):
    rows=[]
    for o in obs:
        b=bucket_of(o,field)
        if b not in ("W","G","D","X") or o["marking"]!=marking: continue
        rows.append({"survival":1 if surv(b) else 0,
                     "is_deontic":1 if o["content_type"]=="deontic" else 0,
                     "salience":o.get("salience_score") or 3.0,
                     "position":o.get("position",0.5),
                     "conv_id":o["conversation_id"]})
    df=pd.DataFrame(rows)
    if df.survival.nunique()<2: return None
    res=GEE.from_formula("survival ~ is_deontic + salience + position",
                         groups="conv_id",data=df,family=Binomial()).fit()
    b=res.params["is_deontic"]; se=res.bse["is_deontic"]; p=res.pvalues["is_deontic"]
    return b,se,p,(b-1.96*se,b+1.96*se)

final=jload(BI/"verdicts_20260602_214506_final.json")
v4=jload(BI/"verdicts_20260602_214506_v2.json")
v3=jload(BI/"verdicts_20260602_214506_v3labels.json")
v1=jload(BI/"verdicts_20260602_214506.json")

print("#"*70); print("# SECTION A — GEE + survival (both label sets)"); print("#"*70)
for tag,man,field in [("HUMAN-CANONICAL (judge_final)",final,"judge_final"),
                      ("v4-REPRODUCIBLE (judge_v2)",v4,"judge_v2")]:
    print(f"\n--- {tag} ---")
    cc=cell_counts(man["observations"],field)
    for marking in ("unmarked","marked"):
        g=gee(man["observations"],field,marking)
        d=cc[(marking,"deontic")]; e=cc[(marking,"epistemic")]
        print(f"  {marking}: deontic {d['surv']}/{d['n']} ({d['surv']/d['n']*100:.0f}%)  "
              f"epistemic {e['surv']}/{e['n']} ({e['surv']/e['n']*100:.0f}%)")
        if g: print(f"     GEE beta={g[0]:.3f} SE={g[1]:.3f} p={g[2]:.2e} CI=[{g[3][0]:.3f},{g[3][1]:.3f}]")

print("\n"+"#"*70); print("# SECTION B — W/G/D/X cells (HUMAN-CANONICAL judge_final)"); print("#"*70)
cc=cell_counts(final["observations"],"judge_final")
print(f"  {'cell':<24} {'W':>3}{'G':>3}{'D':>3}{'X':>3}  {'n':>4} {'surv%':>6}")
for marking in ("unmarked","marked"):
    for ct in ("deontic","epistemic"):
        d=cc[(marking,ct)]
        print(f"  {marking+'/'+ct:<24} {d['W']:>3}{d['G']:>3}{d['D']:>3}{d['X']:>3}  {d['n']:>4} {d['surv']/d['n']*100:>5.0f}%")
print("  (epistemic X = 'facts mostly drop' claim: see X column above)")

print("\n"+"#"*70); print("# Progression of MARKED EPISTEMIC survival (the inflation story)"); print("#"*70)
for tag,man,field in [("v1 original judge",v1,"judge"),("v3 'clearly implied'",v3,"judge_v2"),
                      ("v4 presence-not-inference",v4,"judge_v2"),("human-adjudicated",final,"judge_final")]:
    cc=cell_counts(man["observations"],field)
    me=cc[("marked","epistemic")]
    print(f"  {tag:<28}: marked epistemic survived = {me['surv']}/{me['n']}  (W{me['W']} G{me['G']} D{me['D']} X{me['X']})")

print("\n"+"#"*70); print("# SECTION C — replay, judge-corrections, salience, deontic-unmoved"); print("#"*70)
rep=jload(BI/"behavioral_replay_20260602_214506.json")
disarmed=[r for r in rep if "DISARMED" in r["verdict"]]
armed=[r for r in rep if "STILL ARMED" in r["verdict"]]
incon=[r for r in rep if "INCONCLUSIVE" in r["verdict"]]
print(f"  C9 replay: {len(rep)} tested | disarmed={len(disarmed)} armed={len(armed)} inconclusive={len(incon)}")
for r in rep: print(f"      case{r['case_num']} [{r['case_type']}] {r['item_id']}: {r['verdict']}")

# deontic unmoved under inference fix
moved=[o for o in final["observations"] if o["content_type"]=="deontic"
       and bucket_of(o,"judge_v2")!=o.get("judge_final")]
print(f"\n  C12 deontic labels moved under presence-not-inference fix: {len(moved)}")

# salience gate (re-authored pairs: 213530 + 214234 re-rate)
print("\n  C11 salience gate (re-authored pairs):")
for run in ["salience_20260602_213530.json","salience_20260602_214234.json"]:
    s=jload(SL/run)
    print(f"    {run}: admitted={s['admitted']}  dropped={s['dropped']}")
    for det in s["details"]:
        if det["variant"]=="unmarked":
            print(f"       {det['pair_id']:<28} D={det['d_mean']} E={det['e_mean']} gap={det['gap']} admit={det['admitted']}")

print("\n"+"#"*70); print("# SECTION D/E — STRESS PROBE 45k (run 20260531_231543)"); print("#"*70)
sp=jload(SP/"verdicts_20260531_231543.json")
print(f"  {'cell':<24} {'W':>3}{'G':>3}{'D':>3}{'X':>3}  {'tracer%':>8} {'referent%':>9}")
for cid,blob in sp["cells"].items():
    recs=[r for r in blob["records"] if "judge" in r and r["judge"].get("bucket") in ("W","G","D","X")]
    if not recs: continue
    W=sum(1 for r in recs if r["judge"]["bucket"]=="W"); G=sum(1 for r in recs if r["judge"]["bucket"]=="G")
    D=sum(1 for r in recs if r["judge"]["bucket"]=="D"); X=sum(1 for r in recs if r["judge"]["bucket"]=="X")
    tr=sum(r["tracers"]["rate"] for r in recs)/len(recs)
    ref=sum(1 for r in recs if r.get("deterministic",{}).get("target_present"))/len(recs)
    print(f"  {cid:<24} {W:>3}{G:>3}{D:>3}{X:>3}  {tr*100:>7.0f}% {ref*100:>8.0f}%")

print("\n  Claude 150k (run 20260601_230516) tracer survival + digest lengths:")
c150=jload(SP/"verdicts_20260601_230516.json")
for cid,blob in c150["cells"].items():
    recs=[r for r in blob["records"] if "tracers" in r]
    tr=[r["tracers"]["rate"] for r in recs]
    print(f"    {cid}: tracer mean={sum(tr)/len(tr)*100:.0f}% range={min(tr)*100:.0f}-{max(tr)*100:.0f}%")

# digest char lengths
def charlens(dirpath, prefix):
    fs=glob.glob(str(dirpath/f"{prefix}*.txt"))
    L=[len(Path(f).read_text(encoding='utf-8')) for f in fs]
    return (min(L),max(L),len(L)) if L else (0,0,0)
d45=SP/"digests/20260531_231543"; d150=SP/"digests/20260601_230516"
print("\n  E19 digest CHAR lengths (chars; ~/4 = tokens):")
for m in ["llama","qwen","claude"]:
    lo,hi,n=charlens(d45,m); print(f"    45k {m:<7}: {lo}-{hi} chars  (n={n})")
lo,hi,n=charlens(d150,"claude"); print(f"    150k claude : {lo}-{hi} chars (n={n})")

# the outlier
out=d150/"claude__turn0_explicit__01.txt"
if out.exists():
    txt=out.read_text(encoding="utf-8")
    # find the rule bullet
    rule_line=[l for l in txt.splitlines() if "prod-east-2" in l]
    print(f"\n  E21 outlier claude__turn0_explicit__01.txt: {len(txt)} chars")
    print(f"       rule line: {rule_line[0].strip() if rule_line else '(not found)'}")

print("\n"+"#"*70); print("# SECTION F — metadata"); print("#"*70)
env=(R/".env").read_text(encoding="utf-8")
for line in env.splitlines():
    if any(line.startswith(k) for k in ["CLAUDE_MODEL","TOGETHER_MODEL_1","TOGETHER_MODEL_2","QWEN_ENABLE_THINKING","FILLER_TOKEN_BUDGET"]):
        print("  "+line.strip())
print(f"  between-items: n_conversations={final.get('n_conversations')} items_per_conv={final.get('items_per_conv')} budget={final.get('budget')}")
print(f"  tracer count = 18 (TRACERS); stress filler 45k main / 150k claude")
