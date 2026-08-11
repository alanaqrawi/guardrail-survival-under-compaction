"""D-vs-W(-vs-G) behavioral-replay contrast (v13), existing data only.

The interpretable §4.4 finding is the contrast between degraded residues (D) and textually intact
welded survivors (W) on replay, using W as the control for how often a replay model disobeys an
intact in-context rule. This script reports, per replay model, disarm rate by bucket; the
family-stratified D-vs-W contrast that answers the family-mix confound (an exact conditional
stratified test across the action families with headroom in both buckets, replacing the earlier
unstratified within-family pool; the pooled odds ratio is NOT reported, since six of seven strata
carry a structural zero and any common-OR estimate rests on one P03 observation); and the pooled
textual-severing
denominator (deontic G generalizations whose exact target string is absent, pooled across the Qwen
unmarked, Qwen marked, and Llama runs, with the two-sided 95% Clopper-Pearson upper bound).
Run: .venv/Scripts/python.exe tracer/_replay_contrast.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
except Exception:
    StratifiedTable = None

P = Path(__file__).resolve().parent.parent / "results/between_items_probe"


def load(f):
    return json.loads((P / f).read_text(encoding="utf-8"))


def v(x):
    return x["verdict"].split(" (")[0]


qwen_d = [x for x in load("behavioral_replay_allD_20260602_214506.json") if x["case_type"] == "D-CENSUS"]
qwen_wg = load("behavioral_replay_WG_perobs_20260602_214506.json")
llama = load("behavioral_replay_WGD_llama_20260602_214506.json")
qwen_w = [x for x in qwen_wg if x["case_type"] == "W"]


def rate(rows):
    concl = [x for x in rows if v(x) in ("DISARMED", "STILL ARMED")]
    dis = sum(1 for x in concl if v(x) == "DISARMED")
    return dis, len(concl)


groups = {
    "Qwen D": qwen_d,
    "Qwen W": qwen_w,
    "Qwen G": [x for x in qwen_wg if x["case_type"] == "G"],
    "Llama D": [x for x in llama if x["case_type"] == "D"],
    "Llama W": [x for x in llama if x["case_type"] == "W"],
    "Llama G": [x for x in llama if x["case_type"] == "G"],
}
print("=== disarm rate by bucket (disarmed / conclusive) ===")
for k, rows in groups.items():
    d, n = rate(rows)
    print(f"  {k:<8} {d}/{n} = {100*d/n:.0f}%" if n else f"  {k:<8} -")


def family_tables(drows, wrows):
    fam = defaultdict(lambda: {"D": [0, 0], "W": [0, 0]})
    for x in drows:
        if v(x) in ("DISARMED", "STILL ARMED"):
            fam[x["pair_id"]]["D"][1] += 1
            fam[x["pair_id"]]["D"][0] += v(x) == "DISARMED"
    for x in wrows:
        if v(x) in ("DISARMED", "STILL ARMED"):
            fam[x["pair_id"]]["W"][1] += 1
            fam[x["pair_id"]]["W"][0] += v(x) == "DISARMED"
    return {p: c for p, c in fam.items() if c["D"][1] and c["W"][1]}


shared = family_tables(qwen_d, qwen_w)
print("\n=== Qwen D-vs-W stratified on action family (shared-headroom families) ===")
print(f"  {'family':26} {'D dis/n':>9} {'W dis/n':>9}")
tables = []
for p in sorted(shared):
    c = shared[p]
    Dd, Dn = c["D"]
    Wd, Wn = c["W"]
    print(f"  {p:26} {Dd}/{Dn:<7} {Wd}/{Wn:<7}")
    tables.append(np.array([[Dd, Dn - Dd], [Wd, Wn - Wd]]))

# Exact conditional (stratified) test of association: condition on each stratum's margins
# (a_k ~ hypergeometric), convolve the per-stratum null distributions of sum_k a_k, and read off
# the tail probability at the observed sum. This is the reported test; it does not rely on
# asymptotics and does not depend on the fragile P03 cell the way a pooled OR does.
from scipy.stats import hypergeom, chi2

obs = 0
dist = {0: 1.0}
Emean = Var = 0.0
for t in tables:
    (a, b), (c, d) = t.tolist()
    n = a + b + c + d
    nD = a + b            # D row total
    ndis = a + c          # disarmed column total
    obs += a
    rv = hypergeom(n, nD, ndis)
    lo, hi = max(0, ndis - (n - nD)), min(nD, ndis)
    xs = range(lo, hi + 1)
    Emean += rv.mean()
    Var += rv.var()
    nd = {}
    for s, pv in dist.items():
        for x in xs:
            nd[s + x] = nd.get(s + x, 0.0) + pv * rv.pmf(x)
    dist = nd
tot = sum(dist.values())
p_exact = sum(p for s, p in dist.items() if s >= obs) / tot   # one-sided, D disarms more
print("\n=== family-stratified association (REPORTED test) ===")
print(f"  observed sum a_k (D disarmed) = {obs}; null E[sum] = {Emean:.2f}")
print(f"  exact conditional test p = {p_exact:.2e}   <-- reported")
print(f"  asymptotic CMH p (no cc) = {chi2.sf((obs - Emean) ** 2 / Var, 1):.2e}")
total_concl = rate(qwen_d)[1] + rate(qwen_w)[1]
retained = sum(c["D"][1] + c["W"][1] for c in shared.values())
print(f"  (conclusive obs dropped by shared-headroom restriction: {total_concl - retained})")

if StratifiedTable is not None:
    st = StratifiedTable(tables)
    print("  --- NOT reported (fragile at these counts): six of seven strata have a structural")
    print("      zero, so the pooled OR rests on the single P03 welded observation ---")
    print(f"      Mantel-Haenszel OR = {st.oddsratio_pooled:.2f} (undefined if P03's W outcome flips)")
    print(f"      homogeneity-of-OR test p = {st.test_equal_odds().pvalue:.3f} (detects sparsity, not variation)")

if fisher_exact:
    print("\n=== naive Fisher exact, D vs W (direction only; NOT the reported test) ===")
    qd, qn = rate(qwen_d)
    qw, qwn = rate(qwen_w)
    ld, ln = rate([x for x in llama if x["case_type"] == "D"])
    lw, lwn = rate([x for x in llama if x["case_type"] == "W"])
    _, p_q = fisher_exact([[qd, qn - qd], [qw, qwn - qw]])
    _, p_l = fisher_exact([[ld, ln - ld], [lw, lwn - lw]])
    print(f"  Qwen {qd}/{qn} vs {qw}/{qwn}: p = {p_q:.2e}")
    print(f"  Llama {ld}/{ln} vs {lw}/{lwn}: p = {p_l:.2e}")


# --- pooled textual-severing denominator (label-based, not replay) ---
def deontic_g_target_absent(fname):
    obs = json.loads((P / fname).read_text(encoding="utf-8"))["observations"]
    out = defaultdict(list)  # marking -> list of (item_id, target_present)
    for o in obs:
        if o["content_type"] != "deontic" or o.get("judge_final") != "G":
            continue
        out[o["marking"]].append((o["item_id"], o.get("deterministic_target_present")))
    return out


print("\n=== pooled textual-severing denominator (deontic G, exact target absent) ===")
main_g = deontic_g_target_absent("verdicts_20260602_214506_final.json")
llama_g = deontic_g_target_absent("verdicts_20260610_150139_final.json")
counts = {
    "Qwen unmarked": sum(1 for _, p in main_g["unmarked"] if not p),
    "Qwen marked": sum(1 for _, p in main_g["marked"] if not p),
    "Llama": sum(1 for _, p in (llama_g["unmarked"] + llama_g["marked"]) if not p),
}
for k, n in counts.items():
    print(f"  {k:<14} {n}")
N = sum(counts.values())
cp_upper = 1 - 0.025 ** (1 / N) if N else float("nan")
print(f"  pooled severing opportunities = {N}; observed non-covering = 0")
print(f"  two-sided 95% Clopper-Pearson upper bound on non-covering rate = {100*cp_upper:.0f}%")
