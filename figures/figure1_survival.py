"""Figure 1 — Rule vs. fact survival under single-cycle compaction.

Renders two panels COMPUTED from the artifacts (matches PAPER_NUMBERS_FINAL.md):
  (a) Between-items probe, unmarked, author-adjudicated canonical labels
      (judge_final in results/between_items_probe/verdicts_20260602_214506_final.json):
      deontic (rules) 27% (50/185) vs epistemic (facts) 4% (7/175),
      with the incidental-tracer base rate (0-3%) as reference — the per-cell
      mean tracer survival across the pressured cells of the stress probe.
      Error bars are the conversation cluster-bootstrap 95% CIs from
      tracer/_reanalysis_v7_clustered.py (deontic [17, 38], epistemic [1, 7]).
  (b) Single-rule stress probe (results/stress_probe/verdicts_20260531_231543.json),
      per-cell rule-referent survival (deterministic.target_present) under
      confirmed compression pressure: Llama turn-0 50%, Qwen mid 50%,
      Qwen turn-0 38%, Llama mid 0% (rule died at the base rate in the
      hardest-compression cell). Error bars are per-cell Wilson 95% CIs (n=8).

Outputs figures/figure1_survival.png (300 dpi) and figures/figure1_survival.pdf.
Run: .venv/Scripts/python.exe figures/figure1_survival.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent

# ---- values computed from artifacts ---------------------------------------
def whole(x):  # display convention: whole percents, half rounds up
    return int(x + 0.5)

def survived(bucket):
    return bucket in ("W", "G")

def wilson(k, n, z=1.96):
    """Wilson score 95% interval, returned as (lo%, hi%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))

# Panel (a) conversation cluster-bootstrap 95% CIs (percent), from
# tracer/_reanalysis_v7_clustered.py (resamples whole conversations, seed 7).
CI_A = {"deontic": (17.0, 38.0), "epistemic": (1.0, 7.0)}

# (a) between-items probe: unmarked survival by content type, canonical labels
final = json.loads((ROOT / "results/between_items_probe/verdicts_20260602_214506_final.json")
                   .read_text(encoding="utf-8"))
ab = {}
for ct in ("deontic", "epistemic"):
    sub = [o for o in final["observations"]
           if o["marking"] == "unmarked" and o["content_type"] == ct
           and o.get("judge_final") in ("W", "G", "D", "X")]
    ab[ct] = (sum(1 for o in sub if survived(o["judge_final"])), len(sub))
d_s, d_n = ab["deontic"]
e_s, e_n = ab["epistemic"]

# stress probe: pressured cells = per-cell mean tracer survival <= threshold
stress = json.loads((ROOT / "results/stress_probe/verdicts_20260531_231543.json")
                    .read_text(encoding="utf-8"))
tracer_mean, referent = {}, {}          # per pressured cell, in percent
ref_k, ref_n = {}, {}                    # per pressured cell, raw counts (for Wilson)
for name, cell in stress["cells"].items():
    recs = cell["records"]
    t = 100.0 * sum(r["tracers"]["rate"] for r in recs) / len(recs)
    if t > 100.0 * stress["pressure_threshold"]:
        continue  # compression pressure not confirmed (claude cells at 45k)
    tracer_mean[name] = t
    k = sum(1 for r in recs if r["deterministic"]["target_present"])
    ref_k[name], ref_n[name] = k, len(recs)
    referent[name] = 100.0 * k / len(recs)

base_lo = whole(min(tracer_mean.values()))
base_hi = whole(max(tracer_mean.values()))

print(f"(a) deontic {d_s}/{d_n} ({whole(100 * d_s / d_n)}%)   "
      f"epistemic {e_s}/{e_n} ({whole(100 * e_s / e_n)}%)   "
      f"tracer base {base_lo}-{base_hi}%")
print("(b) " + "   ".join(f"{n} {ref_k[n]}/{ref_n[n]} {referent[n]:.1f}% "
                          f"CI{tuple(round(x) for x in wilson(ref_k[n], ref_n[n]))}"
                          for n in referent))

# ---- colours -------------------------------------------------------------
C_RULE = "#1f4e79"   # deep blue  (normative / rules)
C_FACT = "#c0504d"   # muted red  (epistemic / facts)
C_BASE = "#9e9e9e"   # grey       (incidental base rate)
C_ERR = "#333333"    # error-bar ink

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25]}
)

# ===== Panel A: between-items rule vs fact ================================
# bars drawn at the whole-percent values shown in the annotations (paper convention);
# tracer base plotted at the band midpoint; error bars = conversation cluster bootstrap
pd_val = float(whole(100 * d_s / d_n))
pe_val = float(whole(100 * e_s / e_n))
labelsA = ["Rules\n(deontic)", "Facts\n(epistemic)", "Incidental\ntracer base"]
valsA = [pd_val, pe_val, (base_lo + base_hi) / 2]
colorsA = [C_RULE, C_FACT, C_BASE]

barsA = axA.bar(labelsA, valsA, color=colorsA, width=0.62, edgecolor="white", zorder=2)

# asymmetric error bars for the two estimated rates (base band has no CI)
lo_d, hi_d = CI_A["deontic"]
lo_e, hi_e = CI_A["epistemic"]
axA.errorbar([0, 1], [pd_val, pe_val],
             yerr=[[pd_val - lo_d, pe_val - lo_e], [hi_d - pd_val, hi_e - pe_val]],
             fmt="none", ecolor=C_ERR, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)

annotA = [(0, hi_d, f"{whole(100 * d_s / d_n)}%\n({d_s}/{d_n})"),
          (1, hi_e, f"{whole(100 * e_s / e_n)}%\n({e_s}/{e_n})"),
          (2, valsA[2], f"{base_lo}–{base_hi}%")]
for x, top, txt in annotA:
    axA.text(x, top + 1.4, txt, ha="center", va="bottom", fontsize=9, linespacing=1.0)

axA.set_ylim(0, 46)
axA.set_ylabel("Survival rate (%)")
axA.set_title("(a) Between-items probe\n(unmarked, canonical labels)", fontsize=10)
axA.axhline(base_hi, color=C_BASE, lw=0.8, ls=":", zorder=0)
axA.margins(x=0.08)

# ===== Panel B: stress-probe per-cell rule-referent survival =============
PLACEMENT_LABEL = {"turn0_explicit": "turn-0", "mid_casual": "mid"}
orderB = sorted(referent, key=lambda n: -referent[n])  # stable: ties keep cell order
labelsB = [f"{stress['cells'][n]['model'].capitalize()}\n"
           f"{PLACEMENT_LABEL[stress['cells'][n]['placement']]}" for n in orderB]
valsB = [float(whole(referent[n])) for n in orderB]
ciB = [wilson(ref_k[n], ref_n[n]) for n in orderB]

barsB = axB.bar(labelsB, valsB, color=C_RULE, width=0.62, edgecolor="white", zorder=2)
yerrB = [[max(0.0, v - lo) for v, (lo, hi) in zip(valsB, ciB)],
         [max(0.0, hi - v) for v, (lo, hi) in zip(valsB, ciB)]]
axB.errorbar(range(len(valsB)), valsB, yerr=yerrB, fmt="none",
             ecolor=C_ERR, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
for i, (n, (lo, hi)) in enumerate(zip(orderB, ciB)):
    axB.text(i, hi + 1.8, f"{whole(referent[n])}%", ha="center", va="bottom", fontsize=9)

# shaded incidental-fact base band
axB.axhspan(base_lo, base_hi, color=C_BASE, alpha=0.25, zorder=0)
axB.text(0.98, 0.98, f"shaded band: incidental-fact\nbase rate ({base_lo}–{base_hi}%)",
         transform=axB.transAxes, ha="right", va="top", fontsize=7.5,
         color="#5a5a5a", linespacing=1.1)

axB.set_ylim(0, 88)
axB.set_ylabel("Rule-referent presence (%)")
axB.set_title("(b) Single-rule stress probe\n(per cell, confirmed pressure)", fontsize=10)
axB.margins(x=0.06)

legend_handles = [
    Patch(facecolor=C_RULE, label="Rules (a: W+G survival; b: referent presence)"),
    Patch(facecolor=C_FACT, label="Facts (a: W+G survival)"),
    Patch(facecolor=C_BASE, label="Incidental-tracer base rate"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.06, "Error bars: 95% CI (panel a: conversation cluster bootstrap; "
         "panel b: Wilson, n=8 per cell).",
         ha="center", va="top", fontsize=7.5, color="#5a5a5a")

fig.suptitle("Rule vs. fact survival under single-cycle compaction",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.tight_layout(rect=(0, 0.05, 1, 1))

png = OUT / "figure1_survival.png"
pdf = OUT / "figure1_survival.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"wrote {png}")
print(f"wrote {pdf}")
