"""Apply the human reviewer's corrections to the marked epistemic-G bucket,
fix the rubric error uniformly (epistemic + target-absent => not survived),
re-fit GEE for both conditions, and report agreement honestly.

Reviewer's 10 hand labels (marked epistemic G cases):
  X = inference-from-siblings, target absent (cases 1,6,9) — rubric error
  D = target in a generic list, no specific fact (cases 5,7) — boundary, not survived
  W/G = genuine survival (cases 2,3,4,8,10)
"""
import json
from pathlib import Path

import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results/between_items_probe/verdicts_20260602_214506_v2.json"
man = json.loads(P.read_text(encoding="utf-8"))

# Reviewer's authoritative labels, keyed by (item_id, conversation_id)
REVIEW = {
    ("P01_db_destructive_epistemic", "qwen__marked__conv016"): "X",
    ("P01_db_destructive_epistemic", "qwen__marked__conv017"): "W",
    ("P01_db_destructive_epistemic", "qwen__marked__conv020"): "G",
    ("P01_db_destructive_epistemic", "qwen__marked__conv021"): "G",
    ("P02_vendor_pricing_epistemic", "qwen__marked__conv027"): "D",
    ("P01_db_destructive_epistemic", "qwen__marked__conv027"): "X",
    ("P04_branch_pci_epistemic",     "qwen__marked__conv032"): "D",
    ("P09_person_comp_epistemic",    "qwen__marked__conv033"): "G",
    ("P01_db_destructive_epistemic", "qwen__marked__conv039"): "X",
    ("P06_credential_rotate_epistemic", "qwen__marked__conv040"): "W",
}

def survived(b): return b in ("W", "G")

# 1. Build judge_final = judge_v2, then apply review labels + uniform rubric fix
n_review, n_rubric = 0, 0
for o in man["observations"]:
    b = o.get("judge_v2", {}).get("bucket")
    if b not in ("W", "G", "D", "X"):
        o["judge_final"] = b
        continue
    final = b
    key = (o["item_id"], o["conversation_id"])
    if key in REVIEW:
        final = REVIEW[key]
        n_review += 1
    elif (o["content_type"] == "epistemic"
          and b in ("W", "G")
          and o.get("deterministic_target_present") is False):
        # uniform rubric fix: epistemic fact whose target is absent did not survive
        final = "X"
        n_rubric += 1
    o["judge_final"] = final

print(f"Applied {n_review} reviewer labels + {n_rubric} uniform rubric fixes (epistemic target-absent -> X)")

# 2. Agreement on the 10-case marked-epistemic-G bucket (judge_v2 said all G=survived)
#    reviewer: 5 survived (2,3,4,8,10), 5 not (1,5,6,7,9)
agree = sum(1 for (iid, cid), lbl in REVIEW.items()
            if survived("G") == survived(lbl))  # judge_v2 said G(survived) for all 10
print(f"\nMarked epistemic-G bucket agreement (judge_v2 G=survived vs reviewer): {agree}/10 = {agree*10}%")
print("  -> below 90% gate: the v3 marked epistemic-G labels do NOT count as reported.")

# 3. Corrected survival tables + GEE, per marking, both content types
rows = []
for o in man["observations"]:
    b = o.get("judge_final")
    if b not in ("W", "G", "D", "X") or o.get("error"):
        continue
    rows.append({
        "model": o["model"], "marking": o["marking"], "content_type": o["content_type"],
        "survival": 1 if survived(b) else 0,
        "is_deontic": 1 if o["content_type"] == "deontic" else 0,
        "salience": o.get("salience_score") or 3.0,
        "position": o.get("position", 0.5),
        "conv_id": o["conversation_id"], "bucket": b,
    })
df = pd.DataFrame(rows)

print("\n=== CORRECTED SURVIVAL (judge_final; marked/unmarked separate) ===")
for marking in ("unmarked", "marked"):
    for ct in ("deontic", "epistemic"):
        sub = df[(df.marking == marking) & (df.content_type == ct)]
        n = len(sub); s = int(sub.survival.sum())
        print(f"  {marking:<9} {ct:<10} n={n:>3}  survived={s:>3} ({s/n:.0%})")

print("\n=== GEE RE-FIT (corrected labels) ===")
for marking in ("unmarked", "marked"):
    sub = df[df.marking == marking].copy()
    nd = int(sub.is_deontic.sum()); ne = len(sub) - nd
    sd = int(sub[sub.is_deontic == 1].survival.sum())
    se_ = int(sub[sub.is_deontic == 0].survival.sum())
    print(f"\n  [{marking}] deontic survived={sd}/{nd}, epistemic survived={se_}/{ne}")
    if sub.survival.nunique() < 2:
        print("    degenerate — cannot fit"); continue
    try:
        res = GEE.from_formula("survival ~ is_deontic + salience + position",
                               groups="conv_id", data=sub, family=Binomial()).fit()
        beta = res.params["is_deontic"]; bse = res.bse["is_deontic"]; p = res.pvalues["is_deontic"]
        lo, hi = beta - 1.96*bse, beta + 1.96*bse
        flag = ""
        if bse > 5:
            flag = "  <<< SE huge -> near-separation, NOT trustworthy"
        elif ne > 0 and se_ <= 8:
            flag = f"  <<< only {se_} epistemic survivals -> fragile fit"
        print(f"    beta={beta:.3f}  SE={bse:.3f}  p={p:.4f}  95%CI=[{lo:.3f},{hi:.3f}]{flag}")
    except Exception as e:
        print(f"    fit failed: {type(e).__name__}: {e}")

# persist corrected verdicts
out = ROOT / "results/between_items_probe/verdicts_20260602_214506_final.json"
out.write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"\nSaved corrected verdicts -> {out.name}")
