"""Side-by-side: v4-regenerated judge labels vs human-adjudicated canonical labels.
Reproduction check — NOT a tuning pass. We report both and note the residual gap.
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial

ROOT = Path(__file__).resolve().parent.parent
v4 = json.loads((ROOT / "results/between_items_probe/verdicts_20260602_214506_v2.json").read_text(encoding="utf-8"))      # fresh v4 judge labels (judge_v2)
hu = json.loads((ROOT / "results/between_items_probe/verdicts_20260602_214506_final.json").read_text(encoding="utf-8"))   # human-adjudicated (judge_final)

def survived(b): return b in ("W", "G")

def counts_and_gee(observations, label_field, tag):
    rows = []
    for o in observations:
        raw = o.get(label_field)
        b = raw.get("bucket") if isinstance(raw, dict) else raw  # judge_v2 is dict, judge_final is str
        if b not in ("W", "G", "D", "X"):
            continue
        rows.append({
            "marking": o["marking"], "content_type": o["content_type"],
            "survival": 1 if survived(b) else 0,
            "is_deontic": 1 if o["content_type"] == "deontic" else 0,
            "salience": o.get("salience_score") or 3.0,
            "position": o.get("position", 0.5),
            "conv_id": o["conversation_id"],
        })
    df = pd.DataFrame(rows)
    print(f"\n===== {tag} =====")
    for marking in ("unmarked", "marked"):
        line = []
        for ct in ("deontic", "epistemic"):
            sub = df[(df.marking == marking) & (df.content_type == ct)]
            n = len(sub); s = int(sub.survival.sum())
            line.append(f"{ct} {s}/{n} ({s/n:.0%})")
        print(f"  {marking:<9}: " + "   ".join(line))
        sub = df[df.marking == marking]
        if sub.survival.nunique() < 2:
            print(f"             GEE: degenerate"); continue
        try:
            res = GEE.from_formula("survival ~ is_deontic + salience + position",
                                   groups="conv_id", data=sub, family=Binomial()).fit()
            b_ = res.params["is_deontic"]; se = res.bse["is_deontic"]; p = res.pvalues["is_deontic"]
            print(f"             GEE deontic-vs-epistemic: beta={b_:.3f} SE={se:.3f} p={p:.4f} "
                  f"CI=[{b_-1.96*se:.2f},{b_+1.96*se:.2f}]")
        except Exception as e:
            print(f"             GEE failed: {e}")

counts_and_gee(v4["observations"], "judge_v2", "v4 JUDGE (regenerated end-to-end; no override, no human corr)")
counts_and_gee(hu["observations"], "judge_final", "HUMAN-ADJUDICATED CANONICAL (v3 + human + behavioral override)")

# residual disagreements on the 20 adjudicated cases (v4 vs human) — computed
# from the artifacts, not hard-coded. Case list reconstructed exactly as in
# _make_blind_read.py: the 15 borderline-G cases in manifest order, then the
# 5 seeded SAMPLE-E-D cases (random seed 42).
man = json.loads((ROOT / "results/between_items_probe/verdicts_20260602_214506.json").read_text(encoding="utf-8"))
borderline_g = [o for o in man["observations"]
                if o.get("judge", {}).get("bucket") == "G"
                and o.get("deterministic_target_present") is True
                and o.get("digest_file")]
edee = [o for o in man["observations"]
        if o.get("judge", {}).get("bucket") == "D"
        and o["content_type"] == "epistemic"
        and o.get("deterministic_target_present") is False
        and o.get("digest_file")]
cases = borderline_g + random.Random(42).sample(edee, min(5, len(edee)))

# cross-check: case numbering must match the blind-read sheet the human used
blind = (ROOT / "results/between_items_probe/blind_read_cases.txt").read_text(encoding="utf-8")
sheet = re.findall(r"item_id\s+: (\S+)\ntype\s+: \S+\s+target: .*?\s+marking: (\S+)", blind)
assert [(o["item_id"], o["marking"]) for o in cases] == sheet, \
    "reconstructed case list does not match blind_read_cases.txt order"

hu_by_id = {o["obs_id"]: o for o in hu["observations"]}
v4_by_id = {o["obs_id"]: o for o in v4["observations"]}

# agreement at the survival level (W/G = survived), the paper's headline metric;
# W-vs-G is a within-survived refinement, not a survival disagreement
disagreements = []
for i, o in enumerate(cases, 1):
    h = hu_by_id[o["obs_id"]]["judge_final"]
    v = v4_by_id[o["obs_id"]]["judge_v2"]["bucket"]
    if survived(v) != survived(h):
        disagreements.append((i, o, v, h))

print(f"\n===== residual gap (v4 vs human on the {len(cases)} adjudicated cases) =====")
print(f"survival-level agreement: {len(cases) - len(disagreements)}/{len(cases)}")
for i, o, v, h in disagreements:
    print(f"  case{i:>2} {o['item_id']:<31} {o['marking']:<8}: v4={v} human={h}")
if disagreements and all(survived(h) and not survived(v) for _, _, v, h in disagreements):
    print("=> all disagreements are v4 STRICTER than human: v4 under-counts survival vs human; never inflates. Conservative reproduction.")
elif disagreements:
    print("=> WARNING: at least one disagreement has v4 MORE LENIENT than human — inspect before claiming a conservative reproduction.")
