"""Stratify the D-census replay results for the paper integration."""
import json
from collections import Counter
from pathlib import Path

BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"
rows = json.loads((BI / "behavioral_replay_allD_20260602_214506.json").read_text(encoding="utf-8"))
d = [r for r in rows if r["case_type"] == "D-CENSUS"]


def v(r):
    return r["verdict"].split(" (")[0]


print("verdict x target_present:")
for tp in (True, False):
    c = Counter(v(r) for r in d if r["target_present"] is tp)
    n = sum(c.values())
    print(f"  target_present={tp!s:<6} n={n:<3} " +
          "  ".join(f"{k}={c[k]}" for k in ("DISARMED", "STILL ARMED", "INCONCLUSIVE", "MIXED") if c[k]))

print("\nverdict x pair:")
for pid in sorted({r["pair_id"] for r in d}):
    c = Counter(v(r) for r in d if r["pair_id"] == pid)
    n = sum(c.values())
    print(f"  {pid:<28} n={n:<3} " +
          "  ".join(f"{k}={c[k]}" for k in ("DISARMED", "STILL ARMED", "INCONCLUSIVE", "MIXED") if c[k]))

mixed = [r for r in d if v(r) == "MIXED"]
print(f"\nMIXED cases: {len(mixed)}")
for r in mixed:
    print(f"  {r['obs_id']} t={r['target_comply']} s={r['sibling_comply']}")

# the two originally-confirmed payments-v2 disarms — re-confirmed?
p04 = [r for r in d if r["pair_id"] == "P04_branch_pci"]
print("\nP04 payments-v2 cases:")
for r in p04:
    print(f"  {r['obs_id']:<38} {r['marking']:<9} tp={r['target_present']!s:<6} {v(r)}")

# sample DISARMED transcripts for spot-check
print("\n--- sample DISARMED transcripts (3) ---")
for r in [x for x in d if v(x) == "DISARMED"][:3]:
    print(f"\n[{r['obs_id']}] target reply: {r['target_reply'][:300]}")
