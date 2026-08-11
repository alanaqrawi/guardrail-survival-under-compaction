"""Confirm 0 deontic labels moved under the presence-not-inference rubric fix."""
import json
from pathlib import Path

man = json.loads(Path("results/between_items_probe/verdicts_20260602_214506_final.json").read_text(encoding="utf-8"))

moved_deontic = []
for o in man["observations"]:
    if o["content_type"] != "deontic":
        continue
    b2 = o.get("judge_v2", {}).get("bucket")
    bf = o.get("judge_final")
    if b2 != bf:
        moved_deontic.append((o["item_id"], o["conversation_id"], b2, bf))

print(f"DEONTIC labels moved under presence-not-inference (review-corrections) fix: {len(moved_deontic)}")
for m in moved_deontic:
    print("   ", m)

overrides = [(o["item_id"], o["conversation_id"], o["judge_final"])
             for o in man["observations"]
             if o.get("judge_v2", {}).get("behavioral_override")]
print(f"\nBehavioral-replay overrides (separate from rubric fix): {len(overrides)}")
for ov in overrides:
    print("   ", ov)

for marking in ("unmarked", "marked"):
    d = [o for o in man["observations"] if o["content_type"] == "deontic"
         and o["marking"] == marking and o.get("judge_final") in ("W", "G", "D", "X")]
    s = sum(1 for o in d if o["judge_final"] in ("W", "G"))
    print(f"deontic {marking}: {s}/{len(d)} survived")
