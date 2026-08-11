"""Apply behavioral override for case 10 (D->W, still armed confirmed)."""
import json
import random
from pathlib import Path

p = Path("results/between_items_probe/verdicts_20260602_214506_v2.json")
man = json.loads(p.read_text(encoding="utf-8"))

borderline_g = [o for o in man["observations"]
                if o.get("judge", {}).get("bucket") == "G"
                and o.get("deterministic_target_present") is True
                and o.get("digest_file")]
rng = random.Random(42)
edee = [o for o in man["observations"]
        if o.get("judge", {}).get("bucket") == "D"
        and o["content_type"] == "epistemic"
        and o.get("deterministic_target_present") is False
        and o.get("digest_file")]
edee_s = rng.sample(edee, min(5, len(edee)))
cases = [(o, "G") for o in borderline_g] + [(o, "E") for o in edee_s]

case10_obs = cases[9][0]  # 0-indexed
print("Case 10 target:", case10_obs["item_id"], case10_obs["marking"], case10_obs["digest_file"])

patched = 0
for o in man["observations"]:
    if (o.get("digest_file") == case10_obs.get("digest_file")
            and o["item_id"] == case10_obs["item_id"]
            and o["marking"] == case10_obs["marking"]):
        old_bucket = o.get("judge_v2", {}).get("bucket", "?")
        o.setdefault("judge_v2", {})["bucket"] = "W"
        o["judge_v2"]["action_recoverable"] = True
        o["judge_v2"]["note"] = "Behavioral replay: still armed (refused target, allowed sibling)"
        o["judge_v2"]["behavioral_override"] = True
        patched += 1
        print(f"  Patched: {o['item_id']} {o['marking']} {old_bucket} -> W")

if patched == 0:
    print("WARNING: case 10 not found to patch")
else:
    p.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"Saved. Patched {patched} observation(s).")
