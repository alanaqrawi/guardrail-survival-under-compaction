"""Surface epistemic G cases (marked condition) for human spot-check."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
man = json.loads((ROOT / "results/between_items_probe/verdicts_20260602_214506_v2.json").read_text(encoding="utf-8"))
digests_dir = ROOT / "results/between_items_probe/digests/20260602_214506"

import sys
sys.path.insert(0, str(ROOT))
from config.deontic_pairs import PAIRS_BY_ID

cases = [o for o in man["observations"]
         if o["content_type"] == "epistemic"
         and o["marking"] == "marked"
         and o.get("judge_v2", {}).get("bucket") == "G"
         and o.get("digest_file")]

print(f"Epistemic G cases (marked condition): {len(cases)}")
print("For each: item text, judge note, digest excerpt. Call W / G / D.")
print("W = specific fact recoverable  G = general category of fact present  D = no useful fact content")
print("=" * 80)

for i, o in enumerate(cases, 1):
    pair = PAIRS_BY_ID.get(o["pair_id"])
    item_text = pair.epistemic_marked if pair else "(unknown)"
    note = o.get("judge_v2", {}).get("note", "")
    action_rec = o.get("judge_v2", {}).get("action_recoverable", "?")
    target = o["target"]

    # Pull just the sentence(s) from the digest that mention the target
    digest = ""
    dp = digests_dir / o["digest_file"]
    if dp.exists():
        full = dp.read_text(encoding="utf-8")
        # find lines mentioning the target
        hits = [ln.strip() for ln in full.splitlines()
                if target.lower() in ln.lower() and len(ln.strip()) > 10]
        digest = "\n  ".join(hits[:4]) if hits else "(target not found in digest text)"

    print(f"\nCASE {i:>2}: {o['item_id']}  (conv: {o['conversation_id'][-10:]})")
    print(f"  item text : {item_text}")
    print(f"  judge note: {note}  [action_recoverable={action_rec}]")
    print(f"  target mentions in digest:")
    print(f"  {digest}")
    print(f"  >>> Your call: W / G / D ?")
    print()
