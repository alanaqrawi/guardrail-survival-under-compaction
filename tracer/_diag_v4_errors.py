"""Diagnose the 420 v4 judge errors."""
import json
from pathlib import Path
from collections import Counter

man = json.loads(Path("results/between_items_probe/verdicts_20260602_214506_v2.json").read_text(encoding="utf-8"))

errs = [o for o in man["observations"] if o.get("judge_v2", {}).get("bucket") not in ("W", "G", "D", "X")]
ok = [o for o in man["observations"] if o.get("judge_v2", {}).get("bucket") in ("W", "G", "D", "X")]
print(f"errored: {len(errs)}   ok: {len(ok)}")

# error type tally
types = Counter(o.get("judge_v2", {}).get("error", "(no error field)") for o in errs)
for t, c in types.most_common():
    print(f"  {c:>4}  {t}")

# show 3 raw samples
print("\n--- sample raw outputs from errored calls ---")
for o in errs[:3]:
    raw = o.get("judge_v2", {}).get("raw", "(no raw)")
    print(f"\n[{o['item_id']} {o['marking']}]  len(raw)={len(str(raw))}")
    print("  raw:", str(raw)[:400])
