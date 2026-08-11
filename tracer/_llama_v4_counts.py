"""v4-rubric label counts for the Llama replication run (all vs valid digests)."""
import json
from collections import Counter
from pathlib import Path

BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"
man = json.loads((BI / "verdicts_20260610_150139_v2.json").read_text(encoding="utf-8"))
cls = json.loads((BI / "nonsummary_classification_20260610_150139.json").read_text(encoding="utf-8"))
nonsum = {d["digest_file"] for d in cls["digests"] if d["nonsummary"]}


def b(o):
    r = o.get("judge_v2") or {}
    return r.get("bucket") if isinstance(r, dict) else r


for scope, keep in [("ALL", lambda o: True),
                    ("VALID", lambda o: o["digest_file"] not in nonsum)]:
    for mk in ("unmarked", "marked"):
        for ct in ("deontic", "epistemic"):
            sub = [o for o in man["observations"]
                   if o["marking"] == mk and o["content_type"] == ct
                   and keep(o) and b(o) in ("W", "G", "D", "X")]
            c = Counter(b(o) for o in sub)
            print(f"{scope:<6}{mk:<9}{ct:<10} W={c['W']:<3} G={c['G']:<3} "
                  f"D={c['D']:<3} X={c['X']:<4} n={len(sub)} surv={c['W'] + c['G']}")
bad = [o["obs_id"] for o in man["observations"] if b(o) not in ("W", "G", "D", "X")]
print(f"non-parseable v4 labels: {len(bad)}")
