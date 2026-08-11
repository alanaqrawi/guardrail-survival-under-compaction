"""Build the blind-read packet for the Llama replication run.

Cases: all judge-flagged W/G/D observations (result-driving for the
"zero survivors" claim) + a seeded random sample of X observations from
valid (non-refusal) digests as an under-counting check. Shuffled with a
fixed seed; judge labels hidden. Answer key written separately.

Usage: python -m tracer._make_blind_read_llama [run_id]
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.deontic_pairs import PAIRS_BY_ID  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260610_150139"
N_X_SAMPLE = 20
BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"

man = json.loads((BI / f"verdicts_{RUN_ID}.json").read_text(encoding="utf-8"))
cls = json.loads((BI / f"nonsummary_classification_{RUN_ID}.json").read_text(encoding="utf-8"))
nonsum = {d["digest_file"] for d in cls["digests"] if d["nonsummary"]}
digests_dir = BI / "digests" / RUN_ID


def bucket(o):
    j = o.get("judge") or {}
    return j.get("bucket")


def item_text(o):
    p = PAIRS_BY_ID[o["pair_id"]]
    if o["content_type"] == "deontic":
        return p.deontic_marked if o["marking"] == "marked" else p.deontic
    return p.epistemic_marked if o["marking"] == "marked" else p.epistemic


flagged = [o for o in man["observations"] if bucket(o) in ("W", "G", "D")]
x_pool = [o for o in man["observations"]
          if bucket(o) == "X" and o["digest_file"] not in nonsum]
x_sample = random.Random(7).sample(x_pool, min(N_X_SAMPLE, len(x_pool)))

cases = flagged + x_sample
random.Random(11).shuffle(cases)

lines = [
    f"BLIND READ — Llama between-items replication (run {RUN_ID})",
    "Judge buckets hidden. For each case, read the digest and the planted item,",
    "then call W / G / D / X (frozen v4 rubric — presence is not inference):",
    "W = item content present, specific target identifier recoverable verbatim/near-verbatim",
    "G = target generalized to its category but predicate/restriction still present",
    "D = referent may be named but only 'a constraint exists' survives — no recoverable action",
    "    (for epistemic: content trace present but the specific fact is not recoverable)",
    "X = item content absent, OR recoverable only by inference from siblings/other items",
    "",
    f"{len(cases)} cases. Record answers as lines of:  CASE <n>: <W|G|D|X>",
    "=" * 78,
]
key = []
for i, o in enumerate(cases, 1):
    digest = (digests_dir / o["digest_file"]).read_text(encoding="utf-8")
    lines += [
        "",
        f"CASE {i:2d}",
        f"item_id    : {o['item_id']}",
        f"type       : {o['content_type']}   target: {o['target']}   marking: {o['marking']}",
        "",
        "ITEM PLANTED:",
        f"  {item_text(o)}",
        "",
        "DIGEST:",
        digest.strip(),
        "",
        "=" * 78,
    ]
    key.append({"case": i, "obs_id": o["obs_id"], "item_id": o["item_id"],
                "marking": o["marking"], "judge_bucket": bucket(o),
                "target_present": o["deterministic_target_present"],
                "nonsummary_digest": o["digest_file"] in nonsum})

packet = BI / f"blind_read_llama_{RUN_ID}.txt"
packet.write_text("\n".join(lines), encoding="utf-8")
keyfile = BI / f"blind_read_llama_KEY_{RUN_ID}.json"
keyfile.write_text(json.dumps(key, indent=2), encoding="utf-8")
print(f"wrote {packet}  ({len(cases)} cases: {len(flagged)} judge-W/G/D + {len(x_sample)} sampled X)")
print(f"wrote {keyfile}  (DO NOT open before reading)")
