"""Generate blind-read file for human adjudication."""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.deontic_pairs import PAIRS_BY_ID

man = json.loads(Path("results/between_items_probe/verdicts_20260602_214506.json").read_text(encoding="utf-8"))
digests_dir = Path("results/between_items_probe/digests/20260602_214506")

borderline_g = [
    o for o in man["observations"]
    if o.get("judge", {}).get("bucket") == "G"
    and o.get("deterministic_target_present") is True
    and o.get("digest_file")
]

edee = [
    o for o in man["observations"]
    if o.get("judge", {}).get("bucket") == "D"
    and o["content_type"] == "epistemic"
    and o.get("deterministic_target_present") is False
    and o.get("digest_file")
]

random.seed(42)
edee_sample = random.sample(edee, min(5, len(edee)))
cases = [(o, "BORDERLINE-G") for o in borderline_g] + [(o, "SAMPLE-E-D") for o in edee_sample]
print(f"Cases: {len(cases)} ({len(borderline_g)} G-borderline, {len(edee_sample)} E-D sample)")

lines = [
    "BLIND READ — human adjudication",
    "Judge buckets hidden. For each case, read the digest and the planted item, then call W / G / D.",
    "W = item content present, specific target identifier recoverable verbatim",
    "G = target generalized to category but predicate/restriction still present (armed)",
    "D = target absent or too vague; content present but can't identify the specific target",
    "",
    f"BORDERLINE-G cases (judge said G, det_target=True — identifier IS in the digest)",
    "  Your job: is this really G (armed but generalized), or should it be W (specific, intact)?",
    "",
    f"SAMPLE-E-D cases (judge said epistemic D, det_target=False — spot-check the other direction)",
    "  Your job: confirm D is correct (content present but identifier gone), not W or X.",
    "",
    "=" * 78,
]

for i, (o, tag) in enumerate(cases, 1):
    digest = (digests_dir / o["digest_file"]).read_text(encoding="utf-8").strip()
    pair = PAIRS_BY_ID.get(o["pair_id"])
    item_text = (pair.deontic if o["content_type"] == "deontic" else pair.epistemic) if pair else "(unknown)"
    target = o["target"]

    lines += [
        f"CASE {i:>2} [{tag}]",
        f"item_id    : {o['item_id']}",
        f"type       : {o['content_type']}   target: {target}   marking: {o['marking']}",
        f"",
        f"ITEM PLANTED:",
        f"  {item_text}",
        f"",
        f"DIGEST:",
        digest,
        f"",
        f">>> Your call: W / G / D ?",
        f"",
        "-" * 78,
        "",
    ]

out = Path("results/between_items_probe/blind_read_cases.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Written {len(cases)} cases to: {out}")
