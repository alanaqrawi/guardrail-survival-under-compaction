"""Deterministic digest-validity classifier (non-summary detection).

A digest is a NON-SUMMARY if it contains ZERO conversation-specific identifiers.
Identifier vocabulary (fixed, pre-committed before the Llama between-items analysis):
tracer keys, filler systems/projects/vendors/customers, sibling identifiers, and
deontic-pair targets. Pure refusals ("There is no conversation to summarize ...")
name only generic categories and therefore match zero identifiers; a digest that
opens with a refusal but then summarizes real content names at least one identifier
and counts as VALID.

Usage:
  python -m tracer._classify_nonsummaries stress  20260531_231543
  python -m tracer._classify_nonsummaries stress  20260601_230516
  python -m tracer._classify_nonsummaries between 20260602_214506
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.deontic_pairs import PAIRS  # noqa: E402
from tracer.stress_probe import (  # noqa: E402
    TRACERS, _SYSTEMS, _PROJECTS, _VENDORS, _CUSTOMERS, SIBLINGS,
)

ROOT = Path(__file__).resolve().parent.parent / "results"

# Fixed identifier vocabulary. Case-insensitive matching; every term is a
# multi-character, conversation-specific token (no generic words, no bare
# first names) so a zero-hit digest cannot be a faithful summary.
VOCAB: set[str] = set()
VOCAB.update(key for key, _ in TRACERS)
VOCAB.update(_SYSTEMS)
VOCAB.update(_PROJECTS)
VOCAB.update(_VENDORS)
VOCAB.update(_CUSTOMERS)
VOCAB.update(SIBLINGS)
for p in PAIRS:
    VOCAB.add(p.target)
    VOCAB.update(p.siblings)
VOCAB.add("prod-east-2")


def identifiers_in(text: str) -> list[str]:
    low = text.lower()
    return sorted({v for v in VOCAB if v.lower() in low})


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("stress", "between"):
        sys.exit(__doc__)
    probe = "stress_probe" if sys.argv[1] == "stress" else "between_items_probe"
    run_id = sys.argv[2]
    digests_dir = ROOT / probe / "digests" / run_id
    if not digests_dir.exists():
        sys.exit(f"no digests dir: {digests_dir}")

    # bucket per digest from the verdicts manifest (stress: cells->records,
    # one rule obs per digest; between: observations, 8 item obs per digest)
    buckets: dict[str, str] = {}
    vpath = ROOT / probe / f"verdicts_{run_id}.json"
    if vpath.exists():
        man = json.loads(vpath.read_text(encoding="utf-8"))
        per: dict[str, list[str]] = defaultdict(list)
        if "observations" in man:
            for o in man["observations"]:
                j = o.get("judge") or {}
                per[o.get("digest_file", "")].append(j.get("bucket") or "?")
        elif "cells" in man:
            for cell in man["cells"].values():
                for r in cell.get("records", []):
                    j = r.get("judge") or {}
                    per[r.get("file", "")].append(j.get("bucket") or "?")
        buckets = {k: ",".join(v) for k, v in per.items()}

    rows = []
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # cell -> [n, nonsummary]
    for f in sorted(digests_dir.glob("*.txt")):
        ids = identifiers_in(f.read_text(encoding="utf-8"))
        nonsummary = len(ids) == 0
        cell = "__".join(f.stem.split("__")[:2])
        cells[cell][0] += 1
        cells[cell][1] += int(nonsummary)
        rows.append({
            "digest_file": f.name,
            "cell": cell,
            "n_identifiers": len(ids),
            "identifiers": ids[:8],
            "nonsummary": nonsummary,
            "buckets": buckets.get(f.name, ""),
        })
        flag = "NON-SUMMARY" if nonsummary else f"valid ({len(ids)} ids)"
        print(f"  {f.name:<42} {flag:<16} buckets={buckets.get(f.name, '-')}")

    print("\nPer-cell non-summary counts:")
    for cell, (n, ns) in sorted(cells.items()):
        print(f"  {cell:<28} {ns}/{n} non-summaries")

    out = ROOT / probe / f"nonsummary_classification_{run_id}.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "rule": "non-summary iff zero conversation-specific identifiers "
                "(fixed vocabulary: tracer keys, systems, projects, vendors, "
                "customers, siblings, deontic targets)",
        "digests": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
