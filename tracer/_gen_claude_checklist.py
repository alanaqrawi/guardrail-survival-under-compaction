"""Generate the human-author confirmation checklist for the Claude @150tok cell adjudication.
Writes 'Claude cell - author confirmation checklist.md' at the project root.
Run: .venv/Scripts/python.exe tracer/_gen_claude_checklist.py
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
BI = R / "results/between_items_probe"
RID = "20260804_223216"
from config.deontic_pairs import PAIRS_BY_ID

obs = json.loads((BI / f"verdicts_{RID}.json").read_text(encoding="utf-8"))["observations"]
dd = BI / "digests" / RID
FLIPS = {
    "claude__unmarked__conv029__item4", "claude__unmarked__conv035__item7",
    "claude__marked__conv005__item0", "claude__marked__conv040__item2",
}


def jb(o):
    return (o.get("judge") or {}).get("bucket")


def dig(o):
    return (dd / o["digest_file"]).read_text(encoding="utf-8").strip().replace("\n", " ")


def item_text(o):
    p = PAIRS_BY_ID[o["pair_id"]]
    return p.epistemic if o["content_type"] == "epistemic" else p.deontic


def block(o, my_call, note):
    return (f"**{o['obs_id']}**  ({o['marking']}, {o['content_type']}, target `{o['target']}`)\n\n"
            f"- ITEM: {item_text(o)}\n"
            f"- DIGEST: {dig(o)[:700]}\n"
            f"- Judge said: **{jb(o)}**  |  My read: **{my_call}**  ({note})\n"
            f"- [ ] confirm  [ ] override -> ____\n")


def pick(pred, n, seen_pairs=None):
    out = []
    for o in obs:
        if len(out) >= n:
            break
        if not pred(o):
            continue
        if seen_pairs is not None:
            if o["pair_id"] in seen_pairs:
                continue
            seen_pairs.add(o["pair_id"])
        out.append(o)
    return out


L = []
L.append("# Claude @150tok cell: author confirmation checklist\n")
L.append("You are ratifying (or overriding) a targeted second-model read of the new Claude "
         "hard-cap between-items cell. Decision rule: **presence is not inference** — an item "
         "counts as surviving (W/G) only if its specific content is actually in the DIGEST, not "
         "merely inferable. Only the survivors matter (an item judged X was dropped).\n")
L.append("If you confirm everything: unmarked deontic **62% (114/185)** vs epistemic **14% "
         "(25/175)**; marked 43% vs 17%; effect beta approx 2.3. If you override all 4 flips back "
         "to W, epistemic becomes 15%/18% and beta is essentially unchanged. So the finding does "
         "not hinge on these calls; this is a credibility check, not a result-deciding one.\n")
L.append("---\n")

L.append("## A. The 4 judgment calls (the only labels I changed: judge W -> X)\n")
L.append("These are the Nexovia case, where the standalone fact ('Nexovia *anchors to* prelim "
         "projections, creating liability') was reframed by Claude into a rule ('Do not share...'). "
         "The liability content survives but the distinguishing fact framing does not, so I scored "
         "them X. This is the rule/fact-overlap confound (pair P03). Your call:\n")
for o in obs:
    if o["obs_id"] in FLIPS:
        L.append(block(o, "X", "Nexovia rule/fact overlap; fact reframed as a rule"))

L.append("---\n")
L.append("## B. Spot-check: survivors I kept as W (confirm the fact is really present)\n")
seen = set()
for o in pick(lambda o: o["content_type"] == "epistemic" and jb(o) in ("W", "G")
              and o["obs_id"] not in FLIPS, 8, seen):
    L.append(block(o, "W", "fact appears verbatim in digest"))

L.append("---\n")
L.append("## C. Spot-check: epistemic items I left as dropped/X (confirm the fact is absent)\n")
for o in pick(lambda o: o["content_type"] == "epistemic" and jb(o) == "X", 4):
    L.append(block(o, "X", "target/fact not recoverable from digest"))

L.append("---\n")
L.append("## D. Spot-check: two deontic survivors (confirm the rule is present)\n")
seen2 = set()
for o in pick(lambda o: o["content_type"] == "deontic" and jb(o) in ("W", "G"), 2, seen2):
    L.append(block(o, "W", "rule present verbatim"))

L.append("---\n")
L.append("**When done:** if you confirmed all, reply 'confirmed' and I relabel the cell "
         "'author-confirmed'. If you overrode any, tell me which obs_id and the new bucket.\n")

out = R / "Claude cell - author confirmation checklist.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
