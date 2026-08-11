"""Score the returned blind-read labels for the Llama replication packet.

Compares rater labels against (a) first-pass judge buckets and (b) frozen-v4
buckets, at survival level and exact-bucket level; resolves the decisive cases
to their digests and validity classification.
"""
import json
from pathlib import Path

BI = Path(__file__).resolve().parent.parent / "results" / "between_items_probe"
RUN = "20260610_150139"

RATER = {1: "X", 2: "X", 3: "X", 4: "X", 5: "X", 6: "X", 7: "X", 8: "X",
         9: "X", 10: "X", 11: "X", 12: "X", 13: "X", 14: "X", 15: "X",
         16: "X", 17: "X", 18: "X", 19: "X", 20: "X", 21: "G", 22: "X",
         23: "X", 24: "G", 25: "X", 26: "X", 27: "X", 28: "X", 29: "X",
         30: "X", 31: "X", 32: "X", 33: "X", 34: "X", 35: "X"}

key = json.loads((BI / f"blind_read_llama_KEY_{RUN}.json").read_text(encoding="utf-8"))
v2 = json.loads((BI / f"verdicts_{RUN}_v2.json").read_text(encoding="utf-8"))
v2_by_obs = {}
for o in v2["observations"]:
    r = o.get("judge_v2") or {}
    v2_by_obs[o["obs_id"]] = r.get("bucket") if isinstance(r, dict) else r


def surv(b):
    return b in ("W", "G")


agree_j, agree_v4 = 0, 0
print(f"{'case':<5}{'item_id':<36}{'mk':<9}{'judge':<6}{'v4':<5}{'rater':<6}"
      f"{'tgt':<6}{'nonsum':<7}")
for k in key:
    c = k["case"]
    rater = RATER[c]
    j = k["judge_bucket"]
    v4 = v2_by_obs.get(k["obs_id"], "?")
    sj = surv(j) == surv(rater)
    sv = surv(v4) == surv(rater)
    agree_j += sj
    agree_v4 += sv
    mark = "" if (sj and sv) else "   <-- disagreement"
    print(f"{c:<5}{k['item_id']:<36}{k['marking']:<9}{j:<6}{v4:<5}{rater:<6}"
          f"{str(k['target_present']):<6}{str(k['nonsummary_digest']):<7}{mark}")

n = len(key)
print(f"\nsurvival-level agreement: rater vs first-pass judge {agree_j}/{n}, "
      f"rater vs v4 {agree_v4}/{n}")

# decisive cases -> digest files
print("\nDecisive cases (G survivals + token-coincidence checks):")
man = json.loads((BI / f"verdicts_{RUN}.json").read_text(encoding="utf-8"))
obs_by_id = {o["obs_id"]: o for o in man["observations"]}
for c in (21, 24, 14, 22, 23):
    k = next(x for x in key if x["case"] == c)
    o = obs_by_id[k["obs_id"]]
    print(f"  case {c}: {k['obs_id']}  digest={o['digest_file']}  "
          f"judge={k['judge_bucket']} v4={v2_by_obs.get(k['obs_id'])} "
          f"rater={RATER[c]} nonsummary={k['nonsummary_digest']}")
