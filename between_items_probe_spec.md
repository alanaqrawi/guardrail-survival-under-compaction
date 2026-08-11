# Between-Items Deontic Probe — Spec

Supersedes the matched-pairs design in `tracer/deontic_probe.py`. The switch is
documented as a methodological choice, not a weakness: only 5 of 160 matched-pair
observations were discordant (3.1%), making McNemar nearly powerless regardless of N.
Between-items removes the paraphrase-distinctness constraint and replaces matched-pair
control with a salience covariate in the regression.

---

## Why between-items

The matched-pairs design required equating salience by including the rule's rationale
in both items. That produced near-paraphrases: 9 of 10 pairs had epistemic items that
were essentially restatements of the rule in a different mood. A judge (or a summarizer)
cannot reliably distinguish survived-rule from survived-fact when the two say the same
thing. Fixing the pairs while keeping matched design is hard: equating stakes forces
shared content. Between-items avoids this: rules and facts are about *different* targets,
so there is no paraphrase problem, and salience is controlled statistically rather than
by matching.

---

## Design

### Items
- **10 deontic items** (rules): one per target, kept from the original 10 pairs.
- **10 epistemic items** (facts): re-authored — each is a standalone fact about its target
  (when introduced, what layer it operates at, usage statistics, contract terms, etc.),
  NOT the rule's rationale. A labeler who has never seen the rule must be able to tell
  that the content came from the fact.
- Total pool: 20 items across 10 targets.

### Conversation structure
- Each conversation contains **~8 items** drawn from the pool of 20.
- **Hard constraint:** a rule and a fact about the *same* target NEVER appear in the same
  conversation. Targets rotate so each appears as a rule in some conversations and a fact
  in others.
- Items are embedded at varied positions in the dense filler (~45k tokens, same as the
  stress probe). Position is recorded per item for the regression.
- Tracers planted as before (manipulation check).

### Models
- **Primary quantitative:** Qwen (28–40% survival at 45k — estimable, not floored).
- **Llama optional:** run separately at ~15–25k filler density (floored at 45k). Report
  per-model, never pooled.
- **Claude:** warm rig; add if time/budget allows; report separately if pressure not
  confirmed.

---

## Analysis

Mixed-effects logistic regression (GEE with robust SEs in Python; equivalent to GLMM
for binary outcomes):

```
survival ~ content_type + salience_rating + position + (1|conversation) + (1|target)
```

- `survival`: 1 = item survived (W or G), 0 = dropped (D or X).
- `content_type`: deontic vs epistemic (the primary predictor).
- `salience_rating`: continuous covariate from the salience rater (1–5); replaces
  matched-pair control. Partials out the salience effect on survival.
- `position`: continuous, fraction of filler budget at which item was planted (0.0–1.0).
- `(1|conversation)`, `(1|target)`: random intercepts for conversation and target identity.
- Per-model, never pooled across models.

---

## N

- Target: **~150–200 observations per content type** (~300–400 total items).
- With ~8 items per conversation: **~40–50 Qwen conversations**.
- Run once, examine. Bump N **once only** if the coefficient CI is too wide to resolve
  the effect.
- Budget: ~$15–20.

---

## Guardrails (unchanged from §0)

1. Manipulation check gates interpretation (≥60% tracer drop required).
2. Judge = first-pass filter only; blind human primary on result-driving cells.
3. ≥90% judge↔human agreement gate on decisive items.
4. Raw digests written separately from verdicts; blind protocol preserved.
5. Behavioral replay on a subset of surviving rules (still fires?) and surviving facts
   (still usable?).

---

## Pre-committed decision table

| Outcome | Meaning | Action |
|---|---|---|
| content_type coeff significant (p<0.05), deontic > epistemic | Normative privileging beyond salience | Headline finding. Write it. |
| content_type coeff not significant | Privileging is salience-mediated | Real finding + recipe ("mark constraints saliently"). Write that. |
| content_type coeff significant, deontic < epistemic | Rules less sticky than matched facts | Double-check via human + replay, then write. |
| Manipulation check fails | Pressure not achieved | Adjust filler density, rerun. DO NOT interpret. |
| CI too wide to resolve (borderline) | Underpowered | Bump N once, then call it. |

**Pre-commit:** the result stands whichever way it lands, including "no difference → write
that, done." No further conditions after the one optional N-bump.

---

## Limitations to state in paper

1. Salience rater scale tops out at 5 (ceiling effect); between-items design controls for
   measured salience but cannot rule out supra-ceiling asymmetry.
2. Single filler budget (45k) on Qwen; survival rates are budget-dependent.
3. Between-items design removes within-pair control; the regression covariate handles
   salience but not all item-level confounds.
4. The switch from matched-pairs was motivated by the paired design being underpowered
   (5/160 discordant = 3.1%); state this plainly as a justified methodological choice.

---

## P05 adjudication (parallel, old run)

The P05/marked cell from run 20260601_235256 goes to the independent human (done in
parallel, does not block the new run). The human reviewer confirmed D=W (prohibition
present). This corrects the marked-condition tally to b=1/c=1, not b=1/c=2 as the judge
had it. Recorded in the old run's results; does not affect the new design.

---

## Files

- `config/deontic_pairs.py` — updated with re-authored epistemic items.
- `tracer/between_items_probe.py` — new harness (between-items builder, GEE report,
  guardrails, behavioral replay).
- `tracer/salience_rater.py` — re-run on new pairs; ratings used as regression covariate.
- `results/between_items_probe/` — output directory.
