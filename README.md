# AI Guardrail Survival under Single-Cycle Agentic Self-Summarization

Reproducibility repository for the paper (`paper.md` / `paper.pdf`).

**What it studies.** When a long-running agent *compacts* its context into a model-generated
summary, what happens to a standing safety rule? The central finding is that **a presence check is
not a safety check**: when compaction does not drop a rule outright, it often leaves a residue that
looks like a rule but does not act like one. On behavioral replay, degraded residues let a model
perform the prohibited action far more often than intact welded rules do; category-level survival
behaves like a residue; and even textually intact rules sometimes fail to fire. Separately, rule-form
items are retained more often than prominence-matched facts, which is exactly why presence-based
auditing feels adequate even though survival is not protection. All results concern a **single**
compaction cycle.

## Models

| role | provider | model id |
|------|----------|----------|
| Judge and salience rater | Anthropic | `claude-sonnet-4-6` |
| Canonical summarizer and replay model | Together AI | `Qwen/Qwen3.5-9B` (`enable_thinking=false`) |
| Second summarizer and second replay model | Together AI | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Second-summarizer cell (hard output cap) | Anthropic | `claude-sonnet-4-6` |
| Independent blind reader (Appendix H.3) | Anthropic | Claude Opus 4.8 |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # then add ANTHROPIC_API_KEY and TOGETHER_API_KEY
```

## Reproduce

```powershell
# Between-items deontic-vs-epistemic probe (canonical run)
python -m tracer.between_items_probe

# Re-score under the frozen conservative rubric + behavioral replay
python tracer/rescore_between_items.py

# Behavioral-replay census (D-census, W/G survivors, second replay model)
python tracer/_replay_all_D.py --buckets W,G
python tracer/_replay_all_D.py --replay-model llama
python tracer/_replay_contrast.py      # D-vs-W contrast, G subtypes, severing tally

# Build the paper PDF (markdown is authoritative)
python build_pdf.py
```

Canonical labels: `results/between_items_probe/verdicts_20260602_214506_final.json` (first-pass LLM-judge
labels with targeted author adjudication of decisive and contested cases). A conservative second label
set (`_v2.json`, frozen rubric) is reported alongside every headline number. Scoring is an LLM judge
used as a first-pass filter, validated by targeted author adjudication and behavioral replay; there was
no separate human-rater panel.

## Layout

```
paper.md / paper.pdf / paper.docx   the paper (markdown is authoritative)
config/         prompts.py (frozen prompts), deontic_pairs.py (item set)
tracer/         between_items_probe.py, rescore_between_items.py, stress_probe.py,
                _replay_all_D.py, _replay_contrast.py, _analyze_claude_cell.py, ...
results/between_items_probe/   digests, per-item verdicts, behavioral-replay transcripts
results/salience_ratings/      prominence ratings used for the matched-pair covariate
figures/        figure1_survival.{png,pdf,py}
```

## Citation

The arXiv identifier and archive DOI will be added on posting.
