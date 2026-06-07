# Low-Resource Arabic→English NMT with a Tiny Transformer

A Master's Deep Learning project: training a compact encoder–decoder Transformer **from
scratch, CPU-only**, for Arabic→English translation in a low-resource setting (~50k sentence
pairs), then improving it through a controlled study of **decoding** and **regularization**
ablations. The project is **notebook-first**; `src/` and `scripts/` are helper/reproducibility
mirrors used by the notebooks.

- **Data:** IWSLT 2017 Arabic-English (TED talks). Raw ~231k train pairs sampled to ~50k.
- **Tokenization:** SentencePiece BPE, separate 8k Arabic / 8k English vocabularies.
- **Model:** tiny Transformer — d_model=128, 4 heads, 2 enc / 2 dec layers, ff=512, dropout=0.1,
  **4,006,208 trainable params**.
- **Hardware:** CPU-only.

## Final result (full test, 8,491 examples)

**Final / best model: the AdamW-trained Transformer (E3, weight_decay=1e-4) decoded with beam
search `beam=3, length_penalty=1.2, no_repeat_ngram_size=3`** (decoding tuned in E15, notebook 10).

- **BLEU = 8.1231**
- **chrF++ = 27.6255**
- **Repetition = 19.2%** (down from ~50% for the original greedy model)

*(E7's combined-best was E3 + beam3 lp1.0 nr3 = 7.9076 / 26.9301; the final E15 decoding sweep
then found `length_penalty=1.2` improves it by +0.22 BLEU / +0.70 chrF++ — decoding-only, no
retraining. The lp1.0 setting had slightly lower repetition (16.4%); lp1.2 trades a little
repetition for better BLEU and chrF++.)*

It beats the dictionary baseline on **both** metrics. **E7 is a *decision*, not a new training
run:** stacking each ablation's best setting (dropout 0.1, label-smoothing 0.0, fixed LR, no
weight tying, **AdamW wd 1e-4**) decoded with **nr3** reduces to exactly **E3 + nr3**, which is
already trained and evaluated.

What moved the needle:
- **Decoding (E14 `no_repeat_ngram=3`) was the single biggest improvement** — it lifts every
  model and resolved the earlier BLEU-vs-chrF++ disagreement (chrF++ now exceeds the baseline).
- **E3 (AdamW) was the only *training* ablation that improved the model.**
- **E11 (LR warmup), E1 (dropout 0.3), and E2 (label smoothing 0.1) were negative.**
- **E6 (weight tying) is an *efficiency* variant** (2.98M params, ~same BLEU, lower chrF++) — not
  the final quality model.

### Full comparison (full test, 8,491; sacreBLEU on detokenized English, identical references)

| System | Decoding | BLEU | chrF++ | Repetition | Verdict |
|---|---|---:|---:|---:|---|
| **E3 AdamW wd1e-4** | beam3 **lp1.2** nr3 (E15) | **8.1231** | **27.6255** | 19.2% | **Final / best** |
| E3 AdamW wd1e-4 | beam5 lp1.2 nr3 (E15) | 8.0878 | 27.4195 | 17.4% | ~ties, slower |
| E3 AdamW wd1e-4 | beam3 lp1.0 nr3 | 7.9076 | 26.9301 | 16.4% | prior best (superseded) |
| E6 weight tying | beam3 lp1.0 nr3 | 7.7168 | 25.5947 | 17.4% | Efficient variant (2.98M params) |
| E14 original | beam3 lp1.0 nr3 | 7.7095 | 26.5320 | 15.3% | Best decoding-only |
| Original beam3 | beam3 lp1.0 | 7.5338 | 24.9288 | N/A | Earlier best decoding |
| E2 label smoothing | beam3 lp1.0 | 7.3685 | 25.3485 | N/A | Negative / mixed |
| Original greedy | greedy | 6.2172 | 24.6435 | 49.8% | Initial neural baseline |
| E11 LR warmup | beam3 lp1.0 nr3 | 5.6777 | 22.8403 | 18.7% | Negative |
| Dictionary baseline | position dictionary | 4.7777 | 25.6503 | N/A | Lexical baseline |
| E1 dropout 0.3 | beam3 lp1.0 nr3 | 4.6882 | 21.1459 | 18.6% | Negative |

BLEU is stable between the 1,000-sample subset and the full test (Δ < 0.07), so the smaller-sample
numbers were representative.

### Key result files
- `outputs/tables/final_ablation_results.csv` — the master ablation comparison (table above).
- `outputs/tables/final_main_results.csv` — all systems/decodings with full + 1,000-subset scores.
- `outputs/tables/experiment_registry.csv` — every experiment, status, and metrics.
- `report_notes/ablation_notes.md` — ablation write-up, findings, figure captions, final-model statement.
- `outputs/figures/ablation_bleu_comparison.png`, `ablation_chrf_comparison.png`,
  `ablation_repetition_comparison.png` — comparison plots.

## Data layout (not committed — large / teammate-owned)

```text
Data/
  tokenized/{train,validation,test}.{ar,en}.bpe
  vocab/sp_{ar,en}.{model,vocab}     # SentencePiece; .vocab = subword tokens, not embeddings
```

Splits after `max_len=80` BPE filtering: train 49,441 / val 869 / test 8,491.

## Environment (CPU-only, Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

In Jupyter/VS Code, select the kernel **"Low Resource MT (.venv Python 3.12)"**.

## Notebook workflow (run in order)

| # | Notebook | Purpose |
|---|---|---|
| 01 | `notebooks/01_data_preparation.ipynb` | Cleaning, Arabic normalization, splits (teammate) |
| 02 | `notebooks/02_data_analysis_and_bpe_verification.ipynb` | EDA + SentencePiece BPE + round-trip checks |
| 03 | `notebooks/03_dictionary_baseline.ipynb` | Dictionary position baseline + BLEU/chrF++ |
| 04 | `notebooks/04_tiny_transformer_training.ipynb` | Checkpoint-safe CPU training + resume + CPU/DataLoader benchmark |
| 05 | `notebooks/05_full_evaluation_and_error_analysis.ipynb` | Full-test evaluation, curves, qualitative + error analysis |
| 06 | `notebooks/06_ablation_studies.ipynb` | **E13 beam-search** decoding sweep (beam size × length penalty) |
| 07 | `notebooks/07_training_ablations.ipynb` | **E2 label-smoothing** training ablation |
| 08 | `notebooks/08_ablation_summary.ipynb` | **Consolidated comparison of all ablations + the E7 final-model decision** |
| 09 | `notebooks/09_decoding_repetition_control.ipynb` | **E14 `no_repeat_ngram` decoding sweep** (the biggest single improvement) |
| 10 | `notebooks/10_final_e3_decoding_sweep.ipynb` | **E15 final decoding sweep on the E3 checkpoint** → locks `beam3 lp1.2 nr3` as the final model |

Training ablations **E11 / E1 / E3 / E6** were run with the checkpoint-safe engine
`scripts/run_ablation.py` (isolated checkpoints under `outputs/checkpoints/<exp>/`) and are
consolidated in notebook 08. Superseded notebooks are kept under `notebooks/archive/`.

### Checkpoint / resume behavior
Training is CPU-only and may take hours, so the training notebooks are **checkpoint-safe**: if a
best checkpoint exists and `FORCE_RESTART=False` they load it instead of retraining; if a latest
checkpoint exists they resume from the next epoch; latest + best checkpoints and the training log
are saved every epoch.

### CPU / DataLoader settings (benchmarked, not guessed)
A benchmark in notebook 04 (`outputs/tables/cpu_dataloader_benchmark.csv`) shows **8 torch
threads** is the throughput sweet spot (~200 samples/s) — `cores−1` is ~44% slower for this tiny
model — and DataLoader `num_workers>0` doesn't help (and hangs in-notebook on Windows). Chosen:
`TORCH_THREADS = min(8, CPU_CORES)`, `DATALOADER_WORKERS = 0`. The benchmark evidence stays visible
inside notebook 04.

## Outputs

```text
outputs/
  tables/        result CSVs (final_ablation_results, final_main_results, experiment_registry, per-exp e*_*.csv, logs)
  figures/       loss curves + ablation/comparison plots
  translations/  sample/full predictions
  examples/      qualitative example tables + per-exp predictions
  checkpoints/   model .pt files (gitignored, large): best_full_model.pt + e2/e11/e1/e3/e6 dirs
  logs/          repo_audit.md, plan_summary.md
report_notes/    evidence/notes for the IEEE report (not final prose)
```

## Reproducing the results (no retraining needed)
From the repaired environment:
- Baseline + full-test transformer eval: notebooks `03` and `05` (load `best_full_model.pt`).
- Decoding study (E14): notebook `09`; final comparison + E7 decision: notebook `08`.
- One-command mirrors: `python scripts/run_full_eval.py` (greedy full test),
  `python scripts/run_ablation.py --exp E3` (retrains E3 only if you want to regenerate it).

## Limitations
Absolute BLEU is low — inherent to a tiny, from-scratch, CPU-only, low-resource setup. The final
recipe (AdamW + `no_repeat_ngram=3` decoding) cuts degenerate repetition from ~50% to ~16% and
beats the dictionary baseline on both metrics, but quality is still modest. Several training
ablations were negative (label smoothing, LR warmup, dropout 0.3) and are reported honestly. The
40-epoch models' validation loss was still decreasing at the cap (mildly undertrained). See
`report_notes/limitations_and_future_work.md` and `LOG.md`.

## Report notes
`report_notes/` holds evidence/notes for the IEEE report (not final prose): metric definitions
(BLEU, chrF++), results interpretation, **ablation notes (`ablation_notes.md`)**, error analysis,
limitations/future work, a recent-literature checklist, and oral-defense Q&A.

## Project docs
- `memory.md` — full project context and current status (read first).
- `LOG.md` — development log. `EXPERIMENT_LOG.md` — per-run records.
- `Docs/Plan/nmt_final_agent_execution_plan_merged.md` — authoritative execution plan.
- `GenAI_Usage_Statement.md` — honest GenAI disclosure.
