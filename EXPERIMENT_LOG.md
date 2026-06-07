# Experiment Log

This file records reproducible experiment runs for the low-resource Arabic-English machine translation project.

Each run should include:

- Date
- Owner
- Data split or subset size
- Model/baseline
- Main configuration
- Output files
- Metrics
- Notes

## Runs

### Run 001 - README Repository Test

- Date: 2026-05-13
- Owner: Ahmed Sherif
- Purpose: Verify GitHub repository setup and push workflow.
- Commit: `289bd03 Add project README`
- Notes: Only `README.md` was committed and pushed. No data files were committed.

### Run 002 - Dictionary Baseline Smoke Evaluation

- Date: 2026-05-13
- Owner: Codex / Ahmed Sherif
- Model: Dictionary position baseline
- Evaluation subset: 100 test examples
- Metric: BLEU `5.63`
- Output: `outputs/translations/dictionary_baseline_smoke.bpe`
- Notes: Smoke test only. Later removed from lightweight deliverables to keep CSV outputs clean.

### Run 003 - Notebook Baseline Evaluation

- Date: 2026-05-13
- Owner: Codex / Ahmed Sherif
- Model: Dictionary position baseline
- Evaluation subset: 1000 test examples
- BLEU: `4.688637`
- BLEU text type: Detokenized English using `sp_en.model`
- Outputs:
  - `outputs/tables/baseline_results.csv`
  - `outputs/translations/baseline_samples.csv`
- Notes: This is the current standardized baseline result.

### Run 004 - Tiny Transformer Debug Run

- Date: 2026-05-13
- Owner: Codex / Ahmed Sherif
- Model: Tiny Transformer, greedy decoding
- Training subset: 1000 examples
- Validation subset: 300 examples
- Epochs: 1
- Train loss: `8.228888`
- Validation loss: `7.526851`
- BLEU: `0.008628`
- Notes: Debug run only. Output showed severe undertraining and repeated punctuation.

### Run 005 - Tiny Overfit Diagnostic

- Date: 2026-05-13
- Owner: Codex / Ahmed Sherif
- Model: Tiny Transformer
- Training subset: 50 examples
- Epochs: 60
- Initial train loss: `8.707735`
- Final train loss: `0.184858`
- Outputs:
  - `outputs/tables/overfit_training_log.csv`
  - `outputs/translations/overfit_samples.csv`
- Notes: The model successfully memorized several training examples, confirming the core model/training implementation can learn.

### Run 006 - Tiny Transformer 10k Preliminary Run

- Date: 2026-05-13
- Owner: Codex / Ahmed Sherif
- Model: Tiny Transformer, greedy decoding
- Training subset: 10000 examples
- Validation subset: full filtered validation set, 869 examples
- Epochs: 5
- Batch size: 32
- Dropout: 0.1
- d_model: 128
- Encoder layers: 2
- Decoder layers: 2
- Attention heads: 4
- FFN dimension: 512
- Learning rate: 3e-4
- Train loss by epoch:
  - Epoch 1: `6.396010`
  - Epoch 2: `5.681049`
  - Epoch 3: `5.462038`
  - Epoch 4: `5.301968`
  - Epoch 5: `5.171597`
- Validation loss by epoch:
  - Epoch 1: `5.893121`
  - Epoch 2: `5.663630`
  - Epoch 3: `5.506257`
  - Epoch 4: `5.395732`
  - Epoch 5: `5.318613`
- BLEU: `0.479979`
- Evaluation subset: 200 test examples
- Outputs:
  - `outputs/tables/training_log.csv`
  - `outputs/tables/transformer_results.csv`
  - `outputs/figures/loss_curve.png`
  - `outputs/translations/transformer_samples.csv`
- Notes: Punctuation-only repetition mostly disappeared, but generic phrase repetition remained. More training is needed before using this as the main project result.

## Final-Phase Runs (2026-06)

All runs below were AI-assisted; GenAI use is disclosed in `GenAI_Usage_Statement.md`. All
evaluation is sacreBLEU (BLEU) and chrF++ on detokenized English over the filtered test set
(8,491 examples) unless a 1,000-example subset is noted.

### Run 007 - Full 50k / 40-epoch Tiny Transformer (main model)
- Date: 2026-05-14 | Owner: Ahmed Sherif
- Train 49,441 / val 869; 40 epochs, CPU, ~5h 9m; ~4.0M params (4,006,208 trainable).
- Best epoch 40; best/final val loss 4.0166.
- Output: `outputs/checkpoints/best_full_model.pt`, `outputs/tables/full_training_log.csv`.

### Run 008 - Full-test evaluation + chrF++
- Owner: Eid Abdelrahim
- Dictionary baseline: BLEU 4.7777 / chrF++ 25.6503.
- Tiny Transformer greedy: BLEU 6.2172 / chrF++ 24.6435 (repetition ~49.8%).
- Sampling variance: BLEU 1,000-subset vs full differs < 0.07 (subset was representative).
- Output: `full_test_bleu_results.csv`, `sample_vs_full_eval.csv`, `full_test_predictions.csv`.

### Run 009 - CPU / DataLoader benchmark
- Owner: Ahmed Sherif
- 8 torch threads fastest (~200 samples/s); `cores-1` ~44% slower; DataLoader workers give no
  gain and hang in-notebook on Windows. Chosen: `TORCH_THREADS=min(8,CPU_CORES)`, `workers=0`.
- Output: `outputs/tables/cpu_dataloader_benchmark.csv` (visible in notebook 04).

### Run 010 - E13 beam search (decoding, no retrain)
- Owner: Ahmed Sherif | beam 1/3/5, length penalty 0.6/1.0.
- Best beam3 lp1.0: full-test BLEU 7.5338 / chrF++ 24.9288.

### Run 011 - E14 no_repeat_ngram decoding (no retrain)
- Owner: Ahmed Sherif | beam3 lp1.0 no_repeat_ngram_size=3.
- Full-test BLEU 7.7095 / chrF++ 26.5320; repetition 49.8% -> 15.3%. Biggest single improvement.

### Run 012 - E2 label smoothing 0.1 (training)
- Owner: Ahmed Sherif | 40 epochs, only change ls=0.1.
- greedy 6.08/25.06; beam3-nr3 7.37/25.35; repetition rose to 54.1%. NEGATIVE for the goal.

### Run 013 - E11 LR warmup (training)
- Owner: Ahmed Sherif | inverse-sqrt warmup, warmup_steps=2000, peak 3e-4.
- greedy 3.60/20.25; beam3-nr3 5.68/22.84. NEGATIVE (LR decayed too low -> undertrained).

### Run 014 - E1 dropout 0.3 (training)
- Owner: Ahmed Abdelhameed | greedy 2.65/18.31; beam3-nr3 4.69/21.15. NEGATIVE (over-regularized).

### Run 015 - E3 AdamW, weight_decay 1e-4 (training)
- Owner: Ahmed Abdelhameed | best val loss 4.0062 (beats original 4.0166).
- greedy 5.75/24.58; beam3-nr3 7.91/26.93. Only training change that helped.

### Run 016 - E6 weight tying (training)
- Owner: Ahmed Sherif | 2,982,208 params (-25%); beam3-nr3 7.72/25.59. Efficiency, not quality.

### Run 017 - E15 final decoding sweep on the E3 checkpoint (no retrain)
- Owner: Ahmed Sherif | beam/length-penalty/no_repeat sweep on `outputs/checkpoints/e3/best.pt`.
- **Best: beam3 lp1.2 nr3 -> full-test BLEU 8.1231 / chrF++ 27.6255 (repetition 19.2%).**

### Final model
**E3 (AdamW, weight_decay 1e-4) + beam=3, length_penalty=1.2, no_repeat_ngram_size=3 ->
BLEU 8.1231 / chrF++ 27.6255** on the 8,491-sentence test set; beats the dictionary baseline on
both metrics. Decoding (`no_repeat_ngram`) was the biggest lever; among training changes only
AdamW helped. See `outputs/tables/final_ablation_results.csv` and `report_notes/ablation_notes.md`.

## Pending Runs

### Planned - Tiny Transformer 10k Extended Run (SUPERSEDED by Run 007, the full 40-epoch run)

- Training subset: 10000 examples
- Validation subset: full filtered validation set, 869 examples
- Epochs: 15-20
- Evaluation subset: 1000 test examples
- Required outputs:
  - `outputs/tables/training_log.csv`
  - `outputs/tables/transformer_results.csv`
  - `outputs/tables/final_results.csv`
  - `outputs/figures/loss_curve.png`
  - `outputs/figures/bleu_comparison.png`
  - `outputs/translations/transformer_samples.csv`
  - `outputs/translations/sample_translations.csv`
- Notes: Do not move to full 50k training until this run is inspected.

## Commit Log Guidance

Use clear commit messages such as:

- `Add preprocessed tokenized dataset`
- `Add data inspection notebook`
- `Add dictionary baseline evaluation`
- `Add tiny transformer training notebook`
- `Add preliminary experiment outputs`
- `Add midterm report draft`

Data commits should be made by the teammate responsible for preprocessing.
 
- Initial commit: Added data preparation and tokenization notebooks. Consolidated pipeline and plots for Ar/En comparison. 
 
- Fix: Uploaded Week 1 notebooks (01_data_prep and 02_analysis) for dataset cleaning, BPE tokenization, and EDA plots. 
