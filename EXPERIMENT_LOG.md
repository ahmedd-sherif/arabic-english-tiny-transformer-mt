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

## Pending Runs

### Planned - Tiny Transformer 10k Extended Run

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
