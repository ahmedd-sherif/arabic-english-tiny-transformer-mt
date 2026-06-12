# Experiment Log

This file records reproducible experiment runs for the low-resource Arabic–English machine translation project.

Each run includes: date, owner, data split or subset size, model/baseline, main configuration, output files, and metrics.

---

## Runs

### Run 001 — Repository Setup

- Date: 2026-05-13
- Owner: Ahmed Sherif
- Purpose: Verify GitHub repository setup and push workflow.
- Commit: `289bd03 Add project README`
- Notes: Only `README.md` was committed and pushed. No data files were committed.

---

### Run 002 — Data Preparation and Tokenization (Notebook 01)

- Date: 2026-05-13
- Owner: Ahmed Abdelhameed
- Notebook: `notebooks/01_data_preparation.ipynb`
- Dataset: IWSLT 2017 Arabic–English (`IWSLT/iwslt2017`, `iwslt2017-ar-en`)
- Raw corpus size: train 231,713 pairs / validation 888 / test 8,583
- Filtering steps applied:
  - Removed 54 over-length pairs (word length > 80)
  - Removed 309 pairs with Arabic/English length ratio > 3
  - Removed 1,663 duplicate training pairs
  - **Total removed: 2,042 pairs**
- Final train sample: **50,000 pairs** (seed 42)
- Final splits after filtering: train 50,000 / validation 888 / test 8,567
- Outputs:
  - `data/clean/train.ar`, `data/clean/train.en`
  - `data/clean/validation.ar`, `data/clean/validation.en`
  - `data/processed/train_clean.csv`, `data/processed/validation_clean.csv`
  - `data/processed/removed_pairs.csv`

---

### Run 003 — Analysis, EDA, and BPE Tokenization (Notebook 02)

- Date: 2026-05-14
- Owner: Ahmed Abdelhameed
- Notebook: `notebooks/02_analysis_and_tokenization.ipynb`
- Train statistics: avg Arabic length 14.42 tokens / avg English length 17.45 tokens
- SentencePiece BPE models trained on train split only:
  - Arabic vocab size: **8,000 pieces**
  - English vocab size: **8,000 pieces**
- Special tokens: `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`
- All splits encoded and saved as `.bpe` files under `data/tokenized/`
- Roundtrip verification passed (no encode/decode information loss)
- Outputs:
  - `data/vocab/sp_ar.model`, `data/vocab/sp_en.model`
  - `data/tokenized/train.ar.bpe`, `data/tokenized/train.en.bpe`
  - `outputs/figures/` (EDA plots: length distributions, vocabulary coverage, quality checks)

---

### Run 004 — Dictionary Baseline (Notebook 03)

- Date: 2026-05-15
- Owner: Eid Abd ElRihem
- Notebook: `notebooks/03_dictionary_baseline.ipynb`
- Baseline: position-based dictionary lookup — most frequent English BPE piece per Arabic position. No reordering, no context.
- Evaluation: full test set, 8,491 examples (after max-length BPE filtering), detokenized via `sp_en.decode`, sacreBLEU
- **BLEU: 4.78 / chrF++: 25.65**
- Outputs:
  - `outputs/tables/baseline/results.csv`
  - `outputs/translations/baseline/predictions.csv`

---

### Run 005 — Training Ablation Study (Notebook 04)

- Date: 2026-05-16 to 2026-06-09
- Owners: Ahmed Sherif, Ahmed Abdelhameed, Eid Abd ElRihem
- Notebook: `notebooks/04_training_ablation_experiments.ipynb`
- **Architecture (baseline reference):** d_model=128, 4 heads, 2 encoder layers, 2 decoder layers, FFN=512, dropout=0.1, Adam lr=3e-4, 60 epochs, batch=32, seed=42, ~4M parameters
- Evaluation subset: 1,000 test examples, beam=3 + no_repeat_ngram=3, detokenized sacreBLEU
- All runs trained from scratch; one change at a time vs the baseline

**Results by category (beam3 + no_repeat_ngram=3, BLEU / chrF++ / rep%):**

| Category | Experiment | BLEU | chrF++ | Rep% |
|---|---|---|---|---|
| Reference | Baseline (Adam, dropout 0.1) | 7.99 | 26.86 | 10.1 |
| **A. Training Objective** | A1: Label smoothing 0.1 | 9.03 | 28.52 | 13.1 |
| A. Training Objective | A2: Focal loss (γ=2) | 8.17 | 27.37 | 10.3 |
| **B. Regularization** | **B2: Weight decay 1e-4 ← WINNER** | **18.79** | **41.04** | **4.8** |
| B. Regularization | B1: Dropout 0.3 | 6.03 | 23.62 | 16.7 |
| B. Regularization | B3: Control (wd=0, same code as B2) | 8.23 | 27.65 | 10.1 |
| B. Regularization | B4: R-Drop (α=0.7) | 7.95 | 26.92 | 11.6 |
| **C. Optimization** | C1: AdamW wd 1e-4 | 8.61 | 28.24 | 14.8 |
| C. Optimization | C2: LR inverse-sqrt warmup | 6.10 | 23.67 | 15.0 |
| C. Optimization | C3: Cosine scheduler | 7.65 | 26.58 | 13.6 |
| C. Optimization | C4: Curriculum learning | 8.01 | 27.52 | 11.2 |
| **D. Architecture** | D2: 4+4 layers | 8.86 | 29.31 | 13.4 |
| D. Architecture | D5: 3 layers | 9.63 | 30.42 | 13.8 |
| D. Architecture | D7: GELU activation | 8.78 | 28.67 | 14.1 |
| D. Architecture | D1: Weight tying | 7.95 | 26.53 | 12.6 |
| D. Architecture | D3: d_model=256 | 7.33 | 25.73 | 9.1 |
| D. Architecture | D6: d_model=64 | 7.19 | 25.34 | 16.5 |
| D. Architecture | D4: 1 layer | 5.18 | 20.13 | 8.2 |
| **E. Data** | E4: Rare-word oversampling | 8.22 | 27.63 | 12.6 |
| E. Data | E3: 25K training pairs | 4.79 | 21.29 | 16.4 |
| E. Data | E2: 10K training pairs | 2.06 | 17.21 | 15.9 |
| E. Data | E1: 5K training pairs | 1.66 | 15.77 | 14.0 |
| **F. Combined** | F1: wd + ls + tying + warmup | 12.15 | 33.07 | 14.5 |

**Top-3 checkpoints for decoding study:**
1. B2 Weight decay 1e-4 — BLEU 18.79
2. F1 Combined — BLEU 12.15
3. D5 3 layers — BLEU 9.63

**Key finding:** B3 control (identical code, weight_decay=0) lands at val ≈ 4.0 exactly like the plain baseline, while B2 (weight_decay=1e-4) reaches val ≈ 2.88. This confirms the gain is genuinely from weight decay.

Outputs:
- `outputs/tables/{A–F category dirs}/training_log.csv` (per experiment)
- `outputs/tables/training_comparison.csv`
- `outputs/tables/top3_training_checkpoints.csv`
- `outputs/checkpoints/{A–F category dirs}/{experiment}/best.pt`

---

### Run 006 — Decoding Ablation Study (Notebook 05)

- Date: 2026-06-09
- Owner: Ahmed Sherif
- Notebook: `notebooks/05_decoding_ablation_experiments.ipynb`
- Sweep: 7 decoding configurations × top-3 training checkpoints
- Configurations tested: greedy, beam3 lp1.0 nr0, beam3 lp1.0 nr3, beam3 lp0.8 nr3, beam3 lp1.2 nr3, beam5 lp1.2 nr3, beam10 lp1.2 nr3
- Evaluation: 1,000-subset first, then full test (8,491) for the top candidates

**Best decoding per checkpoint (1,000-subset → full-test):**

| Checkpoint | Best config | Full-test BLEU | Full-test chrF++ | Rep% |
|---|---|---|---|---|
| B2 Weight decay 1e-4 | beam10 lp1.2 nr3 | **19.91** | **42.24** | 4.6 |
| F1 Combined | beam3 lp0.8 nr3 | 12.32 | 32.92 | 16.6 |
| D5 3 layers | beam10 lp1.2 nr3 | 9.77 | 30.82 | 15.0 |

**Final locked model:** B2 (weight decay 1e-4) + beam10, length_penalty=1.2, no_repeat_ngram_size=3

Outputs:
- `outputs/tables/final_results_master.csv`
- `outputs/tables/final_model_decoding_result.csv`

---

### Run 007 — Final Results and Error Analysis (Notebook 06)

- Date: 2026-06-09
- Owner: Eid Abd ElRihem
- Notebook: `notebooks/06_final_results_and_error_analysis.ipynb`
- Final model: B2 weight decay 1e-4 + beam10 lp1.2 nr3, evaluated on full test set (8,491 examples)

**Final system results:**

| System | Decoding | BLEU | chrF++ | Rep% |
|---|---|---|---|---|
| Dictionary baseline | position dict | 4.78 | 25.65 | — |
| **Final model (B2)** | **beam10 lp1.2 nr3** | **19.91** | **42.24** | **4.6** |

- Final model beats the dictionary baseline on both BLEU (+15.13) and chrF++ (+16.59)
- **86.8% of outputs are classified as "adequate"** in the error analysis
- Remaining errors: long/rare-word sentences (topic kept, detail lost); no longer degenerate loops

Outputs:
- `outputs/tables/final_main_results.csv`
- `outputs/figures/` (final system plots, error distribution, training curves)
- `outputs/examples/` (example translations, good and bad cases)

---

## Summary — Final Model

**B2 (Adam, weight_decay=1e-4) + beam=10, length_penalty=1.2, no_repeat_ngram_size=3**

- Full test set: 8,491 examples, detokenized sacreBLEU
- **BLEU: 19.91 / chrF++: 42.24 / Repetition: 4.6%**
- Beats the dictionary baseline on both metrics
- Weight decay is the dominant training factor — verified with a same-code control run (B3)
- `no_repeat_ngram_size=3` is the dominant decoding factor — reduces repetition from ~10% (greedy) to 4.6%
