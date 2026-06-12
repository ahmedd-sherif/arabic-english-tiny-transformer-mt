# Low-Resource Arabic→English NMT with a Tiny Transformer

A Master's Deep Learning project: training a compact encoder–decoder Transformer **from scratch,
CPU-first**, for Arabic→English translation in a low-resource setting (~50k sentence pairs), then
improving it through a controlled study of training and decoding choices grouped into **six
categories (A–F), 22 experiments**. Every notebook is **self-contained** — the model, training
loop and decoding are defined inside them; no project `.py` imports needed.

- **Data:** IWSLT 2017 Arabic–English (TED talks); 231,713 raw pairs → 50,000 sampled after filtering.
- **Tokenization:** SentencePiece BPE, separate 8,000-piece Arabic and English vocabularies.
- **Model:** tiny Transformer — d_model=128, 4 heads, 2 encoder / 2 decoder layers, FFN=512 (~4M parameters).
- **Training:** 60 epochs, seed 42, Adam lr=3e-4, early stopping. CPU-first by design (see *Compute disclosure* below).

---

## Final result (full test set — 8,491 examples)

**Final system: weight decay 1e-4, decoded with beam=10, length-penalty=1.2, no_repeat_ngram_size=3.**

| Metric | Dictionary baseline | Final model |
|---|---|---|
| BLEU | 4.78 | **19.91** |
| chrF++ | 25.65 | **42.24** |
| Repetition | — | 4.6% |

The final model far outperforms the dictionary baseline on both metrics.

**Key finding — weight decay is the dominant factor.** Among all 22 training ablations, adding Adam
weight decay `1e-4` produces by far the largest improvement (BLEU 18.79 vs ~6–10 for all other
changes). This was **verified with a control run (B3):** identical code, weight decay turned off —
the control lands at validation loss ≈ 4.0 exactly like the plain baseline, while the
weight-decay run reaches ≈ 2.88. The gain is genuinely weight decay.

---

## Notebook workflow (run in order)

| # | Notebook | Purpose |
|---|---|---|
| 01 | `notebooks/01_data_preparation.ipynb` | Download IWSLT 2017, filter, deduplicate, sample 50k, save clean splits |
| 02 | `notebooks/02_analysis_and_tokenization.ipynb` | EDA plots, train SentencePiece BPE (8K vocab each), encode all splits, verify roundtrip |
| 03 | `notebooks/03_dictionary_baseline.ipynb` | Non-neural dictionary baseline — most frequent English piece per Arabic position, BLEU + chrF++ |
| 04 | `notebooks/04_training_ablation_experiments.ipynb` | **Training ablations (categories A–F, 22 experiments)** — one Config per experiment, shared `load_or_train`; leaderboard + top-3 checkpoints |
| 05 | `notebooks/05_decoding_ablation_experiments.ipynb` | **Decoding sweep** on the top-3 checkpoints — 7 configs × 3 models → full-test evaluation → locks the final model |
| 06 | `notebooks/06_final_results_and_error_analysis.ipynb` | Final model vs baseline, ablation summary, categorized error analysis, example translations |

Each notebook loads its inputs from the previous step's outputs. Checkpoints are loaded if they
exist (`load_or_train` logic), so re-running is fast. Original exploratory drafts are in
`notebooks/archive_original/`.

---

## Experiment categories

Checkpoints and per-experiment training logs mirror the category hierarchy:

```
outputs/checkpoints/
  baseline/                              # plain Adam (reference)
  _control/                              # B3 same-code control (wd=0)
  A_training_objective/
    exp1_label_smoothing/                # A1: label smoothing 0.1
    exp2_focal_loss/                     # A2: focal loss γ=2
  B_regularization/
    exp1_dropout/                        # B1: dropout 0.3
    exp2_weight_decay/                   # B2: weight decay 1e-4  ← winner
    exp3_rdrop/                          # B4: R-Drop α=0.7
  C_optimization/
    exp1_adamw/                          # C1: AdamW wd 1e-4
    exp2_lr_warmup/                      # C2: inverse-sqrt warmup
    exp3_cosine_scheduler/               # C3: cosine annealing
    exp4_curriculum/                     # C4: curriculum learning
  D_architecture/
    exp1_weight_tying/                   # D1: weight tying
    exp2_layers/                         # D2: 4+4 layers
    exp3_embedding_size/                 # D3: d_model=256
    exp4_depth1/                         # D4: 1 layer
    exp5_depth3/                         # D5: 3 layers
    exp6_width64/                        # D6: d_model=64
    exp7_gelu/                           # D7: GELU activation
  E_data/
    exp1_size5k/  exp2_size10k/  exp3_size25k/  exp4_rareword/
    bpe_vocab_4000/  bpe_vocab_12000/
  F_combined/
    exp1_combined/                       # F1: wd + ls + tying + warmup
  baselines/
    lstm_seq2seq/                        # LSTM seq2seq comparison
  legacy_40epoch/                        # original 40-epoch midterm checkpoint
```

`.pt` files are gitignored. Training logs are at
`outputs/tables/<category>/<experiment>/training_log.csv`.

---

## Ablation results (top results by training BLEU, beam3 nr3 on 1k subset)

| Rank | Experiment | BLEU | chrF++ | Rep% |
|---|---|---|---|---|
| 1 | **B2: Weight decay 1e-4** | **18.79** | **41.04** | 4.8 |
| 2 | F1: Combined (wd+ls+tie+warmup) | 12.15 | 33.07 | 14.5 |
| 3 | D5: 3 encoder/decoder layers | 9.63 | 30.42 | 13.8 |
| 4 | A1: Label smoothing 0.1 | 9.03 | 28.52 | 13.1 |
| 5 | D2: 4+4 layers | 8.86 | 29.31 | 13.4 |
| — | Reference (plain Adam) | 7.99 | 26.86 | 10.1 |

Full results in `outputs/tables/training_comparison.csv`.

---

## Compute disclosure

The project is CPU-first by design; all evaluation and decoding ran on CPU.
**Some training experiments were run on an NVIDIA RTX 5000 GPU** to reduce wall-clock time
(specifically: B2 weight decay, D2 4+4 layers, D3 d_model=256, B4 R-Drop, C4 Curriculum,
D4 depth-1, D5 depth-3, D6 d_model=64, E1–E4 data-size, F1 combined — 13 experiments total).
All other experiments ran on CPU. The training logs carry a `device` column (e.g.,
`cuda (RTX 5000)` or `cpu`) and real per-epoch times. Checkpoints are saved CPU-portable and
load on any hardware.

---

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Select the Jupyter kernel **"Low Resource MT (.venv Python 3.12)"**.

A separate `.venv_gpu` with a CUDA build of torch was used only during the GPU training runs;
it is gitignored.

---

## Repository layout

```
notebooks/          # 6 numbered notebooks (run in order)
src/                # model, dataset, training utilities (mirrored in notebooks)
config/             # hyperparameter YAML
Data/               # tokenized BPE files and vocab (gitignored)
outputs/
  checkpoints/      # saved .pt checkpoints (gitignored)
  tables/           # CSVs: training logs, evaluation results, ablation summary
  figures/          # plots (loss curves, BLEU comparison, error analysis)
  translations/     # prediction CSV files
  examples/         # qualitative translation examples
report_notes/       # evidence notes for the written report
```

---

## Limitations

Absolute BLEU is still modest for a tiny, from-scratch model on 50k pairs. The final recipe
(weight decay + `no_repeat_ngram` decoding) reaches BLEU ~20 / chrF++ ~42 with ~5% repetition
and ~87% adequate outputs. Future work: more training data, human evaluation, attention
heatmaps, COMET metric, checkpoint ensembling.
