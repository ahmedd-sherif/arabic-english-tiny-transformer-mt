# Arabic-English Tiny Transformer MT

Low-resource Arabic-to-English machine translation using a CPU-friendly Transformer trained on pre-tokenized SentencePiece BPE data.

## Current Data Layout

The project expects the prepared data files below:

```text
Data/
  tokenized/
    train.ar.bpe
    train.en.bpe
    validation.ar.bpe
    validation.en.bpe
    test.ar.bpe
    test.en.bpe
  vocab/
    sp_ar.vocab
    sp_ar.model
    sp_en.vocab
    sp_en.model
```

The `.vocab` files contain SentencePiece subword tokens and scores, not neural embeddings. The model learns embeddings during training.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --config config/transformer_tiny.yaml
python src/evaluate.py --config config/transformer_tiny.yaml --checkpoint outputs/checkpoints/best_model.pt
```

## Notes

- The existing data files should be committed by the teammate responsible for preprocessing.
- Generated checkpoints, logs, figures, and translations are ignored by default.

