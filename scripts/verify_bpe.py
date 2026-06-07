"""Read-only BPE verification (does NOT retrain anything).

Loads the existing SentencePiece models and the committed .bpe files and checks:
  - vocabulary sizes and special-token IDs,
  - detokenize -> re-tokenize round-trip on sample lines,
  - that every token in the .bpe files is in the model vocabulary.

Outputs:
  outputs/tables/bpe_config.csv
  outputs/tables/bpe_roundtrip_check.csv
  outputs/tables/bpe_file_verification.csv
"""

import sys
from pathlib import Path
import pandas as pd
import sentencepiece as spm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

AR_MODEL = "Data/vocab/sp_ar.model"
EN_MODEL = "Data/vocab/sp_en.model"
TOK = "Data/tokenized"


def load(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def config_rows():
    rows = []
    for lang, mp in [("ar", AR_MODEL), ("en", EN_MODEL)]:
        sp = load(mp)
        rows.append({
            "lang": lang, "model": mp, "vocab_size": sp.get_piece_size(),
            "pad_id": sp.pad_id(), "unk_id": sp.unk_id(),
            "bos_id": sp.bos_id(), "eos_id": sp.eos_id(),
        })
    return pd.DataFrame(rows)


def roundtrip_rows(n=5):
    rows = []
    for lang, mp, fn in [("ar", AR_MODEL, f"{TOK}/test.ar.bpe"), ("en", EN_MODEL, f"{TOK}/test.en.bpe")]:
        sp = load(mp)
        lines = Path(fn).read_text(encoding="utf-8").splitlines()[:n]
        for i, line in enumerate(lines):
            pieces = line.split()
            text = sp.decode(pieces)              # subwords -> natural text
            re_pieces = sp.encode(text, out_type=str)  # natural text -> subwords
            rows.append({
                "lang": lang, "idx": i,
                "decoded_text": text[:80],
                "n_pieces": len(pieces), "n_repieces": len(re_pieces),
                "pieces_match": pieces == re_pieces,
            })
    return pd.DataFrame(rows)


def file_verification_rows():
    rows = []
    for lang, mp in [("ar", AR_MODEL), ("en", EN_MODEL)]:
        sp = load(mp)
        vocab = {sp.id_to_piece(i) for i in range(sp.get_piece_size())}
        for split in ["train", "validation", "test"]:
            fn = f"{TOK}/{split}.{lang}.bpe"
            lines = Path(fn).read_text(encoding="utf-8").splitlines()
            oov = 0
            checked = 0
            for line in lines:
                for tok in line.split():
                    checked += 1
                    if tok not in vocab:
                        oov += 1
            rows.append({"split": split, "lang": lang, "lines": len(lines),
                         "tokens_checked": checked, "tokens_not_in_vocab": oov})
    return pd.DataFrame(rows)


def main():
    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)

    cfg = config_rows()
    cfg.to_csv(out / "bpe_config.csv", index=False)
    rt = roundtrip_rows()
    rt.to_csv(out / "bpe_roundtrip_check.csv", index=False)
    fv = file_verification_rows()
    fv.to_csv(out / "bpe_file_verification.csv", index=False)

    print("== bpe_config =="); print(cfg.to_string(index=False))
    print("\n== roundtrip (pieces_match should be True) =="); print(rt.to_string(index=False))
    print("\n== file verification (tokens_not_in_vocab should be 0) =="); print(fv.to_string(index=False))


if __name__ == "__main__":
    main()
