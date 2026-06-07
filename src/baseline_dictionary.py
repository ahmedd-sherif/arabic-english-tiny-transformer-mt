import argparse
from collections import Counter, defaultdict
from pathlib import Path

import sacrebleu


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def build_position_dictionary(src_lines, tgt_lines):
    counts = defaultdict(Counter)
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        src_pieces = src_line.split()
        tgt_pieces = tgt_line.split()
        if not src_pieces or not tgt_pieces:
            continue
        for index, src_piece in enumerate(src_pieces):
            tgt_index = min(round(index * (len(tgt_pieces) - 1) / max(1, len(src_pieces) - 1)), len(tgt_pieces) - 1)
            counts[src_piece][tgt_pieces[tgt_index]] += 1
    return {src_piece: tgt_counts.most_common(1)[0][0] for src_piece, tgt_counts in counts.items()}


def translate(line, dictionary):
    return " ".join(dictionary.get(piece, "<unk>") for piece in line.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", default="Data/tokenized/train.ar.bpe")
    parser.add_argument("--train-tgt", default="Data/tokenized/train.en.bpe")
    parser.add_argument("--eval-src", default="Data/tokenized/test.ar.bpe")
    parser.add_argument("--eval-tgt", default="Data/tokenized/test.en.bpe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="outputs/translations/dictionary_baseline.bpe")
    args = parser.parse_args()

    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt)
    eval_src = read_lines(args.eval_src)
    eval_tgt = read_lines(args.eval_tgt)
    if args.limit:
        eval_src = eval_src[: args.limit]
        eval_tgt = eval_tgt[: args.limit]

    dictionary = build_position_dictionary(train_src, train_tgt)
    hypotheses = [translate(line, dictionary) for line in eval_src]
    bleu = sacrebleu.corpus_bleu(hypotheses, [eval_tgt])
    print(bleu)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    print(f"Wrote baseline translations to {output_path}")


if __name__ == "__main__":
    main()

