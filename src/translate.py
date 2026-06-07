import argparse
from pathlib import Path
from typing import List

import sentencepiece as spm
import torch

from model import TinyTransformerMT
from utils import load_config, resolve_device
from vocab import load_vocab


def load_model(config, checkpoint_path, device):
    src_vocab = load_vocab(config["data"]["src_vocab"])
    tgt_vocab = load_vocab(config["data"]["tgt_vocab"])
    model = TinyTransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=int(config["data"]["max_len"]) + 2,
        **config["model"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab


def translate_lines(lines: List[str], config, checkpoint_path, device):
    src_sp = spm.SentencePieceProcessor(model_file=config["data"]["src_spm"])
    tgt_sp = spm.SentencePieceProcessor(model_file=config["data"]["tgt_spm"])
    model, src_vocab, tgt_vocab = load_model(config, checkpoint_path, device)

    outputs = []
    for line in lines:
        pieces = src_sp.encode(line.strip(), out_type=str)
        src_ids = src_vocab.encode_pieces(pieces, add_bos=True, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        pred_ids = model.greedy_decode(
            src,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=int(config["decoding"]["max_len"]),
        )[0].tolist()
        pred_pieces = tgt_vocab.decode_ids(pred_ids, skip_special=True)
        outputs.append(tgt_sp.decode(pred_pieces))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/transformer_tiny.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="UTF-8 text file with raw Arabic sentences.")
    parser.add_argument("--output", default="outputs/translations/predictions.txt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"]["device"])
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    predictions = translate_lines(lines, config, args.checkpoint, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} translations to {output_path}")


if __name__ == "__main__":
    main()

