import argparse
from functools import partial
from pathlib import Path

import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TranslationDataset, collate_batch
from model import TinyTransformerMT
from utils import load_config, resolve_device
from vocab import load_vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/transformer_tiny.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of examples for quick evaluation.")
    parser.add_argument("--output", default="outputs/translations/eval_samples.tsv")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"]["device"])
    src_vocab = load_vocab(config["data"]["src_vocab"])
    tgt_vocab = load_vocab(config["data"]["tgt_vocab"])

    src_key = "val_src" if args.split == "validation" else "test_src"
    tgt_key = "val_tgt" if args.split == "validation" else "test_tgt"
    dataset = TranslationDataset(
        config["data"][src_key],
        config["data"][tgt_key],
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
    )
    if args.limit:
        dataset.examples = dataset.examples[: args.limit]

    collate = partial(collate_batch, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate,
    )

    model = TinyTransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=int(config["data"]["max_len"]) + 2,
        **config["model"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    hypotheses = []
    references = []
    sources = []

    for batch in tqdm(loader, desc="decode"):
        src = batch["src"].to(device)
        pred = model.greedy_decode(
            src,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=int(config["decoding"]["max_len"]),
        )
        for src_ids, pred_ids, ref_ids in zip(src.cpu().tolist(), pred.cpu().tolist(), batch["tgt_output"].tolist()):
            sources.append(" ".join(src_vocab.decode_ids(src_ids, skip_special=True)))
            hypotheses.append(" ".join(tgt_vocab.decode_ids(pred_ids, skip_special=True)))
            references.append(" ".join(tgt_vocab.decode_ids(ref_ids, skip_special=True)))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(bleu)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(config["decoding"]["samples"]), len(hypotheses))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("source_bpe\treference_bpe\thypothesis_bpe\n")
        for idx in range(sample_count):
            handle.write(f"{sources[idx]}\t{references[idx]}\t{hypotheses[idx]}\n")
    print(f"Wrote samples to {output_path}")


if __name__ == "__main__":
    main()

