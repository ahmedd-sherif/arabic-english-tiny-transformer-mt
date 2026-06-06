import argparse
import csv
import json
import math
import sys
import random
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline_dictionary import build_position_dictionary, translate as baseline_translate
from dataset import TranslationDataset, collate_batch
from model import TinyTransformerMT
from utils import ensure_dir, load_config, resolve_device, set_seed
from vocab import load_vocab


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def decode_bpe_lines(lines, model_path):
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return [processor.decode(line.split()) for line in lines]


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def load_history_from_csv(path):
    if not Path(path).exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_epoch(model, loader, criterion, optimizer, device, clip_grad_norm=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(loader, start=1):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

        tokens = tgt_output.ne(model.tgt_pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

        if train and batch_idx % 100 == 0:
            print(f"  batch {batch_idx}/{len(loader)} loss={total_loss / max(1, total_tokens):.4f}", flush=True)

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def greedy_predict_bpe(model, dataset, src_vocab, tgt_vocab, device, max_len, batch_size):
    collate = partial(collate_batch, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    src_bpe, ref_bpe, hyp_bpe = [], [], []

    model.eval()
    for batch in loader:
        src = batch["src"].to(device)
        pred = model.greedy_decode(src, bos_id=tgt_vocab.bos_id, eos_id=tgt_vocab.eos_id, max_len=max_len)
        for src_ids, pred_ids, ref_ids in zip(src.cpu().tolist(), pred.cpu().tolist(), batch["tgt_output"].tolist()):
            src_bpe.append(" ".join(src_vocab.decode_ids(src_ids, stop_at_eos=True)))
            hyp_bpe.append(" ".join(tgt_vocab.decode_ids(pred_ids, stop_at_eos=True)))
            ref_bpe.append(" ".join(tgt_vocab.decode_ids(ref_ids, stop_at_eos=True)))

    return src_bpe, ref_bpe, hyp_bpe


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    best_epoch,
    bad_epochs,
    config,
    history,
    train_loss=None,
    val_loss=None,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "config": config,
            "history": history,
            "seed": config["seed"],
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def merge_history(checkpoint_history, csv_history):
    by_epoch = {}
    for row in checkpoint_history or []:
        by_epoch[int(row["epoch"])] = row
    for row in csv_history or []:
        by_epoch[int(row["epoch"])] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def load_latest_if_available(path, model, optimizer, device, log_path):
    if not path.exists():
        return 1, float("inf"), 0, 0, []

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint["model_state"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint["epoch"])
    best_val_loss = float(checkpoint["best_val_loss"])
    best_epoch = int(checkpoint.get("best_epoch", epoch))
    bad_epochs = int(checkpoint.get("bad_epochs", 0))
    csv_history = load_history_from_csv(log_path)
    history = merge_history(checkpoint.get("history", []), csv_history)
    print(f"Resuming training from epoch {epoch}", flush=True)
    print(f"Loaded {len(csv_history)} previous epochs from {log_path}", flush=True)
    print(f"Next epoch: {epoch + 1}, best_val_loss={best_val_loss:.4f}", flush=True)
    return epoch + 1, best_val_loss, best_epoch, bad_epochs, history


def save_training_outputs(history, tables_dir, figures_dir):
    log_path = tables_dir / "full_training_log.csv"
    df = pd.DataFrame(history)
    df.to_csv(log_path, index=False)

    if not df.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Full Tiny Transformer Training Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "full_loss_curve.png", dpi=200)
        plt.close()


def evaluate_and_save(model, config, src_vocab, tgt_vocab, device, split_size, batch_size):
    tables_dir = ensure_dir("outputs/tables")
    figures_dir = ensure_dir("outputs/figures")
    translations_dir = ensure_dir("outputs/translations")

    test_dataset = TranslationDataset(
        config["data"]["test_src"],
        config["data"]["test_tgt"],
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
    )
    if split_size and split_size < len(test_dataset.examples):
        test_dataset.examples = test_dataset.examples[:split_size]

    src_bpe, ref_bpe, hyp_bpe = greedy_predict_bpe(
        model,
        test_dataset,
        src_vocab,
        tgt_vocab,
        device,
        max_len=int(config["decoding"]["max_len"]),
        batch_size=batch_size,
    )

    src_text = decode_bpe_lines(src_bpe, config["data"]["src_spm"])
    ref_text = decode_bpe_lines(ref_bpe, config["data"]["tgt_spm"])
    hyp_text = decode_bpe_lines(hyp_bpe, config["data"]["tgt_spm"])
    transformer_bleu = sacrebleu.corpus_bleu(hyp_text, [ref_text])

    train_src = read_lines(config["data"]["train_src"])
    train_tgt = read_lines(config["data"]["train_tgt"])
    test_src = read_lines(config["data"]["test_src"])[: len(hyp_text)]
    test_tgt = read_lines(config["data"]["test_tgt"])[: len(hyp_text)]
    dictionary = build_position_dictionary(train_src, train_tgt)
    baseline_bpe = [baseline_translate(line, dictionary) for line in test_src]
    baseline_text = decode_bpe_lines(baseline_bpe, config["data"]["tgt_spm"])
    baseline_ref_text = decode_bpe_lines(test_tgt, config["data"]["tgt_spm"])
    baseline_src_text = decode_bpe_lines(test_src, config["data"]["src_spm"])
    baseline_bleu = sacrebleu.corpus_bleu(baseline_text, [baseline_ref_text])

    transformer_results = pd.DataFrame(
        [
            {
                "model": "tiny_transformer_full_greedy",
                "decoding": "greedy",
                "examples": len(hyp_text),
                "bleu": transformer_bleu.score,
                "bleu_summary": str(transformer_bleu),
            }
        ]
    )
    transformer_results.to_csv(tables_dir / "transformer_results.csv", index=False)

    final_results = pd.DataFrame(
        [
            {
                "model": "dictionary_position_baseline",
                "decoding": "position_dictionary",
                "examples": len(baseline_text),
                "bleu": baseline_bleu.score,
                "bleu_summary": str(baseline_bleu),
            },
            {
                "model": "tiny_transformer_full_greedy",
                "decoding": "greedy",
                "examples": len(hyp_text),
                "bleu": transformer_bleu.score,
                "bleu_summary": str(transformer_bleu),
            },
        ]
    )
    final_results.to_csv(tables_dir / "final_results.csv", index=False)

    predictions = pd.DataFrame(
        {
            "arabic_input": src_text,
            "reference_english": ref_text,
            "transformer_output": hyp_text,
        }
    )
    predictions.to_csv(translations_dir / "full_transformer_predictions.csv", index=False)

    sample_count = min(25, len(hyp_text), len(baseline_text))
    samples = pd.DataFrame(
        {
            "arabic_input": baseline_src_text[:sample_count],
            "reference_english": baseline_ref_text[:sample_count],
            "baseline_output": baseline_text[:sample_count],
            "transformer_output": hyp_text[:sample_count],
        }
    )
    samples.to_csv(translations_dir / "full_transformer_samples.csv", index=False)
    samples.to_csv(translations_dir / "sample_translations.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(final_results["model"], final_results["bleu"])
    plt.ylabel("BLEU")
    plt.title("BLEU Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "bleu_comparison.png", dpi=200)
    plt.close()

    return baseline_bleu, transformer_bleu, samples


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/transformer_tiny.yaml")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--continue-to", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--eval-size", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    config["seed"] = 42
    config["data"]["max_len"] = 80
    config["model"].update(
        {
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1,
        }
    )
    config["training"].update(
        {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "early_stopping_patience": args.patience,
        }
    )
    config["decoding"]["max_len"] = 80

    set_seed(config["seed"])
    device = resolve_device(config["training"]["device"])
    tables_dir = ensure_dir("outputs/tables")
    figures_dir = ensure_dir("outputs/figures")
    checkpoints_dir = ensure_dir("outputs/checkpoints")
    best_path = checkpoints_dir / "best_full_model.pt"
    latest_path = checkpoints_dir / "latest_full_model.pt"

    src_vocab = load_vocab(config["data"]["src_vocab"])
    tgt_vocab = load_vocab(config["data"]["tgt_vocab"])

    train_dataset = TranslationDataset(
        config["data"]["train_src"],
        config["data"]["train_tgt"],
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
    )
    val_dataset = TranslationDataset(
        config["data"]["val_src"],
        config["data"]["val_tgt"],
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
    )

    collate = partial(collate_batch, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
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

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_id,
        label_smoothing=float(config["training"]["label_smoothing"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    log_path = tables_dir / "full_training_log.csv"
    start_epoch, best_val_loss, best_epoch, bad_epochs, history = load_latest_if_available(
        latest_path, model, optimizer, device, log_path
    )
    if history:
        save_training_outputs(history, tables_dir, figures_dir)

    print(f"Device: {device}", flush=True)
    print(f"Train examples used: {train_dataset.stats.kept}/{train_dataset.stats.total}", flush=True)
    print(f"Validation examples used: {val_dataset.stats.kept}/{val_dataset.stats.total}", flush=True)
    print(f"Max epochs: {args.max_epochs}, continue-to cap: {args.continue_to}, patience: {args.patience}", flush=True)

    previous_elapsed = float(history[-1].get("total_elapsed_time", 0.0)) if history else 0.0
    completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.continue_to + 1):
        if epoch > args.max_epochs and bad_epochs > 0:
            print("Reached max_epochs and validation is not still improving. Stopping before continue-to cap.", flush=True)
            break

        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.continue_to}", flush=True)
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            clip_grad_norm=float(config["training"]["clip_grad_norm"]),
            train=True,
        )
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_epoch,
                bad_epochs,
                config,
                history,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print("New best checkpoint saved.", flush=True)
        else:
            bad_epochs += 1
            print(f"No improvement. Patience counter: {bad_epochs}/{args.patience}", flush=True)

        epoch_time = time.time() - epoch_start
        elapsed = previous_elapsed + epoch_time
        previous_elapsed = elapsed
        completed_epoch = epoch
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "learning_rate": get_lr(optimizer),
            "epoch_time": epoch_time,
            "total_elapsed_time": elapsed,
            "checkpoint_improved": improved,
        }
        history.append(row)
        epoch_path = checkpoints_dir / f"full_model_epoch_{epoch:03d}.pt"
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            best_epoch,
            bad_epochs,
            config,
            history,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            best_epoch,
            bad_epochs,
            config,
            history,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        save_training_outputs(history, tables_dir, figures_dir)

        avg_epoch_time = elapsed / max(1, epoch - start_epoch + 1)
        remaining_by_patience = max(0, args.patience - bad_epochs)
        remaining_epochs = min(args.continue_to - epoch, remaining_by_patience if bad_epochs else args.continue_to - epoch)
        eta = avg_epoch_time * max(0, remaining_epochs)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val_loss={best_val_loss:.4f} elapsed={format_duration(elapsed)} "
            f"epoch_time={format_duration(epoch_time)} eta~{format_duration(eta)} improved={improved}",
            flush=True,
        )

        if bad_epochs >= args.patience:
            print("Early stopping triggered.", flush=True)
            break

    print("\nLoading best checkpoint for evaluation...", flush=True)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint["model_state"]))
    baseline_bleu, transformer_bleu, samples = evaluate_and_save(
        model,
        config,
        src_vocab,
        tgt_vocab,
        device,
        split_size=args.eval_size,
        batch_size=int(config["training"]["batch_size"]),
    )

    final = history[-1] if history else {}
    total_time = float(final.get("total_elapsed_time", previous_elapsed))
    summary = {
        "train_examples": train_dataset.stats.kept,
        "validation_examples": val_dataset.stats.kept,
        "epochs_completed": completed_epoch,
        "total_training_time": format_duration(total_time),
        "best_epoch": best_epoch,
        "final_train_loss": final.get("train_loss"),
        "best_validation_loss": best_val_loss,
        "final_validation_loss": final.get("val_loss"),
        "transformer_bleu": transformer_bleu.score,
        "baseline_bleu": baseline_bleu.score,
        "transformer_beats_baseline": transformer_bleu.score > baseline_bleu.score,
    }
    (tables_dir / "full_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal summary", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print("\nSample translations", flush=True)
    print(samples.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
