import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TranslationDataset, collate_batch
from model import TinyTransformerMT
from utils import count_parameters, ensure_dir, load_config, resolve_device, set_seed
from vocab import load_vocab


def run_epoch(model, loader, criterion, optimizer, device, clip_grad_norm=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, leave=False, desc="train" if train else "valid")
    for batch in progress:
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
        progress.set_postfix(loss=total_loss / max(1, total_tokens))

    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/transformer_tiny.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config["training"]["device"])

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

    output_dir = ensure_dir(config["training"]["output_dir"])
    best_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"

    print(f"Device: {device}")
    print(f"Train examples: {train_dataset.stats.kept}/{train_dataset.stats.total}")
    print(f"Validation examples: {val_dataset.stats.kept}/{val_dataset.stats.total}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
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
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["training"]["early_stopping_patience"]):
                print("Early stopping triggered.")
                break

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

