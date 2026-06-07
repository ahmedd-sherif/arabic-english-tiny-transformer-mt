"""Shared evaluation helpers for the notebooks.

Keeps the heavy logic in one place so the notebooks stay readable: build the model,
load a checkpoint, greedy-decode a split to detokenized English, and score it with
sacreBLEU (BLEU) and chrF++ (chrF with word_order=2). Also wraps the dictionary baseline.

All decoding/scoring matches the original `full_train.py` path: predictions are
detokenized with the English SentencePiece model before scoring, so numbers are
comparable to the midterm results.
"""

import math
import time
from functools import partial
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_batch
from model import TinyTransformerMT
from vocab import load_vocab
from baseline_dictionary import build_position_dictionary, translate as dict_translate


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_vocabs(config):
    src_vocab = load_vocab(config["data"]["src_vocab"])
    tgt_vocab = load_vocab(config["data"]["tgt_vocab"])
    return src_vocab, tgt_vocab


def build_model_config():
    """The architecture actually used for the 40-epoch run (see full_train.py)."""
    return dict(d_model=128, nhead=4, num_encoder_layers=2,
                num_decoder_layers=2, dim_feedforward=512, dropout=0.1)


def load_transformer(config, checkpoint_path, src_vocab, tgt_vocab, device="cpu"):
    model = TinyTransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=int(config["data"]["max_len"]) + 2,
        **build_model_config(),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def build_test_dataset(config, src_vocab, tgt_vocab, split="test"):
    src_key = {"test": "test_src", "validation": "val_src"}[split]
    tgt_key = {"test": "test_tgt", "validation": "val_tgt"}[split]
    return TranslationDataset(
        config["data"][src_key], config["data"][tgt_key],
        src_vocab, tgt_vocab, max_len=int(config["data"]["max_len"]),
    )


def _detok(lines, spm_path):
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    return [sp.decode(line.split()) for line in lines]


@torch.no_grad()
def transformer_predict(model, dataset, config, src_vocab, tgt_vocab,
                        device="cpu", limit=None, batch_size=32, max_len=80):
    """Greedy-decode `dataset` and return detokenized (src, ref, hyp) text lists."""
    examples = dataset.examples if limit is None else dataset.examples[:limit]
    sub = list(examples)
    collate = partial(collate_batch, src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)

    class _Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(sub)
        def __getitem__(self, i): return sub[i]

    loader = DataLoader(_Wrap(), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    src_bpe, ref_bpe, hyp_bpe = [], [], []
    for batch in loader:
        src = batch["src"].to(device)
        pred = model.greedy_decode(src, bos_id=tgt_vocab.bos_id,
                                   eos_id=tgt_vocab.eos_id, max_len=max_len)
        for s, p, r in zip(src.cpu().tolist(), pred.cpu().tolist(), batch["tgt_output"].tolist()):
            src_bpe.append(" ".join(src_vocab.decode_ids(s, stop_at_eos=True)))
            hyp_bpe.append(" ".join(tgt_vocab.decode_ids(p, stop_at_eos=True)))
            ref_bpe.append(" ".join(tgt_vocab.decode_ids(r, stop_at_eos=True)))

    src_text = _detok(src_bpe, config["data"]["src_spm"])
    ref_text = _detok(ref_bpe, config["data"]["tgt_spm"])
    hyp_text = _detok(hyp_bpe, config["data"]["tgt_spm"])
    return src_text, ref_text, hyp_text


def filter_test_pairs(config, split="test"):
    """Apply the same max_len filter as TranslationDataset and return the kept
    source/target BPE lines (strings), in dataset order. Single source of truth
    so baseline and transformer are scored on identical references."""
    src_key = {"test": "test_src", "validation": "val_src"}[split]
    tgt_key = {"test": "test_tgt", "validation": "val_tgt"}[split]
    max_len = int(config["data"]["max_len"])
    src_lines = read_lines(config["data"][src_key])
    tgt_lines = read_lines(config["data"][tgt_key])
    kept_src, kept_tgt = [], []
    for s, t in zip(src_lines, tgt_lines):
        if len(s.split()) > max_len or len(t.split()) > max_len:
            continue
        kept_src.append(s)
        kept_tgt.append(t)
    return kept_src, kept_tgt


@torch.no_grad()
def transformer_predict_from_bpe(model, src_bpe_lines, config, src_vocab, tgt_vocab,
                                 device="cpu", batch_size=32, max_len=80):
    """Greedy-decode from raw source BPE lines (guarantees positional alignment).
    Returns (detokenized hyp text, raw hyp BPE strings)."""
    enc = [src_vocab.encode_pieces(l.split(), add_bos=True, add_eos=True) for l in src_bpe_lines]
    hyp_bpe = []
    for i in range(0, len(enc), batch_size):
        chunk = enc[i:i + batch_size]
        smax = max(len(x) for x in chunk)
        src = torch.full((len(chunk), smax), src_vocab.pad_id, dtype=torch.long)
        for r, ids in enumerate(chunk):
            src[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        pred = model.greedy_decode(src.to(device), bos_id=tgt_vocab.bos_id,
                                   eos_id=tgt_vocab.eos_id, max_len=max_len)
        for p in pred.cpu().tolist():
            hyp_bpe.append(" ".join(tgt_vocab.decode_ids(p, stop_at_eos=True)))
    return _detok(hyp_bpe, config["data"]["tgt_spm"]), hyp_bpe


def detok_lines(lines, spm_path):
    return _detok(lines, spm_path)


# --- Beam search decoding (E13) -------------------------------------------------
# Encodes the source once and decodes with a fixed-width beam. length_penalty is the
# exponent in the GNMT-style normaliser: final_score = sum_logprob / length**length_penalty.
# length_penalty=1.0 is plain length normalisation (mean log-prob); 0.6 is milder.
# Sanity check: beam_size=1 reproduces greedy decoding.

@torch.no_grad()
def _encode_src(model, src):
    src_kpm = src.eq(model.src_pad_id)
    src_emb = model.positional_encoding(model.src_embedding(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_kpm)
    return memory, src_kpm


@torch.no_grad()
def _last_step_logprobs(model, ys, memory, memory_kpm):
    L = ys.size(1)
    tgt_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ys.device), diagonal=1)
    tgt_emb = model.positional_encoding(model.tgt_embedding(ys) * math.sqrt(model.d_model))
    out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                    memory_key_padding_mask=memory_kpm)
    return torch.log_softmax(model.output_projection(out[:, -1]), dim=-1)


@torch.no_grad()
def beam_search_one(model, src_ids, bos_id, eos_id, beam_size, max_len, length_penalty, device="cpu"):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, src_kpm = _encode_src(model, src)
    memory = memory.expand(beam_size, -1, -1).contiguous()
    src_kpm = src_kpm.expand(beam_size, -1).contiguous()

    ys = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0                      # only the first beam is active at step 0
    finished = []                        # (normalised_score, token_ids)

    def norm(score, seq_len):
        length = max(1, seq_len - 1)     # exclude BOS
        return score / (length ** length_penalty)

    for _ in range(max_len):
        logp = _last_step_logprobs(model, ys, memory, src_kpm)      # (k, V)
        vocab = logp.size(-1)
        cand = (scores.unsqueeze(1) + logp).view(-1)                # (k*V,)
        top_scores, top_idx = cand.topk(beam_size)
        beam_ids = torch.div(top_idx, vocab, rounding_mode="floor")
        tok_ids = top_idx % vocab
        ys = torch.cat([ys[beam_ids], tok_ids.unsqueeze(1)], dim=1)
        scores = top_scores.clone()

        for b in range(beam_size):
            if tok_ids[b].item() == eos_id:
                finished.append((norm(scores[b].item(), ys.size(1)), ys[b].tolist()))
                scores[b] = float("-inf")
        if len(finished) >= beam_size or torch.isinf(scores).all():
            break

    if not finished:
        b = int(scores.argmax())
        finished.append((norm(scores[b].item(), ys.size(1)), ys[b].tolist()))
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]


@torch.no_grad()
def transformer_beam_predict_from_bpe(model, src_bpe_lines, config, src_vocab, tgt_vocab,
                                      beam_size, length_penalty=1.0, device="cpu", max_len=80):
    """Beam-decode each source BPE line. Returns (detok hyp text, raw hyp BPE strings)."""
    hyp_bpe = []
    for line in src_bpe_lines:
        ids = src_vocab.encode_pieces(line.split(), add_bos=True, add_eos=True)
        out_ids = beam_search_one(model, ids, tgt_vocab.bos_id, tgt_vocab.eos_id,
                                  beam_size, max_len, length_penalty, device)
        hyp_bpe.append(" ".join(tgt_vocab.decode_ids(out_ids, stop_at_eos=True)))
    return _detok(hyp_bpe, config["data"]["tgt_spm"]), hyp_bpe


def dictionary_baseline_predict(config, limit=None):
    """Build the position dictionary on train and translate the test split."""
    train_src = read_lines(config["data"]["train_src"])
    train_tgt = read_lines(config["data"]["train_tgt"])
    test_src = read_lines(config["data"]["test_src"])
    test_tgt = read_lines(config["data"]["test_tgt"])
    if limit is not None:
        test_src, test_tgt = test_src[:limit], test_tgt[:limit]

    dictionary = build_position_dictionary(train_src, train_tgt)
    hyp_bpe = [dict_translate(line, dictionary) for line in test_src]

    src_text = _detok(test_src, config["data"]["src_spm"])
    ref_text = _detok(test_tgt, config["data"]["tgt_spm"])
    hyp_text = _detok(hyp_bpe, config["data"]["tgt_spm"])
    return src_text, ref_text, hyp_text


def score(hyp_text, ref_text):
    """BLEU and chrF++ (chrF with word_order=2) via sacreBLEU."""
    bleu = sacrebleu.corpus_bleu(hyp_text, [ref_text])
    chrfpp = sacrebleu.corpus_chrf(hyp_text, [ref_text], word_order=2)
    return {"bleu": bleu.score, "chrf_pp": chrfpp.score,
            "bleu_str": str(bleu), "chrf_str": str(chrfpp), "n": len(hyp_text)}


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
