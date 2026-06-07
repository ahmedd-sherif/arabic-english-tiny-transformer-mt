"""Batch A full-test evaluation (no training).

Greedy-decodes the full filtered test set (8,491) with the saved best Tiny Transformer
checkpoint, scores BLEU + chrF++, does the same for the dictionary baseline, and also
reports the first-1,000 subset scores so sampling variance can be discussed.

Both models are scored against the SAME references: the true human references obtained by
detokenizing the kept target BPE lines (one filtered source of truth -> identical order).
The expensive transformer decode is CACHED so downstream errors never waste it; delete the
cache or set FORCE_REDECODE=1 to recompute.

Outputs:
  outputs/tables/full_test_bleu_results.csv
  outputs/tables/sample_vs_full_eval.csv
  outputs/examples/full_test_predictions.csv
"""

import os, sys, time
sys.path.insert(0, "src")
import torch
torch.set_num_threads(os.cpu_count() or 1)

import yaml
import pandas as pd
import eval_utils as E
from baseline_dictionary import build_position_dictionary, translate as dict_translate

T0 = time.time()
config = yaml.safe_load(open("config/transformer_tiny.yaml", encoding="utf-8"))
device = "cpu"
SUBSET = 1000
CACHE = "outputs/examples/_tf_hyp_bpe_cache.txt"
FORCE = os.environ.get("FORCE_REDECODE") == "1"

src_vocab, tgt_vocab = E.load_vocabs(config)

# One filtered source of truth: kept source/target BPE lines (dataset order).
kept_src_bpe, kept_tgt_bpe = E.filter_test_pairs(config, "test")
n_full = len(kept_src_bpe)
src_text = E.detok_lines(kept_src_bpe, config["data"]["src_spm"])
ref_text = E.detok_lines(kept_tgt_bpe, config["data"]["tgt_spm"])   # true references
print(f"[info] kept test pairs: {n_full}", flush=True)

# --- Transformer: decode (cached) ---
if os.path.exists(CACHE) and not FORCE:
    tf_hyp_bpe = open(CACHE, encoding="utf-8").read().splitlines()
    assert len(tf_hyp_bpe) == n_full, "cache length mismatch; set FORCE_REDECODE=1"
    tf_hyp = E.detok_lines(tf_hyp_bpe, config["data"]["tgt_spm"])
    tf_decode_s = 0.0
    print(f"[info] loaded cached transformer predictions ({n_full})", flush=True)
else:
    model, ckpt = E.load_transformer(config, "outputs/checkpoints/best_full_model.pt",
                                     src_vocab, tgt_vocab, device)
    print(f"[info] checkpoint epoch={ckpt.get('epoch')} — decoding {n_full} ...", flush=True)
    t = time.time()
    tf_hyp, tf_hyp_bpe = E.transformer_predict_from_bpe(
        model, kept_src_bpe, config, src_vocab, tgt_vocab, device=device,
        batch_size=int(config["training"]["batch_size"]), max_len=int(config["decoding"]["max_len"]))
    tf_decode_s = time.time() - t
    open(CACHE, "w", encoding="utf-8").write("\n".join(tf_hyp_bpe))   # cache immediately
    print(f"[info] transformer decoded in {tf_decode_s/60:.1f} min (cached)", flush=True)

# --- Baseline: dictionary on the SAME kept source lines ---
train_src = E.read_lines(config["data"]["train_src"])
train_tgt = E.read_lines(config["data"]["train_tgt"])
dictionary = build_position_dictionary(train_src, train_tgt)
base_hyp_bpe = [dict_translate(l, dictionary) for l in kept_src_bpe]
base_hyp = E.detok_lines(base_hyp_bpe, config["data"]["tgt_spm"])

# --- Scores: full and first-1000 subset (same references for both models) ---
tf_full, tf_sub = E.score(tf_hyp, ref_text), E.score(tf_hyp[:SUBSET], ref_text[:SUBSET])
bl_full, bl_sub = E.score(base_hyp, ref_text), E.score(base_hyp[:SUBSET], ref_text[:SUBSET])

results = pd.DataFrame([
    {"model": "dictionary_position_baseline", "decoding": "position_dictionary",
     "examples": bl_full["n"], "bleu": round(bl_full["bleu"], 4),
     "chrf_pp": round(bl_full["chrf_pp"], 4), "bleu_str": bl_full["bleu_str"]},
    {"model": "tiny_transformer_best", "decoding": "greedy",
     "examples": tf_full["n"], "bleu": round(tf_full["bleu"], 4),
     "chrf_pp": round(tf_full["chrf_pp"], 4), "bleu_str": tf_full["bleu_str"]},
])
results.to_csv("outputs/tables/full_test_bleu_results.csv", index=False)

sample_vs_full = pd.DataFrame([
    {"model": "dictionary_position_baseline", "bleu_1000": round(bl_sub["bleu"], 4),
     "bleu_full": round(bl_full["bleu"], 4), "delta_bleu": round(bl_full["bleu"] - bl_sub["bleu"], 4),
     "chrf_1000": round(bl_sub["chrf_pp"], 4), "chrf_full": round(bl_full["chrf_pp"], 4)},
    {"model": "tiny_transformer_best", "bleu_1000": round(tf_sub["bleu"], 4),
     "bleu_full": round(tf_full["bleu"], 4), "delta_bleu": round(tf_full["bleu"] - tf_sub["bleu"], 4),
     "chrf_1000": round(tf_sub["chrf_pp"], 4), "chrf_full": round(tf_full["chrf_pp"], 4)},
])
sample_vs_full.to_csv("outputs/tables/sample_vs_full_eval.csv", index=False)

pd.DataFrame({
    "sample_id": range(n_full), "source_ar": src_text, "reference_en": ref_text,
    "transformer_output": tf_hyp, "baseline_output": base_hyp,
    "source_length": [len(s.split()) for s in src_text],
    "target_length": [len(r.split()) for r in ref_text],
}).to_csv("outputs/examples/full_test_predictions.csv", index=False, encoding="utf-8")

print("\n==== FULL-TEST RESULTS (8,491) ====", flush=True)
print(results[["model", "decoding", "examples", "bleu", "chrf_pp"]].to_string(index=False), flush=True)
print("\n==== 1000 vs FULL (sampling variance) ====", flush=True)
print(sample_vs_full.to_string(index=False), flush=True)
print(f"\n[done] total {time.time()-T0:.0f}s | transformer decode {tf_decode_s:.0f}s", flush=True)
