#!/usr/bin/env python3
"""
run_baseline.py  --  Reproduce the MoNLI DAS experiment
from Geiger et al. (2022) "Inducing Causal Structure for
Interpretable Neural Networks" (ICML 2022).

Runs end-to-end on Google Colab (T4 GPU) or any CUDA machine:
    !python run_baseline.py

Use --retrain to delete saved checkpoints and retrain from scratch:
    !python run_baseline.py --retrain

Experiment:
  - Fine-tune ishan/bert-base-uncased-mnli on MoNLI factual task
  - Run Distributed Alignment Search (DAS) with the
    "Negation + Lexical Entailment" causal model
  - Layer 9, subspace width 256, 3 random seeds
  - Report IIA (Interchange Intervention Accuracy)
"""

# ── 0. Install dependencies ────────────────────────────────────
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.36.2", "torch", "scikit-learn", "pandas", "scipy"],
    stdout=subprocess.DEVNULL)

# ── 1. Repo & data setup ─────────────────────────────────────
import glob, os, time, urllib.request

# Navigate into the repo root if we aren't already there
if not os.path.exists("layered_intervenable_model.py"):
    if os.path.exists("GeigerExperiment/layered_intervenable_model.py"):
        os.chdir("GeigerExperiment")
    else:
        print("Cloning GeigerExperiment repository ...")
        subprocess.check_call(
            ["git", "clone", "-q",
             "https://github.com/shengweiming/GeigerExperiment.git"])
        os.chdir("GeigerExperiment")

sys.path.insert(0, os.getcwd())

# Download MoNLI data (Geiger, Richardson & Potts, 2020)
os.makedirs("datasets", exist_ok=True)
_MONLI_URL = "https://raw.githubusercontent.com/atticusg/MoNLI/master"
for _fname in ["pmonli.jsonl", "nmonli_train.jsonl", "nmonli_test.jsonl"]:
    _path = os.path.join("datasets", _fname)
    if not os.path.exists(_path):
        print(f"  Downloading {_fname} ...")
        urllib.request.urlretrieve(f"{_MONLI_URL}/{_fname}", _path)

os.makedirs("saved_models_nli", exist_ok=True)

# Handle --retrain flag
if "--retrain" in sys.argv:
    for f in glob.glob("saved_models_nli/factual-*.bin"):
        os.remove(f)
        print(f"  Deleted stale checkpoint: {f}")

# ── 2. Imports ──────────────────────────────────────────────
import copy, json, random
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel

import dataset_nli
import utils
from LIM_bert import LIMBERTClassifier
from trainer import BERTLIMTrainer

# ── 3. Experiment configuration ────────────────────────────
WEIGHTS_NAME   = "ishan/bert-base-uncased-mnli"
MAX_LENGTH     = 128
SEEDS          = [42, 66, 77]

# Layer 9 in the paper = index 8 in the 0-indexed analysis_model
IIT_LAYER      = 8
DIM_PER_VAR    = 256          # rotation-subspace width per variable

# Rotation is applied to the first 786 dims of the flattened
# (hidden_dim * max_length) representation, matching the notebook.
TARGET_DIMS    = {"start": 0, "end": 786}

# Factual fine-tuning (Appendix A.2)
FACTUAL_EPOCHS = 5
FACTUAL_LR     = 2e-5
FACTUAL_BS     = 32
FACTUAL_TRAIN  = 10_000
FACTUAL_TEST   = 1_000

# DAS rotation-matrix training
DAS_EPOCHS     = 5
DAS_LR         = 2e-3
DAS_BS         = 64
DAS_TRAIN      = 24_000       # split three ways: V1, V2, V1+V2
DAS_TEST       = 1_920

# ── 4. Data-loading helpers (from experiment notebook) ─────
TOKENIZER = BertTokenizer.from_pretrained(WEIGHTS_NAME)

_ENCODE_CACHE = {}

def _encode(X):
    """Tokenise a (premise, hypothesis) pair into BERT inputs.
    Results are cached so repeated calls with the same text are free."""
    key = (X[0], X[1])
    if key in _ENCODE_CACHE:
        return _ENCODE_CACHE[key]
    text = [". ".join(X)] if X[0][-1] != "." else [" ".join(X)]
    out = TOKENIZER(
        text, max_length=MAX_LENGTH, add_special_tokens=True,
        padding="max_length", truncation=True,
        return_attention_mask=True, return_tensors="pt")
    result = (out["input_ids"], out["attention_mask"])
    _ENCODE_CACHE[key] = result
    return result


def load_factual(n, split):
    """Load n factual (base-only) MoNLI examples."""
    ds = dataset_nli.IIT_MoNLIDataset(_encode, split, n)
    X, y = ds.create_factual_pairs()
    return X, torch.tensor(y)


def _load_iit(method_name, n, split):
    """Call a create_neghyp_* method and tensorise labels/ids."""
    ds = dataset_nli.IIT_MoNLIDataset(_encode, split, n)
    X, y, srcs, yi, ids = getattr(ds, method_name)()
    return X, torch.tensor(y), srcs, torch.tensor(yi), torch.tensor(ids)


def load_combined_iit(n, split):
    """
    Build the full "Negation + Lexical Entailment" IIT dataset by
    concatenating V1 (negation), V2 (lex-entailment) and V1+V2 (both)
    sub-datasets, each of size n/3.  Two source slots are created so
    that the V1+V2 condition can swap both variables simultaneously.
    """
    assert n % 3 == 0
    sub = n // 3
    v1 = _load_iit("create_neghyp_V1",    sub, split)
    v2 = _load_iit("create_neghyp_V2",    sub, split)
    vb = _load_iit("create_neghyp_V1_V2", sub, split)

    X_base = (v1[0][0] + v2[0][0] + vb[0][0],
              v1[0][1] + v2[0][1] + vb[0][1])
    y_base = torch.cat([v1[1], v2[1], vb[1]])

    # Source slot 0: V1 src | V2 src | V1V2-source-1
    # Source slot 1: V1 src | V2 src | V1V2-source-2
    # (For single-variable interventions both slots carry the same source.)
    X_srcs = [
        (v1[2][0][0] + v2[2][0][0] + vb[2][0][0],
         v1[2][0][1] + v2[2][0][1] + vb[2][0][1]),
        (v1[2][0][0] + v2[2][0][0] + vb[2][1][0],
         v1[2][0][1] + v2[2][0][1] + vb[2][1][1]),
    ]
    y_iit = torch.cat([v1[3], v2[3], vb[3]])
    ids   = torch.cat([v1[4], v2[4], vb[4]])
    return X_base, y_base, X_srcs, y_iit, ids


def remap_factual_weights(state_dict, iit_layer):
    """
    The DAS model inserts a LinearLayer + InverseLinearLayer into the
    analysis_model after `iit_layer`.  Keys for analysis_model layers
    beyond that index must be shifted by +2 so they land in the right
    slots.  Keys outside analysis_model (normal_model, embeddings,
    classifier, pooler) are copied as-is.
    """
    remapped = {}
    for key, val in state_dict.items():
        if "analysis_model" not in key:
            remapped[key] = val
        else:
            layer_idx = int(key.split(".")[2])
            if layer_idx <= iit_layer:
                remapped[key] = val
            else:
                parts = key.split(".")
                parts[2] = str(layer_idx + 2)
                remapped[".".join(parts)] = val
    return remapped


# ── 5. Main experiment ─────────────────────────────────────
def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be very slow.\n")
    print(f"Device: {device}")
    print(f"Config: layer={IIT_LAYER + 1}, dims={DIM_PER_VAR}, "
          f"seeds={SEEDS}\n")

    # Alignment dict: maps intervention-id -> list of coordinates
    #   0 = intervene on V1 (negation)
    #   1 = intervene on V2 (lexical entailment)
    #   2 = intervene on both V1 and V2
    alignment = {
        0: [{"layer": IIT_LAYER, "start": 0,
             "end": DIM_PER_VAR}],
        1: [{"layer": IIT_LAYER, "start": DIM_PER_VAR,
             "end": 2 * DIM_PER_VAR}],
        2: [{"layer": IIT_LAYER, "start": 0,
             "end": DIM_PER_VAR},
            {"layer": IIT_LAYER, "start": DIM_PER_VAR,
             "end": 2 * DIM_PER_VAR}],
    }

    results = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"  SEED {seed}")
        print(f"{'=' * 60}")
        utils.fix_random_seeds(seed)

        # ── A.  Fine-tune BERT on factual MoNLI ────────────
        ckpt_path = f"saved_models_nli/factual-{seed}.bin"

        if os.path.exists(ckpt_path):
            print(f"\n[1/3] Found saved checkpoint {ckpt_path}, "
                  f"skipping fine-tuning.")
            fact_f1 = None
        else:
            print("\n[1/3] Fine-tuning BERT on MoNLI factual task ...")

            t0 = time.time()
            bert = BertModel.from_pretrained(WEIGHTS_NAME, attn_implementation="eager")
            factual_model = LIMBERTClassifier(
                n_classes=2, bert=bert, max_length=MAX_LENGTH,
                debug=False, target_dims=TARGET_DIMS, target_layers=[],
                device=device, static_search=False,
                nested_disentangle_inplace=False)

            factual_trainer = BERTLIMTrainer(
                factual_model,
                warm_start=False, max_iter=FACTUAL_EPOCHS,
                batch_size=FACTUAL_BS, n_iter_no_change=10000,
                shuffle_train=True, eta=FACTUAL_LR,
                device=device, seed=seed)

            t1 = time.time()
            print(f"      Loading data ...")
            X_tr, y_tr = load_factual(FACTUAL_TRAIN, "train")
            X_te, y_te = load_factual(FACTUAL_TEST,  "test")
            print(f"      Data loaded in {time.time()-t1:.1f}s "
                  f"(cache size: {len(_ENCODE_CACHE)})")

            factual_trainer.fit(X_tr, y_tr)

            preds = factual_trainer.predict(X_te)
            fact_report = classification_report(
                y_te, preds.cpu(), output_dict=True)
            fact_f1 = fact_report["weighted avg"]["f1-score"]
            print(f"\n      Factual test F1 = {fact_f1:.4f}  "
                  f"({time.time()-t0:.1f}s total)")

            torch.save(factual_model.state_dict(), ckpt_path)
            del factual_model, factual_trainer, bert
            torch.cuda.empty_cache()

        # ── B.  Build IIT datasets ─────────────────────
        print("\n[2/3] Building interchange-intervention datasets ...")
        _ENCODE_CACHE.clear()
        utils.fix_random_seeds(seed)

        t0 = time.time()
        train_iit = load_combined_iit(DAS_TRAIN, "train")
        t1 = time.time()
        print(f"      Train set: {t1-t0:.1f}s  "
              f"(cache: {len(_ENCODE_CACHE)} entries)")

        test_iit  = load_combined_iit(DAS_TEST,  "test")
        t2 = time.time()
        print(f"      Test set:  {t2-t1:.1f}s  "
              f"(cache: {len(_ENCODE_CACHE)} entries)")

        # ── C.  Train DAS rotation matrix ────────────────
        print(f"\n[3/3] Training DAS rotation "
              f"(layer {IIT_LAYER + 1}, {DIM_PER_VAR} dims) ...")

        bert_das = BertModel.from_pretrained(WEIGHTS_NAME, attn_implementation="eager")
        das_model = LIMBERTClassifier(
            n_classes=2, bert=bert_das, max_length=MAX_LENGTH,
            debug=False, target_dims=TARGET_DIMS,
            target_layers=[IIT_LAYER], device=device,
            static_search=False, nested_disentangle_inplace=False)

        # Load factual weights, remapping analysis_model keys
        saved_sd = torch.load(ckpt_path, map_location="cpu",
                              weights_only=True)
        das_model.load_state_dict(
            remap_factual_weights(saved_sd, IIT_LAYER), strict=False)

        # Freeze BERT, unfreeze only the rotation matrices
        das_model.set_analysis_mode(True)

        das_trainer = BERTLIMTrainer(
            das_model,
            warm_start=False, max_iter=DAS_EPOCHS,
            batch_size=DAS_BS, n_iter_no_change=10000,
            shuffle_train=False, eta=DAS_LR,
            device=device, seed=seed)

        t0 = time.time()
        das_trainer.fit(
            train_iit[0], train_iit[1],
            iit_data=(train_iit[2], train_iit[3], train_iit[4]),
            intervention_ids_to_coords=alignment)
        print(f"\n      DAS training: {time.time()-t0:.1f}s")

        # ── D.  Evaluate IIA ───────────────────────
        print("      Evaluating IIA on test set ...")

        # Factual accuracy (should stay high)
        test_base_preds = das_trainer.predict(test_iit[0])
        test_fact_report = classification_report(
            test_iit[1], test_base_preds.cpu(), output_dict=True)
        test_fact_f1 = test_fact_report["weighted avg"]["f1-score"]

        # IIA  =  weighted-F1 of interchange-intervention predictions
        test_iit_preds = das_trainer.iit_predict(
            test_iit[0], test_iit[2], test_iit[4], alignment)
        test_iit_report = classification_report(
            test_iit[3], test_iit_preds.cpu(), output_dict=True)
        test_iia = test_iit_report["weighted avg"]["f1-score"]

        result = {
            "seed":         seed,
            "factual_f1":   fact_f1,
            "das_fact_f1":  test_fact_f1,
            "test_iia":     test_iia,
        }
        results.append(result)

        print(f"\n      Seed {seed} results:")
        if fact_f1 is not None:
            print(f"        Factual test F1         = {fact_f1:.4f}")
        else:
            print(f"        Factual test F1         = (loaded from checkpoint)")
        print(f"        DAS factual test F1     = {test_fact_f1:.4f}")
        print(f"        DAS test IIA            = {test_iia:.4f}")

        del das_model, das_trainer, bert_das
        torch.cuda.empty_cache()

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS:  MoNLI DAS  (Layer 9, 256 dims per variable)")
    print("  High-level model: Negation + Lexical Entailment")
    print("=" * 60)
    hdr = f"{'Seed':>6}  {'Factual F1':>11}  {'DAS Fact F1':>12}  {'Test IIA':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        f1_str = f"{r['factual_f1']:>11.4f}" if r['factual_f1'] is not None else "       ckpt"
        print(f"{r['seed']:>6}  {f1_str}  "
              f"{r['das_fact_f1']:>12.4f}  {r['test_iia']:>10.4f}")

    best = max(results, key=lambda r: r["test_iia"])
    print("-" * len(hdr))
    print(f"\n  >>> BEST TEST IIA = {best['test_iia']:.4f}  (seed {best['seed']})")
    print()

    return results


if __name__ == "__main__":
    run()
