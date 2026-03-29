#!/usr/bin/env python3
"""
progressive_randomization.py  --  Cascading randomization sanity check

Following Adebayo et al. (2018) "Sanity Checks for Saliency Maps":
progressively randomize BERT layers from top down and measure how
DAS's IIA degrades. If DAS captures meaningful causal structure,
IIA should drop as learned weights are destroyed.

Steps:
  0:  No randomization (baseline, should match ~0.95 IIA)
  1:  Randomize classifier head only
  2:  Randomize classifier + encoder layer 11
  3:  Randomize classifier + encoder layers 10-11
  ...
  13: Randomize classifier + encoder layers 0-11
  14: Randomize everything (including embeddings + pooler)

Uses seed 77 checkpoint (best IIA from baseline), layer 9, dim 256.

Usage on Colab (L4 GPU):
    !python progressive_randomization.py
"""

# ── 0. Install dependencies ────────────────────────────────────
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.36.2", "torch", "scikit-learn", "pandas", "scipy"],
    stdout=subprocess.DEVNULL)

# ── 1. Repo & data setup ─────────────────────────────────────
import glob, os, time, urllib.request

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

os.makedirs("datasets", exist_ok=True)
_MONLI_URL = "https://raw.githubusercontent.com/atticusg/MoNLI/master"
for _fname in ["pmonli.jsonl", "nmonli_train.jsonl", "nmonli_test.jsonl"]:
    _path = os.path.join("datasets", _fname)
    if not os.path.exists(_path):
        print(f"  Downloading {_fname} ...")
        urllib.request.urlretrieve(f"{_MONLI_URL}/{_fname}", _path)

SAVE_DIR = "saved_models_nli_progressive"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 2. Imports ──────────────────────────────────────────────
import copy, json, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel

import dataset_nli
import utils
from LIM_bert import LIMBERTClassifier
from trainer import BERTLIMTrainer

# ── 3. Configuration ───────────────────────────────────────
WEIGHTS_NAME   = "ishan/bert-base-uncased-mnli"
MAX_LENGTH     = 40
SEED           = 77
IIT_LAYER      = 8    # Layer 9 (0-indexed)
DIM_PER_VAR    = 256
TARGET_DIMS    = {"start": 0, "end": 786}

FACTUAL_CKPT   = "saved_models_nli/factual-77.bin"

DAS_EPOCHS     = 5
DAS_LR         = 2e-3
DAS_BS         = 64
DAS_TRAIN      = 24_000
DAS_TEST       = 1_920
FACTUAL_TEST   = 1_000

# ── 4. Data-loading helpers ──────────────────────────────────
TOKENIZER = BertTokenizer.from_pretrained(WEIGHTS_NAME)
_ENCODE_CACHE = {}


def _encode(X):
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
    ds = dataset_nli.IIT_MoNLIDataset(_encode, split, n)
    X, y = ds.create_factual_pairs()
    return X, torch.tensor(y)


def _load_iit(method_name, n, split):
    ds = dataset_nli.IIT_MoNLIDataset(_encode, split, n)
    X, y, srcs, yi, ids = getattr(ds, method_name)()
    return X, torch.tensor(y), srcs, torch.tensor(yi), torch.tensor(ids)


def load_combined_iit(n, split):
    assert n % 3 == 0
    sub = n // 3
    v1 = _load_iit("create_neghyp_V1",    sub, split)
    v2 = _load_iit("create_neghyp_V2",    sub, split)
    vb = _load_iit("create_neghyp_V1_V2", sub, split)

    X_base = (v1[0][0] + v2[0][0] + vb[0][0],
              v1[0][1] + v2[0][1] + vb[0][1])
    y_base = torch.cat([v1[1], v2[1], vb[1]])
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


# ── 5. Randomization helpers ────────────────────────────────
def _reinit_module(module):
    """Reinitialize a module's weights: Normal(0, 0.02) for weights,
    zeros for biases — matching BertPreTrainedModel._init_weights."""
    for name, param in module.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                param.data.zero_()
            elif "weight" in name:
                param.data.normal_(mean=0.0, std=0.02)


def build_randomization_steps():
    """Return a list of (step_index, label, description) tuples.
    Each step describes which components to randomize."""
    steps = []
    # Step 0: no randomization
    steps.append((0, "baseline", "No randomization"))
    # Step 1: classifier only
    steps.append((1, "cls", "Classifier head"))
    # Steps 2-13: classifier + encoder layers from top down
    for i in range(12):
        top_layer = 11
        bottom_layer = top_layer - i
        if bottom_layer == top_layer:
            layers_str = f"layer {top_layer}"
        else:
            layers_str = f"layers {bottom_layer}-{top_layer}"
        steps.append((
            i + 2,
            f"cls+L{bottom_layer}-{top_layer}",
            f"Classifier + encoder {layers_str}",
        ))
    # Step 14: everything
    steps.append((14, "all", "Everything (embeddings + encoder + pooler + classifier)"))
    return steps


def randomize_model(model, step_idx):
    """Apply progressive randomization to the model in-place.

    step 0:  nothing
    step 1:  classifier_layer
    step 2:  classifier_layer + encoder layer 11
    step 3:  classifier_layer + encoder layers 10-11
    ...
    step 13: classifier_layer + encoder layers 0-11
    step 14: everything (embeddings + pooler + classifier + all layers)

    model_layers is a ModuleList of 12 LIMBertLayer objects (indices 0-11).
    These are shared between normal_model and analysis_model, so
    randomizing them once affects both paths.
    """
    if step_idx == 0:
        return

    # Always randomize classifier from step 1 onwards
    _reinit_module(model.classifier_layer)

    if step_idx >= 14:
        # Randomize embeddings and pooler too
        _reinit_module(model.embeddings)
        _reinit_module(model.pooler)

    # Randomize encoder layers from top down
    # step 2 -> randomize layer 11
    # step 3 -> randomize layers 10-11
    # step 13 -> randomize layers 0-11
    n_layers_to_randomize = max(0, step_idx - 1)  # step 2->1, step 13->12
    if step_idx >= 14:
        n_layers_to_randomize = 12

    for i in range(n_layers_to_randomize):
        layer_idx = 11 - i  # start from top
        _reinit_module(model.model_layers[layer_idx])


# ── 6. Main experiment ─────────────────────────────────────
def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be very slow.\n")
    print(f"Device: {device}")
    print(f"Config: seed={SEED}, layer={IIT_LAYER + 1}, "
          f"dim={DIM_PER_VAR}\n")

    # Verify factual checkpoint exists
    if not os.path.exists(FACTUAL_CKPT):
        print(f"ERROR: Factual checkpoint not found: {FACTUAL_CKPT}")
        print("Run 'python run_baseline.py' first to generate it.")
        sys.exit(1)

    RESULTS_FILE = os.path.join(SAVE_DIR, "results_so_far.json")

    # Load any results saved from a previous run
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    completed_steps = {r["step"] for r in results}

    steps = build_randomization_steps()

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

    # Load factual state dict once
    factual_sd = torch.load(FACTUAL_CKPT, map_location="cpu",
                            weights_only=True)

    # Build IIT datasets once (reused for all steps)
    print("Building interchange-intervention datasets ...")
    utils.fix_random_seeds(SEED)
    _ENCODE_CACHE.clear()

    t0 = time.time()
    train_iit = load_combined_iit(DAS_TRAIN, "train")
    t1 = time.time()
    print(f"  Train set: {t1-t0:.1f}s  "
          f"(cache: {len(_ENCODE_CACHE)} entries)")

    test_iit = load_combined_iit(DAS_TEST, "test")
    t2 = time.time()
    print(f"  Test set:  {t2-t1:.1f}s  "
          f"(cache: {len(_ENCODE_CACHE)} entries)")

    # Also load factual test set for task accuracy
    X_te, y_te = load_factual(FACTUAL_TEST, "test")
    print(f"  Factual test set loaded ({FACTUAL_TEST} examples)")

    for step_idx, label, description in steps:
        if step_idx in completed_steps:
            print(f"\n  Step {step_idx:>2} ({label}): already completed, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Step {step_idx:>2}: {description}")
        print(f"{'=' * 60}")

        utils.fix_random_seeds(SEED)

        # Build a fresh DAS model and load factual weights
        bert = BertModel.from_pretrained(WEIGHTS_NAME,
                                         attn_implementation="eager")
        das_model = LIMBERTClassifier(
            n_classes=2, bert=bert, max_length=MAX_LENGTH,
            debug=False, target_dims=TARGET_DIMS,
            target_layers=[IIT_LAYER], device=device,
            static_search=False, nested_disentangle_inplace=False)

        das_model.load_state_dict(
            remap_factual_weights(factual_sd, IIT_LAYER), strict=False)

        # Apply progressive randomization
        randomize_model(das_model, step_idx)

        # ── Measure task accuracy before DAS ──────────
        print("\n  Measuring task accuracy ...")
        das_model.eval()
        das_model.set_analysis_mode(False)

        task_trainer = BERTLIMTrainer(
            das_model, warm_start=False, max_iter=1,
            batch_size=DAS_BS, n_iter_no_change=10000,
            shuffle_train=False, eta=DAS_LR,
            device=device, seed=SEED)

        preds = task_trainer.predict(X_te)
        task_report = classification_report(
            y_te, preds.cpu(), output_dict=True)
        task_f1 = task_report["weighted avg"]["f1-score"]
        print(f"  Task F1 = {task_f1:.4f}")

        # ── Train DAS rotation ──────────────────────
        print(f"\n  Training DAS rotation "
              f"(layer {IIT_LAYER + 1}, {DIM_PER_VAR} dims) ...")

        # Freeze BERT, unfreeze only rotation matrices
        das_model.set_analysis_mode(True)

        das_trainer = BERTLIMTrainer(
            das_model, warm_start=False, max_iter=DAS_EPOCHS,
            batch_size=DAS_BS, n_iter_no_change=10000,
            shuffle_train=False, eta=DAS_LR,
            device=device, seed=SEED)

        t0 = time.time()
        das_trainer.fit(
            train_iit[0], train_iit[1],
            iit_data=(train_iit[2], train_iit[3], train_iit[4]),
            intervention_ids_to_coords=alignment)
        das_time = time.time() - t0
        print(f"\n  DAS training: {das_time:.1f}s")

        # ── Evaluate IIA ──────────────────────────────
        print("  Evaluating IIA ...")

        test_base_preds = das_trainer.predict(test_iit[0])
        das_fact_report = classification_report(
            test_iit[1], test_base_preds.cpu(), output_dict=True)
        das_fact_f1 = das_fact_report["weighted avg"]["f1-score"]

        test_iit_preds = das_trainer.iit_predict(
            test_iit[0], test_iit[2], test_iit[4], alignment)
        test_iit_report = classification_report(
            test_iit[3], test_iit_preds.cpu(), output_dict=True)
        test_iia = test_iit_report["weighted avg"]["f1-score"]

        result = {
            "step":         step_idx,
            "label":        label,
            "description":  description,
            "task_f1":      task_f1,
            "das_fact_f1":  das_fact_f1,
            "test_iia":     test_iia,
            "das_time_s":   round(das_time, 1),
        }
        results.append(result)

        print(f"\n  Step {step_idx} results:")
        print(f"    Task F1      = {task_f1:.4f}")
        print(f"    DAS Fact F1  = {das_fact_f1:.4f}")
        print(f"    Test IIA     = {test_iia:.4f}")

        del das_model, das_trainer, task_trainer, bert
        torch.cuda.empty_cache()

        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {RESULTS_FILE}")

    # ── Final summary ────────────────────────────────────────
    # Sort by step for clean display
    results.sort(key=lambda r: r["step"])

    print("\n" + "=" * 70)
    print("  PROGRESSIVE RANDOMIZATION (Adebayo et al. 2018 style)")
    print("  Seed 77, Layer 9, 256 dims per variable")
    print("=" * 70)
    hdr = (f"{'Step':>4}  {'Label':>16}  {'Task F1':>8}  "
           f"{'DAS Fact':>9}  {'IIA':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['step']:>4}  {r['label']:>16}  "
              f"{r['task_f1']:>8.4f}  "
              f"{r['das_fact_f1']:>9.4f}  {r['test_iia']:>8.4f}")
    print("-" * len(hdr))

    # Show IIA drop from baseline
    baseline_iia = next(
        (r["test_iia"] for r in results if r["step"] == 0), None)
    if baseline_iia is not None:
        print(f"\n  Baseline IIA: {baseline_iia:.4f}")
        for r in results:
            if r["step"] > 0:
                drop = baseline_iia - r["test_iia"]
                print(f"  Step {r['step']:>2} ({r['label']:>16}): "
                      f"IIA={r['test_iia']:.4f}  "
                      f"(drop={drop:+.4f})")

    print()
    return results


if __name__ == "__main__":
    run()
