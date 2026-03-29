#!/usr/bin/env python3
"""
causal_model_test.py  --  Test DAS specificity to correct causal models

Runs DAS on the fine-tuned BERT (seed 77) with three conditions:
  1. Correct causal model (Negation + Lexical Entailment) — positive control
  2. Shuffled counterfactual labels — same structure, random targets
  3. Random binary variables — consistent causal function on random vars

If DAS is specific, only Condition 1 should achieve high IIA.

Usage on Colab (L4 GPU):
    !python causal_model_test.py
"""

# ── 0. Install dependencies ────────────────────────────────────
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.36.2", "torch", "scikit-learn", "pandas", "scipy"],
    stdout=subprocess.DEVNULL)

# ── 0.5. Google Drive for persistent storage ─────────────────
import glob, os, time, urllib.request

DRIVE_DIR = "/content/drive/MyDrive/DAS_experiment"

if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(DRIVE_DIR, exist_ok=True)
    SAVE_DIR = os.path.join(DRIVE_DIR, "causal_model_test")
    BASELINE_DIR = os.path.join(DRIVE_DIR, "saved_models_nli")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Google Drive detected. Saving to {SAVE_DIR}")
else:
    SAVE_DIR = "causal_model_test"
    BASELINE_DIR = "saved_models_nli"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"No Google Drive. Saving locally to {SAVE_DIR}")

# ── 1. Repo & data setup ─────────────────────────────────────

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

# ── 3. Configuration ───────────────────────────────────────
WEIGHTS_NAME   = "ishan/bert-base-uncased-mnli"
MAX_LENGTH     = 40
SEED           = 77
IIT_LAYER      = 8    # Layer 9 (0-indexed)
DIM_PER_VAR    = 256
TARGET_DIMS    = {"start": 0, "end": 786}

FACTUAL_CKPT   = os.path.join(BASELINE_DIR, "factual-77.bin")

DAS_EPOCHS     = 5
DAS_LR         = 2e-3
DAS_BS         = 64
DAS_TRAIN      = 24_000
DAS_TEST       = 1_920

ALIGNMENT = {
    0: [{"layer": IIT_LAYER, "start": 0, "end": DIM_PER_VAR}],
    1: [{"layer": IIT_LAYER, "start": DIM_PER_VAR, "end": 2 * DIM_PER_VAR}],
    2: [{"layer": IIT_LAYER, "start": 0, "end": DIM_PER_VAR},
        {"layer": IIT_LAYER, "start": DIM_PER_VAR, "end": 2 * DIM_PER_VAR}],
}

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


# ── 5. Condition-specific data builders ──────────────────────

def build_condition1(split):
    """Condition 1: Correct causal model (standard baseline)."""
    return load_combined_iit(DAS_TRAIN if split == "train" else DAS_TEST, split)


def build_condition2(train_data, test_data):
    """Condition 2: Shuffle counterfactual labels.

    Same base inputs, source inputs, and intervention IDs.
    Only y_iit is randomly permuted.
    """
    def shuffle_labels(data):
        X_base, y_base, X_srcs, y_iit, ids = data
        perm = torch.randperm(len(y_iit))
        return X_base, y_base, X_srcs, y_iit[perm], ids

    return shuffle_labels(train_data), shuffle_labels(test_data)


def build_condition3(n_train, n_test, split_train, split_test):
    """Condition 3: Random binary variables.

    Assign random R1, R2 to each MoNLI example. Construct IIT data
    with causal function: output = R1 XOR R2.

    For V1 interventions: swap R1 from source -> output = src_R1 XOR base_R2
    For V2 interventions: swap R2 from source -> output = base_R1 XOR src_R2
    For V1+V2:           swap both          -> output = src_R1 XOR src_R2
    """
    def _build(n, split):
        assert n % 3 == 0
        sub = n // 3

        # Load factual MoNLI examples as a pool of tokenized inputs
        ds = dataset_nli.IIT_MoNLIDataset(_encode, split, n)
        X_pool, _ = ds.create_factual_pairs()
        pool_ids = list(X_pool[0])      # list of input_id tensors
        pool_masks = list(X_pool[1])    # list of attention_mask tensors
        n_pool = len(pool_ids)

        # Assign random binary labels to each example in the pool
        R1 = torch.randint(0, 2, (n_pool,))
        R2 = torch.randint(0, 2, (n_pool,))

        # Build sub-datasets for each intervention type
        all_base_ids, all_base_masks = [], []
        all_src0_ids, all_src0_masks = [], []
        all_src1_ids, all_src1_masks = [], []
        all_y_base, all_y_iit, all_ids = [], [], []

        for intervention_id, n_examples in [(0, sub), (1, sub), (2, sub)]:
            for _ in range(n_examples):
                # Pick random base and source(s)
                bi = random.randint(0, n_pool - 1)
                si = random.randint(0, n_pool - 1)

                base_R1, base_R2 = R1[bi].item(), R2[bi].item()
                src_R1, src_R2 = R1[si].item(), R2[si].item()

                # Base label from causal function
                base_label = base_R1 ^ base_R2

                # Counterfactual label depends on intervention type
                if intervention_id == 0:    # swap R1
                    iit_label = src_R1 ^ base_R2
                elif intervention_id == 1:  # swap R2
                    iit_label = base_R1 ^ src_R2
                else:                       # swap both
                    iit_label = src_R1 ^ src_R2

                all_base_ids.append(pool_ids[bi])
                all_base_masks.append(pool_masks[bi])
                all_src0_ids.append(pool_ids[si])
                all_src0_masks.append(pool_masks[si])

                if intervention_id == 2:
                    # For V1+V2, pick a second source
                    si2 = random.randint(0, n_pool - 1)
                    all_src1_ids.append(pool_ids[si2])
                    all_src1_masks.append(pool_masks[si2])
                    # Recompute with both sources
                    src2_R2 = R2[si2].item()
                    iit_label = src_R1 ^ src2_R2
                else:
                    # For single-var, both source slots are the same
                    all_src1_ids.append(pool_ids[si])
                    all_src1_masks.append(pool_masks[si])

                all_y_base.append(base_label)
                all_y_iit.append(iit_label)
                all_ids.append(intervention_id)

        X_base = (all_base_ids, all_base_masks)
        y_base = torch.tensor(all_y_base)
        X_srcs = [
            (all_src0_ids, all_src0_masks),
            (all_src1_ids, all_src1_masks),
        ]
        y_iit = torch.tensor(all_y_iit)
        ids = torch.tensor(all_ids)
        return X_base, y_base, X_srcs, y_iit, ids

    return _build(n_train, split_train), _build(n_test, split_test)


def print_examples(name, data, n=5):
    """Print a few examples from an IIT dataset for sanity checking."""
    X_base, y_base, X_srcs, y_iit, ids = data
    int_names = {0: "V1", 1: "V2", 2: "V1+V2"}
    print(f"\n  {name} — first {n} examples:")
    print(f"  {'Idx':>4}  {'Intervention':>12}  {'Base Label':>10}  {'IIT Label':>9}")
    for i in range(min(n, len(y_iit))):
        base_text = TOKENIZER.decode(X_base[0][i].squeeze(), skip_special_tokens=True)[:60]
        src_text = TOKENIZER.decode(X_srcs[0][0][i].squeeze(), skip_special_tokens=True)[:60]
        print(f"  {i:>4}  {int_names[ids[i].item()]:>12}  "
              f"{y_base[i].item():>10}  {y_iit[i].item():>9}")
        print(f"        base: {base_text}")
        print(f"        src:  {src_text}")

    # Label distribution
    for iid in [0, 1, 2]:
        mask = ids == iid
        if mask.sum() > 0:
            n_match = (y_base[mask] == y_iit[mask]).sum().item()
            n_total = mask.sum().item()
            print(f"  {int_names[iid]}: {n_total} examples, "
                  f"label flipped in {n_total - n_match}/{n_total} "
                  f"({100*(n_total - n_match)/n_total:.1f}%)")


# ── 6. Run one DAS condition ─────────────────────────────────

def run_condition(name, train_data, test_data, factual_sd, device):
    """Train DAS and evaluate IIA for one condition. Returns result dict."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    utils.fix_random_seeds(SEED)

    # Build fresh DAS model and load factual weights
    bert = BertModel.from_pretrained(WEIGHTS_NAME,
                                     attn_implementation="eager")
    das_model = LIMBERTClassifier(
        n_classes=2, bert=bert, max_length=MAX_LENGTH,
        debug=False, target_dims=TARGET_DIMS,
        target_layers=[IIT_LAYER], device=device,
        static_search=False, nested_disentangle_inplace=False)

    das_model.load_state_dict(
        remap_factual_weights(factual_sd, IIT_LAYER), strict=False)

    das_model.set_analysis_mode(True)

    das_trainer = BERTLIMTrainer(
        das_model, warm_start=False, max_iter=DAS_EPOCHS,
        batch_size=DAS_BS, n_iter_no_change=10000,
        shuffle_train=False, eta=DAS_LR,
        device=device, seed=SEED)

    t0 = time.time()
    das_trainer.fit(
        train_data[0], train_data[1],
        iit_data=(train_data[2], train_data[3], train_data[4]),
        intervention_ids_to_coords=ALIGNMENT)
    das_time = time.time() - t0
    print(f"\n  DAS training: {das_time:.1f}s")

    # Get final training error from trainer
    final_error = das_trainer.errors[-1] if hasattr(das_trainer, 'errors') and das_trainer.errors else None

    # Evaluate IIA
    print("  Evaluating IIA ...")
    test_iit_preds = das_trainer.iit_predict(
        test_data[0], test_data[2], test_data[4], ALIGNMENT)
    test_iit_report = classification_report(
        test_data[3], test_iit_preds.cpu(), output_dict=True)
    test_iia = test_iit_report["weighted avg"]["f1-score"]

    print(f"  Test IIA = {test_iia:.4f}")

    result = {
        "condition":    name,
        "test_iia":     test_iia,
        "final_error":  final_error,
        "das_time_s":   round(das_time, 1),
    }

    del das_model, das_trainer, bert
    torch.cuda.empty_cache()

    return result


# ── 7. Main experiment ─────────────────────────────────────
def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be very slow.\n")
    print(f"Device: {device}")
    print(f"Config: seed={SEED}, layer={IIT_LAYER + 1}, dim={DIM_PER_VAR}\n")

    if not os.path.exists(FACTUAL_CKPT):
        print(f"ERROR: Factual checkpoint not found: {FACTUAL_CKPT}")
        print("Run 'python run_baseline.py' first to generate it.")
        sys.exit(1)

    RESULTS_FILE = os.path.join(SAVE_DIR, "causal_model_results.json")

    # Load any results from a previous run
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    completed = {r["condition"] for r in results}

    factual_sd = torch.load(FACTUAL_CKPT, map_location="cpu",
                            weights_only=True)

    # ── Build datasets for all conditions ─────────────────
    print("Building datasets ...")
    _ENCODE_CACHE.clear()
    utils.fix_random_seeds(SEED)

    t0 = time.time()
    train_c1 = load_combined_iit(DAS_TRAIN, "train")
    test_c1 = load_combined_iit(DAS_TEST, "test")
    print(f"  Condition 1 (correct model): {time.time()-t0:.1f}s")

    # Condition 2: shuffle labels
    utils.fix_random_seeds(SEED + 1)  # different seed for shuffle
    train_c2, test_c2 = build_condition2(train_c1, test_c1)
    print(f"  Condition 2 (shuffled labels): built from Condition 1")

    # Condition 3: random binary variables
    utils.fix_random_seeds(SEED + 2)
    train_c3, test_c3 = build_condition3(DAS_TRAIN, DAS_TEST, "train", "test")
    print(f"  Condition 3 (random variables): {time.time()-t0:.1f}s")

    # ── Sanity-check examples ─────────────────────────────
    print_examples("Condition 1: Correct model", test_c1)
    print_examples("Condition 2: Shuffled labels", test_c2)
    print_examples("Condition 3: Random binary vars", test_c3)

    # ── Run each condition ────────────────────────────────
    conditions = [
        ("1. Correct model", train_c1, test_c1),
        ("2. Shuffled labels", train_c2, test_c2),
        ("3. Random binary variables", train_c3, test_c3),
    ]

    for name, train_data, test_data in conditions:
        if name in completed:
            print(f"\n  {name}: already completed, skipping.")
            continue

        result = run_condition(name, train_data, test_data, factual_sd, device)
        results.append(result)

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {RESULTS_FILE}")

    # ── Final comparison table ────────────────────────────
    print("\n" + "=" * 60)
    print("  CAUSAL MODEL SPECIFICITY TEST")
    print("  Seed 77, Layer 9, 256 dims per variable")
    print("=" * 60)
    hdr = f"{'Condition':<32} | {'IIA':>7} | {'DAS Error':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        err_str = f"{r['final_error']:.1f}" if r['final_error'] is not None else "N/A"
        print(f"{r['condition']:<32} | {r['test_iia']:>7.4f} | {err_str:>10}")
    print("-" * len(hdr))
    print()

    return results


if __name__ == "__main__":
    run()
