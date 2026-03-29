#!/usr/bin/env python3
"""
monli_sanity_check.py  --  Sanity check: DAS on randomly-initialized BERT

Control experiment for the MoNLI DAS baseline. Uses the same BERT
architecture but with ALL weights randomly initialized (no pretrained
weights). Expected results:
  - Task accuracy ~50% (chance level on binary classification)
  - IIA should be substantially lower than the pretrained baseline

Runs DAS with intervention sizes [64, 128, 256] across 3 seeds.

Usage on Colab (L4 GPU):
    !python monli_sanity_check.py

Use --retrain to clear all saved checkpoints and results:
    !python monli_sanity_check.py --retrain
"""

# ── 0. Install dependencies ────────────────────────────────────
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.36.2", "torch", "scikit-learn", "pandas", "scipy"],
    stdout=subprocess.DEVNULL)

# ── 0.5. Mount Google Drive for persistent storage ────────────
import glob, os, time, urllib.request

DRIVE_DIR = "/content/drive/MyDrive/DAS_experiment"

try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    SAVE_DIR = os.path.join(DRIVE_DIR, "saved_models_nli_sanity")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Google Drive mounted. Saving to {SAVE_DIR}")
except ImportError:
    SAVE_DIR = "saved_models_nli_sanity"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Not on Colab. Saving to {SAVE_DIR}")

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

# Handle --retrain flag
if "--retrain" in sys.argv:
    for f in glob.glob(os.path.join(SAVE_DIR, "random-*.bin")):
        os.remove(f)
        print(f"  Deleted: {f}")
    _results_path = os.path.join(SAVE_DIR, "results_so_far.json")
    if os.path.exists(_results_path):
        os.remove(_results_path)
        print(f"  Deleted: {_results_path}")

# ── 2. Imports ──────────────────────────────────────────────
import copy, json, random
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, BertConfig

import dataset_nli
import utils
from LIM_bert import LIMBERTClassifier
from trainer import BERTLIMTrainer

# ── 3. Experiment configuration ────────────────────────────
WEIGHTS_NAME   = "ishan/bert-base-uncased-mnli"
MAX_LENGTH     = 40
SEEDS          = [42, 66, 77]
DIM_SIZES      = [64, 128, 256]

IIT_LAYER      = 8   # Layer 9 (0-indexed)

# DAS rotation-matrix training (same as baseline)
DAS_EPOCHS     = 5
DAS_LR         = 2e-3
DAS_BS         = 64
DAS_TRAIN      = 24_000
DAS_TEST       = 1_920

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


def create_random_bert():
    """Create a BERT model with the same architecture but random weights."""
    config = BertConfig.from_pretrained(WEIGHTS_NAME)
    config.attn_implementation = "eager"
    # BertModel.__init__ calls init_weights() which applies _init_weights
    # to all modules: Normal(mean=0, std=0.02) for weights, zeros for biases
    bert = BertModel(config)
    return bert


# ── 5. Main experiment ─────────────────────────────────────
def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be very slow.\n")
    print(f"Device: {device}")
    print(f"Config: layer={IIT_LAYER + 1}, dim_sizes={DIM_SIZES}, "
          f"seeds={SEEDS}\n")

    RESULTS_FILE = os.path.join(SAVE_DIR, "results_so_far.json")

    # Load any results saved from a previous (interrupted) run
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    completed = {(r["seed"], r["dim_per_var"]) for r in results}

    # ── Evaluate random model task accuracy once ──────────
    print("Checking random-BERT task accuracy (should be ~50%) ...")
    utils.fix_random_seeds(42)
    bert_check = create_random_bert()
    check_model = LIMBERTClassifier(
        n_classes=2, bert=bert_check, max_length=MAX_LENGTH,
        debug=False, target_dims={"start": 0, "end": 786},
        target_layers=[], device=device,
        static_search=False, nested_disentangle_inplace=False)

    check_trainer = BERTLIMTrainer(
        check_model,
        warm_start=False, max_iter=1,
        batch_size=32, n_iter_no_change=10000,
        shuffle_train=False, eta=2e-5,
        device=device, seed=42)

    X_te, y_te = load_factual(1_000, "test")
    preds = check_trainer.predict(X_te)
    report = classification_report(y_te, preds.cpu(), output_dict=True)
    task_acc = report["weighted avg"]["f1-score"]
    print(f"  Random BERT task F1 = {task_acc:.4f}  (expected ~0.50)\n")

    del check_model, check_trainer, bert_check
    torch.cuda.empty_cache()

    # ── Loop over dim sizes and seeds ─────────────────────
    for dim_per_var in DIM_SIZES:
        # TARGET_DIMS must cover at least 2*dim_per_var (V1 + V2)
        target_end = max(786, 2 * dim_per_var)
        target_dims = {"start": 0, "end": target_end}

        alignment = {
            0: [{"layer": IIT_LAYER, "start": 0,
                 "end": dim_per_var}],
            1: [{"layer": IIT_LAYER, "start": dim_per_var,
                 "end": 2 * dim_per_var}],
            2: [{"layer": IIT_LAYER, "start": 0,
                 "end": dim_per_var},
                {"layer": IIT_LAYER, "start": dim_per_var,
                 "end": 2 * dim_per_var}],
        }

        for seed in SEEDS:
            if (seed, dim_per_var) in completed:
                print(f"\n  DIM={dim_per_var}, SEED={seed}: "
                      f"already completed, skipping.")
                continue

            print(f"\n{'=' * 60}")
            print(f"  DIM={dim_per_var}, SEED={seed}")
            print(f"{'=' * 60}")
            utils.fix_random_seeds(seed)

            # ── Build IIT datasets ─────────────────────
            print("\n[1/2] Building interchange-intervention datasets ...")
            _ENCODE_CACHE.clear()
            utils.fix_random_seeds(seed)

            t0 = time.time()
            train_iit = load_combined_iit(DAS_TRAIN, "train")
            t1 = time.time()
            print(f"      Train set: {t1-t0:.1f}s  "
                  f"(cache: {len(_ENCODE_CACHE)} entries)")

            test_iit = load_combined_iit(DAS_TEST, "test")
            t2 = time.time()
            print(f"      Test set:  {t2-t1:.1f}s  "
                  f"(cache: {len(_ENCODE_CACHE)} entries)")

            # ── Create random BERT and save/load checkpoint ──
            ckpt_path = os.path.join(SAVE_DIR, f"random-{seed}.bin")
            if not os.path.exists(ckpt_path):
                utils.fix_random_seeds(seed)
                bert_rand = create_random_bert()
                rand_model = LIMBERTClassifier(
                    n_classes=2, bert=bert_rand, max_length=MAX_LENGTH,
                    debug=False, target_dims=target_dims,
                    target_layers=[], device=device,
                    static_search=False, nested_disentangle_inplace=False)
                torch.save(rand_model.state_dict(), ckpt_path)
                del rand_model, bert_rand
                torch.cuda.empty_cache()

            # ── Train DAS rotation matrix ────────────────
            print(f"\n[2/2] Training DAS rotation "
                  f"(layer {IIT_LAYER + 1}, {dim_per_var} dims) ...")

            bert_das = create_random_bert()
            das_model = LIMBERTClassifier(
                n_classes=2, bert=bert_das, max_length=MAX_LENGTH,
                debug=False, target_dims=target_dims,
                target_layers=[IIT_LAYER], device=device,
                static_search=False, nested_disentangle_inplace=False)

            saved_sd = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=True)
            das_model.load_state_dict(
                remap_factual_weights(saved_sd, IIT_LAYER), strict=False)

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

            # ── Evaluate IIA ───────────────────────
            print("      Evaluating IIA on test set ...")

            test_base_preds = das_trainer.predict(test_iit[0])
            test_fact_report = classification_report(
                test_iit[1], test_base_preds.cpu(), output_dict=True)
            test_fact_f1 = test_fact_report["weighted avg"]["f1-score"]

            test_iit_preds = das_trainer.iit_predict(
                test_iit[0], test_iit[2], test_iit[4], alignment)
            test_iit_report = classification_report(
                test_iit[3], test_iit_preds.cpu(), output_dict=True)
            test_iia = test_iit_report["weighted avg"]["f1-score"]

            result = {
                "seed":         seed,
                "dim_per_var":  dim_per_var,
                "task_f1":      task_acc,
                "das_fact_f1":  test_fact_f1,
                "test_iia":     test_iia,
            }
            results.append(result)

            print(f"\n      DIM={dim_per_var}, Seed {seed} results:")
            print(f"        Random task F1          = {task_acc:.4f}")
            print(f"        DAS factual test F1     = {test_fact_f1:.4f}")
            print(f"        DAS test IIA            = {test_iia:.4f}")

            del das_model, das_trainer, bert_das
            torch.cuda.empty_cache()

            # Save results incrementally
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"      Results saved to {RESULTS_FILE}")

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SANITY CHECK: DAS on Random BERT (no pretraining)")
    print("  High-level model: Negation + Lexical Entailment")
    print("=" * 60)
    hdr = (f"{'Dim':>5}  {'Seed':>6}  {'Task F1':>8}  "
           f"{'DAS Fact F1':>12}  {'Test IIA':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['dim_per_var']:>5}  {r['seed']:>6}  "
              f"{r['task_f1']:>8.4f}  "
              f"{r['das_fact_f1']:>12.4f}  {r['test_iia']:>10.4f}")

    # Best per dim size
    print("-" * len(hdr))
    for dim in DIM_SIZES:
        dim_results = [r for r in results if r["dim_per_var"] == dim]
        if dim_results:
            best = max(dim_results, key=lambda r: r["test_iia"])
            print(f"  DIM={dim:>3}: Best IIA = {best['test_iia']:.4f}  "
                  f"(seed {best['seed']})")

    print()
    return results


if __name__ == "__main__":
    run()
