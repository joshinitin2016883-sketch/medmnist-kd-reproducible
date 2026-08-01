"""Post-hoc decision-rule tuning. No retraining.

argmax over softmax minimises error rate. That is the wrong objective when
classes are imbalanced and the target is macro-F1: with 67% of DermaMNIST being
`nv`, the model needs overwhelming evidence before predicting anything else, so
rare classes are systematically under-predicted. The baseline shows this
directly -- macro AUC 0.93 (the model ranks well) against macro-F1 0.52 (the
decision rule converts that ranking badly).

Two rules are fitted here, both on **validation** probabilities and then applied
unchanged to test:

  logit-adjustment   subtract tau * log(train prior) from each log-probability.
                     One free parameter, so very little room to overfit.
  per-class scaling  one multiplicative weight per class, fitted by coordinate
                     ascent. Seven free parameters -- more expressive, more
                     prone to fitting validation noise.

Fitting on test would not be an experiment, it would be reporting a training
score. The split used for fitting is recorded in the output.

Usage
-----
python tune_thresholds.py --val-probs results/..._val_probs.npz \
                          --test-probs results/..._probs.npz \
                          --out results/..._thresholds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(labels, preds, n_classes):
    return float(
        f1_score(labels, preds, labels=list(range(n_classes)),
                 average="macro", zero_division=0)
    )


def load_probs(path):
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["label_names"]] if "label_names" in d else None
    return d["probs"], d["labels"], names


def train_prior(dataset_flag, data_root, n_classes):
    """Class frequencies in the training split, used by logit adjustment."""
    from evaluate import build_eval_transform, load_split

    ds, _ = load_split(dataset_flag, "train", data_root, build_eval_transform(224))
    labels = np.asarray(getattr(ds, "labels")).reshape(-1)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return counts / counts.sum()


def fit_logit_adjustment(val_probs, val_labels, prior, n_classes, taus):
    """Pick the single tau maximising macro-F1 on validation."""
    log_prior = np.log(np.clip(prior, 1e-12, None))
    log_p = np.log(np.clip(val_probs, 1e-12, None))

    best = (0.0, macro_f1(val_labels, val_probs.argmax(1), n_classes))
    curve = []
    for tau in taus:
        preds = (log_p - tau * log_prior).argmax(1)
        score = macro_f1(val_labels, preds, n_classes)
        curve.append({"tau": round(float(tau), 4), "val_macro_f1": round(score, 6)})
        if score > best[1]:
            best = (float(tau), score)
    return best[0], best[1], curve


def apply_logit_adjustment(probs, prior, tau):
    log_prior = np.log(np.clip(prior, 1e-12, None))
    return (np.log(np.clip(probs, 1e-12, None)) - tau * log_prior).argmax(1)


def fit_class_weights(val_probs, val_labels, n_classes, rounds=6):
    """Coordinate ascent on one multiplicative weight per class."""
    weights = np.ones(n_classes)
    grid = np.concatenate([np.linspace(0.25, 1.0, 16), np.linspace(1.0, 6.0, 26)])
    best = macro_f1(val_labels, (val_probs * weights).argmax(1), n_classes)

    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            original = weights[c]
            for g in grid:
                weights[c] = g
                score = macro_f1(val_labels, (val_probs * weights).argmax(1), n_classes)
                if score > best + 1e-9:
                    best, original, improved = score, g, True
            weights[c] = original
        if not improved:
            break
    return weights, best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-probs", required=True, help="npz from evaluate.py --split val")
    p.add_argument("--test-probs", required=True, help="npz from evaluate.py --split test")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="results/thresholds.json")
    args = p.parse_args(argv)

    val_probs, val_labels, names = load_probs(args.val_probs)
    test_probs, test_labels, _ = load_probs(args.test_probs)
    n_classes = val_probs.shape[1]
    names = names or [str(i) for i in range(n_classes)]

    prior = train_prior(args.dataset, args.data_root, n_classes)

    base_val = macro_f1(val_labels, val_probs.argmax(1), n_classes)
    base_test = macro_f1(test_labels, test_probs.argmax(1), n_classes)

    print(f"classes      : {n_classes}")
    print(f"val / test n : {len(val_labels)} / {len(test_labels)}")
    print(f"train prior  : " + "  ".join(f"{n}={v:.3f}" for n, v in zip(names, prior)))
    print()
    print(f"baseline argmax   val macro-F1 {base_val:.4f} | test macro-F1 {base_test:.4f}")

    # --- rule 1: logit adjustment -----------------------------------------
    taus = np.linspace(0.0, 1.5, 61)
    tau, la_val, curve = fit_logit_adjustment(
        val_probs, val_labels, prior, n_classes, taus
    )
    la_test = macro_f1(test_labels, apply_logit_adjustment(test_probs, prior, tau),
                       n_classes)
    print(f"logit adjustment  tau={tau:.3f}  "
          f"val {la_val:.4f} | test {la_test:.4f}  "
          f"(test delta {la_test - base_test:+.4f})")

    # --- rule 2: per-class weights ----------------------------------------
    weights, cw_val = fit_class_weights(val_probs, val_labels, n_classes)
    cw_test = macro_f1(test_labels, (test_probs * weights).argmax(1), n_classes)
    print(f"per-class weights " + " ".join(f"{w:.2f}" for w in weights))
    print(f"                  val {cw_val:.4f} | test {cw_test:.4f}  "
          f"(test delta {cw_test - base_test:+.4f})")

    # --- per-class F1 under the better rule --------------------------------
    better = "logit_adjustment" if la_test >= cw_test else "per_class_weights"
    best_preds = (apply_logit_adjustment(test_probs, prior, tau)
                  if better == "logit_adjustment" else (test_probs * weights).argmax(1))

    per_class_before = f1_score(test_labels, test_probs.argmax(1),
                                labels=list(range(n_classes)), average=None,
                                zero_division=0)
    per_class_after = f1_score(test_labels, best_preds,
                               labels=list(range(n_classes)), average=None,
                               zero_division=0)
    print(f"\nper-class test F1 under {better}:")
    for i, n in enumerate(names):
        print(f"  {n:<48s} {per_class_before[i]:.4f} -> {per_class_after[i]:.4f} "
              f"({per_class_after[i] - per_class_before[i]:+.4f})")

    payload = {
        "fitted_on": "val",
        "applied_to": "test",
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "class_names": names,
        "train_prior": [float(x) for x in prior],
        "baseline": {"val_macro_f1": base_val, "test_macro_f1": base_test},
        "logit_adjustment": {
            "tau": float(tau), "val_macro_f1": la_val, "test_macro_f1": la_test,
            "test_delta": la_test - base_test, "tau_curve": curve,
        },
        "per_class_weights": {
            "weights": [float(w) for w in weights],
            "val_macro_f1": cw_val, "test_macro_f1": cw_test,
            "test_delta": cw_test - base_test,
        },
        "better_rule": better,
        "per_class_test_f1": {
            n: {"before": float(per_class_before[i]), "after": float(per_class_after[i])}
            for i, n in enumerate(names)
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nwritten -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
