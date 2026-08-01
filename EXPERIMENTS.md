# Experiments

Experiment log for improving **macro-F1** on DermaMNIST (7-class, severely
imbalanced). Every number in the Results section comes from an actual run of
`evaluate.py`. Nothing here is estimated, projected, or filled in by hand.

Target metric is **macro-F1**, not accuracy and not weighted-F1. On a split
where `nv` is 67% of samples, a model that predicts `nv` for every input scores
~67% accuracy and ~0.11 macro-F1. Weighted-F1 has the same blind spot: it
averages per-class F1 weighted by support, so the majority class dominates it
almost entirely. Macro-F1 weights all seven classes equally, which is the only
one of the three that notices melanoma being missed.

---

## Protocol

1. One change per experiment. Never two.
2. Baseline stays runnable — every change goes behind a flag with the old
   behaviour as the default.
3. Same seed, same split, same eval command across compared runs.
4. Keep the change only if macro-F1 improves **by more than the seed-variance
   band measured in E0**. A gain smaller than the noise floor is not a gain.
5. Negative results are logged here, not deleted. A change that didn't work is
   a finding.

Every run is recorded as:

```powershell
python train.py    <flags> --run-name <id> --out checkpoints/<id>.pt
python evaluate.py --checkpoint checkpoints/<id>.pt --out results/<id>.json `
                   --cm-png results/<id>_cm.png --save-probs results/<id>_probs.npz
```

---

## E0 — Seed variance (prerequisite, not an improvement)

**Run this before any other experiment.**

The notebook set no seed, so its published numbers are unreproducible even by
its author. Worse, without knowing how much macro-F1 moves between identical
runs that differ only by seed, we cannot tell a real improvement from noise.

`df` has 23 test images. One image flipping from wrong to right moves that
class's recall by 4.3 percentage points, which moves macro-F1 by ~0.6 points on
its own. Small-class noise is structurally large here.

Protocol: train the baseline three times with `--seed 42`, `--seed 1337`,
`--seed 2024`, changing nothing else. Record mean and spread of macro-F1. That
spread is the bar every later experiment must clear.

Cost: 2 extra training runs. Buys: the ability to make any claim at all.

---

## Ranked backlog

Ranked by expected macro-F1 gain per unit of cost and risk, for this specific
problem: 7 classes, ~58:1 imbalance, small images, 6 GB VRAM.

**VRAM is mostly not the binding constraint.** Every tensor already arrives at
224x224 regardless of source resolution, so most of these cost no extra VRAM.
Training wall-clock and data-loading throughput are the real budget.

| # | Change | Expected macro-F1 | Train time | VRAM | Risk |
|---|--------|------------------|-----------|------|------|
| 1 | Native 224px source data | **large** | ↑↑ | none | low, but changes benchmark |
| 2 | Per-class threshold / logit adjustment | **large** | none | none | val overfit |
| 3 | Test-time augmentation | small–moderate | none | none | very low |
| 4 | Cosine LR schedule | small–moderate | none | none | low |
| 5 | Untangle focal / weights / smoothing | unknown, diagnostic | 3 runs | none | low |
| 6 | Discriminative LR + progressive unfreeze | small–moderate | ↓ slightly | ↓ | low |
| 7 | Class-weight power sweep | small–moderate | 1 run each | none | low |
| 8 | Checkpoint ensembling / SWA | small | none | none | low |
| 9 | Stronger augmentation policy | small | ↑ slightly | none | **domain risk** |
| 10 | Mixup | uncertain | ↑ (needs longer) | none | moderate |
| 11 | CutMix | uncertain | ↑ | none | **high — see notes** |
| 12 | Balanced sampler instead of loss weights | small | none | none | low |

Deliberately not ranked: swapping the backbone. DenseNet121 already beat
ResNet50 in the notebook, but that is an architecture comparison, not a
training-recipe improvement, and it belongs in its own table.

---

### 1. Native 224px source data — `--medmnist-size 224`

The single largest lever, and it is not a training trick.

DermaMNIST is HAM10000 **downsampled to 28x28**. `transforms.Resize((224,224))`
then blows each image up 8x. That upsample adds no information — the model is
looking at 784 pixels of lesion, stretched. Dermoscopic diagnosis depends on
fine structure: pigment networks, dots and globules, streaks, blue-white veil.
None of that survives 28x28. This is a hard ceiling on every number in the repo,
and no amount of loss engineering lifts it.

MedMNIST+ ships `dermamnist` at native 64/128/224. At 224 the model finally sees
what the classifier was pretrained to see.

- **VRAM: unchanged.** Tensors are already 224x224 either way.
- **Time: up substantially** — real JPEG decode instead of a trivial upsample.
  Expect data loading, not GPU, to become the bottleneck. `--num-workers 4`
  becomes worth testing here.
- **Disk: a few GB** to download.
- **Risk:** low methodologically, but this changes the dataset. Results are
  **not** comparable to the 28px baseline or to the published table. It must be
  reported as a separate benchmark, never merged into the same row.

### 2. Per-class thresholds / logit adjustment

Zero training cost, operates entirely on the `_probs.npz` files already saved.

`argmax` over softmax minimises error rate, which is the wrong objective when
classes are imbalanced and you care about macro-F1. With a 67% prior on `nv`,
the model needs strong evidence before predicting anything else, so rare classes
are systematically under-predicted — visible in the baseline as high `nv` recall
alongside poor `akiec`/`bkl` recall.

Two variants, both cheap:
- **Logit adjustment:** subtract `tau * log(train_prior)` from each logit. One
  scalar to tune.
- **Per-class thresholds:** tune seven decision offsets directly.

- **Risk:** thresholds must be fitted on the **validation** split and then
  applied unchanged to test. Fitting on test is not an experiment, it is
  reporting your training score. With 7 free parameters and a small val split,
  logit adjustment's single `tau` is the safer starting point.

### 3. Test-time augmentation

Average predictions over the original image plus horizontal flip, vertical flip,
and both. Dermoscopy has **no canonical orientation** — a lesion rotated 180° is
the same lesion — so flips are genuinely label-preserving here. That is not true
for, say, chest X-rays, where a horizontal flip creates dextrocardia.

4x inference cost on a test set that evaluates in seconds. No extra VRAM. Among
the lowest risk-to-reward ratios available.

### 4. Cosine LR schedule

Baseline runs Adam at a flat 1e-4 and early-stops around epoch 13. Cosine
annealing to near-zero typically buys a small but reliable gain by letting the
model settle into a flatter minimum.

**Subtlety worth understanding before running it:** a cosine schedule is defined
over a fixed horizon. Combined with early stopping at patience 5, the run may
halt at epoch 13 of a 50-epoch cosine curve, at which point the LR has barely
decayed and you have tested nothing. Either set the horizon to the expected stop
epoch, or switch to fixed-epoch training with best-checkpoint selection.

### 5. Untangle focal / class weights / label smoothing

Not a performance change — a diagnostic that makes every later result
interpretable. The baseline stacks three imbalance corrections at once:
`FocalLoss(gamma=1.5)` + damped class weights + `label_smoothing=0.1` welded
inside the loss. Nobody can say which is helping.

Three runs isolate them: plain weighted CE; focal with
`--label-smoothing 0.0`; CE with smoothing only. One of these may well beat the
bundle — combined regularisers often over-correct.

### 6. Discriminative LR + progressive unfreezing

Baseline trains all 25.6M ResNet-50 parameters at a uniform 1e-4 from step one,
on 7,007 images. Early conv layers already encode edges and textures that
transfer fine; hammering them at full LR mostly destroys pretrained structure.

Layer-wise decay (head at 1e-4, `layer4` at 3e-5, `layer1` at 1e-6) or freezing
the stem for the first few epochs both act as regularisation. Frozen layers need
no gradient buffers, so this is one of the few changes that *reduces* both time
and VRAM.

### 7. Class-weight power sweep

Measured on DermaMNIST proportions, the current damping chain does far less than
it looks like it does. `balanced` weights span 58:1. The `** 0.5` takes that to
sqrt(58) ≈ 7.6:1, the mean-normalise is a pure rescale, and **no class ever
reaches the 2.0 clamp — the clamp is inert.** So rare classes are weighted 7.6x
less than true inverse frequency.

Sweep `power` over {0.0, 0.25, 0.5, 0.75, 1.0}. Note that full inverse frequency
is not obviously better: `df` has ~80 training images, and multiplying their
gradients by 58 makes a handful of samples dominate each step, which raises
variance and invites memorisation of those specific images.

### 8. Checkpoint ensembling / SWA

Average either the weights (SWA) or the predictions of the top-k validation
checkpoints. Nearly free given checkpoints already exist, and it reduces the
seed-to-seed variance that E0 will quantify. Modest but dependable.

### 9. Stronger augmentation — **domain caution**

Current policy: hflip, vflip, rot30, `ColorJitter(0.2, 0.2, 0.2)`.

Adding RandAugment or RandomResizedCrop is standard practice and probably helps.
**But be careful with colour.** Melanoma diagnosis depends substantially on
colour variegation — multiple distinct pigment shades within one lesion is a
malignancy criterion. Aggressive brightness/saturation jitter can wash out
exactly the signal being classified. The existing 0.2 is already non-trivial;
pushing to 0.4 may cost more than it buys. Test colour and geometric
augmentation separately.

### 10–11. Mixup and CutMix

**Mixup** (convex blends of image pairs and their labels) is a reasonable
regulariser for limited data, but it needs longer training to pay off and
interacts badly with patience-5 early stopping — mixup raises training loss by
construction and can trip the stopper before it helps.

**CutMix is actively risky here** and I would run it last, if at all. It pastes
a rectangular patch from one image over another and mixes labels by patch area.
On dermoscopy the lesion is a single centred object against plain skin. A patch
can easily cover the entire lesion or none of it, so the area-proportional label
becomes simply wrong: an image showing 100% melanoma gets labelled 60% melanoma
/ 40% nevus. Label noise, not regularisation.

### 12. Balanced sampler instead of loss weights

`WeightedRandomSampler` rebalances at the batch level rather than the loss level.
Roughly equivalent in effect, but it changes what an epoch means — rare images
are drawn repeatedly, common ones skipped. Worth one run as an alternative to
#7, not in addition to it.

---

## Results

All figures below are read from `results/*.json`, produced by `evaluate.py`.

| Run | Change | macro-F1 | weighted-F1 | Accuracy | Balanced acc | macro AUC | Train time | Verdict |
|-----|--------|---------:|------------:|---------:|-------------:|----------:|-----------:|---------|
| `baseline_resnet_derma_seed42` | — (notebook recipe, seeded) | **0.5224** | 0.7450 | 0.7332 | 0.6077 | 0.9307 | 13.7 min | baseline |

<sub>ResNet-50, DermaMNIST 28px→224, Adam 1e-4, batch 64, FocalLoss(γ=1.5,
ls=0.1) + damped class weights, early stop patience 5. Stopped at epoch 11,
best val loss at epoch 6. RTX 4050 6 GB.</sub>

### Per-class, baseline

| Class | Precision | Recall | F1 | Support | AUC |
|-------|----------:|-------:|---:|--------:|----:|
| actinic keratoses / intraepithelial carcinoma | 0.3465 | 0.5303 | 0.4192 | 66 | 0.9539 |
| basal cell carcinoma | 0.4895 | 0.6796 | 0.5691 | 103 | 0.9555 |
| benign keratosis-like lesions | 0.5270 | 0.3545 | 0.4239 | 220 | 0.8635 |
| dermatofibroma | 0.4667 | 0.3043 | 0.3684 | 23 | 0.9429 |
| **melanoma** | 0.4582 | **0.6637** | 0.5421 | 223 | 0.8875 |
| melanocytic nevi | 0.9286 | 0.8248 | 0.8736 | 1341 | 0.9264 |
| vascular lesions | 0.3095 | 0.8966 | 0.4602 | 29 | 0.9851 |

### Observations from the baseline run

**The headline metric is 0.22 higher than the honest one.** Weighted-F1 0.7450
vs macro-F1 0.5224. The entire gap is `nv` (F1 0.8736, 67% of the split)
carrying the weighted average while five of seven classes sit below F1 0.55.

**Melanoma recall is 0.6637.** The model misses roughly one melanoma in three.
Of 223 melanomas, 32 are called `nv` and 18 are called `bkl` — 50 malignant
lesions labelled benign, 22% of all melanoma cases. This is the number that
matters clinically and it is invisible in every metric the README currently
reports.

**AUC looks excellent and is misleading on its own.** Macro AUC 0.9307, with
every class above 0.86 — yet macro-F1 is 0.5224. AUC is threshold-free and
measures ranking; the model *ranks* classes well but `argmax` at the default
threshold converts that ranking into poor decisions under a 67% majority prior.
That gap is precisely the opportunity in experiment #2 (threshold / logit
adjustment), and it is now measured rather than assumed.

**Selection on val loss cost macro-F1.** Epoch 7 had the best val macro-F1
(0.5767) but epoch 6 had the best val loss (0.2995) and was the checkpoint kept.
Epochs 4→5 show the same divergence: val loss improved while val macro-F1 fell
0.5752→0.5600. Early stopping is optimising something other than the target
metric.

**The recipe reproduces the notebook.** Train losses match closely per epoch
(e.g. 0.3947/0.3335/0.3099 here vs 0.3956/0.3343/0.3119 in the notebook), and
test accuracy lands at 0.7332 against the notebook's 0.7332 — despite a
different, and in the notebook's case absent, seed. `train.py` is a faithful
extraction.

### Not yet run

E0 (seed variance) still needs `--seed 1337` and `--seed 2024`. Until those
exist there is no noise floor, so no Phase 2 result can be called an
improvement.
