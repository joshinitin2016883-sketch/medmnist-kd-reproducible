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

---

## E0 — Seed variance (RUN)

Three runs, identical configuration, differing only by seed.

| Seed | macro-F1 | weighted-F1 | Accuracy | Balanced acc | macro AUC | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.5224 | 0.7450 | 0.7332 | 0.6077 | 0.9307 | 11 |
| 1337 | **0.5944** | 0.7588 | 0.7451 | 0.6212 | 0.9355 | 20 |
| 2024 | 0.5594 | 0.7561 | 0.7392 | 0.6624 | 0.9340 | 16 |

| Metric | Mean | SD | Range |
|---|---:|---:|---:|
| **macro-F1** | 0.5587 | **0.0360** | **0.0720** |
| weighted-F1 | 0.7533 | 0.0073 | 0.0138 |
| accuracy | 0.7392 | 0.0060 | 0.0120 |

### This is the most important result in the file

**Macro-F1 has a standard deviation of 0.036 from seed alone** — five times
noisier than weighted-F1 (0.0073) and six times noisier than accuracy (0.0060).
The three runs span 0.5224 to 0.5944, a range of 0.072.

Consequences, in order of how much they matter:

**1. The original single-seed baseline was the worst of the three.** Seed 42's
0.5224 sits 0.036 below the mean. Any Phase 2 change measured against it from a
single run would inherit that bias.

**2. Any unpaired single-run comparison needs a difference above ~0.07 to mean
anything.** Most techniques in the backlog are worth 0.01–0.03. They are
individually undetectable this way. Multi-seed runs, or paired comparisons, are
mandatory — not a nicety.

**3. The original README's KD claim is now measurably noise.** It reported the
DermaMNIST student beating its teacher by **0.65 accuracy points**. The
seed-only range on accuracy is **1.20 points**. The claimed effect is roughly
half the noise floor of the measurement. This was previously an argument from
sample size; it is now a measurement.

**4. Small classes drive nearly all of it.** Per-class F1 range across seeds:

| Class | n | s42 | s1337 | s2024 | range |
|---|---:|---:|---:|---:|---:|
| vasc | 29 | 0.4602 | 0.8070 | 0.6067 | **0.3468** |
| df | 23 | 0.3684 | 0.5333 | 0.3614 | **0.1719** |
| bkl | 220 | 0.4239 | 0.4916 | 0.5280 | 0.1041 |
| akiec | 66 | 0.4192 | 0.4048 | 0.4774 | 0.0727 |
| mel | 223 | 0.5421 | 0.4973 | 0.5179 | 0.0449 |
| bcc | 103 | 0.5691 | 0.5439 | 0.5521 | 0.0252 |
| nv | 1341 | 0.8736 | 0.8830 | 0.8726 | 0.0104 |

`vasc` swings by 0.35 on 29 test images — a handful of images changing outcome.
Macro-F1 weights that class equally with `nv`, so the metric we care about
inherits the instability of the smallest classes. That is a property of the
benchmark, not a fixable defect, and it is why the protocol below now requires
paired comparisons wherever possible.

### Protocol amendment

Prefer **paired** experiments — same checkpoints, only the inference or
decision procedure changed. Seed variance cancels, and effects an order of
magnitude smaller than the unpaired noise floor become visible. Techniques
requiring retraining must be run at ≥3 seeds and compared mean to mean.

---

## Phase 2 results

### EXP-1 — Test-time augmentation (flips) — **KEPT**

Paired: same three checkpoints, 4-view flip averaging at inference only.

| Seed | baseline | +TTA | Δ |
|---|---:|---:|---:|
| 42 | 0.5224 | 0.5325 | +0.0101 |
| 1337 | 0.5944 | 0.6245 | +0.0300 |
| 2024 | 0.5594 | 0.5706 | +0.0111 |
| **mean** | 0.5587 | **0.5759** | **+0.0171** |

Positive on all three seeds. Paired *t*-test: t=2.64, **p=0.119**, 95% CI
−0.011 to +0.045.

**Verdict: kept, with the caveat stated.** The direction is consistent across
every seed and the mechanism is sound — dermoscopy has no canonical
orientation, so flips are label-preserving, and averaging four views reduces
variance in the decision. But n=3 does not reach conventional significance, and
the confidence interval includes zero. This is promising, not proven. Confirming
it properly needs more seeds.

Cost: 4× inference (~9s → ~37s on the full test split). No extra VRAM, no
retraining, no change to training code.

Per-class effect, averaged over seeds:

| Class | n | baseline | +TTA | Δ |
|---|---:|---:|---:|---:|
| akiec | 66 | 0.4338 | 0.4869 | **+0.0531** |
| bcc | 103 | 0.5550 | 0.5839 | +0.0289 |
| vasc | 29 | 0.6246 | 0.6532 | +0.0285 |
| bkl | 220 | 0.4812 | 0.5027 | +0.0215 |
| nv | 1341 | 0.8764 | 0.8787 | +0.0023 |
| mel | 223 | 0.5191 | 0.5149 | −0.0042 |
| df | 23 | 0.4211 | 0.4105 | −0.0106 |

The gain is concentrated in the rare classes, which is where macro-F1 has room.
Note melanoma does **not** improve — the clinically important class is untouched
by this change.

### EXP-2 — Logit adjustment — **REJECTED**

Subtract `tau · log(train prior)` from each log-probability; `tau` fitted on
validation.

| Seed | fitted tau | test Δ |
|---|---:|---:|
| 42 | 0.050 | −0.0070 |
| 1337 | 0.000 | +0.0000 |
| 2024 | 0.025 | −0.0014 |
| **mean** | | **−0.0028** |

**Verdict: rejected. No effect.** Validation selected `tau` at or near zero on
every seed — that is, the search found the *unadjusted* rule was already best on
validation. The prior correction I expected to help does not, and the honest
reading is that the baseline's class-weighted focal loss has already absorbed
most of the prior correction during training. Applying it again at decision time
double-counts.

This was ranked #2 in the backlog on the strength of the macro-AUC 0.93 vs
macro-F1 0.52 gap. That gap is real, but logit adjustment is not the way to
close it. A negative result, logged.

### EXP-3 — Per-class decision weights — **REJECTED (overfits validation)**

Seven multiplicative weights, fitted by coordinate ascent on validation.

| Seed | val Δ | test Δ |
|---|---:|---:|
| 42 | +0.0647 | +0.0204 |
| 1337 | +0.0500 | −0.0339 |
| 2024 | +0.0553 | +0.0234 |
| **mean** | **+0.0566** | **+0.0033** |

Paired *t*-test on test deltas: t=0.18, **p=0.876**.

**Verdict: rejected.** This is the cleanest demonstration of overfitting in the
project. The rule gains **+0.057 macro-F1 on validation, every single time**,
and delivers **+0.003 on test** — nothing. Seven free parameters fitted against
1,003 validation images, where four classes have fewer than 60 examples, is
enough capacity to memorise validation noise.

Run against EXP-2's single parameter, which gained nothing anywhere, the pair
makes the point precisely: the 7-parameter rule did not fail because tuning
thresholds is wrong, it failed because it fitted noise. Anyone reporting the
validation number here would be reporting a +0.057 improvement that does not
exist.

### EXP-4 — Native 224px source data — **LARGEST EFFECT MEASURED**

`--medmnist-size 224`. MedMNIST+ ships DermaMNIST at native 224×224 (1,041 MB)
instead of the default 28×28 (18.8 MB) upsampled 8× at load time. Same official
splits (7007/1003/2005), same architecture, same recipe, three seeds.

| Seed | macro-F1 | weighted-F1 | Accuracy | Balanced acc | macro AUC |
|---|---:|---:|---:|---:|---:|
| 42 | 0.7809 | 0.8794 | 0.8823 | 0.7761 | 0.9796 |
| 1337 | **0.7933** | 0.8610 | 0.8544 | 0.8386 | 0.9788 |
| 2024 | 0.7455 | 0.8462 | 0.8399 | 0.7924 | 0.9730 |

| Metric | 28px (mean±sd) | 224px (mean±sd) | Δ |
|---|---:|---:|---:|
| **macro-F1** | 0.5587 ± 0.0360 | **0.7733 ± 0.0248** | **+0.2145** |
| weighted-F1 | 0.7533 ± 0.0073 | 0.8622 ± 0.0166 | +0.1089 |
| accuracy | 0.7392 ± 0.0060 | 0.8589 ± 0.0216 | +0.1197 |

**+0.2145 macro-F1 — roughly six times the 0.036 seed-noise SD.** This is not a
result that needs careful statistics. It is larger than every other effect in
this file combined, by an order of magnitude.

Cost: mean training time 15.7 min → 19.5 min (+24%), and a 1 GB download.
**VRAM is unchanged** — tensors were already 224×224 in both conditions, so GPU
compute is identical. The only extra cost is data loading. Fits the same 6 GB
card at the same batch size.

#### Per-class F1, mean of 3 seeds

| Class | n | 28px | 224px | Δ |
|---|---:|---:|---:|---:|
| bkl | 220 | 0.4812 | 0.7572 | **+0.2760** |
| akiec | 66 | 0.4338 | 0.7082 | **+0.2744** |
| df | 23 | 0.4211 | 0.6904 | **+0.2693** |
| vasc | 29 | 0.6246 | 0.8813 | **+0.2567** |
| bcc | 103 | 0.5550 | 0.7838 | +0.2288 |
| **mel** | 223 | 0.5191 | 0.6632 | **+0.1441** |
| nv | 1341 | 0.8764 | 0.9286 | +0.0523 |

Every class improves. The gains concentrate exactly where the resolution
hypothesis predicts: `nv`, the easy majority class distinguishable by gross
colour and shape, gains least (+0.05); the classes requiring fine dermoscopic
structure gain four to five times as much.

#### The caveat that matters clinically

**Melanoma benefits least of all the lesion classes** (+0.144 against +0.23 to
+0.28 for the others), and its recall is the least stable across seeds:

| Seed | 28px mel recall | 224px mel recall |
|---|---:|---:|
| 42 | 0.6637 | **0.6233** (worse) |
| 1337 | 0.6099 | 0.8161 |
| 2024 | 0.6502 | 0.7175 |

Mean recall rises 0.641 → 0.719, but one seed got *worse*, and the spread
(0.62–0.82) is far wider than the other classes'. Resolution does not solve
melanoma. Melanoma-vs-nevus is the genuinely hard discrimination in this task,
and it remains the weakest link at any resolution.

#### What this means for the rest of the project

This reframes every prior result in this repository and in the original
notebook. All of the loss engineering, class weighting, focal loss and knowledge
distillation was optimising within an artificial constraint that costs **0.21
macro-F1** — larger than any of those techniques could plausibly recover.

Concretely: the original README compared teacher and student models at
differences of **0.6 accuracy points**, on data whose resolution ceiling costs
**12 accuracy points and 21 macro-F1 points**. Those comparisons were measuring
sub-noise differences inside a badly handicapped setting.

**Verdict: adopt 224px as the default benchmark for all further work.** The
28px results remain in this file as the historical baseline and as the
measurement of what the ceiling cost. They are a *different benchmark*, not a
worse run of the same one, and the two must never be merged into one table.

---

## Still not run

The remaining backlog items all require retraining, and per the protocol
amendment each needs ≥3 seeds to be measurable against a 0.036 SD — roughly
45 minutes per experiment.

| # | Change | Status |
|---|---|---|
| 1 | Native 224px source data | not run — highest expected gain |
| 4 | Cosine LR schedule | not run |
| 5 | Untangle focal / weights / smoothing | not run |
| 6 | Discriminative LR + progressive unfreeze | not run |
| 7 | Class-weight power sweep | not run |
| 8 | Checkpoint ensembling / SWA | not run |
| 9 | Stronger augmentation | not run |
| 10–12 | Mixup, CutMix, balanced sampler | not run |

### New item, added from E0 evidence

**Early-stop on validation macro-F1 rather than validation loss.** Observed
independently on all three seeds: the epoch with the best val macro-F1 was
rejected in favour of the epoch with the best val loss.

| Seed | best-loss epoch (kept) | best-macro-F1 epoch (discarded) | val macro-F1 given up |
|---|---|---|---:|
| 42 | 6 (0.5569) | 7 (0.5767) | −0.0198 |
| 1337 | 13 (0.5829) | 9 (0.6021) | −0.0192 |

Selecting on val loss optimises a quantity dominated by `nv`, which is not the
target metric. Cheap to test: it changes only the early-stopping criterion.
