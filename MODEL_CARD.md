# Model Card — Biomedical Image Classification with Knowledge Distillation

## ⚠️ This is not a clinical diagnostic tool

This model must not be used to diagnose, screen, triage, or make any decision
about a real patient. It is a coursework and research artifact built on public
benchmark data. It has no regulatory clearance of any kind — not FDA, not CE,
not CDSCO. It has never been validated on prospective clinical data, in any
clinical workflow, or on any population outside its training benchmark.

A skin lesion classifier that is wrong about melanoma can contribute to a missed
cancer. Treat every output as a number produced by a pattern matcher, not as a
medical opinion.

---

## Model details

| | |
|---|---|
| **Task** | Multi-class classification of biomedical images |
| **Architectures** | ResNet-50, DenseNet-121, EfficientNet-B0 (ImageNet-pretrained, fine-tuned end-to-end) |
| **Distillation** | EfficientNet-B0 student; ResNet-50 or DenseNet-121 teacher; KL divergence at T=3.0, α=0.7 |
| **Input** | 3×224×224, ImageNet normalisation |
| **Output** | Softmax over 7 classes (DermaMNIST) or 2 (PneumoniaMNIST, BreastMNIST) |
| **Framework** | PyTorch |
| **Explainability** | Grad-CAM over the final convolutional block |
| **Licence** | MIT |

All layers are trainable from step one — there is no frozen backbone or staged
unfreezing in the baseline recipe.

---

## Intended use

**In scope**

- Coursework, teaching, and demonstration of transfer learning and knowledge
  distillation on standardised medical imaging benchmarks.
- Methods research: comparing training recipes, loss functions, and compression
  strategies under class imbalance, using MedMNIST as a controlled testbed.
- Reproducing and extending the MedMNIST benchmark results.

**Explicitly out of scope**

- Any clinical or diagnostic use, including "assistive" or "second opinion" use.
- Triage, screening, or prioritisation of real patients.
- Deployment on images from any source other than the MedMNIST benchmark —
  including clinical dermatoscopes, smartphone photographs, or hospital PACS.
- Any use where a person could be harmed by a wrong output.
- Claims of clinical performance. The reported metrics describe performance on a
  benchmark test split, not diagnostic accuracy.

---

## Training data

Three MedMNIST v2 datasets. MedMNIST supplies fixed official train/val/test
splits; this project uses them unchanged and does not resplit.

| Dataset | Domain | Classes | Total images |
|---|---|---|---|
| DermaMNIST | Dermatoscopic skin lesions (from HAM10000) | 7 | 10,015 |
| PneumoniaMNIST | Paediatric chest X-ray | 2 | 5,856 |
| BreastMNIST | Breast ultrasound | 2 | 780 |

### Caveats that materially affect interpretation

**Resolution.** MedMNIST images are **28×28**. The pipeline upsamples them to
224×224 to match the ImageNet backbones. That upsample adds no information.
Dermatoscopic diagnosis relies on fine structure — pigment networks, dots and
globules, streaks, blue-white veil — essentially none of which survives at
28×28. Any accuracy figure here is a ceiling imposed by the data, not a measure
of what these architectures can do on real dermoscopy.

**Population skew (DermaMNIST / HAM10000).** HAM10000 was collected from two
sites: the Medical University of Vienna, Austria, and a skin cancer practice in
Queensland, Australia. Both populations are predominantly light-skinned. Darker
skin tones (Fitzpatrick V–VI) are severely underrepresented. This is a
well-documented and consequential bias in dermatology datasets: models trained
on them perform substantially worse on darker skin, on precisely the population
where melanoma is already diagnosed later and has worse outcomes. **This model
should be assumed not to generalise across skin tones.** Nothing in this
repository measures or corrects for that, because MedMNIST carries no
demographic metadata.

**Possible lesion-level leakage.** HAM10000 contains multiple images of the same
physical lesion (roughly 10,015 images across ~7,500 unique lesions). If the
MedMNIST split partitions by *image* rather than by *lesion*, near-duplicate
views of one lesion can land in both train and test, which inflates test scores
relative to true generalisation. This has not been verified for the split used
here, and should be checked before quoting any number as generalisation
performance.

**Class imbalance.** DermaMNIST is severely imbalanced. In the test split,
melanocytic nevi (`nv`) account for 1,341 of 2,005 images (67%), while
dermatofibroma (`df`) has 23 and vascular lesions (`vasc`) 29. Two consequences:

1. A model that predicts `nv` for every input scores ~67% accuracy. Accuracy is
   nearly uninformative here.
2. Metrics on the smallest classes rest on a couple dozen images. One image
   changing outcome moves `df` recall by 4.3 percentage points.

**Paediatric-only chest X-rays.** PneumoniaMNIST is drawn from paediatric
patients aged 1–5. It says nothing about adult chest radiography.

---

## Evaluation methodology

- Official MedMNIST test split, held out from training and model selection.
- Model selection is by validation loss (early stopping, patience 5, best
  weights restored). The test split is touched exactly once, at the end.
- Metrics via `evaluate.py`: per-class precision/recall/F1/support, macro-F1,
  weighted-F1, accuracy, balanced accuracy, confusion matrix, and per-class
  one-vs-rest AUC.

**Primary metric is macro-F1**, not accuracy and not weighted-F1. Both of those
are dominated by `nv` and will look healthy while the model fails at melanoma.
Any metric that cannot be computed (for example AUC for a class with no positive
test examples) is omitted and recorded in the `warnings` array of
`metrics.json` — never filled with a placeholder.

---

## Quantitative results

**Not yet established for this repository.**

`train.py` and `evaluate.py` were added after the original notebook. No
checkpoint has been produced by them yet, so there is no `metrics.json` to
report from. This section will be filled from real `evaluate.py` output and
nothing else.

The figures currently in `README.md` come from the notebook's saved cell
outputs. They are genuine numbers from a genuine run, but that run set no random
seed and saved no weights, so it cannot be reproduced or audited. See
`EXPERIMENTS.md`.

---

## Known limitations

1. **Benchmark performance is not diagnostic performance.** These are 28×28
   thumbnails from a curated benchmark, not clinical images.
2. **No demographic evaluation.** Performance is not broken down by skin tone,
   age, sex, or site, because the data carries no such labels. Bias is therefore
   unmeasured, not absent.
3. **Rare-class estimates are unstable.** With 23–29 test images, per-class
   metrics for `df` and `vasc` have confidence intervals wide enough that small
   differences between models are not meaningful.
4. **No calibration.** Softmax outputs are not calibrated probabilities and
   should not be read as confidence. No temperature scaling or reliability
   analysis has been performed.
5. **No out-of-distribution detection.** Given an input unlike anything in
   training — a different imaging modality, a photograph, a blank image — the
   model still emits a confident-looking 7-way distribution.
6. **Grad-CAM is not an explanation of correctness.** It shows which spatial
   regions influenced a logit. It cannot show whether the model learned lesion
   morphology or an artifact such as a ruler mark, ink annotation, or vignetting
   — all of which appear in HAM10000 and all of which correlate with class.
7. **Original results are unreproducible.** The notebook set no seed, so its
   published table cannot be regenerated. `train.py` seeds every run.

---

## Ethical considerations

Automated skin lesion classification carries a specific and asymmetric risk: a
false negative on melanoma can delay treatment of an aggressive cancer, while a
false positive costs a biopsy. These are not equivalent errors, and neither
accuracy nor macro-F1 encodes that asymmetry. Any real deployment would need a
decision threshold chosen from the clinical cost of each error type, not from a
symmetric metric.

The population skew described above means the harm from this asymmetry would not
fall evenly. A model trained overwhelmingly on light skin and deployed broadly
would fail most often on the patients already worst served by dermatology.

---

## Citation

If you use the datasets, cite the original sources:

- Yang et al., *MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D
  Biomedical Image Classification*, Scientific Data, 2023.
- Tschandl et al., *The HAM10000 dataset*, Scientific Data, 2018.
- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by
  Image-Based Deep Learning*, Cell, 2018. (PneumoniaMNIST source)
- Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.
  (BreastMNIST source)
