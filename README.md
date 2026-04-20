# Responsible & Explainable AI 
Trust & Safety Bias Audit and Guardrail Pipeline**

---

## Overview

This repository implements a complete bias audit and production guardrail pipeline for a DistilBERT-based toxicity classifier trained on the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset.

The assignment is structured in five sequential parts:

| File | Part | Description |
|---|---|---|
| `part1.ipynb` | Baseline Classifier | Fine-tune DistilBERT, report metrics, justify threshold |
| `part2.ipynb` | Bias Audit | Measure FPR/TPR disparity between Black-associated and White-associated cohorts |
| `part3.ipynb` | Adversarial Attacks | Character-level evasion attack + label-flipping poisoning attack |
| `part4.ipynb` | Mitigation | Reweighing (AIF360), ThresholdOptimizer (Fairlearn), Oversampling; fairness incompatibility proof |
| `part5.ipynb` | Guardrail Pipeline | Three-layer production pipeline demonstration on 1,000 examples |
| `pipeline.py` | Pipeline Module | `ModerationPipeline` class with regex filter, calibrated model, human review routing |
| `requirements.txt` | Dependencies | All libraries with pinned versions |

---

## Environment

| Parameter | Value |
|---|---|
| **Python version** | 3.10.x (Kaggle default) |
| **GPU** | NVIDIA T4 16 GB (Kaggle free tier) |
| **CUDA** | 11.8 |
| **Training time (Part 1)** | ≈ 25–35 min per training run on T4 |
| **Platform** | [Kaggle Notebooks](https://www.kaggle.com/) — GPU runtime |

---

## Dataset Setup

The dataset is **not included** in this repository. You must download it separately from Kaggle.

### Steps

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one.
2. Navigate to the competition page:
   ```
   https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
   ```
3. Accept the competition rules on the **Rules** tab.
4. Go to the **Data** tab and download **only these two files**:
   - `jigsaw-multilingual-toxic-comment-classification.csv` (the main training file with identity columns)
   - `validation.csv` (optional sanity check)
5. Do **not** download `jigsaw-toxic-comment-train.csv`, the `*-seqlen128.csv` files, `test.csv`, `test_labels.csv`, or `sample_submission.csv`.

### On Kaggle (recommended)

Add the competition dataset directly to your notebook:
- In the Kaggle notebook editor → **Add Data** → search **"jigsaw-unintended-bias-in-toxicity-classification"**
- The data will be available at:
  ```
  /kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/
  ```

---

## How to Reproduce (Kaggle)

### 1. Set up the environment

All notebooks install their dependencies in Cell 1 using `!pip install -q`. No manual setup is required beyond enabling the GPU runtime.

To enable GPU in Kaggle:
- **Settings** (right panel) → **Accelerator** → select **GPU T4 x1**

### 2. Run notebooks in order

Each notebook depends on artefacts saved by the previous one. **Run them in sequence:**

```
part1.ipynb  →  part2.ipynb  →  part3.ipynb  →  part4.ipynb  →  part5.ipynb
```

All intermediate artefacts are saved to `/kaggle/working/`:

| Artefact | Written by | Read by |
|---|---|---|
| `distilbert-jigsaw-final/` | Part 1 | Parts 3, 4, 5 |
| `eval_with_preds.csv` | Part 1 | Parts 2, 3, 4, 5 |
| `train_split.csv` | Part 1 | Parts 3, 4 |
| `eval_split.csv` | Part 1 | Parts 3, 4 |
| `attack1_evasion_sample.csv` | Part 3 | Part 5 |
| `distilbert-reweighed-final/` | Part 4 | Part 5 |
| `distilbert-oversampled-final/` | Part 4 | — |
| `mitigation_comparison.csv` | Part 4 | — |
| `pipeline.py` | (this repo) | Part 5 |

### 3. Upload `pipeline.py` for Part 5

`pipeline.py` must be accessible in `/kaggle/working/` when `part5.ipynb` runs.

**Option A (recommended):** Upload `pipeline.py` as a Kaggle Dataset:
- Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** → upload `pipeline.py`
- Attach it to your notebook via **Add Data**
- It will appear at `/kaggle/input/<your-dataset-name>/pipeline.py`
- Cell 1 of `part5.ipynb` will copy it to `/kaggle/working/pipeline.py` automatically.

**Option B:** Paste the contents of `pipeline.py` into a new cell at the top of `part5.ipynb` and run it with `%%writefile /kaggle/working/pipeline.py`.

### 4. Verify outputs

After running all five notebooks, `/kaggle/working/` should contain:

```
distilbert-jigsaw-final/          ← Part 1 clean model
distilbert-jigsaw-poisoned-final/ ← Part 3 poisoned model
distilbert-reweighed-final/       ← Part 4 reweighed model
distilbert-oversampled-final/     ← Part 4 oversampled model
eval_with_preds.csv
train_split.csv
eval_split.csv
attack1_evasion_sample.csv
mitigation_comparison.csv
confusion_matrix.png
roc_curve.png
pr_curve.png
threshold_sweep.png
bias_grouped_bar.png
bias_confusion_matrices.png
bias_fpr_all_cohorts.png
bias_score_distributions.png
attack1_evasion_results.png
attack2_poisoning_results.png
mitigation_*.png
fairness_incompatibility.png
p5_*.png
```

---

## How to Reproduce (Local)

### Prerequisites

- Python 3.10.x
- CUDA-capable GPU with ≥ 8 GB VRAM (16 GB recommended for batch_size=32)
- CUDA 11.8 + cuDNN

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download dataset
# Place train.csv in: ./data/train.csv
# Update DATA_DIR in part1.ipynb Cell 3 to './data/'

# 5. Run notebooks
jupyter notebook
# Open and run in order: part1.ipynb → part2.ipynb → ... → part5.ipynb

# 6. Test pipeline standalone
python pipeline.py
```

---

## Pipeline Architecture

```
Input text
    │
    ▼
Layer 1 ── Regex pre-filter ────────────────────────────► BLOCK  (deterministic)
    │  5 categories, 28 patterns, re.IGNORECASE
    │  (no match)
    ▼
    Unicode sanitisation
    │  NFKD normalisation + zero-width strip + char dedup
    │
    ▼
Layer 2 ── Calibrated DistilBERT ──────────────────────► BLOCK  (confidence ≥ 0.60)
    │  Best mitigated model (Reweighing)                ► ALLOW  (confidence ≤ 0.40)
    │  Isotonic regression calibration
    │  (0.40 < confidence < 0.60)
    ▼
Layer 3 ── Human review queue ──────────────────────────► REVIEW
```

### Blocklist categories

| Category | Patterns | Examples |
|---|---|---|
| `direct_threat` | 7 | "I will kill you", "watch your back" |
| `self_harm_directed` | 5 | "go kill yourself", "nobody would miss you" |
| `doxxing_stalking` | 6 | "I know where you live", "I'll dox you" |
| `dehumanization` | 5 | "[group] are animals", "exterminate all" |
| `coordinated_harassment` | 5 | "everyone report", "raid their channel" |

---

## Key Findings

### Bias Audit (Part 2)
- The baseline classifier flags comments in the high-black cohort at **~2× the false positive rate** of the reference (white) cohort, consistent with the 2019 Stanford NLP finding on this dataset.
- Primary disparity metric: **FPR (Disparate Impact ratio > 1.0)** — the model over-flags non-toxic content in Black-associated discourse.

### Adversarial Attacks (Part 3)
- Character-level evasion achieves a measurable Attack Success Rate by combining Unicode homoglyphs, zero-width space insertion, and character duplication.
- Label-flipping poisoning at 5% flip rate produces a measurable increase in False Negative Rate (more toxic content missed).

### Mitigation (Part 4)
- **Best technique: Reweighing (AIF360)** — reduces black-cohort FPR while preserving near-baseline F1.
- Demographic parity and equalized odds are **mathematically incompatible** when base rates differ: `PPR_B − PPR_W = (TPR − FPR) × (p_B − p_W) ≠ 0`. (Chouldechova, 2017)

### Guardrail Pipeline (Part 5)
- **Chosen uncertainty band: (0.40, 0.60)** — routes genuinely ambiguous content to human review, reduces auto-action bias, and keeps review queue size operationally sustainable.
- Review queue toxic rate ≈ 50% confirms the band correctly captures model uncertainty, not systematic error.

---

## References

- Chouldechova, A. (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.* Big Data, 5(2), 153–163.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). *Inherent trade-offs in the fair determination of risk scores.* ITCS.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). *The risk of racial bias in hate speech detection.* ACL.
- Jigsaw / Conversation AI. (2019). *Jigsaw Unintended Bias in Toxicity Classification.* Kaggle.

---

## Reproducibility Note

All random seeds are set to `SEED = 42` at the top of every notebook. Kaggle's T4 GPU environment introduces minor non-determinism in CUDA operations even with fixed seeds; reported metrics may vary by ±0.002 across runs. This does not affect any qualitative findings.
