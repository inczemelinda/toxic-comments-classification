# Toxic Comments Classification (TF-IDF vs DistilBERT)

A comparative NLP project for toxic comment detection using the **Jigsaw Toxic Comment Classification Challenge** dataset.  
This project compares **TF-IDF + Logistic Regression** baselines with **DistilBERT** fine-tuning for both **binary** and **multi-label** toxicity classification.

**Authors:** Incze Melinda Henrietta, Trandafir Raluca Andreea

---

## Overview

Online platforms rely on automated moderation tools that require accurate and scalable toxicity detection.  
In this project, we compare two major approaches:

- **Classical ML baselines** using TF-IDF vectorization and Logistic Regression (binary and One-vs-Rest multi-label)
- **Transformer-based modeling** using DistilBERT to capture contextual meaning in text

We evaluate performance using **Accuracy**, **Log Loss**, **ROC-AUC**, and **F1** metrics.

---

## Repository Contents

```text
ToxicCommentsClassification.ipynb   # Main Jupyter notebook (preprocessing, training, evaluation)
README.md                           # Project description and results
.gitignore                          # Excludes local dataset and large model artifacts
LICENSE                             # MIT license
```

Local-only (not included in this repository):

```text
Dataset/                            # train.csv, test.csv (download separately from Kaggle)
models/                             # saved artifacts (TF-IDF/LR + DistilBERT checkpoints)
```

---

## Dataset

The dataset is hosted on Kaggle and is not included in this repository:  
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Download:
- `train.csv`
- `test.csv`

Place them locally in:

```text
Dataset/
  train.csv
  test.csv
```

---

## Requirements

- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- Matplotlib
- Joblib
- PyTorch
- Transformers (Hugging Face)
- Jupyter

---

## Environment Setup (Anaconda)

Create and activate a conda environment:

```bash
conda create -n toxic_env python=3.10 -y
conda activate toxic_env
```

Install dependencies:

```bash
pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib tqdm
pip install torch transformers jupyter
```

---

## Usage

### 1. Open the Notebook

```bash
jupyter notebook ToxicCommentsClassification.ipynb
```

### 2. Run Workflow

The notebook is structured into the following sections:
1. **Data loading & inspection** (Kaggle CSV files)  
2. **Preprocessing** (clean text for TF-IDF, raw text + tokenization for DistilBERT)  
3. **Model training** (TF-IDF baselines and DistilBERT fine-tuning)  
4. **Evaluation** (binary and multi-label metrics)  
5. **Saving artifacts** (optional, stored locally in `models/`)  

---

## Results

Notes:
- Binary best is selected by **F1**.
- Multi-label best is selected by **F1 micro**.
- Subset Accuracy = exact match across all labels.

### Binary

| Model | Accuracy | Log Loss | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9562 | 0.1199 | 0.9788 | 0.8858 |
| DistilBERT | 0.9553 | 0.2770 | 0.9807 | 0.8840 |

### Multi-label

| Model | Subset Accuracy | Log Loss (macro) | ROC-AUC (micro) | ROC-AUC (macro) | F1 (micro) | F1 (macro) |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + OvR Logistic Regression | 0.9094 | 0.0594 | 0.9867 | 0.9826 | 0.7466 | 0.6374 |
| DistilBERT | 0.8210 | 0.4980 | 0.9812 | 0.9799 | 0.5132 | 0.4144 |

---

## Future Work

- Tune multi-label decision thresholds per label instead of using a single global threshold.
- Train DistilBERT on more data and experiment with longer fine-tuning schedules.
- Explore alternative transformer backbones and calibration methods.
- Run systematic hyperparameter search for both TF-IDF baselines and transformer training.

---

## References

- Kaggle: Jigsaw Toxic Comment Classification Challenge (dataset source)  
  https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*
