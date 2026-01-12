# Toxic Comments Classification (TF-IDF vs DistilBERT)

This repository contains our implementation and experiments for toxic comment detection using the **Jigsaw Toxic Comment Classification Challenge** dataset (Wikipedia talk-page comments). We compare classical baselines (TF-IDF + Logistic Regression) with transformer fine-tuning (DistilBERT) on:

- **Binary classification:** toxic vs non-toxic  
- **Multi-label classification:** 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)

**Authors:** Incze Melinda Henrietta, Trandafir Andreea Raluca

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

---

## Environment Setup (Anaconda)

Create and activate a conda environment:
```bash
conda create -n toxic_env python=3.10 -y
conda activate toxic_env

pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib tqdm
pip install torch transformers jupyter

---

## Best Results (summary)

Notes:
- Binary best is selected by **F1**.
- Multi-label best is selected by **F1 micro**.
- Subset Accuracy = exact match across all labels.

### Binary
| Model | Accuracy | Log Loss | ROC-AUC | F1 |
|------:|---------:|---------:|--------:|---:|
| TF-IDF + Logistic Regression | 0.9562 | 0.1199 | 0.9788 | 0.8858 |
| DistilBERT | 0.9553 | 0.2770 | 0.9807 | 0.8840 |

### Multi-label
| Model | Subset Accuracy | Log Loss (macro) | ROC-AUC (micro) | ROC-AUC (macro) | F1 (micro) | F1 (macro) |
|------:|----------------:|-----------------:|----------------:|----------------:|-----------:|-----------:|
| TF-IDF + OvR Logistic Regression | 0.9094 | 0.0594 | 0.9867 | 0.9826 | 0.7466 | 0.6374 |
| DistilBERT | 0.8210 | 0.4980 | 0.9812 | 0.9799 | 0.5132 | 0.4144 |

---

## Contents

- `ToxicCommentsClassification.ipynb` – main Jupyter notebook (preprocessing, training, evaluation)
- `.gitignore` – excludes local dataset and model artifacts from being committed
- `LICENSE` – project license (MIT)

Local-only (not included in the repository):
- `Dataset/` – dataset folder (`train.csv`, `test.csv`)
- `models/` – saved model artifacts (TF-IDF/LR and DistilBERT checkpoints)
