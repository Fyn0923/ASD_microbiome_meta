# Meta-Analysis of Effect Sizes in R

## Overview
This repository contains an R script (`meta_analysis.R`) for performing a meta-analysis on effect sizes (Cohen's d) calculated from a CSV dataset (`data.csv`). The analysis compares ASD and TD groups across different genera, filters for genera with at least 3 occurrences, and conducts a random-effects meta-analysis. Significant results (p ≤ 0.05) are exported to `result.csv`.

# Neural Network for Binary Classification

## Overview
This repository contains a PyTorch-based neural network for binary classification (ASD vs. TD) using data from a CSV file. The model includes training with early stopping, evaluation with multiple metrics (AUC, accuracy, precision, recall, F1, MCC), and interpretability analysis using SHAP. It generates ROC curves, and confusion matrices.

## Requirements
- Python 3.8+
- Required packages:
  pip install torch pandas numpy scikit-learn matplotlib seaborn shap tqdm optuna
