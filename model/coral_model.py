import pandas as pd
import numpy as np
import sys
import scipy.linalg
from baseline_models import evaluate_models

data_path = '/Users/oscar/inter-dataset-generalisation-cybersickness/processed_data/aggregated_dataset.csv'

df = pd.read_csv(data_path)

threshold = 0.0848
df['bin_label'] = (df['label_ssq'] >= threshold).astype(int)
y = df['bin_label']

feature_cols = [col for col in df.columns if col not in ['session_id', 'dataset_id', 'label_ssq', 'bin_label']]
X = df[feature_cols].fillna(0)

# CORAL feature alignment par domaine
print("EVALUATING CORAL MODEL")
epsilon = 1e-5

target_dataset = sys.argv[1] if len(sys.argv) > 1 else None

# Choix de la distribution cible (soit le dataset cible, soit la moyenne globale)
if target_dataset and target_dataset in df['dataset_id'].unique():
    X_target = X[df['dataset_id'] == target_dataset]
else:
    # Distribution globale
    X_target = X

mean_target = X_target.mean(axis=0)
X_target_c = X_target - mean_target
Ct = np.cov(X_target_c, rowvar=False) + epsilon * np.eye(X.shape[1])
Ct_half = np.real(scipy.linalg.fractional_matrix_power(Ct, 0.5))

evaluate_models(df, X, y, target_dataset)
