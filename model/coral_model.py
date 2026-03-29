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

for d_id in df['dataset_id'].unique():
    mask = df['dataset_id'] == d_id
    X_source = X.loc[mask]
    
    mean_source = X_source.mean(axis=0)
    X_source_c = X_source - mean_source
    
    Cs = np.cov(X_source_c, rowvar=False) + epsilon * np.eye(X_source.shape[1])
    Cs_inv_half = np.real(scipy.linalg.fractional_matrix_power(Cs, -0.5))
    
    # Transformation (Blanchiment + Recolorisation avec la Target globale/spécifique)
    X_aligned = np.dot(np.dot(X_source_c, Cs_inv_half), Ct_half) + mean_target.values
    
    # Remplacement des valeurs dans X
    X.loc[mask] = X_aligned

evaluate_models(df, X, y, target_dataset)
