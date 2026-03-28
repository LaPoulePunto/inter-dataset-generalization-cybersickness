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

target_dataset = sys.argv[1] if len(sys.argv) > 1 else None

evaluate_models(df, X, y, target_dataset)
