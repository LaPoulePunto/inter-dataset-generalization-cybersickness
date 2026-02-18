import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import sys

def main():
    # Hardcoded absolute path to avoid directory resolution issues
    data_path = '/Users/oscar/inter-dataset-generalisation-cybersickness/processed_data/aggregated_dataset.csv'
    print(f"Loading aggregated dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    # Define features (X) and label (y)
    # The features are the extracted statistics from the EDA and HR signals
    feature_cols = [col for col in df.columns if col not in ['session_id', 'dataset_id', 'label_ssq']]
    
    X = df[feature_cols]
    
    # Logistic Regression requires categorical/binary labels. 
    # Since our label_ssq is continuous in [0, 1], we need to binarize it.
    # We will use the median as a simple threshold to create "Low Sickness" (0) and "High Sickness" (1) classes.
    threshold = df['label_ssq'].median()
    print(f"Binarizing label_ssq using median threshold: {threshold:.4f}")
    
    y = (df['label_ssq'] > threshold).astype(int)
    
    # Optional: Fill any NaN values with 0
    X = X.fillna(0)
    
    # Get unique datasets
    datasets = df['dataset_id'].unique()
    
    # Check if a specific test dataset was requested via command line
    if len(sys.argv) > 1:
        target_dataset = sys.argv[1]
        if target_dataset in datasets:
            datasets = [target_dataset]
            print(f"\nRunning targeted LODO validation for: {target_dataset}\n")
        else:
            print(f"Error: Dataset '{target_dataset}' not found. Available: {list(datasets)}")
            return
    else:
        print(f"\nFound {len(df['dataset_id'].unique())} datasets for full LODO validation: {list(datasets)}\n")
    
    # Store overall predictions for a global metrics report at the end
    all_y_true = []
    all_y_pred = []

    # Leave-One-Dataset-Out (LODO) Loop
    for test_dataset in datasets:
        print(f"{'='*50}")
        print(f"LODO FOLD: Testing on **{test_dataset}**")
        print(f"{'='*50}")

        # Split data based on dataset_id
        train_mask = df['dataset_id'] != test_dataset
        test_mask = df['dataset_id'] == test_dataset

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
        
        # Check if test set only has 1 class (can happen if a dataset is biased)
        if len(y_test.unique()) < 2:
             print(f"Warning: Test dataset '{test_dataset}' only contains class(es): {y_test.unique()}.")

        # Initialize and train the Logistic Regression model
        # Using class_weight='balanced' to handle potential class imbalance
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Evaluate the fold
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold Accuracy ({test_dataset}): {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")

    print(f"{'='*50}")
    print("GLOBAL LODO PERFORMANCE (Concatenated Folds)")
    print(f"{'='*50}")
    print(f"Global Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
    print("\nGlobal Confusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))
    print("\nGlobal Classification Report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

if __name__ == "__main__":
    main()
