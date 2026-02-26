import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score

def main():
    data_path = '/Users/oscar/inter-dataset-generalisation-cybersickness/processed_data/aggregated_dataset.csv'
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    threshold = 0.0848
    df['bin_label'] = (df['label_ssq'] >= threshold).astype(int)
    y = df['bin_label']
    
    feature_cols = [col for col in df.columns if col not in ['session_id', 'dataset_id', 'label_ssq', 'bin_label']]
    X = df[feature_cols].fillna(0)
    
    datasets = df['dataset_id'].unique()
    
    if len(sys.argv) > 1:
        target_dataset = sys.argv[1]
        if target_dataset in datasets:
            datasets = [target_dataset]
        else:
            print(f"Dataset '{target_dataset}' not found. Available: {list(datasets)}")
            return
    
    all_y_true = []
    all_y_pred = []

    for test_dataset in datasets:
        print(f"\n--- Fold: {test_dataset} ---")

        train_mask = df['dataset_id'] != test_dataset
        test_mask = df['dataset_id'] == test_dataset

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        if len(y_test.unique()) < 2:
             print(f"[Warning] Test set contains only class(es): {y_test.unique()}")

        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        try:
             b_acc = balanced_accuracy_score(y_test, y_pred)
             print(f"Balanced Accuracy: {b_acc:.4f}")
        except ValueError:
             print("Balanced Accuracy: N/A")
             
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    print("\n--- GLOBAL PERFORMANCE ---")
    print("Confusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(f"Global Balanced Accuracy: {balanced_accuracy_score(all_y_true, all_y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

if __name__ == "__main__":
    main()
