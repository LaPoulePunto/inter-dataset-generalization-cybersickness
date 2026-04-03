import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier


def evaluate_models(df, X, y, target_dataset=None):
    """
    Entraîne et évalue les modèles Logistic Regression, RandomForest, XGBoost en utilisant une validation croisée LODO.
    Mise à jour Semi-Supervisée : Intègre le Pseudo-Étiquetage pour le dataset WESAD.
    """
    datasets = df['dataset_id'].unique()

    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ),
        
        "xgb": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    }
    
    for model_name, base_model in models.items():
        print(f"\n==============================")
        print(f"MODEL: {model_name}")
        print(f"==============================")
        
        all_y_true = []
        all_y_pred = []

        for test_dataset in datasets:
            # WESAD n'a pas de labels réels, il ne peut jamais servir de set de test
            if test_dataset == 'WESAD':
                continue

            print(f"\n--- Fold: {test_dataset} ---")

            # On isole WESAD du train_mask pour ne l'utiliser que lors du pseudo-labeling
            train_mask = (df['dataset_id'] != test_dataset) & (df['dataset_id'] != 'WESAD')
            test_mask = df['dataset_id'] == test_dataset
            wesad_mask = df['dataset_id'] == 'WESAD'

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            X_wesad = X[wesad_mask]
            
            # sample weights originaux (basés uniquement sur les données labellisées)
            train_datasets = df.loc[train_mask, 'dataset_id']
            dataset_counts = train_datasets.value_counts()
            dataset_weights = len(train_datasets) / (len(dataset_counts) * dataset_counts)
            sample_weights = train_datasets.map(dataset_weights).values
            
            print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples, WESAD: {len(X_wesad)} samples")
            
            if len(y_test.unique()) < 2:
                print(f"[Warning] Test set contains only class(es): {y_test.unique()}")

            model = base_model.__class__(**base_model.get_params())
            
            # 1. Entraînement initial sur les données labellisées
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # --- AJOUT PSEUDO-LABELING ---
            if len(X_wesad) > 0:
                print("Génération des pseudo-labels WESAD et ré-entraînement...")
                # Le modèle génère les prédictions (pseudo-labels) sur WESAD
                y_pseudo = model.predict(X_wesad)
                
                # Concaténation des features et labels
                X_train = pd.concat([X_train, X_wesad], ignore_index=True)
                y_train = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)
                
                # On calcule un poids équilibré pour la section WESAD comme un vrai dataset
                # Le but est que la somme des poids de WESAD = la moyenne des poids des autres datasets
                all_train_dids = pd.concat([df.loc[train_mask, 'dataset_id'], pd.Series(['WESAD']*len(y_pseudo))], ignore_index=True)
                dataset_counts = all_train_dids.value_counts()
                dataset_weights = len(all_train_dids) / (len(dataset_counts) * dataset_counts)
                sample_weights = all_train_dids.map(dataset_weights).values
                
                # 2. Ré-entrainement sur le nouveau dataset étendu
                model = base_model.__class__(**base_model.get_params())
                model.fit(X_train, y_train, sample_weight=sample_weights)
            # -----------------------------
            
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


data_path = '/Users/oscar/inter-dataset-generalisation-cybersickness/processed_data/aggregated_dataset.csv'

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Erreur: Impossible de lire le fichier {data_path}.")
    sys.exit(1)

threshold = 0.0848
df['bin_label'] = (df['label_ssq'] >= threshold).astype(int)
y = df['bin_label']

feature_cols = [col for col in df.columns if col not in ['session_id', 'dataset_id', 'label_ssq', 'bin_label']]
X = df[feature_cols].fillna(0)

target_dataset = sys.argv[1] if len(sys.argv) > 1 else None

print("Evaluating PSEUDO-LABEL MODEL")
evaluate_models(df, X, y, target_dataset)
