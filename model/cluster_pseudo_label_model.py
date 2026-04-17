import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_clustering(X_scaled, cluster_labels, ssq_values, filename):
    """
    Fonction pour afficher les clusters et les scores SSQ.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Réduction de dimension simple
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création de la figure avec 2 graphiques
    plt.figure(figsize=(12, 5))
    
    # 1. Graphique PCA
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("Nuage de points des clusters (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    
    # Zoom pour mieux voir (on ignore les points trop extrêmes)
    plt.xlim(np.percentile(X_pca[:, 0], 1), np.percentile(X_pca[:, 0], 99))
    plt.ylim(np.percentile(X_pca[:, 1], 1), np.percentile(X_pca[:, 1], 99))
    
    # 2. Graphique SSQ par cluster
    plt.subplot(1, 2, 2)
    df_temp = pd.DataFrame({'Cluster': cluster_labels, 'SSQ': ssq_values})
    sns.boxplot(x='Cluster', y='SSQ', data=df_temp)
    plt.title("Scores SSQ par Cluster")
    
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()
    print(f"Graphique sauvegardé : plots/{filename}.png")


def evaluate_models(df, X, y, target_dataset=None):
    """
    Entraîne et évalue les modèles Logistic Regression, RandomForest, XGBoost.
    Mise à jour Semi-Supervisée : Intègre le Pseudo-Étiquetage par CLUSTERING pour le dataset WESAD.
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
    
    clustering_visualized = False # Pour ne générer qu'un seul graphique global

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

            # --- AJOUT PSEUDO-LABELING PAR CLUSTERING ---
            if len(X_wesad) > 0:
                print("Génération des pseudo-labels WESAD via Clustering et entrainement...")
                
                # Le clustering se base sur les distances, on a besoin de standardiser
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_wesad_scaled = scaler.transform(X_wesad)
                
                # On regroupe toutes les données (train + wesad) pour former les clusters
                X_concat = np.vstack((X_train_scaled, X_wesad_scaled))
                
                n_clusters = 5  # Nombre de clusters arbitraire pour séparer l'espace
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(X_concat)
                
                # Récupérer les ids de clusters pour le train et WESAD
                train_clusters = cluster_labels[:len(X_train)]
                wesad_clusters = cluster_labels[len(X_train):]
                
                # Calcul de la majorité pour chaque cluster à partir des labels connus (Train)
                cluster_to_label = {}
                for c in range(n_clusters):
                    labels_in_cluster = y_train.iloc[np.where(train_clusters == c)[0]]
                    if len(labels_in_cluster) > 0:
                        # On prends le label le plus fréquent (mode)
                        cluster_to_label[c] = int(labels_in_cluster.mode()[0])
                    else:
                        # Si aucun point train labelisé dans ce cluster, label par défaut = 0
                        cluster_to_label[c] = 0
                
                # Assignation des pseudo-labels pour WESAD
                y_pseudo = [cluster_to_label[c] for c in wesad_clusters]
                
                # --- VISUALISATION ---
                if not clustering_visualized:
                    # On récupère les scores SSQ pour la visualisation
                    ssq_concat = pd.concat([df.loc[train_mask, 'label_ssq'], df.loc[wesad_mask, 'label_ssq']], ignore_index=True)
                    
                    visualize_clustering(
                        X_concat, 
                        cluster_labels, 
                        ssq_concat, 
                        "clustering_global"
                    )
                    clustering_visualized = True

                
                # Concaténation des features et labels pour l'entrainement
                X_train = pd.concat([X_train, X_wesad], ignore_index=True)
                y_train = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)
                
                # Recalcul des sample_weights pour le dataset fusionné
                all_train_dids = pd.concat([df.loc[train_mask, 'dataset_id'], pd.Series(['WESAD']*len(y_pseudo))], ignore_index=True)
                dataset_counts = all_train_dids.value_counts()
                dataset_weights = len(all_train_dids) / (len(dataset_counts) * dataset_counts)
                sample_weights = all_train_dids.map(dataset_weights).values
            # ---------------------------------------------
            
            # Entrainement du modèle supervisé (RF, XGB, LogReg) sur le gros dataset enrichit
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
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

print("Evaluating CLUSTER PSEUDO-LABEL MODEL")
evaluate_models(df, X, y, target_dataset)
