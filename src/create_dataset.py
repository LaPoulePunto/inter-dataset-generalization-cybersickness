
import os
import pandas as pd
import numpy as np

# Repertoires
input_dir = "processed_data/normalized_sessions"
output_file = "processed_data/aggregated_dataset.csv"

# Liste pour stocker les donnees
data = []

files = os.listdir(input_dir)
print("Nombre de fichiers trouves:", len(files))

for filename in files:
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            
            # Verifier si les colonnes existent
            if 'EDA_z' in df.columns and 'HR_z' in df.columns:
                
                # Info session
                if 'session_id' in df.columns:
                    session_id = df['session_id'].iloc[0]
                else: 
                    session_id = filename.replace("session_", "").replace(".csv", "")
                
                if 'dataset_id' in df.columns:
                    dataset_id = df['dataset_id'].iloc[0]
                else:
                    dataset_id = "Inconnu"
                    
                if 'label_ssq' in df.columns:
                    label = df['label_ssq'].iloc[0]
                else:
                    label = np.nan
                
                # Calcul des features
                # EDA
                mean_eda = df['EDA_z'].mean()
                std_eda = df['EDA_z'].std()
                max_eda = df['EDA_z'].max()
                
                # HR
                mean_hr = df['HR_z'].mean()
                std_hr = df['HR_z'].std()
                max_hr = df['HR_z'].max()
                
                # Stocker dans un dictionnaire
                row = {
                    'session_id': session_id,
                    'dataset_id': dataset_id,
                    'label_ssq': label,
                    'mean_EDA_z': mean_eda,
                    'std_EDA_z': std_eda,
                    'max_EDA_z': max_eda,
                    'mean_HR_z': mean_hr,
                    'std_HR_z': std_hr,
                    'max_HR_z': max_hr
                }
                
                data.append(row)
                
            else:
                print("Colonnes manquantes dans", filename)
                
        except:
            print("Erreur avec", filename)

# Creer le DataFrame final
df_final = pd.DataFrame(data)
print("Taille finale:", df_final.shape)
