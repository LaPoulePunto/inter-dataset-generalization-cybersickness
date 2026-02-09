
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
                print("Fichier OK:", filename)
            else:
                print("Colonnes manquantes dans", filename)
                
        except:
            print("Erreur avec", filename)
