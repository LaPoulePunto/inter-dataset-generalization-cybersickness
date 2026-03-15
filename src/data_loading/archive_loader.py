import os
import glob
import pandas as pd
import numpy as np
import pickle
import warnings
from src.processing import process_and_save_session

def load_archive_data(root_path, output_dir):
    print(f"Chargement des données Archive depuis {root_path}...")
    
    excel_path = os.path.join(root_path, "03 Self-Reported Questionnaires", "02 Post Exposure Ratings.xlsx")
    dat_dir = os.path.join(root_path, "05 ECG-GSR Data", "01 ECG-GSR Data (Pre-Processed)")
    
    if not os.path.exists(excel_path) or not os.path.exists(dat_dir):
        print("Dossier Archive introuvable.")
        return 0

    # 1. Charger les labels (Dizzy) et les normaliser entre 0 et 1
    df_labels = pd.read_excel(excel_path, engine='openpyxl')
    
    if 'POST_Dizzy' in df_labels.columns:
        max_score = df_labels['POST_Dizzy'].max()
        if max_score == 0: max_score = 1
        df_labels['SSQ_score'] = df_labels['POST_Dizzy'] / max_score
    else:
        df_labels['SSQ_score'] = 0.0

    # 2. Parcourir tous les fichiers .dat
    dat_files = glob.glob(os.path.join(dat_dir, "*_ECG_GSR_PreProcessed.dat"))
    count = 0
    
    for file_path in dat_files:
        # Extraire l'ID du participant (ex: "101" depuis "101_ECG_GSR_PreProcessed.dat")
        filename = os.path.basename(file_path)
        participant_id_str = filename.split('_')[0]
        
        if not participant_id_str.isdigit():
            continue
            
        participant_id = int(participant_id_str)

        # Chercher le score de ce participant
        participant_data = df_labels[df_labels['ID'] == participant_id]
        if participant_data.empty:
            continue
        score = float(participant_data['SSQ_score'].iloc[0])
        
        # 3. Charger le fichier .dat
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignorer les warnings de NumPy
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                
        if not isinstance(data, dict) or 'Data' not in data:
            continue
            
        # 4. Traiter chaque essai du participant
        trials = data['Data']
        for trial_idx, trial in enumerate(trials):
            
            # Vérifier que les données sont valides
            if type(trial) is np.ndarray and len(trial.shape) == 2 and trial.shape[1] >= 2:
                
                # Le premier canal (lent) est l'EDA, le deuxième (rapide) est l'HR
                eda = trial[:, 0]
                hr = trial[:, 1]
                
                # Créer l'axe du temps (1000 Hz)
                time_sec = np.arange(len(eda)) / 1000.0
                
                # Préparer les données pour la sauvegarde
                signals = {
                    'EDA': pd.DataFrame({'Time': time_sec, 'EDA': eda}),
                    'HR': pd.DataFrame({'Time': time_sec, 'HR': hr}) 
                }
                
                # Sauvegarder
                session_id = f"{participant_id}_{trial_idx}"
                process_and_save_session(session_id, 'Archive', signals, score, output_dir)
                count += 1
                
    print(f"{count} sessions Archive traitées.")
    return count
