import os
import glob
import pandas as pd
import numpy as np
import pickle
from src.processing import process_and_save_session

def load_archive_data(root_path, output_dir):
    print(f"Loading Archive data from {root_path}...")
    
    excel_path = os.path.join(root_path, "03 Self-Reported Questionnaires", "02 Post Exposure Ratings.xlsx")
    preprocessed_dir = os.path.join(root_path, "05 ECG-GSR Data", "01 ECG-GSR Data (Pre-Processed)")
    
    if not os.path.exists(excel_path) or not os.path.exists(preprocessed_dir):
        print("Archive data not found.")
        return 0
        
    try:
        # Load labels
        df_labels = pd.read_excel(excel_path, engine='openpyxl')
        
        # Use POST_Dizzy as a simple SSQ score proxy
        if 'POST_Dizzy' in df_labels.columns:
            df_labels['SSQ_score'] = df_labels['POST_Dizzy']
        else:
            df_labels['SSQ_score'] = 0.0
            
    except Exception as e:
        print(f"Error reading excel: {e}")
        return 0

    count = 0
    dat_files = glob.glob(os.path.join(preprocessed_dir, "*_ECG_GSR_PreProcessed.dat"))
    
    for dat_file in dat_files:
        filename = os.path.basename(dat_file)
        participant_id_str = filename.split('_')[0]
        
        try:
            participant_id = int(participant_id_str)
        except ValueError:
            continue
            
        # Get label from excel
        participant_labels = df_labels[df_labels['ID'] == participant_id]
        if participant_labels.empty:
            continue
            
        ssq_score = float(participant_labels['SSQ_score'].mean())
        
        try:
            # Load pickled dat file
            with open(dat_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                
            if not isinstance(data, dict) or 'Data' not in data:
                continue
                
            # data['Data'] is a list of trials
            trials = data['Data']
            
            for trial_idx, trial_data in enumerate(trials):
                # trial_data usually has multiple columns, like [ECG, GSR, ...]
                if hasattr(trial_data, 'shape') and len(trial_data.shape) == 2 and trial_data.shape[1] >= 2:
                    
                    # Canal 0 is likely EDA (slow-moving ~7.05), Canal 1 is ECG (fast-moving ~ -0.2)
                    eda_signal = trial_data[:, 0]
                    ecg_signal = trial_data[:, 1]
                    
                    # Create timestamp (Assuming sampling frequency = 1000Hz)
                    time_col = np.arange(len(eda_signal)) / 1000.0
                    
                    signals = {
                        'EDA': pd.DataFrame({'Time': time_col, 'EDA': eda_signal}),
                        'HR': pd.DataFrame({'Time': time_col, 'HR': ecg_signal}) 
                    }
                    
                    # Save session
                    session_id = f"{participant_id}_{trial_idx}"
                    process_and_save_session(session_id, 'Archive', signals, ssq_score, output_dir)
                    count += 1
                
        except Exception as e:
            continue
            
    print(f"Processed {count} sessions from Archive dataset.")
    return count
