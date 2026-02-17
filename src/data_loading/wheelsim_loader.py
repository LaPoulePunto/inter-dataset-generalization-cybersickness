import pandas as pd
import numpy as np
import os
import glob
from src.processing import process_and_save_session

def load_wheelsim_data(root_path, output_dir):
    print(f"Loading WheelSim data from {root_path}...")
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Directory not found: {root_path}")
        
    count = 0
    
    # Restrict to VR experiments
    search_pattern = os.path.join(root_path, "**", "experiment-2-vr", "**", "physiological-data", "e4", "EDA.csv")
    eda_files = glob.glob(search_pattern, recursive=True)
    
    for eda_path in eda_files:
        try:
            # Reconstruct paths
            physio_dir = os.path.dirname(eda_path)
            session_dir = os.path.dirname(os.path.dirname(physio_dir))
            session_id = os.path.basename(session_dir)
            
            hr_path = os.path.join(physio_dir, "HR.csv")
            questionnaire_dir = os.path.join(session_dir, "questionnaire-data")
            
            # Find SSQ file
            ssq_files = glob.glob(os.path.join(questionnaire_dir, "SSQ-*.csv"))
            if not ssq_files:
                continue
            
            # Parse SSQ score
            ssq_df = pd.read_csv(ssq_files[0])
            ssq_score = None
            if 'Type' in ssq_df.columns and 'Var2' in ssq_df.columns:
                 row = ssq_df[ssq_df['Type'] == 'ssq_pos']
                 if not row.empty:
                     ssq_score = float(row.iloc[0]['Var2'])
                     # Normalize SSQ (0-235.62 -> 0-1)
                     ssq_score = ssq_score / 235.62
            
            if ssq_score is None:
                continue
                
            # Load EDA
            with open(eda_path, 'r') as f:
                start_time_eda = float(f.readline().strip())
                fs_eda = float(f.readline().strip())
            
            eda_values = pd.read_csv(eda_path, skiprows=2, header=None).values.flatten()
            time_eda = start_time_eda + np.arange(len(eda_values)) / fs_eda
            eda_df = pd.DataFrame({'Time': time_eda, 'EDA': eda_values})
            
            # Load HR
            if not os.path.exists(hr_path):
                continue
                
            with open(hr_path, 'r') as f:
                start_time_hr = float(f.readline().strip())
                fs_hr = float(f.readline().strip())
            
            hr_values = pd.read_csv(hr_path, skiprows=2, header=None).values.flatten()
            time_hr = start_time_hr + np.arange(len(hr_values)) / fs_hr
            hr_df = pd.DataFrame({'Time': time_hr, 'HR': hr_values})
            
            # Process and Save directly
            process_and_save_session(
                session_id=session_id,
                dataset_id='WheelSim2023',
                signals={'EDA': eda_df, 'HR': hr_df},
                label_ssq=ssq_score,
                output_dir=output_dir
            )
            count += 1
            
        except Exception as e:
            print(f"Error: {e}")
            continue
            
    print(f"Processed {count} sessions from WheelSim dataset.")
    return count
