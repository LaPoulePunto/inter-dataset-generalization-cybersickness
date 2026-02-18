
import pandas as pd
import numpy as np
import os
import glob
from src.processing import process_and_save_session

def load_vr_cybersickness_data(root_path, output_dir):
    """Loads and formats the VR Cybersickness Dataset (Movement -> EDA/HR proxies)."""
    print(f"Loading VR Cybersickness data from {root_path}...")
    
    if not os.path.exists(root_path):
         return 0
    
    count = 0
    participant_dirs = glob.glob(os.path.join(root_path, "[0-9]"*4))
    
    for participant_dir in participant_dirs:
        participant_id = os.path.basename(participant_dir)
        
        # Locate required files
        transforms_files = glob.glob(os.path.join(participant_dir, "*_Transforms.xlsx"))
        subjective_files = glob.glob(os.path.join(participant_dir, "*_SubjectiveCs.xlsx"))
        
        if not transforms_files or not subjective_files:
            continue
            
        try:
            # 1. Extract SSQ Label (Max score)
            ssq_df = pd.read_excel(subjective_files[0], engine='openpyxl')
            
            score_cols = [c for c in ssq_df.columns if c in ['Rating', 'Score'] or 'CS' in str(c).upper()]
            if not score_cols:
                continue
                
            ssq_score = float(ssq_df[score_cols[0]].max())
            
            # 2. Extract Movement Data (Transforms)
            df = pd.read_excel(transforms_files[0], engine='openpyxl')
            
            # Standardize Time column
            time_col = next((c for c in df.columns if c in ['Time', 'Timestamp', 'Millis'] or 'time' in str(c).lower()), None)
            if not time_col:
                continue
                
            if time_col == 'Millis':
                df['Time'] = df['Millis'] / 1000.0
            elif time_col != 'Time':
                df.rename(columns={time_col: 'Time'}, inplace=True)
                
            # Compute speeds as proxy for physiological signals
            dt = df['Time'].diff().bfill().replace(0, 0.001)
            
            pos_cols = [c for c in df.columns if 'HeadPosition' in c]
            rot_cols = [c for c in df.columns if 'HeadRotation' in c]
            
            linear_speed = np.sqrt((df[pos_cols].diff().fillna(0) ** 2).sum(axis=1)) / dt if len(pos_cols) == 3 else pd.Series(0, index=df.index)
            angular_speed = np.sqrt((df[rot_cols].diff().fillna(0) ** 2).sum(axis=1)) / dt if len(rot_cols) == 3 else pd.Series(0, index=df.index)

            signals = {
                'EDA': pd.DataFrame({'Time': df['Time'], 'EDA': linear_speed}),
                'HR': pd.DataFrame({'Time': df['Time'], 'HR': angular_speed})
            }
                
            # 3. Process and save
            process_and_save_session(participant_id, 'VRCybersickness', signals, ssq_score, output_dir)
            count += 1
             
        except Exception as e:
            print(f"Error processing {participant_id}: {e}")
            continue
            
    print(f"Processed {count} sessions from VR Cybersickness dataset.")
    return count
