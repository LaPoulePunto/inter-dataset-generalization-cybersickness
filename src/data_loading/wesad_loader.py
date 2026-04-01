import os
import sys
import numpy as np
import pandas as pd

# Import createDataset classes
sys.path.append(os.getcwd())
try:
    from createDataset import oneTest, Questionnaire, steamVR, E4
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from createDataset import oneTest, Questionnaire, steamVR, E4

from src.processing import process_and_save_session

def read_e4_csv(csv_path, column_name):
    """Reads an Empatica E4 CSV file and returns a DataFrame with Time and values."""
    if not os.path.exists(csv_path):
        return pd.DataFrame({'Time': [], column_name: []})
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    if len(lines) < 3:
        return pd.DataFrame({'Time': [], column_name: []})
    
    try:
        start_time = float(lines[0].strip().split(',')[0])
        sampling_freq = float(lines[1].strip().split(',')[0])
        
        # Some E4 files might have multiple columns, but for EDA/HR it's usually one
        # Use pandas to read the rest efficiently
        data = pd.read_csv(csv_path, skiprows=2, header=None)
        
        # Take the first column as the value (EDA, HR etc)
        values = data.iloc[:, 0].values
        
        num_samples = len(values)
        times = start_time + np.arange(num_samples) / sampling_freq
        
        return pd.DataFrame({'Time': times, column_name: values})
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame({'Time': [], column_name: []})

def load_wesad_data(base_path, output_dir):
    print(f"Loading WESAD data from {base_path}")
    count = 0
    
    if not os.path.isdir(base_path):
        print(f"Directory {base_path} not found.")
        return count
        
    # Sort for consistent order
    for subject_dir in sorted(os.listdir(base_path)):
        subject_path = os.path.join(base_path, subject_dir)
        if not os.path.isdir(subject_path) or not subject_dir.startswith('S'):
            continue
            
        e4_dir = os.path.join(subject_path, f"{subject_dir}_E4_Data")
        if not os.path.isdir(e4_dir):
            continue
            
        eda_path = os.path.join(e4_dir, 'EDA.csv')
        hr_path = os.path.join(e4_dir, 'HR.csv')
        
        eda_df = read_e4_csv(eda_path, 'EDA')
        hr_df = read_e4_csv(hr_path, 'HR')
        
        if eda_df.empty or hr_df.empty:
            print(f"Warning: WESAD subject {subject_dir} missing EDA or HR data.")
            continue
            
        try:
            # We set the SSQ label to np.nan for WESAD so we can do pseudo-labeling later.
            process_and_save_session(
                session_id=subject_dir,
                dataset_id='WESAD',
                signals={'EDA': eda_df, 'HR': hr_df},
                label_ssq=np.nan,
                output_dir=output_dir
            )
            count += 1
        except Exception as e:
            print(f"Error processing WESAD subject {subject_dir}: {e}")
            continue
            
    print(f"Processed {count} sessions from WESAD dataset.")
    return count
