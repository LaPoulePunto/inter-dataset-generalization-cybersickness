import pickle
import sys
import os

# Import createDataset classes
sys.path.append(os.getcwd())
try:
    from createDataset import oneTest, Questionnaire, steamVR, E4
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from createDataset import oneTest, Questionnaire, steamVR, E4

from src.processing import process_and_save_session

def load_wang_data(file_path, output_dir):
    print(f"Loading Wang data")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    count = 0
    for key, sample in data.items():
        try:
            # Direct access to data
            eda_df = sample.Empatica.GSR.copy()
            hr_df = sample.Empatica.HR.copy()
            ssq_score = sample.SicknessLevel.SSQ
            
            # Simple renaming: GSR -> EDA
            if 'GSR' in eda_df.columns:
                eda_df.rename(columns={'GSR': 'EDA'}, inplace=True)
                
            # Process and Save directly
            process_and_save_session(
                session_id=key,
                dataset_id='Wang2020',
                raw_eda=eda_df,
                raw_hr=hr_df,
                label_ssq=ssq_score,
                output_dir=output_dir
            )
            count += 1
            
        except Exception:
            # Minimal error handling
            continue
            
    print(f"Processed {count} sessions from Wang dataset.")
    return count
