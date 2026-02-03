import numpy as np
import pandas as pd
from scipy import interpolate
import os

def align_signals(eda_df, hr_df, target_fs=4.0):
    # Aligns EDA and HR signals to a common time axis (4hz by default)
    t_start = max(eda_df['Time'].min(), hr_df['Time'].min())
    t_end = min(eda_df['Time'].max(), hr_df['Time'].max())
    common_time = np.arange(t_start, t_end, 1/target_fs)
    
    # Interpolate
    f_eda = interpolate.interp1d(eda_df['Time'], eda_df['EDA'], fill_value="extrapolate")
    f_hr = interpolate.interp1d(hr_df['Time'], hr_df['HR'], fill_value="extrapolate")
    
    return pd.DataFrame({
        'Time': common_time,
        'EDA': f_eda(common_time),
        'HR': f_hr(common_time)
    })

def process_and_save_session(session_id, dataset_id, raw_eda, raw_hr, label_ssq, output_dir):
    """Unifies (4Hz), normalizes (Baseline & Z-Score), and saves session."""
    
    # 1. Unification
    df = align_signals(raw_eda, raw_hr)
    
    # 2. Baseline Subtraction (First 60s)
    # If session is shorter than 60s, take whole session as baseline
    baseline_window = df[df['Time'] <= (df['Time'].iloc[0] + 60)]
    if baseline_window.empty:
        baseline_stats = df[['EDA', 'HR']].mean()
    else:
        baseline_stats = baseline_window[['EDA', 'HR']].mean()
        
    df['EDA_bl'] = df['EDA'] - baseline_stats['EDA']
    df['HR_bl'] = df['HR'] - baseline_stats['HR']
    
    # 3. Z-Score Normalization
    for col in ['EDA', 'HR']:
        std = df[col].std()
        if std == 0: std = 1
        df[f'{col}_z'] = (df[col] - df[col].mean()) / std
    
    # 4. Metadata & Save
    df['session_id'] = session_id
    df['dataset_id'] = dataset_id
    df['label_ssq'] = label_ssq
    
    final_cols = ['session_id', 'Time', 'EDA', 'EDA_bl', 'EDA_z', 'HR', 'HR_bl', 'HR_z', 'dataset_id', 'label_ssq']
    
    os.makedirs(output_dir, exist_ok=True)
    df[final_cols].to_csv(os.path.join(output_dir, f"session_{dataset_id}_{session_id}.csv"), index=False)
