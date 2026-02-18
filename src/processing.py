import numpy as np
import pandas as pd
from scipy import interpolate
import os

def align_signals(signals, target_fs=4.0):
    """Aligns multiple signals to a common time axis at a target frequency."""
    
    # 1. Determine common time range
    t_mins, t_maxs = [], []
    for name, df in signals.items():
        if 'Time' not in df.columns:
            raise ValueError(f"Signal {name} missing 'Time' column")
        t_mins.append(df['Time'].min())
        t_maxs.append(df['Time'].max())
    
    if not t_mins:
        return pd.DataFrame({'Time': []})
        
    t_start, t_end = max(t_mins), min(t_maxs)
    common_time = np.arange(t_start, t_end, 1/target_fs)
    result = {'Time': common_time}
    
    # 2. Interpolate each signal to the common time axis
    for name, df in signals.items():
        val_col = name if name in df.columns else [c for c in df.columns if c != 'Time'][0]

        df_clean = df.drop_duplicates(subset=['Time']).sort_values(by='Time')
        
        if len(df_clean) < 2:
            result[name] = np.full_like(common_time, df_clean[val_col].iloc[0]) if len(df_clean) == 1 else np.zeros_like(common_time)
            continue

        f = interpolate.interp1d(df_clean['Time'], df_clean[val_col], fill_value="extrapolate", bounds_error=False)
        result[name] = f(common_time)
        
    return pd.DataFrame(result)

def process_and_save_session(session_id, dataset_id, signals, label_ssq, output_dir):
    """Unifies (4Hz), normalizes (Baseline & Z-Score), and saves session data."""
    
    # 1. Unification
    df = align_signals(signals)
    if df.empty:
         print(f"Skipping {session_id}: diverse time ranges or empty signals")
         return

    signal_names = list(signals.keys())

    # 2. Baseline Subtraction (First 60s)
    baseline_window = df[df['Time'] <= (df['Time'].iloc[0] + 60)]
    baseline_stats = df[signal_names].mean() if baseline_window.empty else baseline_window[signal_names].mean()
        
    for name in signal_names:
        df[f'{name}_bl'] = df[name] - baseline_stats[name]
    
    # 3. Z-Score Normalization
    for name in signal_names:
        std = df[name].std()
        std = 1 if std == 0 else std # Prevent division by zero
        df[f'{name}_z'] = (df[name] - df[name].mean()) / std
    
    # 4. Metadata & Save
    df['session_id'] = session_id
    df['dataset_id'] = dataset_id
    df['label_ssq'] = label_ssq
    
    final_cols = ['session_id', 'Time']
    for name in signal_names:
        final_cols.extend([name, f'{name}_bl', f'{name}_z'])
    final_cols.extend(['dataset_id', 'label_ssq'])
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"session_{dataset_id}_{session_id}.csv")
    df[final_cols].to_csv(out_path, index=False)
