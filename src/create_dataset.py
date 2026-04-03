import os
import sys

# Add project root to Python path so 'src' imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from scipy import stats
from src.data_loading.wang_loader import load_wang_data
from src.data_loading.wheelsim_loader import load_wheelsim_data
from src.data_loading.vr_cybersickness_loader import load_vr_cybersickness_data
from src.data_loading.archive_loader import load_archive_data
from src.data_loading.wesad_loader import load_wesad_data

def main():
    raw_data_dir = os.path.join(project_root, 'raw_data')
    output_dir = os.path.join(project_root, 'processed_data', 'normalized_sessions')
    output_file = os.path.join(project_root, 'processed_data', 'aggregated_dataset.csv')
    
    # 1. Load Datasets
    wang_path = os.path.join(raw_data_dir, 'wang', 'raw_data2020.p')
    if os.path.exists(wang_path): load_wang_data(wang_path, output_dir)
        
    wheelsim_path = os.path.join(raw_data_dir, 'wheelSimPhysio2023')
    if os.path.exists(wheelsim_path): load_wheelsim_data(wheelsim_path, output_dir)

    vr_path = os.path.join(raw_data_dir, 'VR_Cybersickness_Dataset') 
    if os.path.exists(vr_path): load_vr_cybersickness_data(vr_path, output_dir)
        
    archive_path = os.path.join(raw_data_dir, 'archive')
    if os.path.exists(archive_path): load_archive_data(archive_path, output_dir)
        
    wesad_path = os.path.join(raw_data_dir, 'WESAD')
    if os.path.exists(wesad_path): load_wesad_data(wesad_path, output_dir)

    # 2. Aggregate Features
    if not os.path.exists(output_dir):
        print(f"Input directory {output_dir} does not exist.")
        return

    data = []
    files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    print(f"Nombre de fichiers trouves: {len(files)}")

    for filename in files:
        filepath = os.path.join(output_dir, filename)
        try:
            df = pd.read_csv(filepath)
            
            # Ensure required physiological/proxy columns exist
            if 'EDA_z' not in df.columns or 'HR_z' not in df.columns:
                continue
                
            session_id = df.get('session_id', pd.Series([filename.replace("session_", "").replace(".csv", "")])).iloc[0]
            dataset_id = df.get('dataset_id', pd.Series(["Inconnu"])).iloc[0]
            label = df.get('label_ssq', pd.Series([np.nan])).iloc[0]
            
            slope_eda, _, _, _, _ = stats.linregress(df['Time'], df['EDA_z'])
            slope_hr, _, _, _, _ = stats.linregress(df['Time'], df['HR_z'])
            
            data.append({
                'session_id': session_id,
                'dataset_id': dataset_id,
                'label_ssq': label,
                'mean_EDA_z': df['EDA_z'].mean(),
                'std_EDA_z': df['EDA_z'].std(),
                'max_EDA_z': df['EDA_z'].max(),
                'slope_EDA_z': 0 if np.isnan(slope_eda) else slope_eda,
                'mean_HR_z': df['HR_z'].mean(),
                'std_HR_z': df['HR_z'].std(),
                'max_HR_z': df['HR_z'].max(),
                'slope_HR_z': 0 if np.isnan(slope_hr) else slope_hr
            })
                
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

    # 3. Save Final Dataset
    if data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame(data).to_csv(output_file, index=False)
        print(f"Fini ! Sauvegarde dans {output_file}")
    else:
        print("Aucune donnee agregee (compatible EDA/HR).")

if __name__ == "__main__":
    main()
