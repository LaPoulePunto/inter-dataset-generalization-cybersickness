from src.data_loading.wang_loader import load_wang_data
from src.data_loading.wheelsim_loader import load_wheelsim_data
from src.data_loading.wesad_loader import load_wesad_data

# Configuration
WANG_PATH = "raw_data/wang/raw_data2020.p"
WHEELSIM_PATH = "raw_data/wheelSimPhysio2023"
WESAD_PATH = "raw_data/WESAD"
OUTPUT_DIR = "processed_data/normalized_sessions"
    
load_wang_data(WANG_PATH, OUTPUT_DIR)
load_wheelsim_data(WHEELSIM_PATH, OUTPUT_DIR)
load_wesad_data(WESAD_PATH, OUTPUT_DIR)
