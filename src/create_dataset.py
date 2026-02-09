
import os
import pandas as pd
import numpy as np

# Repertoires
input_dir = "processed_data/normalized_sessions"
output_file = "processed_data/aggregated_dataset.csv"

# Liste pour stocker les donnees
data = []

files = os.listdir(input_dir)
print("Nombre de fichiers trouves:", len(files))
