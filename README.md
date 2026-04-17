# Inter-Dataset Generalization for Cybersickness Detection

This project investigates the generalization of cybersickness detection models across multiple datasets. The goal is to develop robust models that can accurately predict cybersickness levels using physiological signals, even when tested on datasets they weren't explicitly trained on.

## Datasets

The project supports several datasets. Due to their size, they are not hosted on GitHub and must be downloaded and placed in the `raw_data/` directory.

| Dataset |
| :--- |
| **Wang et al. (2023)** |
| **WheelSimPhysio-2023** |
| **VR Cybersickness** |
| **Archive** |
| **WESAD** |

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

```bash
# Sync environment and install dependencies
uv sync
```

## Usage Pipeline

### 1. Data Extraction & Feature Aggregation

To process the raw data, normalize physiological signals (EDA, HR), and aggregate them into a single CSV for modeling:

```bash
uv run src/create_dataset.py
```

This script will:
- Load data from all available datasets in `raw_data/`.
- Perform Z-score normalization.
- Calculate features (mean, std, max, slope) for each session.
- Generate `processed_data/aggregated_dataset.csv`.

### 2. Model Training & Evaluation

The models are located in the `model/` directory. You can run them using Leave-One-Dataset-Out (LODO) cross-validation.

#### Baseline Models (Supervised)
Runs Logistic Regression, Random Forest, and XGBoost on labeled data.
```bash
uv run model/baseline_models.py
```

#### Semi-Supervised Models (Using WESAD)
These models leverage the unlabeled WESAD dataset to improve generalization.
- **Pseudo-Labeling**: `uv run model/pseudo_label_model.py`
- **Clustering-based Pseudo-Labeling**: `uv run model/cluster_pseudo_label_model.py`
- **CORAL (Domain Adaptation)**: `uv run model/coral_model.py`

## Project Structure

- `src/`: Core logic for data loading and processing.
  - `data_loading/`: Loaders for different dataset formats.
  - `processing.py`: Normalization and signal processing utilities.
- `model/`: Implementation of various classification and domain adaptation models.
- `raw_data/`: Input datasets (not versioned).
- `processed_data/`: Intermediate and final CSV files (not versioned).
- `docs/`: Documentation and project related files.
- `data_exploration.ipynb`: initial data analysis.
