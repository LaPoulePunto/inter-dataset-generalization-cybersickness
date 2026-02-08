# inter-dataset-generalization-cybersickness

This project investigates the generalization of cybersickness detection models across different datasets (2 for now)

## Datasets

The datasets used in this project are too large to be hosted on GitHub (even with Github LFS because I got a free tier account) and must be downloaded separately:

1.  **Wang et al. (2023)** (Cybersickness_Dataset)
    *   Download: [https://github.com/coreturn/Cybersickness_Dataset](https://github.com/coreturn/Cybersickness_Dataset)
    *   Place the data in `raw_data/wang/`

2.  **WheelSimPhysio-2023**
    *   Download: [https://data.mendeley.com/datasets/z6dfjh596r/2](https://data.mendeley.com/datasets/z6dfjh596r/2)
    *   Place the data in `raw_data/wheelSimPhysio2023/`

## Installation and Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### 1. Install Dependencies

To sync the environment and install dependencies:

```bash
uv sync
```

### 2. Run the Processing Pipeline

To run the data extraction and processing logic (which uses `src/processing.py`):

```bash
uv run run_extraction.py
```