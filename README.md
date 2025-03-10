# AMLS 24/25 Assignment

## Overview

This project is part of the Applied Machine Learning Systems (AMLS) module at UCL, focusing on two primary machine learning tasks involving image classification using datasets from MedMNIST:

1. **Task A**: Binary classification using the BreastMNIST dataset to classify breast tumor images as "Benign" or "Malignant".
2. **Task B**: Multi-class classification using the BloodMNIST dataset to classify blood cell images into eight distinct categories.

## Project Structure

The project folder is organized as follows:

``` bash
AMLS_24_25_SNXXXXXX/
|-- A/
|   |-- (Code related to Task A)
|
|-- B/
|   |-- (Code related to Task B)
|
|-- Datasets/
|   |-- (Empty folder to be populated during assessment)
|
|-- env/
|   |-- environment.yml
|   |-- requirements.txt
|-- main.py
|-- README.md
```

### Files and Folders

- **A/**: Contains all scripts related to Task A, including data preprocessing, model training, and evaluation for the BreastMNIST dataset.
- **B/**: Contains all scripts related to Task B, including data preprocessing, model training, and evaluation for the BloodMNIST dataset.
- **Datasets/**: This folder is left empty for submission and will be populated during assessment. The project will read directly from this folder.
- **env/**: Contains the environment configuration files.
  - **environment.yml**: Conda environment configuration file, which automatically includes dependencies from `requirements.txt`.
  - **requirements.txt**: A list of required packages for the project.
- **[main.py](./main.py)**: The main script to execute the entire project workflow, including training and testing models for both tasks.
- **[README.md](./README.md)**: Provides an overview of the project, file descriptions, and setup instructions (This file).

## Setup Instructions

This project can be set up using two different approaches to manage dependencies and environment:

### Conda Environment with Requirements File

Alternatively, you can use Conda to create a virtual environment and install packages from an `environment.yml` file.

To get started with Conda, run:

```sh
# Create a new conda environment
sudo conda env create -f env/environment.yml

# Activate the environment
conda activate AMLS_Project
```

The `environment.yml` file is configured to include the dependencies listed in `requirements.txt`, and create an virtual conda environment called `AMLS_Env`.

## Required Packages

numpy
matplotlib
medmnist
torch

## Running the Project

To train and evaluate the models for both tasks, run the main script from the terminal:

```sh
python main.py
```

This will execute both the binary classification (Task A) and multi-class classification (Task B) workflows.

## References

- Yang, J., Shi, R., Wei, D. et al. *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Sci Data 10, 41 (2023).
