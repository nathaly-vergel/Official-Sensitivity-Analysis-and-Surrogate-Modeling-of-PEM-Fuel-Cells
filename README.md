# Sensitivity Analysis and Surrogate Modeling of PEM Fuel Cells

This repository was developed as part of our Statistical Consulting project at Ludwig-Maximilians-University Munich, carried out in collaboration with the Center for Solar Energy and Hydrogen Research Baden-Württemberg (ZSW Ulm) and the University of Applied Sciences Munich. The project focused on sensitivity analysis and surrogate modeling of Proton Exchange Membrane (PEM) fuel cells using the open-source AlphaPEM simulator. Our aim was to identify which operating conditions and uncertain physical parameters most influence fuel cell performance and to build surrogate models that are both efficient and interpretable, supporting further research and model calibration in clean energy applications.

## Repository Structure

The repository is organized into the following main folders:

```bash
├── configs                 # Configuration files (parameter ranges, hyperparameter search spaces). Main entry point for user interaction
├── data
│   ├── designs             # Sampling designs (design matrices)
│   ├── external            # External raw data if any
│   ├── processed           # Processed data (e.g. imputation, classification)
│   └── raw                 # Raw simulation outputs from AlphaPEM, including errors and metadata
├── external
│   └── AlphaPEM            # Fixed AlphaPEM version (fuel cell simulator)
│   └── supplementary_2023_ird   # IRD package for model validity analysis
├── models
│   └── pysr_equations      # Symbolic regression results (PySR equations)
├── notebooks               # IMPORTANT: Step-by-step guides for each stage of the project
├── results
│   ├── model_validity      # Results and figures from model validity analyses
│   └── surrogate_models    # Results and figures from surrogate modeling
├── R_scripts
│   └── scripts             # Supporting R scripts (e.g. IRD methods)
├── scripts                 # Executable scripts for tasks (e.g. running AlphaPEM, PySR)
└── src
    ├── analysis            # Analysis routines (e.g. Sobol, SHAP)
    ├── FE                  # Feature engineering.
    ├── sampling            # Sampling strategies (Sobol, LHS, etc)
    ├── surrogate_models    # Surrogate model implementations
    ├── validity            # Model validity checks (e.g. validity criteria for IRD)
    └── visualization       # Plotting and visualization utilities

```

## Installation

The project uses **Python and R** together inside one Conda environment.
Follow the steps below to set up everything.


### 1. Install AlphaPEM

We use a fixed version of AlphaPEM (v1.0) so everyone has the same results.

```bash
cd external
git clone https://github.com/gassraphael/AlphaPEM.git
cd AlphaPEM
git checkout 2b042c3
```

### 2. Create the Conda environment

First, clone this repository. Then run:

```bash
conda env create -f environment.yml
conda activate env_PEM
```

### 3. Install the IRD package (for Model Validity step)

We use the **Interpretable Regional Descriptors (IRD)** method in R to find valid regions in the model input space.
Clone the IRD package repository:

```bash
git clone https://github.com/slds-lmu/supplementary_2023_ird.git
```

### 4. Set up R inside the Conda environment

`run_IRD.R` is an R script that is called from Python.
You need R installed inside the same environment so `Rscript` works directly. In case you're using a different environment than the one provided, please follow these instructions:

Run:

```bash
conda activate env_PEM
conda config --add channels conda-forge
conda install -y -c conda-forge r-base r-essentials \
  r-optparse r-data.table r-mlr3 r-mlr3learners r-mlr3pipelines \
  r-iml r-ranger r-yaml r-jsonlite r-devtools
```

To test everything works well, run these commands:

```bash
Rscript --version
Rscript -e "cat('ok\n')"
```

If you see the R version and the text `ok`, you should ready to run IRD from Python.

