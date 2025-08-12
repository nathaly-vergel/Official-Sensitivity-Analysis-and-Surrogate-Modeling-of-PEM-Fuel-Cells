# Official Sensitivity Analysis and Surrogate Modeling of PEM Fuel Cells

This repository was designed to contain all code and instructions for our consulting project.
We use the **AlphaPEM** fuel cell model to run simulations, perform sensitivity analysis and build surrogate models.

The project uses **Python and R** together inside one Conda environment.
Follow the steps below to set up everything.


## 1. Install AlphaPEM

We use a fixed version of AlphaPEM (v1.0) so everyone has the same results.

```bash
cd external
git clone https://github.com/gassraphael/AlphaPEM.git
cd AlphaPEM
git checkout 2b042c3
```

## 2. Create the Conda environment

First, clone this repository. Then run:

```bash
conda env create -f environment.yml
conda activate env_PEM
```

## 3. Install the IRD package (for Model Validity step)

We use the **Interpretable Regional Descriptors (IRD)** method in R to find valid regions in the model input space.
Clone the IRD package repository:

```bash
git clone https://github.com/slds-lmu/supplementary_2023_ird.git
```

## 4. Set up R inside the Conda environment

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

