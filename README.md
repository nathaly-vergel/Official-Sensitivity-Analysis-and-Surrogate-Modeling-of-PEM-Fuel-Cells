# Official-Sensitivity-Analysis-and-Surrogate-Modeling-of-PEM-Fuel-Cells
Repository for the project partners

### Cloning the irdpackage repository

```bash
git clone https://github.com/slds-lmu/supplementary_2023_ird.git
```

### R Setup (Required for `run_IRD.R` in Model Validity)

`run_IRD.R` is an R script. You must have **R** and its required packages installed in your conda environment so `Rscript` works from Python.

**Install into your env:**

```bash
conda activate env_PEM
conda config --add channels conda-forge
conda install -y -c conda-forge r-base r-essentials \
  r-optparse r-data.table r-mlr3 r-mlr3learners r-mlr3pipelines \
  r-iml r-ranger r-yaml r-jsonlite r-devtools
```

**Test:**

```bash
Rscript --version
Rscript -e "cat('ok\n')"
```

If you see the version and `ok`, you're ready to run IRD from Python.






