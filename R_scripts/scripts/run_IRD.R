#!/usr/bin/env Rscript
# Interpretable Regional Descriptors runner
#
# What this script does:
# - Loads a CSV with many parameters and one classification column
# - Trains a random forest to predict validity
# - Runs one or more IRD methods (PRIM, MaxBox, MAIRE)
# - Post-processes the box
# - Saves a YAML of bounds and a text report for each method
# - Saves Random Forest metrics to a TXT file
#
# Inputs (via CLI):
# - --data: Path to CSV with features + target
# - --target: Target column name (default: classification)
# - --positive: Positive class label (default: valid)
# - --xinterest: Path to JSON or YAML mapping feature -> value
# - --range: Desired probability range 'low,high' (default: 0.8,1.0)
# - --outdir: Output directory (default: results/model_validity)
# - --methods: Comma separated: PRIM,MaxBox,Maire (default: PRIM,MaxBox)
# - --run_name: Run name to tag outputs (default: timestamp)
# - --categorical_overrides: Comma separated features to force as categorical (default: e)
# - --seed: Random seed (default: 42)
# - --ird_pkg_dir: Path to local irdpackage dir (default: external/supplementary_2023_ird/irdpackage)

suppressWarnings({
  needed <- c("optparse","data.table","mlr3","mlr3learners","mlr3pipelines","iml",
              "ranger","yaml","jsonlite","tools")
  miss <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
    install.packages(miss)
  }
})

library(optparse)
library(data.table)
library(mlr3)
library(mlr3learners)
library(iml)
library(jsonlite)

# ---- Helpers: load local irdpackage or from installed namespace ----
use_ird_dev <- function(pkg_dir) {
  message("[1] \"Loading the irdpackage\"")
  if (requireNamespace("irdpackage", quietly = TRUE)) return(invisible(TRUE))
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  Sys.unsetenv("GITHUB_PAT")
  devtools::load_all(pkg_dir)
  invisible(TRUE)
}

# ---- Import our export helpers (write_ird_yaml, write_ird_text_report) ----
# Adjust the path if your helpers live somewhere else
source(file.path("R_scripts", "ird_helpers.R"))

# ---- Parse CLI ----
option_list <- list(
  make_option(c("-d","--data"), type="character", help="Path to CSV with features + target.", metavar="FILE"),
  make_option(c("-t","--target"), type="character", default="classification", help="Target column [default: %default]."),
  make_option(c("-p","--positive"), type="character", default="valid", help="Positive class label [default: %default]."),
  make_option(c("-x","--xinterest"), type="character", help="Path to JSON or YAML with x_interest mapping."),
  make_option(c("-r","--range"), type="character", default="0.8,1.0", help="Desired probability range 'low,high' [default: %default]."),
  make_option(c("-o","--outdir"), type="character", default=file.path("results","model_validity"), help="Output directory [default: %default]."),
  make_option(c("-m","--methods"), type="character", default="PRIM,MaxBox", help="Methods to run: PRIM,MaxBox,Maire [default: %default]."),
  make_option(c("-n","--run_name"), type="character", default=NULL, help="Run name to tag outputs. Default is timestamp."),
  make_option(c("-c","--categorical_overrides"), type="character", default="e", help="Comma separated feature names that must be categorical [default: %default]."),
  make_option(c("-s","--seed"), type="integer", default=42, help="Random seed [default: %default]."),
  make_option(c("--ird_pkg_dir"), type="character", default=file.path("external","supplementary_2023_ird","irdpackage"),
              help="Path to local irdpackage directory [default: %default].")
)
opt <- parse_args(OptionParser(option_list=option_list))

# ---- Validate ----
stopifnot(!is.null(opt$data), file.exists(opt$data))
stopifnot(!is.null(opt$xinterest), file.exists(opt$xinterest))

desired_range <- as.numeric(strsplit(opt$range, ",")[[1]])
if (length(desired_range) != 2 || any(is.na(desired_range))) stop("Invalid --range. Use like: 0.8,1.0")

methods <- trimws(strsplit(opt$methods, ",")[[1]])
methods <- methods[methods %in% c("PRIM","MaxBox","Maire")]
if (!length(methods)) stop("No valid methods selected. Choose any of: PRIM, MaxBox, Maire")

categorical_overrides <- character(0)
if (!is.null(opt$categorical_overrides) && nchar(opt$categorical_overrides)) {
  categorical_overrides <- unique(trimws(strsplit(opt$categorical_overrides, ",")[[1]]))
}

if (is.null(opt$run_name) || !nchar(opt$run_name)) {
  opt$run_name <- format(Sys.time(), "%Y%m%d_%H%M%S")
}

dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

# ---- Load IRD package ----
use_ird_dev(opt$ird_pkg_dir)

# ---- Read x_interest mapping ----
read_mapping <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("json")) return(jsonlite::fromJSON(path))
  if (ext %in% c("yml","yaml")) return(yaml::yaml.load_file(path))
  stop("x_interest must be JSON or YAML.")
}
x_interest_list <- read_mapping(opt$xinterest)

# ---- Load data ----
dt <- fread(opt$data, data.table = FALSE)
df <- as.data.frame(dt)

# Ensure target present
if (!opt$target %in% names(df)) stop("Target column not found: ", opt$target)

# Rename target to 'validity'
names(df)[names(df) == opt$target] <- "validity"

# Coerce target to factor with positive class last
if (!is.factor(df$validity)) df$validity <- factor(df$validity)
labs <- levels(df$validity)
if (!(opt$positive %in% labs)) {
  stop("Positive class '", opt$positive, "' not found in target levels: ", paste(labs, collapse=", "))
}
neg <- setdiff(labs, opt$positive)
if (length(neg) != 1) stop("Target must be binary. Found levels: ", paste(labs, collapse=", "))
df$validity <- factor(df$validity, levels = c(neg, opt$positive))

# ---- Build x_interest and select features (keys in x_interest only) ----
all_features <- setdiff(names(df), "validity")

xi_keys <- names(x_interest_list)
if (!length(xi_keys)) stop("x_interest has no keys.")

# Features used = intersection of data columns and x_interest keys
features <- intersect(all_features, xi_keys)
if (!length(features)) stop("None of the x_interest keys match columns in data.")

# Informative messages
dropped <- setdiff(all_features, features)
if (length(dropped)) {
  message("Dropping ", length(dropped), " columns not listed in x_interest: ",
          paste(dropped, collapse = ", "))
}
missing_in_data <- setdiff(xi_keys, all_features)
if (length(missing_in_data)) {
  message("Ignoring x_interest keys not in data: ",
          paste(missing_in_data, collapse = ", "))
}

# Build x_interest (selected features only)
x_interest <- as.data.frame(as.list(x_interest_list[features]), stringsAsFactors = FALSE)
# Add dummy label
x_interest$validity <- factor(opt$positive, levels = levels(df$validity))

# ---- Type harmonization (critical to avoid IMl type mismatch) ----
# Any feature in categorical_overrides => factor in BOTH df and x_interest (same levels).
# All other features => numeric (double) in BOTH df and x_interest.
harmonize_types <- function(df, x_interest, features, categorical_overrides) {
  for (f in features) {
    if (f %in% categorical_overrides) {
      # factor in df
      if (!is.factor(df[[f]])) df[[f]] <- factor(df[[f]])
      # ensure x_interest value exists in levels; if not, extend
      xi_val_chr <- as.character(x_interest[[f]][1])
      levs <- levels(df[[f]])
      if (!xi_val_chr %in% levs) {
        levels(df[[f]]) <- union(levs, xi_val_chr)
      }
      # factor in x_interest with same levels
      x_interest[[f]] <- factor(x_interest[[f]], levels = levels(df[[f]]))
    } else {
      # coerce both to numeric (double)
      # factors/characters in df are converted safely
      suppressWarnings(df[[f]] <- as.numeric(df[[f]]))
      suppressWarnings(x_interest[[f]] <- as.numeric(x_interest[[f]]))
      # if still not numeric, throw a clear error
      if (!is.numeric(df[[f]]) || !is.numeric(x_interest[[f]])) {
        stop("Feature '", f, "' must be numeric but could not be coerced. ",
             "Consider adding it to --categorical_overrides if it is categorical.")
      }
    }
  }
  # sanity check: classes must match
  df_cls <- vapply(features, function(f) class(df[[f]])[1], "")
  xi_cls <- vapply(features, function(f) class(x_interest[[f]])[1], "")
  mism <- features[df_cls != xi_cls]
  if (length(mism)) {
    stop("Type mismatch after harmonization for: ", paste(mism, collapse = ", "))
  }
  list(df = df, x_interest = x_interest)
}

hz <- harmonize_types(df, x_interest, features, categorical_overrides)
df <- hz$df
x_interest <- hz$x_interest

# Keep only the selected features + target in df
df <- df[, c(features, "validity"), drop = FALSE]

# ---- Train/test, model, predictor ----
set.seed(opt$seed)

task <- TaskClassif$new(id = "pem", backend = df, target = "validity")
mod  <- lrn("classif.ranger", predict_type = "prob")

split <- partition(task, ratio = 0.8, stratify = TRUE)
task_train <- task$clone()$filter(split$train)
task_test  <- task$clone()$filter(split$test)

mod$train(task_train)

prediction_test <- mod$predict(task_test)
test_metrics <- prediction_test$score(msrs(c("classif.acc","classif.precision","classif.recall","classif.fbeta","classif.auc")))
cat("Test set performance:\n")
print(test_metrics)

mod$train(task)
prediction_full <- mod$predict(task)
full_metrics <- prediction_full$score(msrs(c("classif.acc","classif.precision","classif.recall","classif.fbeta","classif.auc")))
cat("Full dataset performance:\n")
print(full_metrics)

# Save RF metrics to TXT
capture_tbl <- function(x) paste(capture.output(print(x)), collapse = "\n")
metrics_txt <- paste0(
  "Random Forest metrics\n",
  "Seed: ", opt$seed, "\n",
  "Data: ", opt$data, "\n",
  "Target: ", opt$target, "    Positive class: ", opt$positive, "\n\n",
  "[Test set]\n",
  capture_tbl(test_metrics), "\n\n",
  "[Full dataset]\n",
  capture_tbl(full_metrics), "\n"
)
metrics_path <- file.path(opt$outdir, paste0("RF_metrics_", opt$run_name, ".txt"))
dir.create(dirname(metrics_path), showWarnings = FALSE, recursive = TRUE)
writeLines(metrics_txt, metrics_path)
message("Saved RF metrics: ", metrics_path)

pred <- Predictor$new(
  model = mod,
  data  = df,
  y     = "validity",
  type  = "classification",
  class = opt$positive
)

# ---- IRD runners ----
run_prim <- function() {
  prim <- Prim$new(predictor = pred)
  bx   <- prim$find_box(x_interest = x_interest, desired_range = desired_range)
  post <- PostProcessing$new(predictor = pred)$find_box(
    x_interest = x_interest,
    desired_range = desired_range,
    box_init = bx$box
  )
  post
}

run_maxbox <- function() {
  mb <- MaxBox$new(predictor = pred, quiet = FALSE, strategy = "traindata")
  bx <- mb$find_box(x_interest = x_interest, desired_range = desired_range)
  post <- PostProcessing$new(predictor = pred)$find_box(
    x_interest = x_interest,
    desired_range = desired_range,
    box_init = bx$box
  )
  post
}

run_maire <- function() {
  if (!requireNamespace("tensorflow", quietly = TRUE)) {
    message("tensorflow not available, skipping MAIRE.")
    return(NULL)
  }
  tensorflow::tf$compat$v1$disable_eager_execution()
  mair <- Maire$new(
    predictor = pred,
    num_of_iterations = 100L,
    convergence = TRUE,
    quiet = FALSE,
    strategy = "traindata"
  )
  bx <- mair$find_box(x_interest = x_interest, desired_range = desired_range)
  post <- PostProcessing$new(predictor = pred, subbox_relsize = 0.3)$find_box(
    x_interest = x_interest,
    desired_range = desired_range,
    box_init = bx$box
  )
  post
}

# ---- Execute selected methods and export ----
for (m in methods) {
  cat("\n=== Running ", m, " ===\n", sep = "")
  post_box <- NULL
  if (m == "PRIM") {
    post_box <- tryCatch(run_prim(), error = function(e) { message("PRIM failed: ", e$message); NULL })
  } else if (m == "MaxBox") {
    post_box <- tryCatch(run_maxbox(), error = function(e) { message("MaxBox failed: ", e$message); NULL })
  } else if (m == "Maire") {
    post_box <- tryCatch(run_maire(), error = function(e) {
      message("MAIRE failed (this happens sometimes). Error: ", e$message); NULL
    })
  }
  
  if (is.null(post_box)) next
  
  base <- paste0(m, "_", opt$run_name)
  yaml_path <- file.path(opt$outdir, paste0("IRD_bounds_", base, ".yaml"))
  txt_path  <- file.path(opt$outdir, paste0("IRD_report_", base, ".txt"))
  
  # Save YAML
  write_ird_yaml(
    post_box_obj = post_box,
    data_pc      = df,
    outfile      = yaml_path,
    categorical_overrides = categorical_overrides,
    round_digits = 6
  )
  
  # Save text report
  write_ird_text_report(
    post_box_obj  = post_box,
    data_pc       = df,
    outfile       = txt_path,
    method        = m,
    desired_range = desired_range,
    desired_class = opt$positive,
    postprocessed = TRUE
  )
  
  cat("Saved:\n  ", yaml_path, "\n  ", txt_path, "\n", sep = "")
}
