#---------------------------------------------------------------------------
#  R-Utils for: Interpretable Regional Descriptors
# --------------------------------------------------------------------------

# Simple helpers to export IRD results.
# - write_ird_yaml(): writes a parameters YAML with continuous/categorical handling
# - write_ird_text_report(): writes a console-style report (like printing the object)

suppressWarnings({
  if (!requireNamespace("yaml", quietly = TRUE)) install.packages("yaml")
})

library(yaml)

# ----------------------------------------------------------
# Boolean handler so YAML shows True/False, not yes/no
# ----------------------------------------------------------

yaml_bool_handler <- function(x) {
  # Emit capitalized booleans without quotes
  structure(ifelse(x, "True", "False"), class = "verbatim")
}

# ----------------------------------------------------------
# Write IRD box to a YAML with continuous/categorical handling
# ----------------------------------------------------------
write_ird_yaml <- function(
  post_box_obj,
  data_pc,
  outfile,
  categorical_overrides = character(0),  # e.g., c("e")
  round_digits = 6
) {
  # Names in the box should align with x_interest columns
  feature_names <- names(post_box_obj$x_interest)
  
  # Pull numeric bounds (may be NA for categoricals)
  lower_rd <- post_box_obj$box$lower
  upper_rd <- post_box_obj$box$upper
  
  # Pull categorical levels (list or named list)
  levels_rd <- post_box_obj$box$levels
  if (is.null(levels_rd)) levels_rd <- vector("list", length(feature_names))
  if (is.null(names(levels_rd))) names(levels_rd) <- feature_names
  
  # Helper to decide if feature is categorical
  is_categorical <- function(f) {
    f %in% categorical_overrides ||
      is.factor(data_pc[[f]]) || is.character(data_pc[[f]])
  }
  
  # Clean a numeric safely (round if numeric-like, otherwise leave as-is)
  safe_round <- function(x) {
    if (is.numeric(x)) {
      return(signif(x, round_digits))
    }
    # try to convert numeric-looking strings
    suppressWarnings({
      xx <- as.numeric(x)
      if (!any(is.na(xx))) return(signif(xx, round_digits))
    })
    x
  }
  
  # Ensure levels are a simple vector with correct types if possible
  normalize_levels <- function(vals) {
    if (is.null(vals)) return(NULL)
    v <- unlist(vals, use.names = FALSE)
    # Attempt numeric conversion if all entries look numeric
    suppressWarnings({
      vv <- as.numeric(v)
      if (!any(is.na(vv))) return(signif(vv, round_digits))
    })
    # Otherwise keep as character
    as.character(v)
  }
  
  params <- lapply(feature_names, function(f) {
    entry <- list(name = f)
    
    if (is_categorical(f)) {
      entry$type <- "categorical"
      vals <- normalize_levels(levels_rd[[f]])
      if (is.null(vals)) {
        # Fallback: if IRD didn't return levels, fall back to observed levels in data
        if (is.factor(data_pc[[f]])) {
          vals <- levels(data_pc[[f]])
        } else {
          vals <- unique(as.character(data_pc[[f]]))
        }
      }
      entry$values <- vals
      entry$fixed <- length(vals) == 1
    } else {
      entry$type <- "continuous"
      low  <- as.numeric(lower_rd[[f]])
      high <- as.numeric(upper_rd[[f]])
      
      # Safety fallback if IRD didn't return bounds (rare)
      if (is.na(low) || is.na(high)) {
        low  <- suppressWarnings(min(as.numeric(data_pc[[f]]), na.rm = TRUE))
        high <- suppressWarnings(max(as.numeric(data_pc[[f]]), na.rm = TRUE))
      }
      
      entry$low   <- safe_round(low)
      entry$high  <- safe_round(high)
      entry$fixed <- isTRUE(all.equal(as.numeric(low), as.numeric(high)))
    }
    
    entry
  })
  
  # Wrap in top-level list
  out_list <- list(parameters = params)
  
  # Write YAML (flow = FALSE gives block style; indent for readability)
  
  yaml_text <- as.yaml(
    out_list,
    indent = 2,
    line.sep = "\n",
    handlers = list(logical = yaml_bool_handler)
  )
  
  writeLines(yaml_text, con = outfile)
  invisible(outfile)
}

# ----------------------------------------------------------
# Save a console-like IRD report to a text file
# ----------------------------------------------------------

write_ird_text_report <- function(
  post_box_obj,
  data_pc,
  outfile,
  method,
  desired_range,
  desired_class = "valid",
  postprocessed = TRUE,
  digits_main = 2,          # for "normal" magnitudes
  digits_scientific = 3,    # for very small/large values
  equal_tol = 1e-12         # tolerance to consider low==high
) {
  stopifnot(length(desired_range) == 2)
  
  # Helpers ---------------------------------------------------------------
  is_scalar <- function(x) is.numeric(x) && length(x) == 1 && !is.na(x)
  
  fmt_num <- function(x) {
    if (!is.numeric(x) || is.na(x)) return(as.character(x))
    ax <- abs(x)
    if ((ax >= 1e5) || (ax > 0 && ax < 1e-3)) {
      return(format(x, scientific = TRUE, digits = digits_scientific, trim = TRUE))
    } else {
      # fixed with nsmall for prettier columns
      return(format(round(x, digits_main), nsmall = digits_main, trim = TRUE, scientific = FALSE))
    }
  }
  
  fmt_set <- function(vals) {
    if (is.null(vals)) return("{}")
    v <- unlist(vals, use.names = FALSE)
    # try numeric conversion
    suppressWarnings({
      vv <- as.numeric(v)
      if (!any(is.na(vv))) {
        return(paste0("{", paste(vapply(vv, fmt_num, character(1)), collapse = ", "), "}"))
      }
    })
    paste0("{", paste(as.character(v), collapse = ", "), "}")
  }
  
  fmt_interval_or_singleton <- function(lo, hi) {
    if (is.na(lo) || is.na(hi)) return("[NA, NA]")
    if (isTRUE(abs(hi - lo) <= equal_tol)) {
      return(paste0("{", fmt_num(lo), "}"))
    }
    paste0("[", fmt_num(lo), ", ", fmt_num(hi), "]")
  }
  
  pad_left  <- function(s, w) sprintf("%-*s", w, s)
  
  # Pull bits from the object ---------------------------------------------
  feature_names <- names(post_box_obj$x_interest)
  
  lower_rd     <- post_box_obj$box$lower
  upper_rd     <- post_box_obj$box$upper
  lower_1d     <- post_box_obj$box_single$lower
  upper_1d     <- post_box_obj$box_single$upper
  levels_rd    <- post_box_obj$box$levels
  levels_1d    <- post_box_obj$box_single$levels
  
  # evaluate() returns named metrics (e.g., impurity, dist)
  metrics <- post_box_obj$evaluate()
  
  # Build table columns ----------------------------------------------------
  col_feature <- feature_names
  col_xint <- vapply(feature_names, function(f) {
    xi <- post_box_obj$x_interest[[f]][1]
    if (is.numeric(xi)) fmt_num(xi) else as.character(xi)
  }, character(1))
  
  is_cat <- vapply(feature_names, function(f) {
    has_levels <- !is.null(levels_rd) && !is.null(levels_rd[[f]]) && length(levels_rd[[f]]) > 0
    has_levels || is.factor(data_pc[[f]]) || is.character(data_pc[[f]])
  }, logical(1))
  
  # Regional descriptor column
  col_rd <- vapply(seq_along(feature_names), function(i) {
    f <- feature_names[i]
    if (is_cat[i]) {
      vals <- if (!is.null(levels_rd[[f]])) levels_rd[[f]] else unique(data_pc[[f]])
      fmt_set(vals)
    } else {
      lo <- suppressWarnings(as.numeric(lower_rd[[f]]))
      hi <- suppressWarnings(as.numeric(upper_rd[[f]]))
      fmt_interval_or_singleton(lo, hi)
    }
  }, character(1))
  
  # 1-dim descriptor column
  col_1d <- vapply(seq_along(feature_names), function(i) {
    f <- feature_names[i]
    if (is_cat[i]) {
      vals <- if (!is.null(levels_1d[[f]])) levels_1d[[f]] else unique(data_pc[[f]])
      fmt_set(vals)
    } else {
      lo <- suppressWarnings(as.numeric(lower_1d[[f]]))
      hi <- suppressWarnings(as.numeric(upper_1d[[f]]))
      fmt_interval_or_singleton(lo, hi)
    }
  }, character(1))
  
  # full observed range from data
  col_range <- vapply(seq_along(feature_names), function(i) {
    f <- feature_names[i]
    if (is_cat[i]) {
      vals <- if (is.factor(data_pc[[f]])) levels(data_pc[[f]]) else unique(as.character(data_pc[[f]]))
      fmt_set(vals)
    } else {
      suppressWarnings({
        lo <- min(as.numeric(data_pc[[f]]), na.rm = TRUE)
        hi <- max(as.numeric(data_pc[[f]]), na.rm = TRUE)
      })
      fmt_interval_or_singleton(lo, hi)
    }
  }, character(1))
  
  # Compose table with widths ---------------------------------------------
  header <- c("feature", "x_interest", "regional descriptor", "1-dim descriptor", "range")
  
  # data matrix of strings
  M <- cbind(col_feature, col_xint, col_rd, col_1d, col_range)
  widths <- pmax(nchar(header), apply(M, 2, function(col) max(nchar(col), na.rm = TRUE)))
  
  fmt_row <- function(v) {
    paste(mapply(pad_left, v, widths), collapse = "  ")
  }
  
  table_lines <- c(
    fmt_row(header),
    vapply(seq_len(nrow(M)), function(i) fmt_row(M[i, ]), character(1))
  )
  
  # Compose metrics block --------------------------------------------------
  if (is.null(names(metrics))) {
    names(metrics) <- paste0("metric_", seq_along(metrics))
  }
  met_names <- names(metrics)
  met_vals  <- vapply(metrics, function(v) {
    if (is.numeric(v)) format(v, digits = 8, scientific = FALSE, trim = TRUE) else as.character(v)
  }, character(1))
  met_widths <- pmax(nchar(met_names), nchar(met_vals)) + 2
  metrics_line1 <- paste(mapply(function(n, w) sprintf("%-*s", w, n), met_names, met_widths), collapse = "")
  metrics_line2 <- paste(mapply(function(v, w) sprintf("%-*s", w, v), met_vals,  met_widths), collapse = "")
  
  # Compose header ---------------------------------------------------------
  header_lines <- c(
    "Regional Descriptors",
    "",
    paste0("Method: ", method),
    paste0("Post-processed: ", if (postprocessed) "True" else "False"),
    paste0("Desired class: ", desired_class),
    paste0("Desired range: [", fmt_num(desired_range[1]), ", ", fmt_num(desired_range[2]), "]"),
    ""
  )
  
  # Stitch and write -------------------------------------------------------
  lines <- c(
    header_lines,
    "Descriptor:",
    table_lines,
    "",
    metrics_line1,
    metrics_line2,
    ""
  )
  
  writeLines(lines, con = outfile)
  invisible(outfile)
}

