# scripts/run_sampler_batch_parallel.py
# Parallel AlphaPEM runner with per-worker checkpoints, final merge, and a metadata log.

# =============================================================================
# Examples
# =============================================================================
# Extended run (explicit arguments shown)
#
# - Input:     2,000 configs from a design matrix pickle
# - Subset:    simulate first 1,200 rows starting at offset 400
# - Output:    CSV results/errors into data/raw
# - Verify:    validate & derive using the YAMLs
# - Parallel:  8 workers, each saving checkpoints every 10 rows
# - Names:     outputs use run_name "test_batch"
# - Layouts:   AlphaPEM + YAML paths relative to repo root
#
# Command:
#   python scripts/run_sampler_batch_parallel.py \
#     --input data/designs/huge_design_matrix.pkl \
#     --n_samples 1200 \
#     --offset 400 \
#     --alpha_pem_root external/AlphaPEM \
#     --param_config_yaml configs/param_config.yaml \
#     --simulator_defaults_yaml configs/simulator_defaults.yaml \
#     --verify \
#     --n_workers 8 \
#     --save_every 10 \
#     --output_dir data/raw \
#     --run_name test_batch \
#     --format csv \
#     --print_errors
#
# This will produce:
#   data/raw/test_batch_simulations.csv
#   data/raw/test_batch_sim_errors.csv
#   data/raw/test_batch_meta.json
#   and a temporary folder under data/raw/temp/test_batch/ during the run
#
# -----------------------------------------------------------------------------
# Minimal run (mostly defaults)
#
# - Input:     first 10 rows of a design matrix
# - Output:    Pickle results/errors into data/raw
# - Verify:    on (default)
# - Parallel:  all cores (default), checkpoints every 10 rows (default)
# - Names:     run_name defaults to input filename stem
#
# Command:
#   python scripts/run_sampler_batch_parallel.py \
#     --input data/designs/sobol_design_matrix.pkl \
#     --n_samples 10 \
#     --format pkl
#
# Outputs:
#   data/raw/sobol_design_matrix_simulations.pkl
#   data/raw/sobol_design_matrix_sim_errors.csv
#   data/raw/sobol_design_matrix_meta.json
# =============================================================================

from __future__ import annotations
import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
from multiprocessing import get_context, cpu_count

# --- Make 'src' importable (repo root assumed one level up from this script) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Reuse components from your new sampler
from src.sampling.bounds import (
    load_param_config,
    validate_sample_against_spec,
    SpecError,
    SampleValidationError,
    SamplingSpec,
)
from src.sampling.sampler import _AlphaPEMRunner, _expand_arrays  # internal helpers


# ----------------------------- Utilities -----------------------------

def load_df(input_path: str) -> pd.DataFrame:
    ext = Path(input_path).suffix.lower()
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(input_path)
    if ext == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported input extension '{ext}'. Use .pkl or .csv.")


def infer_names(input_path: str, output_dir: str, run_name: Optional[str], results_format: str) -> Tuple[str, str, str]:
    """Return (final_results_path, final_errors_path, temp_dir)."""
    os.makedirs(output_dir, exist_ok=True)
    base_in = Path(input_path).stem
    if not run_name:
        run_name = base_in  # default to input filename stem
    ext = ".csv" if results_format.lower() == "csv" else ".pkl"
    results_path = str(Path(output_dir) / f"{run_name}_simulations{ext}")
    errors_path  = str(Path(output_dir) / f"{run_name}_sim_errors.csv")
    temp_dir     = str(Path(output_dir) / "temp" / run_name)
    os.makedirs(temp_dir, exist_ok=True)
    return results_path, errors_path, temp_dir


def split_evenly(df: pd.DataFrame, n_parts: int) -> List[pd.DataFrame]:
    """Split a DataFrame into n_parts with sizes differing by at most 1."""
    if n_parts <= 1:
        return [df.copy()]
    n_parts = min(n_parts, len(df)) or 1
    # np.array_split gives nearly-even chunks
    return [chunk.copy() for chunk in np.array_split(df, n_parts) if len(chunk) > 0]


def write_meta(meta_path: str, data: dict) -> None:
    """Write metadata as pretty JSON (atomic replace)."""
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, meta_path)



def read_text(path: str) -> Optional[str]:
    """Read a text file (UTF-8). Return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

# ----------------------------- Worker -----------------------------

def _worker_run_chunk(args: Tuple[int, pd.DataFrame, str, str, Optional[str], bool,
                                   bool, str, str, Optional[int], int]) -> Tuple[str, str, int, int]:
    """
    Single worker process:
      - builds its own AlphaPEM runner + optional YAML spec,
      - loops rows, runs sim, collects results,
      - saves temp result & error files every `save_every`,
      - returns (res_path, err_path, n_ok, n_err).
    """
    (worker_id,
     df_chunk,
     alpha_pem_root,
     simulator_defaults_yaml,
     param_config_yaml,
     verify,
     print_errors,
     temp_dir,
     run_name,
     n_target,
     save_every) = args

    # Build runner and optional spec in the subprocess
    runner = _AlphaPEMRunner(alpha_pem_root=alpha_pem_root,
                             simulator_defaults_yaml=simulator_defaults_yaml)
    spec: Optional[SamplingSpec] = None
    if verify and param_config_yaml:
        spec = load_param_config(param_config_yaml)

    # Temp outputs for this worker
    res_path = os.path.join(temp_dir, f"worker_{run_name}_core{worker_id}.pkl")
    err_path = os.path.join(temp_dir, f"worker_{run_name}_core{worker_id}_errors.csv")

    results: List[Dict[str, Any]] = []
    errors:  List[Dict[str, Any]] = []
    n_ok = n_err = 0

    for j, (idx, row) in enumerate(df_chunk.iterrows(), start=1):
        rec = row.to_dict()
        try:
            # Coerce integer-like floats before validation (robustness)
            if spec is not None:
                for name, p in spec.spec_index.items():
                    if p.get("type") == "integer" and name in rec and isinstance(rec[name], float) and float(rec[name]).is_integer():
                        rec[name] = int(rec[name])
                rec = validate_sample_against_spec(rec, spec)

            out = runner.run_one(rec)
            results.append({**row.to_dict(), **out})
            n_ok += 1

        except (SampleValidationError, Exception) as e:
            fail = row.to_dict()
            fail.update({"ifc": None, "Ucell": None, "error": str(e)})
            results.append(fail)
            errors.append({"index": idx, "config_id": row.get("config_id", None), "error": str(e)})
            n_err += 1

            if print_errors:
                cid = row.get("config_id", None)
                print(f"Worker {worker_id}: could not simulate index={idx}{'' if cid is None else f', config_id={cid}'}")
                print(f"  Error: {e}")

        # Per-worker checkpoint
        if (j % save_every) == 0:
            df_tmp = pd.DataFrame(results)
            df_tmp = _expand_arrays(df_tmp, "ifc", "ifc", n_target=n_target or 31)
            df_tmp = _expand_arrays(df_tmp, "Ucell", "Ucell", n_target=n_target or 31)
            df_tmp.to_pickle(res_path)
            pd.DataFrame(errors or [], columns=["index", "config_id", "error"]).to_csv(err_path, index=False)
            print(f"Worker {worker_id}: checkpoint ({j} rows in chunk) → {os.path.basename(res_path)}")

    # Final per-worker write
    df_tmp = pd.DataFrame(results)
    df_tmp = _expand_arrays(df_tmp, "ifc", "ifc", n_target=n_target or 31)
    df_tmp = _expand_arrays(df_tmp, "Ucell", "Ucell", n_target=n_target or 31)
    df_tmp.to_pickle(res_path)
    pd.DataFrame(errors or [], columns=["index", "config_id", "error"]).to_csv(err_path, index=False)

    print(f"Worker {worker_id}: done | ok={n_ok}, err={n_err} → {os.path.basename(res_path)}")
    return res_path, err_path, n_ok, n_err


# ----------------------------- Orchestrator (parent) -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Parallel AlphaPEM simulator with per-worker temp checkpoints.")
    ap.add_argument("--input", required=True, help="Path to input .pkl or .csv with configs.")
    ap.add_argument("--n_samples", default="all", help="Number of rows to simulate or 'all'.")
    ap.add_argument("--offset", type=int, default=0, help="Starting row index for subsetting (default 0).")

    # Sampler settings (defaults relative to repo root)
    ap.add_argument("--alpha_pem_root", default="external/AlphaPEM", help="Path to AlphaPEM repo.")
    ap.add_argument("--param_config_yaml", default="configs/param_config.yaml", help="param_config.yaml path.")
    ap.add_argument("--simulator_defaults_yaml", default="configs/simulator_defaults.yaml", help="simulator_defaults.yaml path.")
    ap.add_argument("--verify", action="store_true", help="Validate and derive from YAML.")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.set_defaults(verify=True)

    # Parallelism + checkpoints
    ap.add_argument("--n_workers", type=int, default=0, help="Number of workers (default: all cores).")
    ap.add_argument("--save_every", type=int, default=10, help="Temp save frequency inside each worker (default 10).")
    ap.add_argument("--print_errors", action="store_true", help="Print row-level errors.")
    ap.add_argument("--no-print_errors", dest="print_errors", action="store_false")
    ap.set_defaults(print_errors=True)

    # Output (relative to repo root)
    ap.add_argument("--output_dir", default="data/raw", help="Directory for final results/errors.")
    ap.add_argument("--run_name", default=None, help="Prefix for outputs; default is input filename stem.")
    ap.add_argument("--format", choices=["pkl", "csv"], default="pkl", help="Final results file format.")
    ap.add_argument("--n_target", type=int, default=31, help="Target columns for ifc_/Ucell_ expansion (default 31).")

    args = ap.parse_args()

    # Load and subset input
    df_in = load_df(args.input)
    total = len(df_in)
    if total == 0:
        print("Input is empty. Nothing to do.")
        return

    if isinstance(args.n_samples, str) and args.n_samples.lower() == "all":
        df_sub = df_in.iloc[args.offset:].reset_index(drop=True)
    else:
        n = int(args.n_samples)
        start = args.offset
        end = start + n
        if end > total:
            raise ValueError(f"Requested rows [{start}, {end}) exceed input length {total}.")
        df_sub = df_in.iloc[start:end].reset_index(drop=True)

    # Names and temp dirs
    results_path, errors_path, temp_dir = infer_names(
        input_path=args.input,
        output_dir=args.output_dir,
        run_name=args.run_name,
        results_format=args.format,
    )

    # Timing + meta path
    start_ts = datetime.now().isoformat(timespec="seconds")
    start_t  = datetime.now().timestamp()
    meta_path = os.path.join(args.output_dir, f"{Path(results_path).stem.replace('_simulations','')}_meta.json")

    # Workers/chunks
    n_workers = args.n_workers if args.n_workers and args.n_workers > 0 else cpu_count()
    n_workers = min(n_workers, len(df_sub)) or 1
    chunks = split_evenly(df_sub, n_workers)

    print(f"Parallel AlphaPEM: {len(df_sub)} rows | {len(chunks)} worker chunk(s) | workers={n_workers}")
    print(f"Temp dir: {temp_dir}")

    # Initial meta (state: running)
    initial_meta = {
        "status": "running",
        "design_file": str(Path(args.input).resolve()),
        "results_path": str(Path(results_path).resolve()),
        "errors_path":  str(Path(errors_path).resolve()),
        "meta_path":    str(Path(meta_path).resolve()),
        "temp_dir":     str(Path(temp_dir).resolve()),
        "n_total_in_file": int(len(df_in)),
        "n_requested": args.n_samples if isinstance(args.n_samples, str) else int(args.n_samples),
        "offset": int(args.offset),
        "n_subset": int(len(df_sub)),
        "n_workers": int(n_workers),
        "save_every": int(args.save_every),
        "verify": bool(args.verify),
        "print_errors": bool(args.print_errors),
        "format": args.format,
        "n_target": int(args.n_target),
        "alpha_pem_root": str(args.alpha_pem_root),
        "param_config_yaml": str(args.param_config_yaml),
        "simulator_defaults_yaml": str(args.simulator_defaults_yaml),
        "start_time": start_ts,
        "end_time": None,
        "duration_seconds": None,
        "ok": 0,
        "err": 0,
    }
    write_meta(meta_path, initial_meta)

    # Inline copies of YAML configs in the meta (for reproducibility)
    param_cfg_text = read_text(args.param_config_yaml)
    sim_defs_text  = read_text(args.simulator_defaults_yaml)
    initial_meta.update({
        "param_config_text": param_cfg_text,
        "simulator_defaults_text": sim_defs_text,
    })
    write_meta(meta_path, initial_meta)

    # Build packets
    packets = []
    for worker_id, df_chunk in enumerate(chunks):
        packets.append((
            worker_id,
            df_chunk,
            args.alpha_pem_root,
            args.simulator_defaults_yaml,
            args.param_config_yaml if args.verify else None,
            args.verify,
            args.print_errors,
            temp_dir,
            Path(results_path).stem.replace("_simulations", ""),
            args.n_target,
            args.save_every,
        ))

    # Run pool + merge
    ctx = get_context("spawn")  # Windows-safe
    res_files: List[str] = []
    err_files: List[str] = []
    total_ok = total_err = 0

    try:
        with ctx.Pool(processes=n_workers) as pool:
            for res_path, err_path, n_ok, n_err in pool.imap_unordered(_worker_run_chunk, packets):
                res_files.append(res_path)
                err_files.append(err_path)
                total_ok += n_ok
                total_err += n_err
                print(f"Chunk saved → {os.path.basename(res_path)} (ok={n_ok}, err={n_err})")

        # Merge results
        outs = [pd.read_pickle(p) for p in res_files]
        out_df = pd.concat(outs, axis=0, ignore_index=False)

        # Final write
        if args.format == "csv":
            out_df.to_csv(results_path, index=False)
        else:
            out_df.to_pickle(results_path)

        # Merge errors
        err_dfs = []
        for p in err_files:
            try:
                err_dfs.append(pd.read_csv(p))
            except Exception:
                pass
        if err_dfs:
            pd.concat(err_dfs, axis=0, ignore_index=True).to_csv(errors_path, index=False)
        else:
            pd.DataFrame([], columns=["index","config_id","error"]).to_csv(errors_path, index=False)

        print(f"Final results → {results_path}")
        print(f"Final errors  → {errors_path}")
        print(f"Totals        → ok={total_ok}, err={total_err}")

        # Success meta update
        end_ts = datetime.now().isoformat(timespec="seconds")
        end_t  = datetime.now().timestamp()
        final_meta = dict(initial_meta)
        final_meta.update({
            "status": "success",
            "end_time": end_ts,
            "duration_seconds": round(end_t - start_t, 3),
            "ok": int(total_ok),
            "err": int(total_err),
        })
        write_meta(meta_path, final_meta)

        # Cleanup temp files on success
        for p in res_files + err_files:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

    except KeyboardInterrupt:
        print("Interrupted. Temp files kept.")
        # Interrupted meta update
        end_ts = datetime.now().isoformat(timespec="seconds")
        end_t  = datetime.now().timestamp()
        interrupted_meta = dict(initial_meta)
        interrupted_meta.update({
            "status": "interrupted",
            "end_time": end_ts,
            "duration_seconds": round(end_t - start_t, 3),
            "ok": int(total_ok),
            "err": int(total_err),
        })
        write_meta(meta_path, interrupted_meta)
        raise

    except Exception as e:
        print(f"Run failed: {e}")
        print("Temp files kept for inspection.")
        # Failure meta update
        end_ts = datetime.now().isoformat(timespec="seconds")
        end_t  = datetime.now().timestamp()
        fail_meta = dict(initial_meta)
        fail_meta.update({
            "status": "failed",
            "end_time": end_ts,
            "duration_seconds": round(end_t - start_t, 3),
            "ok": int(total_ok),
            "err": int(total_err),
            "error_message": str(e),
        })
        write_meta(meta_path, fail_meta)
        raise


if __name__ == "__main__":
    # Windows needs this guard + freeze_support for multiprocessing to spawn properly
    import multiprocessing as mp
    mp.freeze_support()
    main()
