from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import yaml

# Use the spec + validators from bounds.py
from src.sampling.bounds import (
    load_param_config,
    validate_sample_against_spec,
    SamplingSpec,
    SpecError,
    SampleValidationError,
)

# ==============================
# AlphaPEM bridge (lazy import)
# ==============================

def _import_alphapem(alpha_pem_root: str):
    alpha_pem_root = os.path.abspath(alpha_pem_root)
    if alpha_pem_root not in sys.path:
        sys.path.append(alpha_pem_root)
    from configuration.settings import (
        current_density_parameters,
        physical_parameters,
        computing_parameters,
        operating_inputs,
    )
    from model.AlphaPEM import AlphaPEM
    return AlphaPEM, current_density_parameters, physical_parameters, computing_parameters, operating_inputs


# ==============================
# Internal runner (single sample)
# ==============================

@dataclass
class _AlphaPEMRunner:
    alpha_pem_root: str
    simulator_defaults_yaml: str

    def __post_init__(self):
        (
            self.AlphaPEM,
            self.current_density_parameters,
            self.physical_parameters,
            self.computing_parameters,
            self.operating_inputs,
        ) = _import_alphapem(self.alpha_pem_root)
        with open(self.simulator_defaults_yaml, "r", encoding="utf-8") as f:
            self.defaults = yaml.safe_load(f) or {}

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run AlphaPEM for a single configuration and return a dict with arrays 'ifc' and 'Ucell'."""
        d = self.defaults

        # 1) current density function + timing
        type_current = d.get("type_current", "polarization")
        (
            t_step,
            i_step,
            delta_pola,
            i_EIS,
            ratio_EIS,
            f_EIS,
            t_EIS,
            current_density,
        ) = self.current_density_parameters(type_current)

        if d.get("override_current_params", False) and d.get("delta_pola") is not None:
            delta_pola = tuple(d["delta_pola"])  # user-enforced timing

        # 2) operating inputs (prefer sample → defaults → helper)
        type_fuel_cell = d.get("type_fuel_cell", "EH-31_2.0")
        (
            _Tfc,
            _Pa_des,
            _Pc_des,
            _Sa,
            _Sc,
            _Phi_a_des,
            _Phi_c_des,
            i_max_pola,
        ) = self.operating_inputs(type_fuel_cell)

        def pick(key: str, helper_val: float) -> float:
            return float(sample[key]) if key in sample else float(d.get(key, helper_val))

        Tfc = pick("Tfc", _Tfc)
        Pa_des = pick("Pa_des", _Pa_des)
        # Pc_des comes from the (possibly validated) sample; fall back to defaults or helper only.
        Pc_des = float(sample["Pc_des"]) if "Pc_des" in sample else float(d.get("Pc_des", _Pc_des))
        Sa = pick("Sa", _Sa)
        Sc = pick("Sc", _Sc)
        Phi_a_des = pick("Phi_a_des", _Phi_a_des)
        Phi_c_des = pick("Phi_c_des", _Phi_c_des)

        # 3) physical parameters (sample → defaults → helper)
        (
            Hcl,
            epsilon_mc_h,
            tau_h,
            Hmem,
            Hgdl,
            epsilon_gdl_h,
            epsilon_c_h,
            Hgc,
            Wgc,
            Lgc,
            Aact_h,
            e_h,
            Re_h,
            i0_c_ref_h,
            kappa_co_h,
            kappa_c_h,
            a_slim_h,
            b_slim_h,
            a_switch_h,
            C_scl_h,
        ) = self.physical_parameters(type_fuel_cell)

        def pick3(key: str, default_yaml: Optional[float], helper_val: float) -> float:
            return float(sample[key]) if key in sample else float(
                default_yaml if default_yaml is not None else helper_val
            )

        epsilon_gdl = pick3("epsilon_gdl", d.get("epsilon_gdl"), epsilon_gdl_h)
        tau = pick3("tau", d.get("tau"), tau_h)
        epsilon_mc = pick3("epsilon_mc", d.get("epsilon_mc"), epsilon_mc_h)
        epsilon_c = pick3("epsilon_c", d.get("epsilon_c"), epsilon_c_h)
        e = pick3("e", d.get("e"), e_h)
        Re = pick3("Re", d.get("Re"), Re_h)
        i0_c_ref = pick3("i0_c_ref", d.get("i0_c_ref"), i0_c_ref_h)
        kappa_co = pick3("kappa_co", d.get("kappa_co"), kappa_co_h)
        kappa_c = pick3("kappa_c", d.get("kappa_c"), kappa_c_h)

        Aact = float(d.get("Aact", Aact_h))
        Hgdl = float(d.get("Hgdl", Hgdl))
        Hmem = float(d.get("Hmem", Hmem))
        Hcl = float(d.get("Hcl", Hcl))
        Hgc = float(d.get("Hgc", Hgc))
        Wgc = float(d.get("Wgc", Wgc))
        Lgc = float(d.get("Lgc", Lgc))
        C_scl = float(d.get("C_scl", C_scl_h))

        # 4) computing parameters
        if d.get("use_computing_from_helpers", True):
            max_step, n_gdl, t_purge = self.computing_parameters(type_current, Hgdl, Hcl)
        else:
            max_step = d.get("max_step", None)
            n_gdl = d.get("n_gdl", None)
            t_purge = tuple(d["t_purge"]) if d.get("t_purge") is not None else None

        # 5) assemble kwargs and run
        params = {
            "current_density": current_density,
            "Tfc": Tfc,
            "Pa_des": Pa_des,
            "Pc_des": Pc_des,
            "Sa": Sa,
            "Sc": Sc,
            "Phi_a_des": Phi_a_des,
            "Phi_c_des": Phi_c_des,
            "t_step": t_step,
            "i_step": i_step,
            "i_max_pola": i_max_pola,
            "delta_pola": delta_pola,
            "i_EIS": i_EIS,
            "ratio_EIS": ratio_EIS,
            "t_EIS": t_EIS,
            "f_EIS": f_EIS,
            "Aact": Aact,
            "Hgdl": Hgdl,
            "Hmem": Hmem,
            "Hcl": Hcl,
            "Hgc": Hgc,
            "Wgc": Wgc,
            "Lgc": Lgc,
            "epsilon_gdl": epsilon_gdl,
            "tau": tau,
            "epsilon_mc": epsilon_mc,
            "epsilon_c": epsilon_c,
            "e": e,
            "Re": Re,
            "i0_c_ref": i0_c_ref,
            "kappa_co": kappa_co,
            "kappa_c": kappa_c,
                        # prefer values coming from param_config.yaml (via sample), then defaults, then helpers
            "a_slim": float(sample.get("a_slim", d.get("a_slim", a_slim_h))),
            "b_slim": float(sample.get("b_slim", d.get("b_slim", b_slim_h))),
            "a_switch": float(sample.get("a_switch", d.get("a_switch", a_switch_h))),
            "C_scl": float(sample.get("C_scl", d.get("C_scl", C_scl_h))),
            "max_step": max_step,
            "n_gdl": n_gdl,
            "t_purge": t_purge,
            "type_fuel_cell": d.get("type_fuel_cell", "EH-31_2.0"),
            "type_current": type_current,
            "type_auxiliary": d.get("type_auxiliary", "no_auxiliary"),
            "type_control": d.get("type_control", "no_control"),
            "type_purge": d.get("type_purge", "no_purge"),
            "type_display": d.get("type_display", "no_display"),
            "type_plot": d.get("type_plot", "fixed"),
        }

        sim = self.AlphaPEM(**params)

        # Extract the polarization curve
        if params["type_plot"] != "fixed":
            raise RuntimeError("type_plot must be 'fixed' to extract polarization curve.")

        t = np.asarray(sim.variables["t"], dtype=float)
        ucell_t = np.asarray(sim.variables["Ucell"], dtype=float)
        current_density_fn = sim.operating_inputs["current_density"]
        p = sim.parameters

        # ifc(t) in A/cm^2
        ifc_t = np.array([current_density_fn(tt, p) / 1e4 for tt in t], dtype=float)

        delta_t_load_pola, delta_t_break_pola, delta_i_pola, delta_t_ini_pola = p["delta_pola"]
        i_max_pola = p["i_max_pola"]
        nb_loads = int(i_max_pola / delta_i_pola + 1)

        ifc_dis = np.zeros(nb_loads, dtype=float)
        ucell_dis = np.zeros(nb_loads, dtype=float)
        for k in range(nb_loads):
            t_load = (
                delta_t_ini_pola
                + (k + 1) * (delta_t_load_pola + delta_t_break_pola)
                - (delta_t_break_pola / 10.0)
            )
            idx = int(np.abs(t - t_load).argmin())
            ifc_dis[k] = ifc_t[idx]
            ucell_dis[k] = ucell_t[idx]

        return {
            "ifc": ifc_dis,
            "Ucell": ucell_dis
        }


# ======================================================
# Public API: run simulations from a design matrix (DF)
# ======================================================

def _expand_arrays(df: pd.DataFrame, col: str, prefix: str, n_target: Optional[int] = 31) -> pd.DataFrame:
    """
    Expand a column of 1D arrays/lists into prefixed scalar columns.
    If n_target is set, pad/truncate to that length for consistent headers (e.g., 31).
    """
    if col not in df.columns:
        return df
    arrs = df[col]
    # determine max length present
    max_len = 0
    for v in arrs:
        try:
            max_len = max(max_len, len(v))
        except Exception:
            pass
    if max_len == 0:
        # nothing to expand
        return df

    L = n_target if (n_target is not None) else max_len
    def pad_or_trunc(x):
        if not isinstance(x, (list, tuple, np.ndarray)):
            return [np.nan] * L
        x = list(x)
        if len(x) >= L:
            return x[:L]
        # pad with NaN
        return x + [np.nan] * (L - len(x))

    expanded = pd.DataFrame([pad_or_trunc(x) for x in arrs], columns=[f"{prefix}_{i+1}" for i in range(L)], index=df.index)
    return pd.concat([df, expanded], axis=1)


def run_alphapem_from_df(
    df: pd.DataFrame,
    *,
    alpha_pem_root: str = "../external/AlphaPEM",
    simulator_defaults_yaml: str = "../configs/simulator_defaults.yaml",
    param_config_yaml: Optional[str] = "../configs/param_config.yaml",
    verify: bool = True,
    run_name: Optional[str] = None,
    output_dir: str = "../data/raw",
    results_format: str = "pkl",  # 'pkl' or 'csv'
    save_every: int = 20,
    print_errors: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run AlphaPEM simulations for each row of a design DataFrame.

    Expectations for `df`:
      - contains at least the variable parameters from param_config.yaml
      - may include fixed and/or derived params; if missing and `param_config_yaml` is provided,
        derived/fixed will be filled by `validate_sample_against_spec`.
      - should include 'index' and 'config_id' columns (they will be preserved if present)

    Behavior:
      - optionally validates each row against `param_config_yaml` (bounds & fixed),
        and computes **derived params strictly from the YAML** (e.g., Pc_des)
      - runs the simulator and appends columns: 'ifc', 'Ucell', plus
        expanded 'ifc_1..ifc_31' and 'Ucell_1..Ucell_31' (padded/truncated to 31)
      - checkpoints to `{output_dir}/{run_name}_simulations.{pkl|csv}`
      - logs errors to `data/raw/{run_name}_sim_errors.csv`
    """

    # Resolve output paths from run_name
    save_path = None
    errors_path = None
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        ext = ".csv" if results_format.lower() == "csv" else ".pkl"
        base = (run_name or "run").strip()
        save_path = os.path.join(output_dir, f"{base}_simulations{ext}")
        errors_path = os.path.join(output_dir, f"{base}_sim_errors.csv")
        os.makedirs(os.path.dirname(errors_path), exist_ok=True)

    # Prepare helpers
    runner = _AlphaPEMRunner(alpha_pem_root=alpha_pem_root, simulator_defaults_yaml=simulator_defaults_yaml)
    spec: Optional[SamplingSpec] = None
    if verify and param_config_yaml:
        try:
            spec = load_param_config(param_config_yaml)
        except SpecError as e:
            raise SpecError(f"Param config YAML invalid: {e}")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total = len(df)
    print(f"[INFO] Running AlphaPEM on {total} configuration(s)...")
    if save_results:
        print(f"[INFO] Directory for results → {save_path}")
        print(f"[INFO] Directory for errors   → {errors_path}")

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        rec = row.to_dict()
        try:
            # Validate/fill strictly via param_config (adds derived & fixed)
            if spec is not None:
                rec = validate_sample_against_spec(rec, spec)
            # else: do not compute any derived here; leave as provided by the caller

            out = runner.run_one(rec)
            results.append({**row.to_dict(), **out})
        except (SampleValidationError, Exception) as e:
            # Save row with empty outputs, but keep all original columns
            fail = row.to_dict()
            fail.update({"ifc": None, "Ucell": None, "error": str(e)})
            results.append(fail)
            errors.append({"index": idx, "config_id": row.get("config_id", None), "error": str(e)})

            if print_errors:
                cid = row.get("config_id", None)
                print(f"[ERROR] Could not simulate index={idx}"
                      f"{'' if cid is None else f', config_id={cid}'}")
                print(f"   Error: {e}")

        # checkpoint
        if (i % save_every == 0) and save_results:
            _write_results(results, save_path)
            _write_errors(errors, errors_path)
            print(f"[INFO] Checkpoint: {i}/{total} → {save_path}")

    # final save
    out_df = pd.DataFrame(results)

    # Always add expanded columns (31 by default)
    out_df = _expand_arrays(out_df, "ifc", "ifc", n_target=31)
    out_df = _expand_arrays(out_df, "Ucell", "Ucell", n_target=31)

    if save_results:
        _write_results(out_df, save_path)
        print(f"[INFO] Saved results: {save_path}")

        _write_errors(errors, errors_path)
        if errors:
            print(f"[INFO] Logged {len(errors)} error(s): {errors_path}")

    return out_df


# ==============================
# I/O helpers
# ==============================

def _write_results(results: List[Dict[str, Any]] | pd.DataFrame, path: str) -> None:
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        df.to_pickle(path)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        # default to pickle
        df.to_pickle(path)


def _write_errors(errors: List[Dict[str, Any]], path: str) -> None:
    if not path:
        return
    if not errors:
        # touch an empty file so downstream knows we tried
        pd.DataFrame([], columns=["index", "config_id", "error"]).to_csv(path, index=False)
        return
    pd.DataFrame(errors).to_csv(path, index=False)
