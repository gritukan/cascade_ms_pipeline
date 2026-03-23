from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..cmd import run_cmd
from ..results import merge_rescored_results
from ..util import common_parent, safe_float_series
from .base import RescoreArtifacts, Rescorer


def _read_table_guess(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=None, engine="python", comment="#", low_memory=False)
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", comment="#", low_memory=False)



def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}. Columns: {list(df.columns)[:80]} ...")
    return None


@dataclass
class OktoberfestRescoreConfig:
    work_dir: Path
    search_results: Path
    spectra: Path
    search_results_type: str = "Sage"
    spectra_type: str = "mzml"
    instrument_type: str = "QE"
    tag: str = ""
    intensity_model: str = "Prosit_2020_intensity_HCD"
    irt_model: str = "Prosit_2019_irt"
    prediction_server: str = "koina.wilhelmlab.org:443"
    ssl: bool = True
    fdr_estimation_method: str = "percolator"
    regressionMethod: str = "spline"
    add_feature_cols: Union[str, List[str]] = "none"
    numThreads: int = 16
    thermoExe: Optional[str] = None
    massTolerance: float = 20.0
    unitMassTolerance: str = "ppm"
    ce_range: Tuple[int, int] = (19, 50)
    use_ransac_model: bool = False
    static_mods: Optional[Dict[str, List[float]]] = None
    var_mods: Optional[Dict[str, List[float]]] = None
    extra_config: Dict[str, object] = field(default_factory=dict)
    config_path: Optional[Path] = None
    python_bin: str = "python"
    output_rel: str = "."

    def to_dict(self) -> Dict[str, object]:
        cfg: Dict[str, object] = {
            "type": "Rescoring",
            "tag": self.tag,
            "output": self.output_rel,
            "inputs": {
                "search_results": str(self.search_results),
                "search_results_type": self.search_results_type,
                "spectra": str(self.spectra),
                "spectra_type": self.spectra_type,
                "instrument_type": self.instrument_type,
            },
            "models": {"intensity": self.intensity_model, "irt": self.irt_model},
            "prediction_server": self.prediction_server,
            "numThreads": int(self.numThreads),
            "fdr_estimation_method": self.fdr_estimation_method,
            "add_feature_cols": self.add_feature_cols,
            "regressionMethod": self.regressionMethod,
            "ssl": bool(self.ssl),
            "massTolerance": float(self.massTolerance),
            "unitMassTolerance": self.unitMassTolerance,
            "ce_alignment_options": {
                "ce_range": [int(self.ce_range[0]), int(self.ce_range[1])],
                "use_ransac_model": bool(self.use_ransac_model),
            },
        }
        if self.thermoExe:
            cfg["thermoExe"] = self.thermoExe
        if self.static_mods is not None:
            cfg["static_mods"] = self.static_mods
        if self.var_mods is not None:
            cfg["var_mods"] = self.var_mods
        cfg.update(self.extra_config or {})
        return cfg



def write_oktoberfest_config(cfg: OktoberfestRescoreConfig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return out_path



def run_oktoberfest_rescoring(cfg: OktoberfestRescoreConfig, *, dry_run: bool = False, log_path: Optional[Path] = None) -> Path:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg.config_path or (cfg.work_dir / "oktoberfest_config.json").resolve()
    if cfg.config_path is None:
        write_oktoberfest_config(cfg, config_path)
    cmd = [cfg.python_bin, "-m", "oktoberfest", "--config_path", str(config_path)]
    run_cmd(cmd, cwd=config_path.parent, dry_run=dry_run, log_path=log_path)
    return config_path



def oktoberfest_psm_output_paths(*, output_dir: Path, fdr_method: str, kind: str = "rescore") -> Tuple[Path, Path]:
    method = str(fdr_method).lower()
    sub = output_dir / "results" / method
    return sub / f"{kind}.{method}.psms.txt", sub / f"{kind}.{method}.decoy.psms.txt"



def load_oktoberfest_psms(psms_path: Path, *, label: int) -> pd.DataFrame:
    df = _read_table_guess(psms_path)
    q_col = _find_col(df, ["q-value", "q_value", "qvalue", "qval", "mokapot q-value", "percolator q-value"], required=False)
    score_col = _find_col(df, ["score", "mokapot score", "mokapot_score", "percolator score", "svm_score"], required=False)
    pep_col = _find_col(df, ["Peptide", "peptide", "sequence", "Sequence", "stripped_peptide", "modified_sequence"], required=False)
    prot_col = _find_col(df, ["Protein", "protein", "Proteins", "proteins", "proteinIds", "protein_id", "ProteinIds"], required=False)
    psm_id_col = _find_col(df, ["PSMId", "psm_id", "psmId"], required=True)
    df["_q"] = safe_float_series(df[q_col]) if q_col is not None else np.nan
    df["_score"] = safe_float_series(df[score_col]) if score_col is not None else np.nan
    df["_peptide"] = df[pep_col].astype(str) if pep_col is not None else ""
    df["_proteins"] = df[prot_col].astype(str) if prot_col is not None else ""
    df["scan"] = df[psm_id_col].str.split("-").str[-3].astype("int64")
    df["label"] = int(label)
    return df



def load_oktoberfest_results(*, output_dir: Path, fdr_method: str = "mokapot", kind: str = "rescore") -> pd.DataFrame:
    tgt_path, dec_path = oktoberfest_psm_output_paths(output_dir=output_dir, fdr_method=fdr_method, kind=kind)
    frames: List[pd.DataFrame] = []
    if tgt_path.exists():
        frames.append(load_oktoberfest_psms(tgt_path, label=1))
    if dec_path.exists():
        frames.append(load_oktoberfest_psms(dec_path, label=-1))
    if not frames:
        raise FileNotFoundError(f"Could not find any Oktoberfest PSM outputs under {output_dir}")
    return pd.concat(frames, ignore_index=True)



def oktoberfest_tab_path(*, output_dir: Path, fdr_method: str, kind: str = "rescore") -> Path:
    method = str(fdr_method).lower()
    sub = output_dir / "results" / method
    fname = "rescore.tab" if kind == "rescore" else "original.tab"
    return sub / fname



def maybe_attach_proteins_from_tab(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    fdr_method: str = "mokapot",
    kind: str = "rescore",
    min_nonempty_fraction: float = 0.01,
) -> pd.DataFrame:
    if "_proteins" not in df.columns or df.empty:
        return df
    nonempty = (df["_proteins"].astype(str).str.strip() != "").mean()
    if nonempty >= float(min_nonempty_fraction):
        return df
    tab_path = oktoberfest_tab_path(output_dir=output_dir, fdr_method=fdr_method, kind=kind)
    if not tab_path.exists():
        return df
    tab = _read_table_guess(tab_path)
    join_key = None
    for k in ["scan", "SpecId", "PSMId", "ScanNr", "scan_number", "SCAN_NUMBER"]:
        if k in df.columns and k in tab.columns:
            join_key = k
            break
    if join_key is None:
        return df
    tab_prot_col = _find_col(tab, ["Protein", "Proteins", "protein", "proteins"], required=False)
    if tab_prot_col is None:
        return df
    tab_map = tab[[join_key, tab_prot_col]].dropna(subset=[join_key]).drop_duplicates(subset=[join_key])
    out = df.merge(tab_map, how="left", on=join_key, suffixes=("", "_from_tab"))
    if "_proteins_from_tab" in out.columns:
        mask = out["_proteins"].astype(str).str.strip().eq("")
        out.loc[mask, "_proteins"] = out.loc[mask, "_proteins_from_tab"].astype(str)
        out = out.drop(columns=["_proteins_from_tab"])
    return out


class OktoberfestRescorer(Rescorer):
    name = "oktoberfest"

    def run(self, cfg, ctx, base_artifacts, current_df):
        if base_artifacts.engine_name != "sage":
            raise ValueError("Oktoberfest integration is currently implemented only for Sage searches")
        out_dir = ctx.step_dir / "rescore" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        spectra_path = Path(str(cfg.params.get("spectra_path", common_parent(ctx.spectra))))
        fdr_method = str(cfg.params.get("fdr_estimation_method", "percolator"))
        kind = str(cfg.params.get("kind", "rescore"))
        search_results_type = str(cfg.params.get("search_results_type", "Sage"))
        search_results = base_artifacts.raw_paths.get("results", base_artifacts.normalized_path).resolve()
        okcfg = OktoberfestRescoreConfig(
            work_dir=out_dir,
            search_results=search_results,
            spectra=spectra_path,
            search_results_type=search_results_type,
            spectra_type=str(cfg.params.get("spectra_type", "mzml")),
            instrument_type=str(cfg.params.get("instrument_type", "QE")),
            tag=str(cfg.params.get("tag", "")),
            intensity_model=str(cfg.params.get("intensity_model", "Prosit_2020_intensity_HCD")),
            irt_model=str(cfg.params.get("irt_model", "Prosit_2019_irt")),
            prediction_server=str(cfg.params.get("prediction_server", "koina.wilhelmlab.org:443")),
            ssl=bool(cfg.params.get("ssl", True)),
            fdr_estimation_method=fdr_method,
            regressionMethod=str(cfg.params.get("regressionMethod", "spline")),
            add_feature_cols=cfg.params.get("add_feature_cols", "none"),
            numThreads=int(cfg.params.get("numThreads", 16)),
            thermoExe=cfg.params.get("thermoExe"),
            massTolerance=float(cfg.params.get("massTolerance", 20.0)),
            unitMassTolerance=str(cfg.params.get("unitMassTolerance", "ppm")),
            ce_range=tuple(cfg.params.get("ce_range", [19, 50])),
            use_ransac_model=bool(cfg.params.get("use_ransac_model", False)),
            static_mods=cfg.params.get("static_mods"),
            var_mods=cfg.params.get("var_mods"),
            extra_config=dict(cfg.params.get("extra_config", {})),
            config_path=Path(str(cfg.params["config_path"])) if "config_path" in cfg.params else None,
            python_bin=ctx.general.binaries.python,
        )
        config_path = run_oktoberfest_rescoring(okcfg, dry_run=ctx.general.dry_run, log_path=ctx.log_path)
        if ctx.general.dry_run:
            merged = current_df.copy()
            merged["score_final"] = merged.get("score_final", merged["score_engine"])
            merged["q_final"] = merged.get("q_final", merged["engine_q"])
            merged["final_score_source"] = merged.get("final_score_source", "engine")
            return RescoreArtifacts(name=self.name, merged_df=merged, raw_paths={"config": config_path}, notes=["dry_run"])

        rescored = load_oktoberfest_results(output_dir=out_dir, fdr_method=fdr_method, kind=kind)
        rescored = maybe_attach_proteins_from_tab(rescored, output_dir=out_dir, fdr_method=fdr_method, kind=kind)
        merged, merge_report = merge_rescored_results(current_df, rescored, rescorer_name=self.name, out_dir=out_dir)
        return RescoreArtifacts(
            name=self.name,
            merged_df=merged,
            raw_paths={"config": config_path, "output_dir": out_dir},
            merge_report=merge_report,
            notes=[f"Merged Oktoberfest results using strategy: {merge_report.strategy}"],
        )
