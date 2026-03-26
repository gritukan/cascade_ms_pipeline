from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..cmd import run_cmd
from ..protease import PROTEASE_CONFIGS
from ..results import choose_first_column
from ..util import dataframe_to_tsv, normalize_peptide_sequence, safe_float_series
from .base import SearchArtifacts, SearchEngine, SearchExecutionContext


@dataclass
class DiannRunConfig:
    diann_bin: str
    raw_files: List[Path]
    fasta_files: List[Path]
    out_report: Path
    cfg_out: Optional[Path] = None
    raw_dir: Optional[Path] = None
    lib: Optional[Path] = None
    out_lib: Optional[Path] = None
    temp_dir: Optional[Path] = None
    fasta_search: bool = False
    predictor: bool = False
    gen_spec_lib: bool = False
    threads: Optional[int] = None
    qvalue: Optional[float] = None
    verbose: Optional[int] = None
    cut: Optional[str] = None
    min_pep_len: Optional[int] = None
    max_pep_len: Optional[int] = None
    xic: Optional[int] = None
    report_decoys: bool = True
    extra_args: List[str] = field(default_factory=list)


_MANAGED_FLAGS = {
    "--fasta-search",
    "--predictor",
    "--gen-spec-lib",
    "--report-decoys",
}

_MANAGED_FLAGS_WITH_VALUE = {
    "--cfg",
    "--dir",
    "--f",
    "--fasta",
    "--lib",
    "--out",
    "--out-lib",
    "--temp",
    "--threads",
    "--verbose",
    "--qvalue",
    "--cut",
    "--min-pep-len",
    "--max-pep-len",
}


_REPORT_SUFFIXES = (".parquet", ".tsv")


def report_path_candidates(requested_path: Path) -> List[Path]:
    candidates: List[Path] = [requested_path]
    if requested_path.suffix in _REPORT_SUFFIXES:
        stem = requested_path.with_suffix("")
        for suffix in _REPORT_SUFFIXES:
            candidate = stem.with_suffix(suffix)
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates



def resolve_existing_report_path(requested_path: Path) -> Path:
    candidates = report_path_candidates(requested_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Could not find DIA-NN main report. Looked for: {searched}"
    )



def resolve_optional_report_path(requested_path: Path) -> Optional[Path]:
    for candidate in report_path_candidates(requested_path):
        if candidate.exists():
            return candidate
    return None



def stats_path_for_report(report_path: Path) -> Path:
    if report_path.suffix in _REPORT_SUFFIXES:
        return report_path.with_suffix(".stats.tsv")
    return report_path.with_name(f"{report_path.name}.stats.tsv")


def _has_raw_inputs(cfg: DiannRunConfig) -> bool:
    return cfg.raw_dir is not None or bool(cfg.raw_files)


def _clean_extra_args(extra_args: List[str]) -> List[str]:
    """
    Remove arguments that are already managed explicitly by DiannRunConfig.
    This avoids duplicate/contradictory flags when we split DIA-NN into
    separate library-generation and search phases.
    """
    cleaned: List[str] = []
    tokens = [str(x) for x in extra_args]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _MANAGED_FLAGS:
            i += 1
            continue
        if token == "--xic":
            i += 1
            if i < len(tokens) and not tokens[i].startswith("--"):
                i += 1
            continue
        if token in _MANAGED_FLAGS_WITH_VALUE:
            i += 1
            if i < len(tokens):
                i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned



def build_diann_cfg_tokens(cfg: DiannRunConfig) -> List[str]:
    tokens: List[str] = []
    if cfg.raw_dir is not None:
        tokens.extend(["--dir", str(cfg.raw_dir)])
    for f in cfg.raw_files:
        tokens.extend(["--f", str(f)])
    return tokens



def write_diann_cfg(tokens: List[str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return out_path



def predicted_library_path(out_lib_template: Path) -> Path:
    if out_lib_template.suffix:
        return out_lib_template.with_suffix(".predicted.speclib")
    return out_lib_template.with_name(f"{out_lib_template.name}.predicted.speclib")



def needs_separate_predicted_library_step(cfg: DiannRunConfig) -> bool:
    return cfg.lib is None and cfg.fasta_search and bool(cfg.fasta_files)



def build_diann_cmd(cfg: DiannRunConfig) -> Tuple[List[str], Optional[Path]]:
    cmd: List[str] = [cfg.diann_bin]
    used_cfg: Optional[Path] = None

    if cfg.cfg_out is not None and _has_raw_inputs(cfg):
        used_cfg = write_diann_cfg(build_diann_cfg_tokens(cfg), cfg.cfg_out)
        cmd.extend(["--cfg", str(used_cfg)])
    else:
        if cfg.raw_dir is not None:
            cmd.extend(["--dir", str(cfg.raw_dir)])
        for f in cfg.raw_files:
            cmd.extend(["--f", str(f)])

    for fa in cfg.fasta_files:
        cmd.extend(["--fasta", str(fa)])
    if cfg.lib is not None:
        cmd.extend(["--lib", str(cfg.lib)])
    if cfg.fasta_search:
        cmd.append("--fasta-search")
    if cfg.predictor:
        cmd.append("--predictor")
    if cfg.gen_spec_lib:
        cmd.append("--gen-spec-lib")

    cmd.extend(["--out", str(cfg.out_report)])
    if cfg.out_lib is not None:
        cmd.extend(["--out-lib", str(cfg.out_lib)])
    if cfg.temp_dir is not None:
        os.makedirs(cfg.temp_dir, exist_ok=True)
        cmd.extend(["--temp", str(cfg.temp_dir)])
    if cfg.threads is not None and int(cfg.threads) > 0:
        cmd.extend(["--threads", str(int(cfg.threads))])
    if cfg.verbose is not None:
        cmd.extend(["--verbose", str(int(cfg.verbose))])
    if cfg.qvalue is not None:
        cmd.extend(["--qvalue", str(float(cfg.qvalue))])
    if cfg.cut is not None:
        cmd.extend(["--cut", cfg.cut])
    if cfg.min_pep_len is not None:
        cmd.extend(["--min-pep-len", str(cfg.min_pep_len)])
    if cfg.max_pep_len is not None:
        cmd.extend(["--max-pep-len", str(cfg.max_pep_len)])
    if cfg.xic is not None:
        cmd.extend(["--xic", str(int(cfg.xic))])
    if cfg.report_decoys:
        cmd.append("--report-decoys")

    cmd.extend(["--qvalue", "1.0"])  # Ensure all results are reported, even if we filter them out later

    cmd.extend(_clean_extra_args(cfg.extra_args))
    return cmd, used_cfg


class DiannEngine(SearchEngine):
    name = "diann"

    @staticmethod
    def _load_report(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                raise ImportError(
                    f"Failed to read DIA-NN parquet report {path}. Install a parquet backend such as pyarrow or fastparquet."
                ) from exc
        return pd.read_csv(path, sep="\t", low_memory=False)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        q_col = choose_first_column(
            work,
            ["Q.Value", "Global.Q.Value", "Lib.Q.Value", "PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value"],
        )
        if q_col is None:
            raise KeyError("Could not find a DIA-NN q-value column")
        decoy_col = choose_first_column(work, ["Decoy", "decoy", "is_decoy", "Is.Decoy"])
        pep_col = choose_first_column(work, ["Stripped.Sequence", "Sequence", "Peptide", "Modified.Sequence", "Precursor.Id"])
        mod_pep_col = choose_first_column(work, ["Modified.Sequence", "Precursor.Id", "Stripped.Sequence"])
        prot_col = choose_first_column(work, ["Protein.Ids", "Protein.Group", "Protein.Names", "Proteins"])
        score_col = choose_first_column(work, ["CScore", "Score", "score"])
        run_col = choose_first_column(work, ["Run", "File.Name", "FileName", "Raw.File"])
        precursor_col = choose_first_column(work, ["Precursor.Id", "PrecursorID", "Precursor"])

        out = pd.DataFrame()
        out["source_file"] = work[run_col].astype(str).map(lambda x: Path(x).name) if run_col else ""
        out["scan"] = np.nan
        out["spectrum_id"] = work[precursor_col].astype(str) if precursor_col else (
            out["source_file"].astype(str) + "::" + work.index.astype(str)
        )
        out["peptide"] = work[pep_col].astype(str).map(normalize_peptide_sequence) if pep_col else ""
        out["modified_peptide"] = work[mod_pep_col].astype(str) if mod_pep_col else out["peptide"]
        out["proteins"] = work[prot_col].astype(str) if prot_col else ""
        if decoy_col is None:
            out["label"] = 1.0
        else:
            ser = work[decoy_col]
            if ser.dtype == bool:
                out["label"] = ser.map(lambda x: -1.0 if bool(x) else 1.0)
            else:
                out["label"] = pd.to_numeric(ser, errors="coerce").fillna(0).astype(int).map(lambda x: -1.0 if x != 0 else 1.0)
        if score_col:
            out["score_engine"] = safe_float_series(work[score_col])
        else:
            pep_col_name = choose_first_column(work, ["PEP", "pep", "Pep"])

            sort_cols = [q_col]
            if pep_col_name:
                sort_cols.append(pep_col_name)

            # Sort properly by both columns
            sorted_df = work.sort_values(sort_cols, ascending=True, na_position='last')

            # Assign rank (0 = best)
            sorted_df["score_engine"] = -pd.Series(range(len(sorted_df)), index=sorted_df.index)

            # Map back to original dataframe
            out["score_engine"] = sorted_df["score_engine"].reindex(work.index)
        out["engine_q"] = safe_float_series(work[q_col])
        out["rank"] = np.nan
        out["row_id"] = [f"prec_{i:08d}" for i in range(len(out))]
        for col in work.columns:
            if col not in out.columns:
                out[col] = work[col]
        return out

    @staticmethod
    def _ensure_requested_tsv_report(
        requested_report_path: Path,
        actual_report_path: Path,
        raw_df: pd.DataFrame,
    ) -> Path:
        if requested_report_path.exists():
            return requested_report_path
        if requested_report_path.suffix == ".tsv" and actual_report_path.suffix == ".parquet":
            dataframe_to_tsv(raw_df, requested_report_path)
            return requested_report_path
        return actual_report_path

    def _base_cfg(self, ctx: SearchExecutionContext, engine_dir: Path, report_path: Path) -> DiannRunConfig:
        protease_cfg = PROTEASE_CONFIGS[ctx.general.protease]
        return DiannRunConfig(
            diann_bin=ctx.general.binaries.diann,
            raw_files=ctx.spectra,
            fasta_files=[ctx.combined_fasta.fasta_path],
            out_report=report_path,
            cfg_out=engine_dir / "diann_search.cfg",
            raw_dir=None,
            lib=Path(str(ctx.step.engine_params["lib"])) if "lib" in ctx.step.engine_params else None,
            out_lib=Path(str(ctx.step.engine_params["out_lib"])) if "out_lib" in ctx.step.engine_params else None,
            temp_dir=engine_dir / "temp",
            fasta_search=bool(ctx.step.engine_params.get("fasta_search", False)),
            predictor=bool(ctx.step.engine_params.get("predictor", False)),
            gen_spec_lib=bool(ctx.step.engine_params.get("gen_spec_lib", False)),
            threads=int(ctx.step.engine_params["threads"]) if "threads" in ctx.step.engine_params else None,
            qvalue=float(ctx.step.engine_params["qvalue"]) if "qvalue" in ctx.step.engine_params else None,
            verbose=int(ctx.step.engine_params["verbose"]) if "verbose" in ctx.step.engine_params else None,
            cut=ctx.step.engine_params.get("cut", protease_cfg.diann_cut),
            min_pep_len=(
                int(ctx.step.engine_params.get("min_pep_len", protease_cfg.min_len))
                if (ctx.step.engine_params.get("min_pep_len", protease_cfg.min_len) is not None)
                else None
            ),
            max_pep_len=(
                int(ctx.step.engine_params.get("max_pep_len", protease_cfg.max_len))
                if (ctx.step.engine_params.get("max_pep_len", protease_cfg.max_len) is not None)
                else None
            ),
            xic=int(ctx.step.engine_params.get("xic", 60)),
            report_decoys=True,
            extra_args=list(ctx.step.engine_params.get("extra_args", [])),
        )

    def _plan_execution(
        self,
        base_cfg: DiannRunConfig,
        engine_dir: Path,
    ) -> Tuple[Optional[DiannRunConfig], DiannRunConfig, dict[str, Path]]:
        raw_paths: dict[str, Path] = {}

        if not needs_separate_predicted_library_step(base_cfg):
            search_cfg = replace(
                base_cfg,
                fasta_search=False,
                predictor=False,
            )
            return None, search_cfg , raw_paths

        temp_root = base_cfg.temp_dir or (engine_dir / "temp")
        predicted_lib_dir = temp_root / "predicted_library"
        predicted_lib_dir.mkdir(parents=True, exist_ok=True)

        predicted_lib_template = predicted_lib_dir / "predicted_library.tsv"
        predicted_lib_path = predicted_library_path(predicted_lib_template)
        lib_generation_report = predicted_lib_dir / "predicted_library_build.tsv"

        library_generation_cfg = replace(
            base_cfg,
            raw_files=[],
            out_report=lib_generation_report,
            cfg_out=None,
            lib=None,
            out_lib=predicted_lib_template,
            temp_dir=predicted_lib_dir,
            fasta_search=True,
            predictor=base_cfg.predictor,
            gen_spec_lib=True,
            xic=None,
            report_decoys=False,
        )

        search_cfg = replace(
            base_cfg,
            lib=predicted_lib_path,
            fasta_search=False,
            predictor=False,
            cfg_out=engine_dir / "diann_search.cfg",
            temp_dir=temp_root,
            xic=base_cfg.xic,
        )

        raw_paths["predicted_lib"] = predicted_lib_path
        raw_paths["predicted_lib_report"] = lib_generation_report
        raw_paths["predicted_lib_template"] = predicted_lib_template
        return library_generation_cfg, search_cfg, raw_paths

    def run(self, ctx: SearchExecutionContext) -> SearchArtifacts:
        if ctx.step.rescorers:
            raise ValueError("DIA-NN rescoring is not implemented in this pipeline yet")

        engine_dir = ctx.step_dir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        report_path = engine_dir / "diann_report.tsv"

        base_cfg = self._base_cfg(ctx, engine_dir, report_path)
        library_generation_cfg, search_cfg, extra_raw_paths = self._plan_execution(base_cfg, engine_dir)

        if library_generation_cfg is not None:
            lib_cmd, lib_used_cfg = build_diann_cmd(library_generation_cfg)
            if not ctx.step.skip_engine:
                run_cmd(lib_cmd, dry_run=ctx.general.dry_run, log_path=ctx.log_path)
            if lib_used_cfg is not None:
                extra_raw_paths["predicted_lib_cfg"] = lib_used_cfg
            predicted_lib_report_path = resolve_optional_report_path(
                extra_raw_paths.get("predicted_lib_report", library_generation_cfg.out_report)
            )
            if predicted_lib_report_path is not None:
                extra_raw_paths["predicted_lib_report"] = predicted_lib_report_path

        search_cmd, used_cfg = build_diann_cmd(search_cfg)
        if not ctx.step.skip_engine:
            run_cmd(search_cmd, dry_run=ctx.general.dry_run, log_path=ctx.log_path)

        normalized_path = ctx.step_dir / "normalized" / "row_base_engine.tsv"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        if ctx.general.dry_run:
            empty = pd.DataFrame(
                columns=[
                    "row_id",
                    "source_file",
                    "scan",
                    "spectrum_id",
                    "peptide",
                    "modified_peptide",
                    "proteins",
                    "label",
                    "score_engine",
                    "engine_q",
                ]
            )
            dataframe_to_tsv(empty, normalized_path)
            raw_paths = {
                "cfg": used_cfg or engine_dir / "diann_search.cfg",
                "report": report_path,
                **extra_raw_paths,
            }
            return SearchArtifacts(engine_name=self.name, row_df=empty, normalized_path=normalized_path, raw_paths=raw_paths)

        actual_report_path = resolve_existing_report_path(report_path)
        raw_df = self._load_report(actual_report_path)
        compatible_report_path = self._ensure_requested_tsv_report(report_path, actual_report_path, raw_df)
        normalized = self._normalize(raw_df)
        dataframe_to_tsv(normalized, normalized_path)
        raw_paths = {
            "cfg": used_cfg or engine_dir / "diann_search.cfg",
            "report": compatible_report_path,
            "report_native": actual_report_path,
            **extra_raw_paths,
        }
        stats_path = stats_path_for_report(actual_report_path)
        if stats_path.exists():
            raw_paths["report_stats"] = stats_path
        return SearchArtifacts(engine_name=self.name, row_df=normalized, normalized_path=normalized_path, raw_paths=raw_paths)
