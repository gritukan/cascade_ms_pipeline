from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..cmd import run_cmd
from ..protease import PROTEASE_CONFIGS
from ..results import choose_first_column, ensure_numeric_label
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
    extra_args: List[str] = field(default_factory=list)



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



def build_diann_cmd(cfg: DiannRunConfig) -> Tuple[List[str], Optional[Path]]:
    cmd: List[str] = [cfg.diann_bin]
    used_cfg: Optional[Path] = None
    if cfg.cfg_out is not None:
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
    cmd.append("--report-decoys")
    cmd.extend([str(x) for x in cfg.extra_args])
    return cmd, used_cfg


class DiannEngine(SearchEngine):
    name = "diann"

    @staticmethod
    def _load_report(path: Path) -> pd.DataFrame:
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
        score_col = choose_first_column(work, ["CScore", "Score", "score", "Mass.Evidence"])
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
        out["score_engine"] = safe_float_series(work[score_col]) if score_col else (-safe_float_series(work[q_col]))
        out["engine_q"] = safe_float_series(work[q_col])
        out["rank"] = np.nan
        out["row_id"] = [f"prec_{i:08d}" for i in range(len(out))]
        for col in work.columns:
            if col not in out.columns:
                out[col] = work[col]
        return out

    def run(self, ctx: SearchExecutionContext) -> SearchArtifacts:
        if ctx.step.rescorers:
            raise ValueError("DIA-NN rescoring is not implemented in this pipeline yet")
        engine_dir = ctx.step_dir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        protease_cfg = PROTEASE_CONFIGS[ctx.general.protease]
        report_path = engine_dir / "diann_report.tsv"
        cfg = DiannRunConfig(
            diann_bin=ctx.general.binaries.diann,
            raw_files=ctx.spectra,
            fasta_files=[ctx.combined_fasta.fasta_path],
            out_report=report_path,
            cfg_out=engine_dir / "diann.cfg",
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
            min_pep_len=int(ctx.step.engine_params.get("min_pep_len", protease_cfg.min_len)) if (ctx.step.engine_params.get("min_pep_len", protease_cfg.min_len) is not None) else None,
            max_pep_len=int(ctx.step.engine_params.get("max_pep_len", protease_cfg.max_len)) if (ctx.step.engine_params.get("max_pep_len", protease_cfg.max_len) is not None) else None,
            extra_args=list(ctx.step.engine_params.get("extra_args", [])),
        )
        cmd, used_cfg = build_diann_cmd(cfg)
        run_cmd(cmd, dry_run=ctx.general.dry_run, log_path=ctx.log_path)

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
            raw_paths = {"cfg": used_cfg or engine_dir / "diann.cfg", "report": report_path}
            return SearchArtifacts(engine_name=self.name, row_df=empty, normalized_path=normalized_path, raw_paths=raw_paths)

        raw_df = self._load_report(report_path)
        normalized = self._normalize(raw_df)
        dataframe_to_tsv(normalized, normalized_path)
        raw_paths = {"cfg": used_cfg or engine_dir / "diann.cfg", "report": report_path}
        return SearchArtifacts(engine_name=self.name, row_df=normalized, normalized_path=normalized_path, raw_paths=raw_paths)
