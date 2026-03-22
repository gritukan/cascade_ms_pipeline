from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..cmd import run_cmd
from ..protease import PROTEASE_CONFIGS
from ..results import best_per_spectrum, choose_first_column, ensure_numeric_label
from ..util import dataframe_to_tsv, normalize_peptide_sequence, parse_scan_number, recursive_update, safe_float_series
from .base import SearchArtifacts, SearchEngine, SearchExecutionContext


DEFAULT_SAGE_CONFIG: Dict[str, Any] = {
    "database": {
        "bucket_size": 8192,
        "enzyme": {
            "missed_cleavages": 2,
            "min_len": 7,
            "max_len": 50,
            "cleave_at": "KR",
            "restrict": "P",
            "c_terminal": True,
            "semi_enzymatic": False,
        },
        "peptide_min_mass": 500.0,
        "peptide_max_mass": 5000.0,
        "ion_kinds": ["b", "y"],
        "min_ion_index": 2,
        "max_variable_mods": 0,
        "static_mods": {},
        "variable_mods": {},
        "decoy_tag": "rev_",
        "generate_decoys": True,
        "fasta": "REPLACED_BY_CLI",
    },
    "deisotope": True,
    "chimera": False,
    "max_fragment_charge": 2,
    "report_psms": 200,
    "precursor_tol": {"ppm": [-10, 10]},
    "fragment_tol": {"ppm": [-10, 10]},
    "isotope_errors": [-1, 3],
    "score_type": "SageHyperScore",
}


class SageEngine(SearchEngine):
    name = "sage"

    def _build_config(self, ctx: SearchExecutionContext) -> Path:
        params = dict(ctx.step.engine_params)
        template_path = params.pop("config_path", None)
        overrides = params.pop("config_overrides", {})
        config_out = ctx.step_dir / "engine" / "sage_config.used.json"
        config_out.parent.mkdir(parents=True, exist_ok=True)

        if template_path:
            cfg = json.loads(Path(str(template_path)).read_text(encoding="utf-8"))
        else:
            cfg = json.loads(json.dumps(DEFAULT_SAGE_CONFIG))

        protease_cfg = PROTEASE_CONFIGS[ctx.general.protease]
        db = cfg.setdefault("database", {})
        enz = db.setdefault("enzyme", {})
        enz["cleave_at"] = protease_cfg.cleave_at
        enz["restrict"] = protease_cfg.restrict
        enz["c_terminal"] = protease_cfg.c_terminal
        if protease_cfg.min_len is not None:
            enz["min_len"] = protease_cfg.min_len
        if protease_cfg.max_len is not None:
            enz["max_len"] = protease_cfg.max_len
        if protease_cfg.forbid_miscleavages:
            enz["missed_cleavages"] = 0

        if ctx.general.fragmentation == "etd":
            db["ion_kinds"] = ["c", "z"]

        db["generate_decoys"] = ctx.combined_fasta.generate_decoys
        db["decoy_tag"] = ctx.combined_fasta.effective_decoy_prefix

        recursive_update(cfg, overrides)
        config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return config_out

    @staticmethod
    def _load_results(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if "label" not in df.columns:
            is_decoy_col = choose_first_column(df, ["is_decoy", "IsDecoy", "decoy"])
            if is_decoy_col is not None:
                df["label"] = df[is_decoy_col].map(lambda x: -1 if bool(x) else 1)
        df["label"] = ensure_numeric_label(df["label"])
        return df

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        score_col = choose_first_column(work, ["sage_discriminant_score", "hyperscore", "score"])
        if score_col is None:
            raise KeyError("Could not find a Sage score column (expected sage_discriminant_score or hyperscore)")
        q_col = choose_first_column(work, ["spectrum_q", "q", "q_value"])
        pep_col = choose_first_column(work, ["stripped_peptide", "peptide"])
        mod_pep_col = choose_first_column(work, ["peptide", "modified_peptide", "stripped_peptide"])
        prot_col = choose_first_column(work, ["proteins", "protein", "protein_ids"])
        file_col = choose_first_column(work, ["filename", "file", "raw_file"])
        scan_col = choose_first_column(work, ["scannr", "scan", "scan_number"])
        specid_col = choose_first_column(work, ["title", "spectrum_id", "spectrum_name"])
        rank_col = choose_first_column(work, ["rank"])

        out = pd.DataFrame()
        out["source_file"] = work[file_col].astype(str).map(lambda x: Path(x).name) if file_col else ""
        out["scan"] = work[scan_col].apply(parse_scan_number) if scan_col else np.nan
        out["spectrum_id"] = work[specid_col].astype(str) if specid_col else (
            out["source_file"].astype(str) + ":" + out["scan"].astype(str)
        )
        out["peptide"] = work[pep_col].astype(str).map(normalize_peptide_sequence) if pep_col else ""
        out["modified_peptide"] = work[mod_pep_col].astype(str) if mod_pep_col else out["peptide"]
        out["proteins"] = work[prot_col].astype(str) if prot_col else ""
        out["label"] = ensure_numeric_label(work["label"]).astype(float)
        out["score_engine"] = safe_float_series(work[score_col])
        out["engine_q"] = safe_float_series(work[q_col]) if q_col else np.nan
        out["rank"] = pd.to_numeric(work[rank_col], errors="coerce") if rank_col else np.nan
        out["charge"] = safe_float_series(work["charge"]) if "charge" in work.columns else np.nan
        out["matched_peaks"] = safe_float_series(work["matched_peaks"]) if "matched_peaks" in work.columns else np.nan
        out["longest_b"] = safe_float_series(work["longest_b"]) if "longest_b" in work.columns else np.nan
        out["longest_y"] = safe_float_series(work["longest_y"]) if "longest_y" in work.columns else np.nan
        out["row_id"] = [f"psm_{i:08d}" for i in range(len(out))]

        # Preserve all original columns for transparency by prefixing where needed.
        for col in work.columns:
            if col not in out.columns:
                out[col] = work[col]
        return out

    def run(self, ctx: SearchExecutionContext) -> SearchArtifacts:
        engine_dir = ctx.step_dir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        sage_cfg = self._build_config(ctx)
        results_path = engine_dir / "results.sage.tsv"

        extra_args = list(ctx.step.engine_params.get("extra_args", []))
        write_pin = bool(ctx.step.rescorers) or bool(ctx.step.engine_params.get("write_pin", False))
        cmd = [
            ctx.general.binaries.sage,
            str(sage_cfg),
            "-o",
            str(engine_dir),
            "-f",
            str(ctx.combined_fasta.fasta_path),
        ]
        if write_pin:
            cmd.append("--write-pin")
        cmd.extend(extra_args)
        cmd.extend([str(p) for p in ctx.spectra])
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
            return SearchArtifacts(engine_name=self.name, row_df=empty, normalized_path=normalized_path, raw_paths={"config": sage_cfg})

        raw_df = self._load_results(results_path)
        normalized = self._normalize(raw_df)
        winners, audit = best_per_spectrum(normalized)
        dataframe_to_tsv(winners, normalized_path)
        audit_path = ctx.step_dir / "normalized" / "winner_selection_audit.tsv"
        dataframe_to_tsv(audit, audit_path)
        raw_all_path = ctx.step_dir / "normalized" / "row_all_candidates_engine.tsv"
        dataframe_to_tsv(normalized, raw_all_path)
        notes = []
        if len(normalized) != len(winners):
            notes.append(
                f"Collapsed {len(normalized)} engine rows to {len(winners)} best-per-spectrum winners before downstream FDR analysis"
            )
        return SearchArtifacts(
            engine_name=self.name,
            row_df=winners,
            normalized_path=normalized_path,
            raw_paths={"config": sage_cfg, "results": results_path, "all_candidates": raw_all_path, "audit": audit_path},
            notes=notes,
        )
