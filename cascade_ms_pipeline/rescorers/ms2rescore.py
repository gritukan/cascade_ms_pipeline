from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..cmd import run_cmd
from ..results import merge_rescored_results, prepare_rescore_keys
from ..util import common_parent, safe_float_series
from .base import RescoreArtifacts, Rescorer


class MS2RescoreRescorer(Rescorer):
    name = "ms2rescore"

    @staticmethod
    def _load_psms(psms_tsv: Path) -> pd.DataFrame:
        df = pd.read_csv(psms_tsv, sep="\t", low_memory=False)
        q_col = None
        for c in ["q_value", "qvalue", "q-value", "qval", "psm_qvalue", "mokapot q-value", "mokapot q_value"]:
            if c in df.columns:
                q_col = c
                break
        if q_col is None:
            for c in df.columns:
                if c.lower().replace("_", "").replace("-", "") in {"qvalue", "qval"}:
                    q_col = c
                    break
        if q_col is not None:
            df["_q"] = safe_float_series(df[q_col])
        else:
            df["_q"] = np.nan

        if "label" in df.columns:
            df["label"] = safe_float_series(df["label"])
        elif "is_decoy" in df.columns:
            df["label"] = df["is_decoy"].map(lambda x: -1 if bool(x) else 1)
        elif "isDecoy" in df.columns:
            df["label"] = df["isDecoy"].map(lambda x: -1 if bool(x) else 1)
        else:
            df["label"] = 1

        score_col = None
        for c in ["score", "mokapot score", "mokapot_score", "posterior_error_prob", "posterior_error_probability"]:
            if c in df.columns:
                score_col = c
                break
        df["_score"] = safe_float_series(df[score_col]) if score_col is not None else np.nan
        return df

    def run(self, cfg, ctx, base_artifacts, current_df):
        if base_artifacts.engine_name != "sage":
            raise ValueError("MS2Rescore integration is currently implemented only for Sage searches")

        out_dir = ctx.step_dir / "rescore" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_dir / "rescore"

        spectrum_path = cfg.params.get("spectra_path")
        if spectrum_path is None:
            spectrum_path = common_parent(ctx.spectra)
        else:
            spectrum_path = Path(str(spectrum_path))

        psm_input = base_artifacts.raw_paths.get("results", base_artifacts.normalized_path)
        cmd = [
            ctx.general.binaries.ms2rescore,
            "-p",
            str(psm_input),
            "-t",
            "sage",
            "-s",
            str(spectrum_path),
            "-f",
            str(ctx.combined_fasta.fasta_path),
            "-o",
            str(out_prefix),
        ]
        if "config_path" in cfg.params:
            cmd.extend(["-c", str(cfg.params["config_path"])])
        if "processes" in cfg.params:
            cmd.extend(["-n", str(int(cfg.params["processes"]))])
        cmd.extend([str(x) for x in cfg.params.get("extra_args", [])])

        run_cmd(cmd, dry_run=ctx.general.dry_run, log_path=ctx.log_path)
        psms_tsv = Path(str(out_prefix) + ".psms.tsv")
        if ctx.general.dry_run:
            merged = current_df.copy()
            merged["score_final"] = merged.get("score_final", merged["score_engine"])
            merged["q_final"] = merged.get("q_final", merged["engine_q"])
            merged["final_score_source"] = merged.get("final_score_source", "engine")
            return RescoreArtifacts(name=self.name, merged_df=merged, raw_paths={"psms": psms_tsv}, notes=["dry_run"]) 

        rescored = self._load_psms(psms_tsv)
        merged, merge_report = merge_rescored_results(current_df, rescored, rescorer_name=self.name, out_dir=out_dir)
        return RescoreArtifacts(
            name=self.name,
            merged_df=merged,
            raw_paths={"psms": psms_tsv},
            merge_report=merge_report,
            notes=[f"Merged MS2Rescore results using strategy: {merge_report.strategy}"],
        )
