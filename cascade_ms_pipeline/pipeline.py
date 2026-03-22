from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .analysis_tables import (
    build_accepted_counts_at_alpha,
    build_entrapment_bounds,
    build_entrapment_bounds_by_length,
    build_identifications_vs_q,
    build_identifications_vs_q_by_length,
    build_score_survival,
    build_score_survival_by_length,
    pairwise_method_overlap,
)
from .cmd import capture_cmd
from .config import PipelineConfig, SearchStepConfig
from .engines import ENGINE_REGISTRY, SearchExecutionContext
from .fasta_groups import build_combined_fasta
from .fdr import FDRResult, run_fdr_method
from .plots import (
    plot_entrapment_bounds,
    plot_entrapment_bounds_by_length,
    plot_gamma_fits,
    plot_identifications_vs_q,
    plot_identifications_vs_q_by_length,
    plot_method_comparison,
    plot_score_distributions,
    plot_score_distributions_by_length,
    plot_score_survival,
    plot_score_survival_by_length,
)
from .protease import PROTEASE_CONFIGS
from .results import aggregate_to_peptide_level, annotate_groups_on_results, select_score_column
from .util import dataframe_to_tsv
from .rescorers import RESCORER_REGISTRY
from .trimming import trim_dda_spectra
from .util import ensure_dir, json_ready, write_json, write_text


class PipelineRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.report_dir = ensure_dir(cfg.general.report_dir)
        self.log_path = self.report_dir / "pipeline.log"
        self.trim_dirs_to_cleanup: List[Path] = []

    def snapshot_config(self) -> None:
        write_json(self.cfg.to_dict(), self.report_dir / "config.resolved.json")

    def capture_versions(self) -> None:
        rows = []
        binaries = self.cfg.general.binaries
        commands = {
            "python": [binaries.python, "-V"],
            "sage": [binaries.sage, "--version"],
            "diann": [binaries.diann, "--version"],
            "ms2rescore": [binaries.ms2rescore, "--version"],
        }
        for name, cmd in commands.items():
            try:
                output = capture_cmd(cmd)
            except Exception as exc:  # pragma: no cover - best effort only
                output = f"ERROR: {exc}"
            rows.append({"tool": name, "command": " ".join(cmd), "output": output})
        dataframe_to_tsv(pd.DataFrame(rows), self.report_dir / "software_versions.tsv")

    def run(self) -> None:
        self.snapshot_config()
        self.capture_versions()
        dataframe_to_tsv(
            pd.DataFrame({"input_spectrum": [str(x) for x in self.cfg.general.spectra]}),
            self.report_dir / "input_spectra.tsv",
        )

        current_spectra = list(self.cfg.general.spectra)
        step_summaries: List[Dict[str, object]] = []

        for idx, step in enumerate(self.cfg.searches, start=1):
            if not step.enabled:
                continue
            step_name = f"{idx:02d}_{step.name}"
            step_dir = ensure_dir(self.report_dir / "steps" / step_name)
            summary = self.run_step(step=step, step_dir=step_dir, spectra=current_spectra)
            step_summaries.append(summary)
            if summary.get("next_spectra"):
                current_spectra = [Path(str(x)) for x in summary["next_spectra"]]

        write_json(step_summaries, self.report_dir / "pipeline_summary.json")

        if not self.cfg.general.keep_intermediate_spectra:
            for trim_dir in self.trim_dirs_to_cleanup:
                if trim_dir.exists():
                    shutil.rmtree(trim_dir, ignore_errors=True)
            if self.trim_dirs_to_cleanup:
                write_text(
                    "Intermediate trimmed spectra were removed because general.keep_intermediate_spectra=false.\n"
                    "Per-step trim manifests and summaries remain in the report directory.\n",
                    self.report_dir / "trim_cleanup_note.txt",
                )

    def run_step(self, *, step: SearchStepConfig, step_dir: Path, spectra: List[Path]) -> Dict[str, object]:
        selected_groups = [self.cfg.fasta_group_map()[name] for name in step.fasta_groups]
        protease_cfg = PROTEASE_CONFIGS[self.cfg.general.protease]
        combined = build_combined_fasta(
            selected_groups,
            step_dir / "combined_fasta",
            protease_cfg=protease_cfg,
        )

        ctx = SearchExecutionContext(
            general=self.cfg.general,
            step=step,
            step_dir=step_dir,
            spectra=spectra,
            combined_fasta=combined,
            log_path=self.log_path,
        )

        engine = ENGINE_REGISTRY[step.engine_type]
        artifacts = engine.run(ctx)

        notes: List[str] = list(artifacts.notes)
        row_df = artifacts.row_df.copy()
        if not row_df.empty:
            row_df = annotate_groups_on_results(
                row_df,
                decoy_prefix=combined.effective_decoy_prefix,
                entrapment_groups=combined.entrapment_groups,
                entrapment_strategy=step.fdr.entrapment_strategy,
            )
            row_df["score_final"] = row_df.get("score_final", row_df["score_engine"])
            row_df["q_final"] = row_df.get("q_final", row_df["engine_q"])
            row_df["final_score_source"] = row_df.get("final_score_source", "engine")

        base_annotated_path = step_dir / "normalized" / "row_base_annotated.tsv"
        dataframe_to_tsv(row_df, base_annotated_path)

        current_df = row_df
        for rescorer_cfg in step.rescorers:
            rescorer = RESCORER_REGISTRY[rescorer_cfg.type]
            res_artifacts = rescorer.run(rescorer_cfg, ctx, artifacts, current_df)
            current_df = res_artifacts.merged_df
            current_df = annotate_groups_on_results(
                current_df,
                decoy_prefix=combined.effective_decoy_prefix,
                entrapment_groups=combined.entrapment_groups,
                entrapment_strategy=step.fdr.entrapment_strategy,
            )
            out_path = step_dir / "normalized" / f"row_after_{rescorer_cfg.type}.tsv"
            dataframe_to_tsv(current_df, out_path)
            notes.extend(res_artifacts.notes)
            if res_artifacts.merge_report is not None:
                notes.append(
                    f"{rescorer_cfg.type}: matched {res_artifacts.merge_report.matched_rows}/{res_artifacts.merge_report.base_rows} rows via {res_artifacts.merge_report.strategy}"
                )

        final_row_path = step_dir / "normalized" / "row_final.tsv"
        dataframe_to_tsv(current_df, final_row_path)

        if current_df.empty:
            write_text("No rows available (dry run or empty engine output).\n", step_dir / "notes.txt")
            return {
                "step": step.name,
                "engine": step.engine_type,
                "notes": notes + ["No downstream FDR analysis was run because the row table is empty."],
                "next_spectra": [str(x) for x in spectra],
            }

        peptide_df = aggregate_to_peptide_level(current_df)
        peptide_df = annotate_groups_on_results(
            peptide_df,
            decoy_prefix=combined.effective_decoy_prefix,
            entrapment_groups=combined.entrapment_groups,
            entrapment_strategy=step.fdr.entrapment_strategy,
        )
        score_col_row = select_score_column(current_df, step.fdr.score_source)
        score_col_pep = select_score_column(peptide_df, step.fdr.score_source)
        dataframe_to_tsv(peptide_df, step_dir / "normalized" / "peptide_final.tsv")

        self._run_level(
            step=step,
            level_name="psm",
            df=current_df,
            score_col=score_col_row,
            step_dir=step_dir,
            r_effective=combined.r_effective,
        )
        self._run_level(
            step=step,
            level_name="peptide",
            df=peptide_df,
            score_col=score_col_pep,
            step_dir=step_dir,
            r_effective=combined.r_effective,
        )

        next_spectra = spectra
        if step.trim.enabled:
            next_spectra = self._run_trimming(step=step, step_dir=step_dir, row_df=current_df, input_spectra=spectra)

        write_text("\n".join(notes) + "\n", step_dir / "notes.txt")
        return {
            "step": step.name,
            "engine": step.engine_type,
            "score_source": step.fdr.score_source,
            "row_score_column": score_col_row,
            "peptide_score_column": score_col_pep,
            "n_row_results": int(len(current_df)),
            "n_peptide_results": int(len(peptide_df)),
            "combined_fasta": str(combined.fasta_path),
            "notes": notes,
            "next_spectra": [str(x) for x in next_spectra],
        }

    def _run_level(
        self,
        *,
        step: SearchStepConfig,
        level_name: str,
        df: pd.DataFrame,
        score_col: str,
        step_dir: Path,
        r_effective: Optional[float],
    ) -> None:
        level_dir = ensure_dir(step_dir / "fdr" / level_name)
        plot_format = self.cfg.general.plot_format

        plot_score_distributions(
            df,
            score_col=score_col,
            out_path=level_dir / f"score_distribution.{plot_format}",
            title=f"{step.name} / {level_name} / score distributions ({score_col})",
        )
        plot_score_distributions_by_length(
            df,
            score_col=score_col,
            out_path=level_dir / f"score_distribution_by_length.{plot_format}",
            title=f"{step.name} / {level_name} / score distributions by peptide length ({score_col})",
        )

        score_survival_df = build_score_survival(df, score_col=score_col)
        score_survival_len_df = build_score_survival_by_length(df, score_col=score_col)
        dataframe_to_tsv(score_survival_df, level_dir / "score_survival.tsv")
        dataframe_to_tsv(score_survival_len_df, level_dir / "score_survival_by_length.tsv")
        plot_score_survival(
            score_survival_df,
            out_path=level_dir / f"score_survival.{plot_format}",
            title=f"{step.name} / {level_name} / cumulative score curves ({score_col})",
        )
        plot_score_survival_by_length(
            score_survival_len_df,
            out_path=level_dir / f"score_survival_by_length.{plot_format}",
            title=f"{step.name} / {level_name} / cumulative score curves by peptide length ({score_col})",
        )

        method_counts: Dict[str, pd.DataFrame] = {}
        accepted_by_method: Dict[str, pd.DataFrame] = {}

        for method in step.fdr.methods:
            method_dir = ensure_dir(level_dir / method)
            result = run_fdr_method(
                method,
                df,
                score_col=score_col,
                correction=step.fdr.correction,
                groupwalk_k=step.fdr.groupwalk_k,
                groupwalk_seed=step.fdr.groupwalk_seed,
                transferred_min_decoys=step.fdr.transferred_min_decoys,
                transferred_min_points=step.fdr.transferred_min_points,
                transferred_clip_min=step.fdr.transferred_clip_min,
            )
            result_df = result.df.copy()
            dataframe_to_tsv(result_df, method_dir / "q_values.tsv")
            for diag_name, diag_df in result.diagnostics.items():
                if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
                    dataframe_to_tsv(diag_df, method_dir / f"diagnostic_{diag_name}.tsv")

            counts_df = build_identifications_vs_q(result_df)
            counts_len_df = build_identifications_vs_q_by_length(result_df)
            dataframe_to_tsv(counts_df, method_dir / "identifications_vs_q.tsv")
            dataframe_to_tsv(counts_len_df, method_dir / "identifications_vs_q_by_length.tsv")
            method_counts[method] = counts_df
            accepted_by_method[method] = result_df

            plot_identifications_vs_q(
                counts_df,
                out_path=method_dir / f"identifications_vs_q.{plot_format}",
                title=f"{step.name} / {level_name} / {method} / accepted IDs vs q",
            )
            plot_identifications_vs_q_by_length(
                counts_len_df,
                out_path=method_dir / f"identifications_vs_q_by_length.{plot_format}",
                title=f"{step.name} / {level_name} / {method} / accepted IDs vs q by length",
            )

            if r_effective is not None and ("is_entrapment" in result_df.columns):
                bounds_df = build_entrapment_bounds(result_df, r_effective=r_effective)
                bounds_len_df = build_entrapment_bounds_by_length(result_df, r_effective=r_effective)
                dataframe_to_tsv(bounds_df, method_dir / "entrapment_bounds.tsv")
                dataframe_to_tsv(bounds_len_df, method_dir / "entrapment_bounds_by_length.tsv")
                plot_entrapment_bounds(
                    bounds_df,
                    out_path=method_dir / f"entrapment_bounds.{plot_format}",
                    title=f"{step.name} / {level_name} / {method} / entrapment bounds",
                )
                plot_entrapment_bounds_by_length(
                    bounds_len_df,
                    out_path=method_dir / f"entrapment_bounds_by_length.{plot_format}",
                    title=f"{step.name} / {level_name} / {method} / entrapment bounds by length",
                )

            if method == "transferred_subgroup" and "gamma_fit_points" in result.diagnostics:
                plot_gamma_fits(
                    result.diagnostics["gamma_fit_points"],
                    out_path=method_dir / f"gamma_fits.{plot_format}",
                    title=f"{step.name} / {level_name} / transferred subgroup gamma fits",
                )

            accepted_counts = build_accepted_counts_at_alpha(result_df, alphas=step.fdr.alpha_grid)
            dataframe_to_tsv(accepted_counts, method_dir / "accepted_counts_at_alpha.tsv")
            accepted_dir = ensure_dir(method_dir / "accepted")
            for alpha in step.fdr.alpha_grid:
                accepted = result_df[(result_df["label"] == 1) & (result_df["q_value"] <= float(alpha))].copy()
                accepted = accepted.sort_values(["q_value", score_col], ascending=[True, False], kind="mergesort")
                keep_cols = [
                    c
                    for c in [
                        "row_id",
                        "source_file",
                        "scan",
                        "spectrum_id",
                        "peptide",
                        "modified_peptide",
                        "proteins",
                        "group_name",
                        score_col,
                        "q_value",
                    ]
                    if c in accepted.columns
                ]
                dataframe_to_tsv(accepted[keep_cols], accepted_dir / f"accepted_alpha_{alpha:.4f}.tsv")

        plot_method_comparison(
            method_counts,
            out_path=level_dir / f"method_comparison.{plot_format}",
            title=f"{step.name} / {level_name} / method comparison",
        )

        overlap_alpha = 0.01 if 0.01 in set(step.fdr.alpha_grid) else step.fdr.alpha_grid[0]
        overlap_df = pairwise_method_overlap(accepted_by_method, alpha=overlap_alpha)
        dataframe_to_tsv(overlap_df, level_dir / f"method_overlap_alpha_{overlap_alpha:.4f}.tsv")

    def _run_trimming(
        self,
        *,
        step: SearchStepConfig,
        step_dir: Path,
        row_df: pd.DataFrame,
        input_spectra: List[Path],
    ) -> List[Path]:
        trim_dir = ensure_dir(step_dir / "trim")
        if self.cfg.general.acquisition != "dda":
            message = (
                "DIA trimming is not implemented in this pipeline yet; the requested trim step was skipped and the next step will reuse the same spectra.\n"
            )
            write_text(message, trim_dir / "NOT_IMPLEMENTED.txt")
            if step.trim.unsupported_action == "error":
                raise NotImplementedError(message)
            return input_spectra

        method_q_path = step_dir / "fdr" / "psm" / step.trim.method / "q_values.tsv"
        if not method_q_path.exists():
            raise FileNotFoundError(
                f"Cannot trim spectra because requested method output is missing: {method_q_path}"
            )
        accepted_df = pd.read_csv(method_q_path, sep="\t", low_memory=False)
        accepted_df = accepted_df[(accepted_df["label"] == 1) & (accepted_df["q_value"] <= float(step.trim.alpha))].copy()
        keep_cols = [c for c in ["row_id", "source_file", "scan", "spectrum_id", "peptide", "group_name", "q_value"] if c in accepted_df.columns]
        dataframe_to_tsv(accepted_df[keep_cols], trim_dir / "accepted_psms_for_trimming.tsv")
        trimmed_dir = trim_dir / "trimmed_spectra"
        trimmed_paths, summary_df = trim_dda_spectra(input_spectra, accepted_df, out_dir=trimmed_dir)
        self.trim_dirs_to_cleanup.append(trimmed_dir)
        return trimmed_paths



def run_pipeline(cfg: PipelineConfig) -> None:
    PipelineRunner(cfg).run()
