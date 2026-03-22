from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fasta_groups import annotate_group_assignment
from .util import (
    dataframe_to_tsv,
    normalize_peptide_sequence,
    parse_scan_number,
    peptide_length,
    safe_float_series,
    stable_desc_order,
)


REQUIRED_BASE_COLUMNS = [
    "row_id",
    "source_file",
    "spectrum_id",
    "scan",
    "peptide",
    "modified_peptide",
    "proteins",
    "label",
    "score_engine",
    "engine_q",
]


@dataclass
class MergeReport:
    strategy: str
    matched_rows: int
    base_rows: int
    rescored_rows: int
    report_path: Path



def choose_first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        hit = lower_map.get(col.lower())
        if hit is not None:
            return hit
    return None



def ensure_numeric_label(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().any():
        return out
    text = series.astype(str).str.lower().str.strip()
    decoy_tokens = {"-1", "decoy", "true", "yes", "d"}
    target_tokens = {"1", "0", "target", "false", "no", "t"}
    return text.map(lambda x: -1 if x in decoy_tokens else (1 if x in target_tokens else 1))



def annotate_groups_on_results(
    df: pd.DataFrame,
    *,
    decoy_prefix: str,
    entrapment_groups: set[str],
    entrapment_strategy: str = "unambiguous",
) -> pd.DataFrame:
    work = df.copy()
    ann = work["proteins"].astype(str).apply(
        lambda value: annotate_group_assignment(
            value,
            decoy_prefix=decoy_prefix,
            entrapment_groups=entrapment_groups,
            entrapment_strategy=entrapment_strategy,
        )
    )
    work["matched_groups"] = ann.map(lambda x: ";".join(x.matched_groups))
    work["group_name"] = ann.map(lambda x: x.group_name)
    work["is_entrapment"] = ann.map(lambda x: x.is_entrapment)
    work["is_ambiguous_entrapment"] = ann.map(lambda x: x.is_ambiguous_entrapment)
    work["peptide_length"] = work["peptide"].apply(peptide_length)
    return work



def best_per_spectrum(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (winners, audit) keeping the best-scoring row per spectrum."""
    work = df.copy()
    work["score_engine"] = safe_float_series(work["score_engine"])
    if "rank" in work.columns and work["rank"].notna().any():
        try:
            rank_num = pd.to_numeric(work["rank"], errors="coerce")
            work = work[(rank_num.isna()) | (rank_num == 1)].copy()
        except Exception:
            pass
    key_cols = [c for c in ["source_file", "scan", "spectrum_id"] if c in work.columns]
    if not key_cols:
        work["__tmp_key"] = np.arange(len(work))
        key_cols = ["__tmp_key"]
    order = stable_desc_order(work["score_engine"].fillna(-np.inf).values)
    work_sorted = work.iloc[order].copy()
    winners = work_sorted.drop_duplicates(subset=key_cols, keep="first").copy()
    audit = pd.DataFrame(
        {
            "n_input_rows": [len(df)],
            "n_after_rank1_filter": [len(work)],
            "n_winner_rows": [len(winners)],
        }
    )
    return winners, audit



def aggregate_to_peptide_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level results to label-aware peptide representatives.

    This is intentionally simple and transparent: one representative row per
    (label, peptide) pair, chosen by the best score in the chosen score columns.
    It is *not* a picked-peptide procedure.
    """
    work = df.copy()
    work["peptide"] = work["peptide"].astype(str).map(normalize_peptide_sequence)
    work = work[work["peptide"].astype(str).str.len() > 0].copy()

    score_cols = [c for c in work.columns if c.startswith("score_")] or ["score_engine"]
    final_score_col = "score_final" if "score_final" in work.columns else score_cols[-1]
    work[final_score_col] = safe_float_series(work[final_score_col])

    key_cols = ["label", "peptide"]
    order = stable_desc_order(work[final_score_col].fillna(-np.inf).values)
    sorted_work = work.iloc[order].copy()
    rep = sorted_work.drop_duplicates(subset=key_cols, keep="first").copy()

    group_union = (
        work.groupby(key_cols)["matched_groups"]
        .apply(lambda s: ";".join(sorted({x for part in s.dropna().astype(str) for x in part.split(";") if x})))
        .reset_index(name="matched_groups_union")
    )
    rep = rep.merge(group_union, on=key_cols, how="left")
    rep["matched_groups"] = rep["matched_groups_union"].fillna(rep.get("matched_groups", ""))
    rep = rep.drop(columns=["matched_groups_union"], errors="ignore")

    rep["peptide_level_row_id"] = [f"pep_{i:08d}" for i in range(len(rep))]
    rep["row_id"] = rep["peptide_level_row_id"]
    rep["peptide_length"] = rep["peptide"].apply(peptide_length)
    rep["source_file"] = rep.get("source_file", pd.Series(["*aggregated*"] * len(rep)))
    rep["spectrum_id"] = rep.get("spectrum_id", pd.Series(["*aggregated*"] * len(rep)))
    return rep



def prepare_rescore_keys(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    src_col = choose_first_column(work, ["source_file", "Raw file", "raw_file", "filename", "file", "run", "Run"])
    if src_col is not None:
        work["source_file"] = work[src_col].astype(str).map(lambda x: Path(x).name)
    scan_col = choose_first_column(
        work,
        ["scan", "Scan", "ScanNr", "scannr", "scan_number", "SCAN_NUMBER", "spectrum_id", "SpecId", "PSMId"],
    )
    if scan_col is not None:
        work["scan"] = work[scan_col].apply(parse_scan_number)
    specid_col = choose_first_column(work, ["spectrum_id", "SpecId", "PSMId", "psm_id", "specid"])
    if specid_col is not None:
        work["psm_id"] = work[specid_col].astype(str)
    pep_col = choose_first_column(work, ["peptide", "_peptide", "Peptide", "sequence", "Sequence", "modified_peptide"])
    if pep_col is not None:
        work["peptide"] = work[pep_col].astype(str).map(normalize_peptide_sequence)
    mod_col = choose_first_column(work, ["modified_peptide", "peptide", "_peptide", "Peptide", "sequence"])
    if mod_col is not None:
        work["modified_peptide"] = work[mod_col].astype(str)
    return work



def merge_rescored_results(
    base_df: pd.DataFrame,
    rescored_df: pd.DataFrame,
    *,
    rescorer_name: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, MergeReport]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = prepare_rescore_keys(base_df)
    rescored = prepare_rescore_keys(rescored_df)

    candidate_keys: List[Tuple[str, List[str]]] = [
        ("psm_id", ["psm_id"]),
        ("source_file_scan_peptide", ["source_file", "scan", "peptide"]),
        ("source_file_scan_modified", ["source_file", "scan", "modified_peptide"]),
        ("scan_peptide", ["scan", "peptide"]),
    ]

    chosen_name = None
    chosen_keys = None
    best_merge = None
    best_match = -1

    for name, keys in candidate_keys:
        if not all(k in base.columns for k in keys):
            continue
        if not all(k in rescored.columns for k in keys):
            continue

        resc_small = rescored[keys + [c for c in ["_score", "_q", "label", "proteins", "_proteins"] if c in rescored.columns]].copy()
        resc_small = resc_small.drop_duplicates(subset=keys, keep="first")
        merged = base.merge(resc_small, on=keys, how="left", suffixes=("", f"__{rescorer_name}"))
        matched = int(merged["_score"].notna().sum() if "_score" in merged.columns else merged["_q"].notna().sum())
        if matched > best_match:
            best_match = matched
            best_merge = merged
            chosen_name = name
            chosen_keys = keys

    if best_merge is None:
        merged = base.copy()
        matched = 0
        chosen_name = "none"
    else:
        merged = best_merge
        matched = best_match

    score_col = f"score_{rescorer_name}"
    q_col = f"q_{rescorer_name}"
    if "_score" in merged.columns:
        merged[score_col] = safe_float_series(merged["_score"])
    else:
        merged[score_col] = np.nan
    if "_q" in merged.columns:
        merged[q_col] = safe_float_series(merged["_q"])
    else:
        merged[q_col] = np.nan

    for prot_col in ["_proteins", "proteins"]:
        if prot_col in merged.columns and "proteins" in merged.columns:
            empty_mask = merged["proteins"].astype(str).str.strip().eq("")
            merged.loc[empty_mask, "proteins"] = merged.loc[empty_mask, prot_col].astype(str)

    keep_cols = [c for c in merged.columns if not c.startswith("_") or c in {"_score", "_q", "_proteins"}]
    merged = merged[keep_cols].copy()

    report_df = pd.DataFrame(
        {
            "strategy": [chosen_name],
            "keys": [";".join(chosen_keys) if chosen_keys else ""],
            "matched_rows": [matched],
            "base_rows": [len(base)],
            "rescored_rows": [len(rescored)],
        }
    )
    report_path = out_dir / f"merge_report_{rescorer_name}.tsv"
    dataframe_to_tsv(report_df, report_path)

    if matched > 0:
        merged["score_final"] = merged[score_col].where(merged[score_col].notna(), merged.get("score_final", merged["score_engine"]))
        merged["q_final"] = merged[q_col].where(merged[q_col].notna(), merged.get("q_final", merged["engine_q"]))
        merged["final_score_source"] = rescorer_name
    else:
        if "score_final" not in merged.columns:
            merged["score_final"] = merged["score_engine"]
        if "q_final" not in merged.columns:
            merged["q_final"] = merged["engine_q"]
        merged["final_score_source"] = merged.get("final_score_source", "engine")

    return merged, MergeReport(
        strategy=chosen_name,
        matched_rows=matched,
        base_rows=len(base),
        rescored_rows=len(rescored),
        report_path=report_path,
    )



def select_score_column(df: pd.DataFrame, score_source: str) -> str:
    if score_source == "final":
        if "score_final" in df.columns:
            return "score_final"
        return "score_engine"
    if score_source == "engine":
        return "score_engine"
    if score_source.startswith("rescorer:"):
        name = score_source.split(":", 1)[1]
        col = f"score_{name}"
        if col not in df.columns:
            raise KeyError(f"Requested score source {score_source!r} but column {col!r} is missing")
        return col
    if score_source not in df.columns:
        raise KeyError(f"Requested score_source {score_source!r} but this column is absent")
    return score_source
