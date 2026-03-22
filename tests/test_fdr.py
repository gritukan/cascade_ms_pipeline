from __future__ import annotations

import numpy as np
import pandas as pd

from cascade_ms_pipeline.fdr import (
    compute_all_together,
    compute_group_walk,
    compute_per_group,
    compute_transferred_subgroup,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(12)],
            "label": [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1],
            "score": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "group_name": ["canonical", "canonical", "canonical", "novel", "novel", "novel", "novel", "entrapment", "entrapment", "entrapment", "entrapment", "entrapment"],
            "peptide": [
                "PEPTIDEA",
                "PEPTIDEB",
                "PEPTIDEC",
                "PEPTIDED",
                "PEPTIDEE",
                "PEPTIDEF",
                "PEPTIDEG",
                "PEPTIDEH",
                "PEPTIDEI",
                "PEPTIDEJ",
                "PEPTIDEK",
                "PEPTIDEL",
            ],
        }
    )


def test_all_together_qvalues_are_monotone_on_sorted_thresholds() -> None:
    df = _toy_df()
    result = compute_all_together(df, score_col="score", correction=1.0)
    diag = result.diagnostics["sorted_thresholds"]
    diffs = np.diff(diag["q_value"].to_numpy(dtype=float))
    assert np.all(diffs >= -1e-12)
    assert result.df["q_value"].between(0, 1).all()


def test_per_group_matches_all_together_for_single_group() -> None:
    df = _toy_df().copy()
    df["group_name"] = "single"
    all_result = compute_all_together(df, score_col="score", correction=1.0).df.sort_values("row_id")
    group_result = compute_per_group(df, score_col="score", correction=1.0).df.sort_values("row_id")
    assert np.allclose(all_result["q_value"].to_numpy(), group_result["q_value"].to_numpy(), equal_nan=True)


def test_transferred_subgroup_returns_finite_qvalues() -> None:
    df = _toy_df()
    result = compute_transferred_subgroup(df, score_col="score", correction=1.0, min_decoys=20)
    assert result.df["q_value"].notna().all()
    assert result.df["q_value"].between(0, 1).all()
    assert "gamma_fit_points" in result.diagnostics
    assert not result.diagnostics["gamma_fit_params"].empty


def test_group_walk_returns_finite_qvalues() -> None:
    df = _toy_df()
    result = compute_group_walk(df, score_col="score", correction=1.0, k=2, seed=1)
    assert result.df["q_value"].notna().all()
    assert result.df["q_value"].between(0, 1).all()
    assert not result.diagnostics["frontiers"].empty
