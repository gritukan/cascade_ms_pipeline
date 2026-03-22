from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .util import monotone_q_from_fdr, q_threshold_grid, safe_float_series, stable_desc_order


@dataclass
class FDRResult:
    method: str
    df: pd.DataFrame
    diagnostics: Dict[str, pd.DataFrame] = field(default_factory=dict)



def _prep(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = df.copy()
    work[score_col] = safe_float_series(work[score_col])
    work["label"] = safe_float_series(work["label"])
    work = work[work[score_col].notna() & work["label"].isin([1, -1])].copy()
    return work



def standard_tdc_qvalues(df: pd.DataFrame, *, score_col: str, correction: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = _prep(df, score_col)
    order = stable_desc_order(work[score_col].values)
    sorted_work = work.iloc[order].copy().reset_index(drop=False).rename(columns={"index": "__orig_index"})
    is_target = (sorted_work["label"] == 1).values
    cum_targets = np.cumsum(is_target)
    cum_decoys = np.cumsum(~is_target)
    fdr_est = (cum_decoys + float(correction)) / np.maximum(cum_targets, 1)
    qvals = monotone_q_from_fdr(fdr_est)
    sorted_work["raw_fdr_estimate"] = fdr_est
    sorted_work["q_value"] = np.clip(qvals, 0.0, 1.0)
    sorted_work["cum_targets"] = cum_targets
    sorted_work["cum_decoys"] = cum_decoys
    back = sorted_work.sort_values("__orig_index", kind="mergesort").drop(columns=["__orig_index"])
    return back, sorted_work



def compute_all_together(df: pd.DataFrame, *, score_col: str, correction: float = 1.0) -> FDRResult:
    out, diag = standard_tdc_qvalues(df, score_col=score_col, correction=correction)
    return FDRResult(method="all_together", df=out, diagnostics={"sorted_thresholds": diag})



def compute_per_group(df: pd.DataFrame, *, score_col: str, correction: float = 1.0, group_col: str = "group_name") -> FDRResult:
    work = _prep(df, score_col)
    frames: List[pd.DataFrame] = []
    diag_frames: List[pd.DataFrame] = []
    for group_name, group_df in work.groupby(group_col, sort=False):
        grp_out, grp_diag = standard_tdc_qvalues(group_df, score_col=score_col, correction=correction)
        grp_out["group_name"] = group_name
        grp_diag["group_name"] = group_name
        frames.append(grp_out)
        diag_frames.append(grp_diag)
    out = pd.concat(frames, ignore_index=True) if frames else work.assign(q_value=np.nan, raw_fdr_estimate=np.nan)
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    return FDRResult(method="per_group", df=out, diagnostics={"sorted_thresholds": diag})



def _fit_linear_gamma(scores: np.ndarray, gamma_obs: np.ndarray, weights: np.ndarray, clip_min: float, min_points: int) -> Tuple[np.ndarray, pd.DataFrame]:
    finite = np.isfinite(scores) & np.isfinite(gamma_obs) & np.isfinite(weights)
    scores = scores[finite]
    gamma_obs = gamma_obs[finite]
    weights = weights[finite]
    diag = pd.DataFrame({"score": scores, "gamma_observed": gamma_obs, "fit_weight": weights})
    if len(scores) < max(2, min_points) or np.unique(scores).size < 2:
        const = float(np.average(gamma_obs, weights=np.maximum(weights, 1.0))) if len(scores) else 0.0
        diag["gamma_fitted"] = np.clip(const, clip_min, 1.0)
        diag["fit_model"] = f"constant:{const:.6g}"
        return np.array([0.0, const], dtype=float), diag
    coeffs = np.polyfit(scores, gamma_obs, deg=1, w=np.sqrt(np.maximum(weights, 1.0)))
    fitted = np.clip(coeffs[0] * scores + coeffs[1], clip_min, 1.0)
    diag["gamma_fitted"] = fitted
    diag["fit_model"] = f"linear:{coeffs[0]:.6g}*score+{coeffs[1]:.6g}"
    return np.asarray(coeffs, dtype=float), diag



def compute_transferred_subgroup(
    df: pd.DataFrame,
    *,
    score_col: str,
    correction: float = 1.0,
    group_col: str = "group_name",
    min_decoys: int = 20,
    min_points: int = 8,
    clip_min: float = 1e-6,
) -> FDRResult:
    work = _prep(df, score_col)
    if work.empty:
        return FDRResult(method="transferred_subgroup", df=work.assign(q_value=np.nan))

    order = stable_desc_order(work[score_col].values)
    sorted_work = work.iloc[order].copy().reset_index(drop=False).rename(columns={"index": "__orig_index"})
    labels = (sorted_work["label"] == 1).values
    groups = sorted_work[group_col].astype(str).values
    scores = sorted_work[score_col].values.astype(float)

    cum_targets_all = np.cumsum(labels)
    cum_decoys_all = np.cumsum(~labels)
    global_fdr = (cum_decoys_all + float(correction)) / np.maximum(cum_targets_all, 1)

    sorted_work["raw_fdr_estimate"] = np.nan
    sorted_work["q_value"] = np.nan

    diag_groups: List[pd.DataFrame] = []
    fit_params: List[Dict[str, object]] = []

    unique_groups = pd.unique(groups)
    for group_name in unique_groups:
        in_group = groups == group_name
        cum_targets_g = np.cumsum(in_group & labels)
        cum_decoys_g = np.cumsum(in_group & (~labels))
        gamma_obs = cum_decoys_g / np.maximum(cum_decoys_all, 1)
        fit_mask = cum_decoys_all >= int(min_decoys)
        if int(fit_mask.sum()) < max(2, int(min_points)):
            fit_mask = cum_decoys_all > 0
        coeffs, diag = _fit_linear_gamma(
            scores[fit_mask],
            gamma_obs[fit_mask],
            cum_decoys_all[fit_mask].astype(float),
            clip_min=clip_min,
            min_points=min_points,
        )
        if diag.empty:
            coeffs = np.array([0.0, 1.0], dtype=float)
            diag = pd.DataFrame({"score": scores, "gamma_observed": gamma_obs, "fit_weight": cum_decoys_all.astype(float)})
            diag["gamma_fitted"] = 1.0
            diag["fit_model"] = "constant:1.0"
        gamma_hat_all = np.clip(coeffs[0] * scores + coeffs[1], clip_min, 1.0)
        raw_est_all_thresholds = (cum_targets_all / np.maximum(cum_targets_g, 1)) * gamma_hat_all * global_fdr
        q_all_thresholds = monotone_q_from_fdr(np.clip(raw_est_all_thresholds, 0.0, np.inf))
        # assign only to rows from this group, but using the group-wise q curve over *all* global thresholds
        sorted_work.loc[in_group, "raw_fdr_estimate"] = raw_est_all_thresholds[in_group]
        sorted_work.loc[in_group, "q_value"] = np.clip(q_all_thresholds[in_group], 0.0, 1.0)

        diag = diag.assign(group_name=group_name)
        diag_groups.append(diag)
        fit_params.append(
            {
                "group_name": group_name,
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "n_fit_points": int(fit_mask.sum()),
                "min_decoys": int(min_decoys),
            }
        )

    back = sorted_work.sort_values("__orig_index", kind="mergesort").drop(columns=["__orig_index"])
    diagnostics = {
        "gamma_fit_points": pd.concat(diag_groups, ignore_index=True) if diag_groups else pd.DataFrame(),
        "gamma_fit_params": pd.DataFrame(fit_params),
        "sorted_thresholds": sorted_work,
    }
    return FDRResult(method="transferred_subgroup", df=back, diagnostics=diagnostics)



def compute_group_walk(
    df: pd.DataFrame,
    *,
    score_col: str,
    correction: float = 1.0,
    group_col: str = "group_name",
    k: int = 40,
    seed: int = 1,
) -> FDRResult:
    """Python implementation of the R `groupwalk` package algorithm.

    Source algorithm adapted from the reference implementation described in the
    Group-walk paper and distributed in the `groupwalk` R package.
    """
    work = _prep(df, score_col)
    if work.empty:
        return FDRResult(method="group_walk", df=work.assign(q_value=np.nan))

    rng = np.random.default_rng(seed)
    ordered_inds = np.argsort(work[score_col].values, kind="mergesort")  # smallest -> largest
    ordered = work.iloc[ordered_inds].copy().reset_index(drop=False).rename(columns={"index": "__orig_index"})

    group_ids = list(pd.unique(ordered[group_col].astype(str)))
    groups_and_labels = pd.DataFrame(
        {
            "ordered_groups": ordered[group_col].astype(str).values,
            "ordered_labels": (ordered["label"] == 1).values,
        }
    )
    totals = [int((groups_and_labels["ordered_groups"] == g).sum()) for g in group_ids]
    labels_sorted: List[np.ndarray] = [
        groups_and_labels.loc[groups_and_labels["ordered_groups"] == g, "ordered_labels"].values.astype(bool)
        for g in group_ids
    ]

    starts = np.ones(len(group_ids), dtype=int)  # 1-based to mirror the reference implementation
    weights = np.zeros(len(group_ids), dtype=int)
    decoys_plus_one = float(sum((~groups_and_labels["ordered_labels"]).astype(int))) + float(correction)
    rejections = int(sum(groups_and_labels["ordered_labels"].astype(int)))
    q_vals = np.ones(len(ordered), dtype=float)
    q_val = 1.0
    switch = True
    frontier_rows: List[Dict[str, object]] = []

    # map group-local positions back to global row positions in ascending-score order
    group_global_positions: Dict[str, np.ndarray] = {
        g: np.where(groups_and_labels["ordered_groups"].values == g)[0] for g in group_ids
    }

    while np.any(starts <= np.asarray(totals)):
        frontier_rows.append({f"frontier_{g}": int(starts[i]) for i, g in enumerate(group_ids)})
        fdr_e = decoys_plus_one / max(rejections, 1)
        q_val = min(q_val, fdr_e)

        active = np.where(starts <= np.asarray(totals))[0]
        if np.any(starts[active] <= k):
            min_start = int(np.min(starts[active]))
            inds = np.where(starts[active] == min_start)[0]
        else:
            if switch:
                for gi in range(len(group_ids)):
                    if starts[gi] <= totals[gi]:
                        lo = starts[gi] - k - 1
                        hi = starts[gi] - 1
                        weights[gi] = int(np.sum(~labels_sorted[gi][lo:hi]))
                switch = False
            else:
                gi = index_update
                weights[gi] = int(
                    weights[gi]
                    - int(not labels_sorted[gi][starts[gi] - k - 2])
                    + int(not labels_sorted[gi][starts[gi] - 2])
                )
            active_weights = weights[active]
            inds = np.where(active_weights == np.max(active_weights))[0]

        if len(inds) == 1:
            randind = int(inds[0])
        else:
            randind = int(rng.choice(inds))
        index_update = int(active[randind])

        global_pos = group_global_positions[group_ids[index_update]][starts[index_update] - 1]
        q_vals[global_pos] = q_val
        label_at_update = bool(labels_sorted[index_update][starts[index_update] - 1])
        decoys_plus_one -= int(not label_at_update)
        rejections -= int(label_at_update)
        starts[index_update] += 1

    ordered["q_value"] = np.clip(q_vals, 0.0, 1.0)
    ordered["raw_fdr_estimate"] = ordered["q_value"]
    back = ordered.sort_values("__orig_index", kind="mergesort").drop(columns=["__orig_index"])
    frontier_df = pd.DataFrame(frontier_rows)
    return FDRResult(method="group_walk", df=back, diagnostics={"frontiers": frontier_df, "sorted_thresholds": ordered})


METHOD_DISPATCH = {
    "all_together": compute_all_together,
    "per_group": compute_per_group,
    "transferred_subgroup": compute_transferred_subgroup,
    "group_walk": compute_group_walk,
}



def run_fdr_method(
    method: str,
    df: pd.DataFrame,
    *,
    score_col: str,
    correction: float,
    group_col: str = "group_name",
    groupwalk_k: int = 40,
    groupwalk_seed: int = 1,
    transferred_min_decoys: int = 20,
    transferred_min_points: int = 8,
    transferred_clip_min: float = 1e-6,
) -> FDRResult:
    method = method.lower()
    if method == "all_together":
        return compute_all_together(df, score_col=score_col, correction=correction)
    if method == "per_group":
        return compute_per_group(df, score_col=score_col, correction=correction, group_col=group_col)
    if method == "transferred_subgroup":
        return compute_transferred_subgroup(
            df,
            score_col=score_col,
            correction=correction,
            group_col=group_col,
            min_decoys=transferred_min_decoys,
            min_points=transferred_min_points,
            clip_min=transferred_clip_min,
        )
    if method == "group_walk":
        return compute_group_walk(
            df,
            score_col=score_col,
            correction=correction,
            group_col=group_col,
            k=groupwalk_k,
            seed=groupwalk_seed,
        )
    raise KeyError(f"Unknown FDR method {method!r}")
