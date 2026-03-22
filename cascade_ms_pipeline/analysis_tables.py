from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .util import q_threshold_grid




def _score_thresholds(values: Sequence[float], max_points: int = 400) -> np.ndarray:
    arr = np.asarray([float(x) for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    uniq = np.unique(arr)
    if uniq.size <= max_points:
        return np.sort(uniq)
    qs = np.linspace(0.0, 1.0, max_points)
    return np.unique(np.quantile(arr, qs))


def build_score_survival(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str = "group_name",
    label_col: str = "label",
    thresholds: Optional[Sequence[float]] = None,
    max_points: int = 400,
) -> pd.DataFrame:
    work = df[df[score_col].notna()].copy()
    if thresholds is None:
        thresholds = _score_thresholds(work[score_col].tolist(), max_points=max_points)
    rows: List[Dict[str, object]] = []
    group_names = list(pd.unique(work[group_col].astype(str)))
    for thr in thresholds:
        accepted = work[work[score_col] >= thr]
        for label_value, label_name in [(1, "target"), (-1, "decoy")]:
            subset = accepted[accepted[label_col] == label_value]
            rows.append(
                {
                    "score_threshold": float(thr),
                    "group_name": "__all__",
                    "label_name": label_name,
                    "n_rows": int(len(subset)),
                }
            )
            counts = subset[group_col].astype(str).value_counts()
            for g in group_names:
                rows.append(
                    {
                        "score_threshold": float(thr),
                        "group_name": g,
                        "label_name": label_name,
                        "n_rows": int(counts.get(g, 0)),
                    }
                )
    return pd.DataFrame(rows)


def build_score_survival_by_length(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str = "group_name",
    label_col: str = "label",
    length_col: str = "peptide_length",
    thresholds: Optional[Sequence[float]] = None,
    max_points: int = 400,
) -> pd.DataFrame:
    work = df[df[score_col].notna() & df[length_col].notna()].copy()
    rows: List[pd.DataFrame] = []
    for length, sub in work.groupby(length_col, sort=True):
        local_thresholds = thresholds
        if local_thresholds is None:
            local_thresholds = _score_thresholds(sub[score_col].tolist(), max_points=max_points)
        part = build_score_survival(
            sub,
            score_col=score_col,
            group_col=group_col,
            label_col=label_col,
            thresholds=local_thresholds,
            max_points=max_points,
        )
        part.insert(0, "peptide_length", int(length))
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["peptide_length", "score_threshold", "group_name", "label_name", "n_rows"]
    )

def build_identifications_vs_q(
    df: pd.DataFrame,
    *,
    q_col: str = "q_value",
    group_col: str = "group_name",
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    targets = df[(df["label"] == 1) & df[q_col].notna()].copy()
    if thresholds is None:
        thresholds = q_threshold_grid(targets[q_col].tolist())
    rows: List[Dict[str, object]] = []
    group_names = list(pd.unique(targets[group_col].astype(str)))
    for thr in thresholds:
        accepted = targets[targets[q_col] <= thr]
        total = int(len(accepted))
        rows.append({"q_threshold": float(thr), "group_name": "__all__", "n_identifications": total})
        counts = accepted[group_col].astype(str).value_counts()
        for g in group_names:
            rows.append({"q_threshold": float(thr), "group_name": g, "n_identifications": int(counts.get(g, 0))})
    return pd.DataFrame(rows)



def build_identifications_vs_q_by_length(
    df: pd.DataFrame,
    *,
    q_col: str = "q_value",
    group_col: str = "group_name",
    length_col: str = "peptide_length",
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    targets = df[(df["label"] == 1) & df[q_col].notna() & df[length_col].notna()].copy()
    if thresholds is None:
        thresholds = q_threshold_grid(targets[q_col].tolist())
    rows: List[Dict[str, object]] = []
    for length, sub in targets.groupby(length_col, sort=True):
        counts_df = build_identifications_vs_q(sub, q_col=q_col, group_col=group_col, thresholds=thresholds)
        counts_df.insert(0, "peptide_length", int(length))
        rows.append(counts_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["peptide_length", "q_threshold", "group_name", "n_identifications"])



def build_entrapment_bounds(
    df: pd.DataFrame,
    *,
    q_col: str = "q_value",
    thresholds: Optional[Sequence[float]] = None,
    r_effective: Optional[float] = None,
) -> pd.DataFrame:
    targets = df[(df["label"] == 1) & df[q_col].notna()].copy()
    if thresholds is None:
        thresholds = q_threshold_grid(targets[q_col].tolist())
    r = float(r_effective) if (r_effective is not None and r_effective > 0) else 1.0
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        accepted = targets[targets[q_col] <= thr]
        n_ent = int(accepted["is_entrapment"].sum()) if "is_entrapment" in accepted.columns else 0
        n_amb = int(accepted["is_ambiguous_entrapment"].sum()) if "is_ambiguous_entrapment" in accepted.columns else 0
        n_non = int((~accepted["is_entrapment"]).sum()) if "is_entrapment" in accepted.columns else len(accepted)
        denom = n_ent + n_non
        lower = n_ent / denom if denom else 0.0
        upper = (n_ent * (1.0 + 1.0 / r)) / denom if denom else 0.0
        rows.append(
            {
                "q_threshold": float(thr),
                "n_target_original": n_non,
                "n_entrapment": n_ent,
                "n_ambiguous": n_amb,
                "lower_bound_fdp": float(lower),
                "combined_upper_bound_fdp": float(upper),
                "r_effective": float(r),
            }
        )
    return pd.DataFrame(rows)



def build_entrapment_bounds_by_length(
    df: pd.DataFrame,
    *,
    q_col: str = "q_value",
    length_col: str = "peptide_length",
    thresholds: Optional[Sequence[float]] = None,
    r_effective: Optional[float] = None,
) -> pd.DataFrame:
    targets = df[(df["label"] == 1) & df[q_col].notna() & df[length_col].notna()].copy()
    if thresholds is None:
        thresholds = q_threshold_grid(targets[q_col].tolist())
    rows: List[pd.DataFrame] = []
    for length, sub in targets.groupby(length_col, sort=True):
        part = build_entrapment_bounds(sub, q_col=q_col, thresholds=thresholds, r_effective=r_effective)
        part.insert(0, "peptide_length", int(length))
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



def build_accepted_counts_at_alpha(
    df: pd.DataFrame,
    *,
    q_col: str = "q_value",
    group_col: str = "group_name",
    alphas: Sequence[float],
) -> pd.DataFrame:
    targets = df[(df["label"] == 1) & df[q_col].notna()].copy()
    rows: List[Dict[str, object]] = []
    for alpha in alphas:
        accepted = targets[targets[q_col] <= alpha]
        rows.append({"alpha": float(alpha), "group_name": "__all__", "n_identifications": int(len(accepted))})
        counts = accepted[group_col].astype(str).value_counts()
        for g, n in counts.items():
            rows.append({"alpha": float(alpha), "group_name": g, "n_identifications": int(n)})
    return pd.DataFrame(rows)



def pairwise_method_overlap(
    accepted_by_method: Dict[str, pd.DataFrame],
    *,
    alpha: float,
    id_col: str = "row_id",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    keys = sorted(accepted_by_method)
    for a, b in combinations(keys, 2):
        set_a = set(accepted_by_method[a].loc[(accepted_by_method[a]["label"] == 1) & (accepted_by_method[a]["q_value"] <= alpha), id_col])
        set_b = set(accepted_by_method[b].loc[(accepted_by_method[b]["label"] == 1) & (accepted_by_method[b]["q_value"] <= alpha), id_col])
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        rows.append(
            {
                "alpha": float(alpha),
                "method_a": a,
                "method_b": b,
                "n_a": len(set_a),
                "n_b": len(set_b),
                "intersection": inter,
                "union": union,
                "jaccard": (inter / union) if union else 0.0,
            }
        )
    return pd.DataFrame(rows)
