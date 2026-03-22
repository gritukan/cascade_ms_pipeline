from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def plot_score_distributions(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str = "group_name",
    label_col: str = "label",
    out_path: Path,
    title: str,
) -> None:
    work = df[df[score_col].notna()].copy()
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = []
    for group_name in sorted(pd.unique(work[group_col].astype(str))):
        for label_value, label_name in [(1, "target"), (-1, "decoy")]:
            subset = work[(work[group_col].astype(str) == group_name) & (work[label_col] == label_value)][score_col].dropna()
            if subset.empty:
                continue
            counts, bins = np.histogram(subset.values.astype(float), bins=50)
            mids = (bins[:-1] + bins[1:]) / 2.0
            ax.plot(mids, np.maximum(counts, 1), label=f"{group_name} / {label_name} (N={len(subset)})")
            cats.append((group_name, label_name))
    if not cats:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)



def plot_score_distributions_by_length(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str = "group_name",
    label_col: str = "label",
    length_col: str = "peptide_length",
    out_path: Path,
    title: str,
) -> None:
    work = df[df[score_col].notna() & df[length_col].notna()].copy()
    lengths = sorted(int(x) for x in pd.unique(work[length_col]))
    if not lengths:
        return
    ncols = min(4, len(lengths))
    nrows = math.ceil(len(lengths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    groups = sorted(pd.unique(work[group_col].astype(str)))
    for idx, length in enumerate(lengths):
        ax = axes[idx // ncols][idx % ncols]
        sub = work[work[length_col] == length]
        for group_name in groups:
            for label_value, label_name in [(1, "target"), (-1, "decoy")]:
                subset = sub[(sub[group_col].astype(str) == group_name) & (sub[label_col] == label_value)][score_col].dropna()
                if subset.empty:
                    continue
                counts, bins = np.histogram(subset.values.astype(float), bins=40)
                mids = (bins[:-1] + bins[1:]) / 2.0
                ax.plot(mids, np.maximum(counts, 1), label=f"{group_name}/{label_name}")
        ax.set_title(f"Length {length}")
        ax.set_yscale("log")
        ax.set_xlabel(score_col)
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)
    for idx in range(len(lengths), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)





def plot_score_survival(curve_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if curve_df.empty:
        return
    work = curve_df[curve_df["group_name"] != "__all__"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name in sorted(pd.unique(work["group_name"].astype(str))):
        for label_name in ["target", "decoy"]:
            sub = work[(work["group_name"].astype(str) == group_name) & (work["label_name"] == label_name)]
            if sub.empty:
                continue
            ax.plot(sub["score_threshold"], np.maximum(sub["n_rows"], 1), label=f"{group_name} / {label_name}")
    ax.set_yscale("log")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Rows with score ≥ threshold")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_score_survival_by_length(curve_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if curve_df.empty:
        return
    lengths = sorted(int(x) for x in pd.unique(curve_df["peptide_length"]))
    if not lengths:
        return
    ncols = min(4, len(lengths))
    nrows = math.ceil(len(lengths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    groups = sorted(x for x in pd.unique(curve_df["group_name"].astype(str)) if x != "__all__")
    for idx, length in enumerate(lengths):
        ax = axes[idx // ncols][idx % ncols]
        sub = curve_df[curve_df["peptide_length"] == length]
        for group_name in groups:
            for label_name in ["target", "decoy"]:
                gsub = sub[(sub["group_name"].astype(str) == group_name) & (sub["label_name"] == label_name)]
                if gsub.empty:
                    continue
                ax.plot(gsub["score_threshold"], np.maximum(gsub["n_rows"], 1), label=f"{group_name}/{label_name}")
        ax.set_yscale("log")
        ax.set_title(f"Length {length}")
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("Rows with score ≥ threshold")
        ax.legend(fontsize=6)
    for idx in range(len(lengths), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_identifications_vs_q(counts_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if counts_df.empty:
        return
    work = counts_df[counts_df["group_name"] != "__all__"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name in sorted(pd.unique(work["group_name"].astype(str))):
        sub = work[work["group_name"].astype(str) == group_name]
        ax.plot(sub["q_threshold"], sub["n_identifications"], label=group_name)
    total = counts_df[counts_df["group_name"] == "__all__"]
    if not total.empty:
        ax.plot(total["q_threshold"], total["n_identifications"], label="__all__", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("q-value threshold")
    ax.set_ylabel("Accepted target identifications")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)



def plot_identifications_vs_q_by_length(counts_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if counts_df.empty:
        return
    lengths = sorted(int(x) for x in pd.unique(counts_df["peptide_length"]))
    if not lengths:
        return
    ncols = min(4, len(lengths))
    nrows = math.ceil(len(lengths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    groups = sorted(x for x in pd.unique(counts_df["group_name"].astype(str)) if x != "__all__")
    for idx, length in enumerate(lengths):
        ax = axes[idx // ncols][idx % ncols]
        sub = counts_df[counts_df["peptide_length"] == length]
        for group_name in groups:
            gsub = sub[sub["group_name"].astype(str) == group_name]
            ax.plot(gsub["q_threshold"], gsub["n_identifications"], label=group_name)
        total = sub[sub["group_name"] == "__all__"]
        if not total.empty:
            ax.plot(total["q_threshold"], total["n_identifications"], label="__all__", linewidth=2)
        ax.set_xscale("log")
        ax.set_title(f"Length {length}")
        ax.set_xlabel("q-value threshold")
        ax.set_ylabel("Accepted IDs")
        ax.legend(fontsize=6)
    for idx in range(len(lengths), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_entrapment_bounds(bounds_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if bounds_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bounds_df["q_threshold"], bounds_df["lower_bound_fdp"], label="lower entrapment bound")
    ax.plot(bounds_df["q_threshold"], bounds_df["combined_upper_bound_fdp"], label="combined upper bound")
    ax.plot(bounds_df["q_threshold"], bounds_df["q_threshold"], label="y=x", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q-value threshold")
    ax.set_ylabel("Estimated FDP")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)



def plot_entrapment_bounds_by_length(bounds_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if bounds_df.empty:
        return
    lengths = sorted(int(x) for x in pd.unique(bounds_df["peptide_length"]))
    if not lengths:
        return
    ncols = min(4, len(lengths))
    nrows = math.ceil(len(lengths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, length in enumerate(lengths):
        ax = axes[idx // ncols][idx % ncols]
        sub = bounds_df[bounds_df["peptide_length"] == length]
        ax.plot(sub["q_threshold"], sub["lower_bound_fdp"], label="lower")
        ax.plot(sub["q_threshold"], sub["combined_upper_bound_fdp"], label="upper")
        ax.plot(sub["q_threshold"], sub["q_threshold"], label="y=x", linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Length {length}")
        ax.set_xlabel("q-value threshold")
        ax.set_ylabel("Estimated FDP")
        ax.legend(fontsize=6)
    for idx in range(len(lengths), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_gamma_fits(diag_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if diag_df.empty or "group_name" not in diag_df.columns:
        return
    groups = sorted(pd.unique(diag_df["group_name"].astype(str)))
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, group_name in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        sub = diag_df[diag_df["group_name"].astype(str) == group_name].copy()
        if sub.empty:
            continue
        ax.scatter(sub["score"], sub["gamma_observed"], s=8, alpha=0.5, label="observed")
        if "gamma_fitted" in sub.columns:
            order = np.argsort(sub["score"].values)
            ax.plot(sub.iloc[order]["score"], sub.iloc[order]["gamma_fitted"], label="fit")
        ax.set_title(group_name)
        ax.set_xlabel("score")
        ax.set_ylabel("gamma")
        ax.legend(fontsize=6)
    for idx in range(len(groups), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_method_comparison(counts_by_method: Dict[str, pd.DataFrame], *, out_path: Path, title: str) -> None:
    if not counts_by_method:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, counts_df in sorted(counts_by_method.items()):
        sub = counts_df[counts_df["group_name"] == "__all__"]
        if sub.empty:
            continue
        ax.plot(sub["q_threshold"], sub["n_identifications"], label=method)
    ax.set_xscale("log")
    ax.set_xlabel("q-value threshold")
    ax.set_ylabel("Accepted target identifications")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)
