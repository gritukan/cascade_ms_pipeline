from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Any) -> Any:
    path = Path(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Any) -> Path:
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return path


def write_text(text: str, path: Any) -> Path:
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def basename(path_like: Any) -> str:
    return Path(str(path_like)).name


def stem(path_like: Any) -> str:
    return Path(str(path_like)).stem


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


_SCAN_RE = re.compile(r"(?:^|[ =])(scan|spectrum|index)=([0-9]+)(?:$|[ ;])", re.IGNORECASE)


def parse_scan_number(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    try:
        return int(text)
    except Exception:
        pass
    m = _SCAN_RE.search(text + " ")
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None


_FLANKING_RE = re.compile(r"^[A-Za-z]\.(.+)\.[A-Za-z]$")


def normalize_peptide_sequence(seq: Any) -> str:
    """Return a best-effort stripped peptide sequence.

    The function removes common flanking residues (K.PEPTIDE.R), strips anything
    that is not a letter, and uppercases the result.
    """
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return ""
    text = str(seq).strip()
    m = _FLANKING_RE.match(text)
    if m:
        text = m.group(1)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^\)]*\)", "", text)
    text = re.sub(r"[^A-Za-z]", "", text)
    return text.upper()


def peptide_length(seq: Any) -> Optional[int]:
    clean = normalize_peptide_sequence(seq)
    return len(clean) if clean else None


def split_proteins_field(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    elif "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in text.split() if p.strip()]
    return [p.split()[0] for p in parts if p]


def recursive_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_dir(path: Any) -> Path:
    path = Path(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataframe_to_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def stable_desc_order(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.argsort(-arr, kind="mergesort")


def monotone_q_from_fdr(fdr_estimates_sorted_desc: Sequence[float]) -> np.ndarray:
    arr = np.asarray(fdr_estimates_sorted_desc, dtype=float)
    arr = np.clip(arr, 0.0, np.inf)
    return np.minimum.accumulate(arr[::-1])[::-1]


def common_parent(paths: Sequence[Path]) -> Path:
    if not paths:
        return Path(".")
    resolved = [p.resolve() for p in paths]
    common = os.path.commonpath([str(p) for p in resolved])
    return Path(common)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if hasattr(value, "__dict__"):
        return json_ready(vars(value))
    return value


def q_threshold_grid(observed_q: Optional[Iterable[float]] = None, max_q: float = 0.1) -> np.ndarray:
    base = np.unique(np.concatenate(([0.0], np.logspace(-4, math.log10(max_q), 60), [max_q])))
    if observed_q is None:
        return base
    observed = np.asarray([float(x) for x in observed_q if pd.notna(x)], dtype=float)
    if observed.size == 0:
        return base
    observed = observed[(observed >= 0.0) & (observed <= max_q)]
    if observed.size > 300:
        quantiles = np.unique(np.quantile(observed, np.linspace(0, 1, 200)))
        return np.unique(np.concatenate([base, quantiles]))
    return np.unique(np.concatenate([base, observed]))


def unique_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
