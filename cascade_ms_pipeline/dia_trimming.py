from __future__ import annotations

import base64
import gzip
import math
import re
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from lxml import etree

PROTON = 1.007276466812
H2O = 18.0105646837
C13_C12 = 1.0033548378

# residue masses in peptide chain
AA_MASS = {
    'A': 71.037113805,
    'R': 156.101111050,
    'N': 114.042927470,
    'D': 115.026943065,
    'C': 103.009184505,
    'E': 129.042593135,
    'Q': 128.058577540,
    'G': 57.021463735,
    'H': 137.058911875,
    'I': 113.084064015,
    'L': 113.084064015,
    'K': 128.094963050,
    'M': 131.040484645,
    'F': 147.068413945,
    'P': 97.052763875,
    'S': 87.032028435,
    'T': 101.047678505,
    'W': 186.079312980,
    'Y': 163.063328575,
    'V': 99.068413945,
}

UNIMOD_MASS = {
    '1': 42.010565,   # acetyl
    '4': 57.021464,   # carbamidomethyl
    '7': 0.984016,    # deamidated
    '21': 79.966331,  # phospho
    '35': 15.994915,  # oxidation
    '121': 114.042927,  # digly
}

NAMED_MOD_MASS = {
    'oxidation': 15.994915,
    'ox': 15.994915,
    'carbamidomethyl': 57.021464,
    'cam': 57.021464,
    'acetyl': 42.010565,
    'phospho': 79.966331,
    'deamidated': 0.984016,
    'digly': 114.042927,
}

FEATURE_RE = re.compile(r'^(?P<ion>[by])(?P<n>\d+)(?:-(?P<loss>H2O|NH3))?\^(?P<z>\d+)$', re.IGNORECASE)
FEATURE_RE_ALT = re.compile(r'^(?P<ion>[by])(?P<n>\d+)\^(?P<z>\d+)(?:-(?P<loss>H2O|NH3))?$', re.IGNORECASE)
PRECURSOR_RE = re.compile(r'^(?P<seq>.*?)(?P<z>\d+)$')
FLANK_RE = re.compile(r'^[A-Z]\.(.+)\.[A-Z]$')

CV_MS_LEVEL = 'MS:1000511'
CV_SCAN_START_TIME = 'MS:1000016'
CV_MZ_ARRAY = 'MS:1000514'
CV_INTENSITY_ARRAY = 'MS:1000515'
CV_32FLOAT = 'MS:1000521'
CV_64FLOAT = 'MS:1000523'
CV_32INT = 'MS:1000519'
CV_64INT = 'MS:1000522'
CV_ZLIB = 'MS:1000574'
CV_NO_COMPRESSION = 'MS:1000576'
CV_TARGET_MZ = 'MS:1000827'
CV_LOWER_OFFSET = 'MS:1000828'
CV_UPPER_OFFSET = 'MS:1000829'

NUMPRESS_ACCESSIONS = {
    'MS:1002312', 'MS:1002313', 'MS:1002314', 'MS:1002746', 'MS:1002747', 'MS:1002748'
}


class UnsupportedAnnotationError(ValueError):
    pass


@dataclass
class DiannDiaTrimConfig:
    q_threshold: float = 0.01
    q_col: Optional[str] = None
    run_col: Optional[str] = None
    precursor_col: str = 'Precursor.Id'
    xic_dir: Optional[Path] = None
    mz_tolerance_ppm: float = 20.0
    rt_max_diff_sec: Optional[float] = None
    ms1_isotopes: int = 3
    ms1_value_mode: str = 'envelope_total'  # envelope_total | monoisotopic
    remove_zero_peaks: bool = True
    zero_floor: float = 0.0


@dataclass
class SpectrumMeta:
    ordinal: int
    native_id: str
    ms_level: int
    rt_sec: float
    iso_low: Optional[float]
    iso_high: Optional[float]


@dataclass
class AssignedFeature:
    spectrum_ordinal: int
    target_mz: float
    intensity: float
    source: str
    rt_sec: float
    precursor_id: str


@dataclass
class TrimStats:
    input_file: Path
    output_file: Path
    total_spectra: int
    modified_spectra: int
    total_subtractions: int
    n_accepted_precursors: int
    n_xic_rows_total: int
    n_xic_rows_used: int
    n_xic_rows_skipped_annotation: int
    n_xic_rows_skipped_rt: int
    n_xic_rows_skipped_window: int


def _local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _open_maybe_gzip(path: Path, mode: str = 'rb'):
    if str(path).lower().endswith('.gz'):
        return gzip.open(path, mode)
    return path.open(mode)


def _guess_q_col(df: pd.DataFrame) -> str:
    for c in ['_q', 'Q.Value', 'Global.Q.Value', 'Lib.Q.Value', 'PG.Q.Value', 'Global.PG.Q.Value', 'Lib.PG.Q.Value', 'q', 'q_value']:
        if c in df.columns:
            return c
    raise KeyError(f'Could not find DIA-NN q-value column in report: {list(df.columns)[:80]}')


def _guess_run_col(df: pd.DataFrame) -> Optional[str]:
    for c in ['Run', 'File.Name', 'FileName', 'Raw.File', 'filename', 'source_file']:
        if c in df.columns:
            return c
    return None


def _basename_no_ext(pathlike: str | Path) -> str:
    p = Path(str(pathlike))
    name = p.name
    for suf in ('.mzML.gz', '.mzml.gz', '.mzML', '.mzml', '.raw', '.RAW'):
        if name.endswith(suf):
            return name[:-len(suf)]
    return p.stem


def guess_diann_xic_dir(report_path: Path) -> Path:
    base = report_path
    if base.suffix.lower() == '.parquet':
        stem = base.name[:-len('.parquet')]
    else:
        stem = base.stem
    return base.with_name(f'{stem}_xic')


def guess_diann_xic_path(report_path: Path, run_path: Path, *, xic_dir: Optional[Path] = None) -> Path:
    xdir = xic_dir if xic_dir is not None else guess_diann_xic_dir(report_path)
    return xdir / f'{_basename_no_ext(run_path)}.xic.parquet'


def load_diann_xic(xic_path: Path) -> pd.DataFrame:
    suf = xic_path.suffix.lower()
    if suf == '.parquet':
        try:
            return pd.read_parquet(xic_path)
        except Exception as exc:
            raise ImportError(
                f'Failed to read DIA-NN XIC parquet {xic_path}. Install a parquet backend such as pyarrow or fastparquet.'
            ) from exc
    if suf in {'.tsv', '.txt'}:
        return pd.read_csv(xic_path, sep='\t', low_memory=False)
    if suf == '.csv':
        return pd.read_csv(xic_path, low_memory=False)
    raise ValueError(f'Unsupported XIC file format: {xic_path}')


def select_accepted_precursors(report_df: pd.DataFrame, cfg: DiannDiaTrimConfig) -> Dict[str, Set[str]]:
    q_col = cfg.q_col or _guess_q_col(report_df)
    run_col = cfg.run_col or _guess_run_col(report_df)
    if cfg.precursor_col not in report_df.columns:
        raise KeyError(f'Report is missing precursor column {cfg.precursor_col!r}')
    work = report_df.copy()
    work[q_col] = pd.to_numeric(work[q_col], errors='coerce')
    work = work[work[q_col].notna() & (work[q_col] <= float(cfg.q_threshold))].copy()
    out: Dict[str, Set[str]] = defaultdict(set)
    if run_col is None:
        out['*all*'] = set(work[cfg.precursor_col].dropna().astype(str))
        return dict(out)
    for run_name, sub in work.groupby(run_col):
        out[_basename_no_ext(run_name)] = set(sub[cfg.precursor_col].dropna().astype(str))
    return dict(out)


def _strip_flanks(seq: str) -> str:
    m = FLANK_RE.match(seq)
    return m.group(1) if m else seq


def _mod_mass(token: str) -> float:
    tok = token.strip().strip('[]()')
    if not tok:
        return 0.0
    if tok.lower().startswith('unimod:'):
        key = tok.split(':', 1)[1]
        if key in UNIMOD_MASS:
            return UNIMOD_MASS[key]
        raise UnsupportedAnnotationError(f'Unsupported UniMod annotation: {tok}')
    low = tok.lower()
    if low in NAMED_MOD_MASS:
        return NAMED_MOD_MASS[low]
    if low.startswith('+') or low.startswith('-'):
        try:
            return float(low)
        except Exception as exc:
            raise UnsupportedAnnotationError(f'Unsupported numeric modification annotation: {tok}') from exc
    raise UnsupportedAnnotationError(f'Unsupported modification annotation: {tok}')


def parse_modified_sequence(seq: str) -> Tuple[List[Tuple[str, float]], float, float]:
    seq = _strip_flanks(str(seq))
    residues: List[Tuple[str, float]] = []
    nterm_shift = 0.0
    cterm_shift = 0.0
    i = 0
    pending_nterm = True
    while i < len(seq):
        ch = seq[i]
        if ch in '[(':
            close = ']' if ch == '[' else ')'
            j = seq.find(close, i + 1)
            if j < 0:
                raise UnsupportedAnnotationError(f'Unclosed modification in sequence: {seq}')
            shift = _mod_mass(seq[i+1:j])
            if pending_nterm and not residues:
                nterm_shift += shift
            elif residues:
                aa, cur = residues[-1]
                residues[-1] = (aa, cur + shift)
            else:
                nterm_shift += shift
            i = j + 1
            continue
        if ch.isalpha() and ch.isupper():
            if ch not in AA_MASS:
                raise UnsupportedAnnotationError(f'Unsupported residue {ch!r} in sequence: {seq}')
            residues.append((ch, 0.0))
            pending_nterm = False
            i += 1
            continue
        if ch in '.-_':
            i += 1
            continue
        raise UnsupportedAnnotationError(f'Unsupported sequence token {ch!r} in sequence: {seq}')
    return residues, nterm_shift, cterm_shift


def parse_precursor_id(precursor_id: str) -> Tuple[str, int]:
    m = PRECURSOR_RE.match(str(precursor_id))
    if not m:
        raise UnsupportedAnnotationError(f'Could not parse precursor ID: {precursor_id}')
    z = int(m.group('z'))
    seq = m.group('seq')
    if z <= 0:
        raise UnsupportedAnnotationError(f'Invalid precursor charge in {precursor_id}')
    return seq, z


def precursor_mz(precursor_id: str) -> float:
    seq, z = parse_precursor_id(precursor_id)
    residues, nshift, cshift = parse_modified_sequence(seq)
    neutral = sum(AA_MASS[aa] + shift for aa, shift in residues) + H2O + nshift + cshift
    return (neutral + z * PROTON) / z


def fragment_mz_from_feature(precursor_id: str, feature: str) -> float:
    seq, _ = parse_precursor_id(precursor_id)
    residues, nshift, cshift = parse_modified_sequence(seq)
    m = FEATURE_RE.match(feature) or FEATURE_RE_ALT.match(feature)
    if not m:
        raise UnsupportedAnnotationError(f'Unsupported feature annotation: {feature}')
    ion = m.group('ion').lower()
    n = int(m.group('n'))
    z = int(m.group('z'))
    loss = m.group('loss')
    if n <= 0 or z <= 0 or n > len(residues):
        raise UnsupportedAnnotationError(f'Invalid fragment annotation: {feature} for {precursor_id}')
    if ion == 'b':
        mass = sum(AA_MASS[aa] + shift for aa, shift in residues[:n]) + nshift
    else:
        mass = sum(AA_MASS[aa] + shift for aa, shift in residues[-n:]) + H2O + cshift
    if loss is not None:
        if loss.upper() == 'H2O':
            mass -= H2O
        elif loss.upper() == 'NH3':
            mass -= 17.026549101
    return (mass + z * PROTON) / z


def precursor_isotope_targets(precursor_id: str, n_isotopes: int, *, value_mode: str = 'envelope_total') -> List[Tuple[float, float]]:
    mz = precursor_mz(precursor_id)
    _, z = parse_precursor_id(precursor_id)
    if n_isotopes <= 1:
        return [(mz, 1.0)]
    seq, _ = parse_precursor_id(precursor_id)
    residues, nshift, cshift = parse_modified_sequence(seq)
    neutral = sum(AA_MASS[aa] + shift for aa, shift in residues) + H2O + nshift + cshift
    n_c = max(1.0, neutral / 111.1254 * 4.9384)
    lam = n_c * 0.0107
    probs = np.array([math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(n_isotopes)], dtype=float)
    probs /= probs.sum()
    if value_mode == 'monoisotopic':
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return [(mz + k * C13_C12 / z, float(probs[k])) for k in range(n_isotopes)]


def _parse_rt_sec_from_spectrum(elem) -> float:
    for child in elem.iter():
        if _local_name(child.tag) != 'cvParam':
            continue
        if child.get('accession') == CV_SCAN_START_TIME or child.get('name') == 'scan start time':
            value = float(child.get('value'))
            unit_acc = child.get('unitAccession', '')
            unit_name = child.get('unitName', '').lower()
            if unit_acc == 'UO:0000031' or unit_name.startswith('minute'):
                return value * 60.0
            return value
    return float('nan')


def _parse_ms_level(elem) -> int:
    for child in elem.iter():
        if _local_name(child.tag) != 'cvParam':
            continue
        if child.get('accession') == CV_MS_LEVEL or child.get('name') == 'ms level':
            return int(float(child.get('value')))
    return 0


def _parse_isolation_window(elem) -> Tuple[Optional[float], Optional[float]]:
    target = low_off = high_off = None
    for child in elem.iter():
        if _local_name(child.tag) != 'cvParam':
            continue
        acc = child.get('accession')
        if acc == CV_TARGET_MZ:
            target = float(child.get('value'))
        elif acc == CV_LOWER_OFFSET:
            low_off = float(child.get('value'))
        elif acc == CV_UPPER_OFFSET:
            high_off = float(child.get('value'))
    if target is None:
        return None, None
    low = target - (low_off if low_off is not None else 0.0)
    high = target + (high_off if high_off is not None else 0.0)
    return low, high


def build_spectrum_index(mzml_path: Path) -> List[SpectrumMeta]:
    metas: List[SpectrumMeta] = []
    with _open_maybe_gzip(mzml_path, 'rb') as fh:
        ctx = etree.iterparse(fh, events=('end',), huge_tree=True)
        ordinal = 0
        for _, elem in ctx:
            if _local_name(elem.tag) != 'spectrum':
                continue
            ms_level = _parse_ms_level(elem)
            rt_sec = _parse_rt_sec_from_spectrum(elem)
            low, high = (None, None)
            if ms_level == 2:
                low, high = _parse_isolation_window(elem)
            metas.append(SpectrumMeta(
                ordinal=ordinal,
                native_id=str(elem.get('id', f'index={ordinal}')),
                ms_level=ms_level,
                rt_sec=rt_sec,
                iso_low=low,
                iso_high=high,
            ))
            ordinal += 1
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return metas


def _nearest_index(sorted_values: np.ndarray, target: float) -> int:
    if sorted_values.size == 0:
        return -1
    pos = int(np.searchsorted(sorted_values, target))
    if pos <= 0:
        return 0
    if pos >= sorted_values.size:
        return int(sorted_values.size - 1)
    if abs(sorted_values[pos] - target) < abs(sorted_values[pos - 1] - target):
        return pos
    return pos - 1


def _group_ms2_windows(metas: List[SpectrumMeta]) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]]:
    ms1 = [m for m in metas if m.ms_level == 1 and math.isfinite(m.rt_sec)]
    ms1_rts = np.array([m.rt_sec for m in ms1], dtype=float)
    ms1_ord = np.array([m.ordinal for m in ms1], dtype=int)
    windows: Dict[Tuple[float, float], List[SpectrumMeta]] = defaultdict(list)
    for m in metas:
        if m.ms_level != 2 or not math.isfinite(m.rt_sec):
            continue
        if m.iso_low is None or m.iso_high is None:
            continue
        key = (round(float(m.iso_low), 5), round(float(m.iso_high), 5))
        windows[key].append(m)
    win_arrays: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    for key, arr in windows.items():
        arr = sorted(arr, key=lambda x: x.rt_sec)
        win_arrays[key] = (np.array([x.rt_sec for x in arr], dtype=float), np.array([x.ordinal for x in arr], dtype=int))
    return ms1_rts, ms1_ord, win_arrays


def _candidate_windows_for_precursor(win_arrays: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]], prec_mz: float) -> List[Tuple[float, float]]:
    hits = [key for key in win_arrays if key[0] - 1e-6 <= prec_mz <= key[1] + 1e-6]
    if not hits:
        return []
    hits.sort(key=lambda key: abs((key[0] + key[1]) / 2.0 - prec_mz))
    return hits


def assign_diann_xics_to_spectra(
    xic_df: pd.DataFrame,
    accepted_precursors: Set[str],
    metas: List[SpectrumMeta],
    cfg: DiannDiaTrimConfig,
) -> Tuple[Dict[int, List[AssignedFeature]], Dict[str, int]]:
    if not {'pr', 'feature', 'rt', 'value'}.issubset(xic_df.columns):
        raise KeyError(f'XIC file is missing required columns. Expected pr, feature, rt, value; got {list(xic_df.columns)[:50]}')
    ms1_rts, ms1_ord, win_arrays = _group_ms2_windows(metas)
    subtractions: Dict[int, List[AssignedFeature]] = defaultdict(list)
    precursor_cache: Dict[str, Tuple[float, List[Tuple[float, float]]]] = {}
    stats = {
        'n_xic_rows_total': int(len(xic_df)),
        'n_xic_rows_used': 0,
        'n_xic_rows_skipped_annotation': 0,
        'n_xic_rows_skipped_rt': 0,
        'n_xic_rows_skipped_window': 0,
    }
    rt_max = cfg.rt_max_diff_sec
    if rt_max is None:
        # no hard cap by default; DIA-NN xic rows should already be close to an apex window
        rt_max = float('inf')

    for row in xic_df.itertuples(index=False):
        pr = str(getattr(row, 'pr'))
        if pr not in accepted_precursors:
            continue
        feature = str(getattr(row, 'feature'))
        rt_sec = float(getattr(row, 'rt'))
        value = float(getattr(row, 'value'))
        if not math.isfinite(rt_sec) or not math.isfinite(value) or value <= 0:
            continue
        try:
            if pr not in precursor_cache:
                pmz = precursor_mz(pr)
                iso_targets = precursor_isotope_targets(pr, cfg.ms1_isotopes, value_mode=cfg.ms1_value_mode)
                precursor_cache[pr] = (pmz, iso_targets)
            prec_mz, iso_targets = precursor_cache[pr]
            if feature.lower() == 'ms1':
                idx = _nearest_index(ms1_rts, rt_sec)
                if idx < 0 or abs(ms1_rts[idx] - rt_sec) > rt_max:
                    stats['n_xic_rows_skipped_rt'] += 1
                    continue
                ord_ = int(ms1_ord[idx])
                for target_mz, frac in iso_targets:
                    subtractions[ord_].append(AssignedFeature(ord_, target_mz, value * frac, 'ms1', rt_sec, pr))
                stats['n_xic_rows_used'] += 1
            else:
                cand_wins = _candidate_windows_for_precursor(win_arrays, prec_mz)
                if not cand_wins:
                    stats['n_xic_rows_skipped_window'] += 1
                    continue
                target_mz = fragment_mz_from_feature(pr, feature)
                # choose nearest RT within best candidate window
                rt_arr, ord_arr = win_arrays[cand_wins[0]]
                idx = _nearest_index(rt_arr, rt_sec)
                if idx < 0 or abs(rt_arr[idx] - rt_sec) > rt_max:
                    stats['n_xic_rows_skipped_rt'] += 1
                    continue
                ord_ = int(ord_arr[idx])
                subtractions[ord_].append(AssignedFeature(ord_, target_mz, value, feature, rt_sec, pr))
                stats['n_xic_rows_used'] += 1
        except UnsupportedAnnotationError:
            stats['n_xic_rows_skipped_annotation'] += 1
            continue
    return dict(subtractions), stats


def _ppm_to_abs(mz: float, ppm: float) -> float:
    return mz * ppm * 1e-6


def _dtype_from_cv(cv_accessions: Set[str]) -> np.dtype:
    if CV_64FLOAT in cv_accessions:
        return np.float64
    if CV_32FLOAT in cv_accessions:
        return np.float32
    if CV_64INT in cv_accessions:
        return np.int64
    if CV_32INT in cv_accessions:
        return np.int32
    return np.float64


def _binary_array_meta(bin_array_elem) -> Tuple[str, np.dtype, bool]:
    cv_accessions = {child.get('accession') for child in bin_array_elem if _local_name(child.tag) == 'cvParam'}
    if cv_accessions & NUMPRESS_ACCESSIONS:
        raise NotImplementedError('MS-Numpress compressed mzML is not supported by the DIA trimmer')
    if CV_MZ_ARRAY in cv_accessions:
        arr_type = 'mz'
    elif CV_INTENSITY_ARRAY in cv_accessions:
        arr_type = 'intensity'
    else:
        arr_type = 'other'
    dtype = _dtype_from_cv(cv_accessions)
    compressed = CV_ZLIB in cv_accessions
    return arr_type, dtype, compressed


def _decode_binary_array(bin_array_elem) -> Tuple[np.ndarray, np.dtype, bool, str]:
    arr_type, dtype, compressed = _binary_array_meta(bin_array_elem)
    binary_elem = next((child for child in bin_array_elem if _local_name(child.tag) == 'binary'), None)
    if binary_elem is None:
        raise ValueError('binaryDataArray is missing <binary> child')
    data = base64.b64decode((binary_elem.text or '').encode('ascii'))
    if compressed:
        data = zlib.decompress(data)
    arr = np.frombuffer(data, dtype=dtype)
    return arr.copy(), dtype, compressed, arr_type


def _encode_binary_array(bin_array_elem, arr: np.ndarray, *, dtype: np.dtype, compressed: bool) -> None:
    binary_elem = next((child for child in bin_array_elem if _local_name(child.tag) == 'binary'), None)
    if binary_elem is None:
        raise ValueError('binaryDataArray is missing <binary> child')
    data = np.asarray(arr, dtype=dtype).tobytes(order='C')
    if compressed:
        data = zlib.compress(data)
    encoded = base64.b64encode(data).decode('ascii')
    binary_elem.text = encoded
    bin_array_elem.set('encodedLength', str(len(encoded)))


def _apply_subtractions_to_arrays(mz_arr: np.ndarray, inten_arr: np.ndarray, assignments: Sequence[AssignedFeature], ppm_tol: float, remove_zero: bool, zero_floor: float) -> Tuple[np.ndarray, np.ndarray, int]:
    if mz_arr.size == 0 or inten_arr.size == 0 or not assignments:
        return mz_arr, inten_arr, 0
    sub_per_index = defaultdict(float)
    for item in assignments:
        tol = _ppm_to_abs(item.target_mz, ppm_tol)
        pos = int(np.searchsorted(mz_arr, item.target_mz))
        candidates = []
        if pos < mz_arr.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best_idx = None
        best_diff = None
        for idx in candidates:
            diff = abs(float(mz_arr[idx]) - item.target_mz)
            if diff <= tol and (best_diff is None or diff < best_diff):
                best_idx = idx
                best_diff = diff
        if best_idx is not None:
            sub_per_index[best_idx] += float(item.intensity)
    n_hit = len(sub_per_index)
    if not n_hit:
        return mz_arr, inten_arr, 0
    inten = inten_arr.astype(np.float64, copy=True)
    for idx, amt in sub_per_index.items():
        inten[idx] = max(float(zero_floor), inten[idx] - amt)
    if remove_zero:
        mask = inten > float(zero_floor)
        return mz_arr[mask], inten[mask].astype(inten_arr.dtype, copy=False), n_hit
    return mz_arr, inten.astype(inten_arr.dtype, copy=False), n_hit


def _prune(elem) -> None:
    parent = elem.getparent()
    elem.clear()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]


def trim_mzml_with_subtractions(
    input_mzml: Path,
    output_mzml: Path,
    assignments_by_ordinal: Dict[int, List[AssignedFeature]],
    cfg: DiannDiaTrimConfig,
) -> TrimStats:
    output_mzml.parent.mkdir(parents=True, exist_ok=True)
    total_spectra = 0
    modified_spectra = 0
    total_subtractions = sum(len(v) for v in assignments_by_ordinal.values())
    with _open_maybe_gzip(input_mzml, 'rb') as fh_in, output_mzml.open('wb') as fh_out:
        fh_out.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        with etree.xmlfile(fh_out, encoding='utf-8') as xf:
            ctx = etree.iterparse(fh_in, events=('start', 'end'), huge_tree=True)
            root_ctx = run_ctx = spectrum_list_ctx = None
            ordinal = 0
            for event, elem in ctx:
                lname = _local_name(elem.tag)
                parent = elem.getparent()
                parent_name = _local_name(parent.tag) if parent is not None else None
                if event == 'start':
                    if lname == 'mzML' and root_ctx is None:
                        root_ctx = xf.element(elem.tag, elem.attrib, nsmap=elem.nsmap)
                        root_ctx.__enter__()
                    elif lname == 'run' and parent_name == 'mzML' and run_ctx is None:
                        run_ctx = xf.element(elem.tag, elem.attrib)
                        run_ctx.__enter__()
                    elif lname == 'spectrumList' and parent_name == 'run' and spectrum_list_ctx is None:
                        spectrum_list_ctx = xf.element(elem.tag, elem.attrib)
                        spectrum_list_ctx.__enter__()
                    continue
                # end events
                if lname == 'spectrum' and parent_name == 'spectrumList':
                    total_spectra += 1
                    if ordinal in assignments_by_ordinal:
                        mz_arr = inten_arr = None
                        mz_meta = inten_meta = None
                        for bda in [child for child in elem if _local_name(child.tag) == 'binaryDataArrayList']:
                            for arr_elem in bda:
                                if _local_name(arr_elem.tag) != 'binaryDataArray':
                                    continue
                                arr, dtype, compressed, arr_type = _decode_binary_array(arr_elem)
                                if arr_type == 'mz':
                                    mz_arr = arr
                                    mz_meta = (arr_elem, dtype, compressed)
                                elif arr_type == 'intensity':
                                    inten_arr = arr
                                    inten_meta = (arr_elem, dtype, compressed)
                        if mz_arr is not None and inten_arr is not None and mz_meta is not None and inten_meta is not None:
                            new_mz, new_int, n_hit = _apply_subtractions_to_arrays(
                                mz_arr.astype(np.float64, copy=False),
                                inten_arr.astype(np.float64, copy=False),
                                assignments_by_ordinal[ordinal],
                                ppm_tol=cfg.mz_tolerance_ppm,
                                remove_zero=cfg.remove_zero_peaks,
                                zero_floor=cfg.zero_floor,
                            )
                            if n_hit:
                                _encode_binary_array(mz_meta[0], np.asarray(new_mz), dtype=mz_meta[1], compressed=mz_meta[2])
                                _encode_binary_array(inten_meta[0], np.asarray(new_int), dtype=inten_meta[1], compressed=inten_meta[2])
                                elem.set('defaultArrayLength', str(len(new_mz)))
                                modified_spectra += 1
                    xf.write(elem)
                    ordinal += 1
                    _prune(elem)
                    continue
                if lname == 'spectrumList' and parent_name == 'run':
                    if spectrum_list_ctx is not None:
                        spectrum_list_ctx.__exit__(None, None, None)
                        spectrum_list_ctx = None
                    _prune(elem)
                    continue
                if parent_name == 'run' and lname != 'spectrumList':
                    xf.write(elem)
                    _prune(elem)
                    continue
                if lname == 'run' and parent_name == 'mzML':
                    if run_ctx is not None:
                        run_ctx.__exit__(None, None, None)
                        run_ctx = None
                    _prune(elem)
                    continue
                if parent_name == 'mzML' and lname not in {'run', 'indexList', 'indexListOffset', 'fileChecksum'}:
                    xf.write(elem)
                    _prune(elem)
                    continue
                if lname == 'mzML':
                    if root_ctx is not None:
                        root_ctx.__exit__(None, None, None)
                        root_ctx = None
                    _prune(elem)
                    continue
                if parent_name not in {'mzML', 'run', 'spectrumList'}:
                    continue
                if parent is not None:
                    _prune(elem)
    return TrimStats(
        input_file=input_mzml,
        output_file=output_mzml,
        total_spectra=total_spectra,
        modified_spectra=modified_spectra,
        total_subtractions=total_subtractions,
        n_accepted_precursors=0,
        n_xic_rows_total=0,
        n_xic_rows_used=0,
        n_xic_rows_skipped_annotation=0,
        n_xic_rows_skipped_rt=0,
        n_xic_rows_skipped_window=0,
    )


def trim_dia_run_from_xic_df(
    input_mzml: Path,
    output_mzml: Path,
    xic_df: pd.DataFrame,
    accepted_precursors: Set[str],
    cfg: DiannDiaTrimConfig,
) -> TrimStats:
    metas = build_spectrum_index(input_mzml)
    assignments, stats = assign_diann_xics_to_spectra(xic_df, accepted_precursors, metas, cfg)
    trim_stats = trim_mzml_with_subtractions(input_mzml, output_mzml, assignments, cfg)
    trim_stats.n_accepted_precursors = int(len(accepted_precursors))
    trim_stats.n_xic_rows_total = int(stats['n_xic_rows_total'])
    trim_stats.n_xic_rows_used = int(stats['n_xic_rows_used'])
    trim_stats.n_xic_rows_skipped_annotation = int(stats['n_xic_rows_skipped_annotation'])
    trim_stats.n_xic_rows_skipped_rt = int(stats['n_xic_rows_skipped_rt'])
    trim_stats.n_xic_rows_skipped_window = int(stats['n_xic_rows_skipped_window'])
    return trim_stats


def trim_dia_runs_from_report(
    report_path: Path,
    spectra: Sequence[Path],
    out_dir: Path,
    *,
    report_df: Optional[pd.DataFrame] = None,
    cfg: Optional[DiannDiaTrimConfig] = None,
) -> Tuple[List[Path], pd.DataFrame]:
    cfg = cfg or DiannDiaTrimConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    rdf = report_df if report_df is not None else (
        pd.read_parquet(report_path) if report_path.suffix.lower() == '.parquet' else pd.read_csv(report_path, sep='\t', low_memory=False)
    )
    accepted = select_accepted_precursors(rdf, cfg)
    rows = []
    outputs: List[Path] = []
    for spec in spectra:
        run_key = _basename_no_ext(spec)
        acc = accepted.get(run_key, accepted.get('*all*', set()))
        xic_path = guess_diann_xic_path(report_path, spec, xic_dir=cfg.xic_dir)
        xic_df = load_diann_xic(xic_path)
        out_name = f'{run_key}.trimmed.mzML'
        out_path = out_dir / out_name
        stats = trim_dia_run_from_xic_df(spec, out_path, xic_df, acc, cfg)
        rows.append({
            'input_file': str(spec),
            'output_file': str(out_path),
            'xic_file': str(xic_path),
            **stats.__dict__,
        })
        outputs.append(out_path)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'diann_trim_summary.tsv', sep='\t', index=False)
    return outputs, summary
