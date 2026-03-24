from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from lxml import etree

from .indexed_mzml import CountingWriter, build_index_footer_prefix, indexed_root_tag, sha1_of_path
from .util import dataframe_to_tsv, local_name, parse_scan_number


@dataclass
class TrimStats:
    input_file: Path
    output_file: Path
    total_spectra: int
    total_ms2: int
    removed_ms2: int
    kept_spectra: int



def _open_xml_stream(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")



def _prune_element(elem) -> None:
    parent = elem.getparent()
    elem.clear()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]



def _spectrum_ms_level(elem) -> Optional[int]:
    for child in elem.iter():
        if local_name(child.tag) != "cvParam":
            continue
        acc = child.get("accession", "")
        name = child.get("name", "")
        if acc == "MS:1000511" or name == "ms level":
            try:
                return int(child.get("value"))
            except Exception:
                return None
    return None



def _should_remove_spectrum(elem, scans_to_remove: Set[int], native_ids_to_remove: Set[str]) -> bool:
    spec_id = elem.get("id", "")
    ms_level = _spectrum_ms_level(elem)
    if ms_level != 2:
        return False
    if spec_id in native_ids_to_remove:
        return True
    scan = parse_scan_number(spec_id)
    if scan is not None and scan in scans_to_remove:
        return True
    return False



def _count_kept_spectra(path: Path, scans_to_remove: Set[int], native_ids_to_remove: Set[str]) -> Tuple[int, int, int, int]:
    total = 0
    total_ms2 = 0
    removed_ms2 = 0
    with _open_xml_stream(path) as fh:
        context = etree.iterparse(fh, events=("end",), huge_tree=True)
        for _, elem in context:
            if local_name(elem.tag) != "spectrum":
                continue
            total += 1
            ms_level = _spectrum_ms_level(elem)
            if ms_level == 2:
                total_ms2 += 1
                if _should_remove_spectrum(elem, scans_to_remove, native_ids_to_remove):
                    removed_ms2 += 1
            _prune_element(elem)
    kept = total - removed_ms2
    return total, total_ms2, removed_ms2, kept



def trim_mzml_remove_ms2(
    input_path: Path,
    output_path: Path,
    *,
    scans_to_remove: Set[int],
    native_ids_to_remove: Optional[Set[str]] = None,
) -> TrimStats:
    native_ids_to_remove = native_ids_to_remove or set()
    total, total_ms2, removed_ms2, kept = _count_kept_spectra(input_path, scans_to_remove, native_ids_to_remove)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_xml_stream(input_path) as fh_in, output_path.open("wb") as raw_out:
        writer = CountingWriter(raw_out)
        writer.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        indexed_root_cm = None
        mzml_cm = None
        run_cm = None
        spectrum_list_cm = None
        mzml_nsmap = None
        mzml_tag = None
        spectrum_offsets = []

        with etree.xmlfile(writer, encoding="utf-8") as xf:
            context = etree.iterparse(fh_in, events=("start", "end"), huge_tree=True)
            for event, elem in context:
                lname = local_name(elem.tag)
                parent = elem.getparent()
                parent_lname = local_name(parent.tag) if parent is not None else None

                if event == "start":
                    if lname == "indexedmzML" and indexed_root_cm is None:
                        indexed_root_cm = xf.element(elem.tag, elem.attrib, nsmap=elem.nsmap)
                        indexed_root_cm.__enter__()
                    elif lname == "mzML":
                        if indexed_root_cm is None:
                            indexed_root_cm = xf.element(indexed_root_tag(elem.tag), {}, nsmap=elem.nsmap)
                            indexed_root_cm.__enter__()
                        if mzml_cm is None:
                            mzml_tag = elem.tag
                            mzml_nsmap = elem.nsmap
                            mzml_cm = xf.element(elem.tag, elem.attrib)
                            mzml_cm.__enter__()
                    elif lname == "run" and parent_lname == "mzML" and run_cm is None:
                        run_cm = xf.element(elem.tag, elem.attrib)
                        run_cm.__enter__()
                    elif lname == "spectrumList" and parent_lname == "run" and spectrum_list_cm is None:
                        attrib = dict(elem.attrib)
                        attrib["count"] = str(kept)
                        spectrum_list_cm = xf.element(elem.tag, attrib)
                        spectrum_list_cm.__enter__()
                    continue

                # end events below
                if lname == "spectrum" and parent_lname == "spectrumList":
                    if not _should_remove_spectrum(elem, scans_to_remove, native_ids_to_remove):
                        xf.flush()
                        spectrum_offsets.append((elem.get("id", ""), writer.tell()))
                        xf.write(elem)
                    _prune_element(elem)
                    continue

                if lname == "spectrumList" and parent_lname == "run":
                    if spectrum_list_cm is not None:
                        spectrum_list_cm.__exit__(None, None, None)
                        spectrum_list_cm = None
                    _prune_element(elem)
                    continue

                if parent_lname == "run" and lname != "spectrumList":
                    xf.write(elem)
                    _prune_element(elem)
                    continue

                if lname == "run" and parent_lname == "mzML":
                    if run_cm is not None:
                        run_cm.__exit__(None, None, None)
                        run_cm = None
                    _prune_element(elem)
                    continue

                if parent_lname == "mzML" and lname not in {"run", "indexList", "indexListOffset", "fileChecksum"}:
                    xf.write(elem)
                    _prune_element(elem)
                    continue

                if lname == "mzML":
                    if mzml_cm is not None:
                        mzml_cm.__exit__(None, None, None)
                        mzml_cm = None
                    _prune_element(elem)
                    continue

                if lname in {"indexedmzML", "indexList", "indexListOffset", "fileChecksum"}:
                    _prune_element(elem)
                    continue

                # For deeper descendants, keep the subtree intact until its direct parent
                # is written or skipped as a whole.
                if parent_lname not in {"mzML", "run", "spectrumList"}:
                    continue
                if parent is not None:
                    _prune_element(elem)

            xf.flush()
            writer.flush()
            index_list_offset = writer.tell()
            footer_prefix = build_index_footer_prefix(index_list_offset, spectrum_offsets)
            writer.write(footer_prefix)
            writer.flush()
            checksum = sha1_of_path(output_path)
            writer.write(checksum.encode("utf-8"))
            writer.write(b"</fileChecksum>")
            if indexed_root_cm is not None:
                indexed_root_cm.__exit__(None, None, None)
                indexed_root_cm = None

    return TrimStats(
        input_file=input_path,
        output_file=output_path,
        total_spectra=total,
        total_ms2=total_ms2,
        removed_ms2=removed_ms2,
        kept_spectra=kept,
    )



def trim_dda_spectra(
    spectra: List[Path],
    accepted_psms: pd.DataFrame,
    *,
    out_dir: Path,
) -> Tuple[List[Path], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = accepted_psms.copy()
    if "source_file" not in work.columns:
        raise KeyError("accepted_psms is missing source_file column required for DDA trimming")
    work["source_file"] = work["source_file"].astype(str).map(lambda x: Path(x).name)
    rows: List[Dict[str, object]] = []
    trimmed_paths: List[Path] = []
    for spec in spectra:
        basename = spec.name
        sub = work[work["source_file"] == basename].copy()
        scans = {int(x) for x in sub["scan"].dropna().astype(int).tolist()} if "scan" in sub.columns else set()
        native_ids = set(sub["spectrum_id"].dropna().astype(str).tolist()) if "spectrum_id" in sub.columns else set()
        out_name = basename[:-3] if basename.lower().endswith(".gz") else basename
        if not out_name.lower().endswith(".mzml"):
            out_name = spec.stem + "_trimmed.mzML"
        else:
            out_name = out_name[:-5] + "_trimmed.mzML"
        out_path = out_dir / out_name
        stats = trim_mzml_remove_ms2(spec, out_path, scans_to_remove=scans, native_ids_to_remove=native_ids)
        trimmed_paths.append(out_path)
        rows.append(
            {
                "input_file": str(stats.input_file),
                "output_file": str(stats.output_file),
                "total_spectra": stats.total_spectra,
                "total_ms2": stats.total_ms2,
                "removed_ms2": stats.removed_ms2,
                "kept_spectra": stats.kept_spectra,
                "n_requested_psms": int(len(sub)),
                "n_requested_scans": int(len(scans)),
            }
        )
    summary_df = pd.DataFrame(rows)
    dataframe_to_tsv(summary_df, out_dir / "trim_summary.tsv")
    return trimmed_paths, summary_df
