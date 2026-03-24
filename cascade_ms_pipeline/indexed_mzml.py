from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import hashlib

from lxml import etree


class CountingWriter:
    """Binary writer wrapper that exposes tell() for xmlfile output."""

    def __init__(self, fh):
        self._fh = fh

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self._fh.write(data)

    def tell(self) -> int:
        return int(self._fh.tell())

    def flush(self) -> None:
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._fh, name)


def namespace_uri(tag: str | None) -> Optional[str]:
    if not tag:
        return None
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None


def qname(ns_uri: Optional[str], local: str) -> str:
    return f"{{{ns_uri}}}{local}" if ns_uri else local


def indexed_root_tag(mzml_tag: str) -> str:
    return qname(namespace_uri(mzml_tag), "indexedmzML")


def build_index_list_element(
    ns_uri: Optional[str],
    spectrum_offsets: Sequence[Tuple[str, int]],
    chromatogram_offsets: Sequence[Tuple[str, int]] | None = None,
):
    chromatogram_offsets = list(chromatogram_offsets or [])
    n_indices = int(bool(spectrum_offsets)) + int(bool(chromatogram_offsets))
    index_list = etree.Element(qname(ns_uri, "indexList"), count=str(n_indices))
    if spectrum_offsets:
        spec_index = etree.SubElement(index_list, qname(ns_uri, "index"), name="spectrum")
        for native_id, offset in spectrum_offsets:
            off = etree.SubElement(spec_index, qname(ns_uri, "offset"), idRef=str(native_id))
            off.text = str(int(offset))
    if chromatogram_offsets:
        chrom_index = etree.SubElement(index_list, qname(ns_uri, "index"), name="chromatogram")
        for native_id, offset in chromatogram_offsets:
            off = etree.SubElement(chrom_index, qname(ns_uri, "offset"), idRef=str(native_id))
            off.text = str(int(offset))
    return index_list


def build_index_list_offset_element(ns_uri: Optional[str], offset: int):
    elem = etree.Element(qname(ns_uri, "indexListOffset"))
    elem.text = str(int(offset))
    return elem


def build_index_footer_prefix(
    index_list_offset: int,
    spectrum_offsets: Sequence[Tuple[str, int]],
    chromatogram_offsets: Sequence[Tuple[str, int]] | None = None,
) -> bytes:
    chromatogram_offsets = list(chromatogram_offsets or [])
    parts: List[str] = []
    n_indices = int(bool(spectrum_offsets)) + int(bool(chromatogram_offsets))
    parts.append(f'<indexList count="{n_indices}">')
    if spectrum_offsets:
        parts.append('<index name="spectrum">')
        for native_id, offset in spectrum_offsets:
            parts.append(f'<offset idRef="{escape(str(native_id), quote=True)}">{int(offset)}</offset>')
        parts.append('</index>')
    if chromatogram_offsets:
        parts.append('<index name="chromatogram">')
        for native_id, offset in chromatogram_offsets:
            parts.append(f'<offset idRef="{escape(str(native_id), quote=True)}">{int(offset)}</offset>')
        parts.append('</index>')
    parts.append('</indexList>')
    parts.append(f'<indexListOffset>{int(index_list_offset)}</indexListOffset>')
    parts.append('<fileChecksum>')
    return ''.join(parts).encode('utf-8')


def sha1_of_path(path: Path) -> str:
    h = hashlib.sha1()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
