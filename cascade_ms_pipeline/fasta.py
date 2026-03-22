from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import re


@dataclass
class FastaRecord:
    header: str
    sequence: str


def read_fasta(path: Path) -> Iterator[FastaRecord]:
    header: Optional[str] = None
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, sequence="".join(seq_parts).upper())
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(re.sub(r"\s+", "", line))
        if header is not None:
            yield FastaRecord(header=header, sequence="".join(seq_parts).upper())


def write_fasta(records: Iterable[FastaRecord], path: Path, wrap: int = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(f">{rec.header}\n")
            seq = rec.sequence
            for i in range(0, len(seq), wrap):
                fh.write(seq[i : i + wrap] + "\n")
    return path


def parse_accession(header: str) -> str:
    return header.split()[0]
