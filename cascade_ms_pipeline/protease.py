from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .fasta import read_fasta


@dataclass(frozen=True)
class ProteaseConfig:
    cleave_at: str
    restrict: str
    c_terminal: bool
    diann_cut: Optional[str]
    min_len: Optional[int] = None
    max_len: Optional[int] = None
    forbid_miscleavages: bool = False


PROTEASE_CONFIGS: Dict[str, ProteaseConfig] = {
    "trypsin": ProteaseConfig(cleave_at="KR", restrict="P", c_terminal=True, diann_cut="K*,R*,!*P"),
    "chymotrypsin": ProteaseConfig(cleave_at="FWY", restrict="P", c_terminal=True, diann_cut="F*,W*,Y*,!*P"),
    "pepsin": ProteaseConfig(cleave_at="FWYL", restrict="", c_terminal=True, diann_cut="F*,W*,Y*,L*"),
    "aspn": ProteaseConfig(cleave_at="D", restrict="", c_terminal=False, diann_cut="*D"),
    "gluc": ProteaseConfig(cleave_at="E", restrict="", c_terminal=True, diann_cut="E*"),
    "lysc": ProteaseConfig(cleave_at="K", restrict="", c_terminal=True, diann_cut="K*"),
    "lysn": ProteaseConfig(cleave_at="K", restrict="", c_terminal=False, diann_cut="*K"),
    "argc": ProteaseConfig(cleave_at="R", restrict="", c_terminal=True, diann_cut="R*"),
    "hla": ProteaseConfig(cleave_at="", restrict="", c_terminal=True, diann_cut=None, min_len=7, max_len=11, forbid_miscleavages=True),
}


def cleavage_sites(seq: str, cleave_at: str = "KR", restrict: str = "P", c_terminal: bool = True) -> List[int]:
    n = len(seq)
    cuts = [0]
    if c_terminal:
        for i, aa in enumerate(seq[:-1]):
            if aa in cleave_at and (not restrict or seq[i + 1] != restrict):
                cuts.append(i + 1)
        cuts.append(n)
    else:
        for i, aa in enumerate(seq[1:], start=1):
            if aa in cleave_at and (not restrict or seq[i - 1] != restrict):
                cuts.append(i)
        cuts.append(n)
    return sorted(set(cuts))


def digest_sequence(
    seq: str,
    *,
    enzyme_cleave_at: str,
    restrict: str,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    c_terminal: bool = True,
    cleave_at_special: Optional[str] = None,
) -> List[str]:
    if cleave_at_special == "$":
        return [seq] if min_len <= len(seq) <= max_len else []
    if enzyme_cleave_at == "":
        peps: List[str] = []
        n = len(seq)
        for i in range(n):
            for L in range(min_len, max_len + 1):
                j = i + L
                if j <= n:
                    peps.append(seq[i:j])
        return peps

    cuts = cleavage_sites(seq, cleave_at=enzyme_cleave_at, restrict=restrict, c_terminal=c_terminal)
    peps: List[str] = []
    for k in range(len(cuts) - 1):
        for m in range(k + 1, min(len(cuts), k + missed_cleavages + 2)):
            pep = seq[cuts[k] : cuts[m]]
            if min_len <= len(pep) <= max_len:
                peps.append(pep)
    return peps


def digest_fasta_unique_peptides(
    fasta_path: Path,
    *,
    enzyme_cleave_at: str = "KR",
    restrict: str = "P",
    missed_cleavages: int = 2,
    min_len: int = 7,
    max_len: int = 50,
    c_terminal: bool = True,
    collapse_il: bool = False,
    cleave_at_special: Optional[str] = None,
    include_record: Optional[callable] = None,
) -> Tuple[int, int]:
    uniq = set()
    total = 0
    for rec in read_fasta(fasta_path):
        if include_record is not None and not include_record(rec):
            continue
        seq = rec.sequence.replace("I", "L") if collapse_il else rec.sequence
        peps = digest_sequence(
            seq,
            enzyme_cleave_at=enzyme_cleave_at,
            restrict=restrict,
            missed_cleavages=missed_cleavages,
            min_len=min_len,
            max_len=max_len,
            c_terminal=c_terminal,
            cleave_at_special=cleave_at_special,
        )
        for pep in peps:
            uniq.add(pep)
            total += 1
    return len(uniq), total
