from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .config import FastaGroupConfig
from .fasta import FastaRecord, parse_accession, read_fasta, write_fasta
from .protease import ProteaseConfig, digest_fasta_unique_peptides
from .util import dataframe_to_tsv, sha256_file, split_proteins_field, unique_preserve_order


GROUP_TOKEN = "grp="


@dataclass
class CombinedFastaInfo:
    fasta_path: Path
    manifest_path: Path
    selected_groups: List[str]
    supplies_decoys: bool
    effective_decoy_prefix: str
    generate_decoys: bool
    entrapment_groups: Set[str]
    r_effective: Optional[float]


@dataclass
class GroupAnnotation:
    matched_groups: List[str]
    group_name: str
    is_entrapment: bool
    is_ambiguous_entrapment: bool



def _rewrite_accession(accession: str, group_name: str, decoy_prefix: str, supplies_decoys: bool) -> str:
    if supplies_decoys and accession.startswith(decoy_prefix):
        tail = accession[len(decoy_prefix) :]
        return f"{decoy_prefix}{GROUP_TOKEN}{group_name}|{tail}"
    return f"{GROUP_TOKEN}{group_name}|{accession}"



def build_combined_fasta(
    selected_groups: Sequence[FastaGroupConfig],
    out_dir: Path,
    *,
    engine_generated_decoy_prefix: str = "rev_",
    protease_cfg: Optional[ProteaseConfig] = None,
    missed_cleavages: int = 2,
    min_len: int = 7,
    max_len: int = 50,
    collapse_il: bool = False,
) -> CombinedFastaInfo:
    """Concatenate selected FASTA groups into a single search FASTA.

    Each accession is rewritten to carry a stable group token while preserving the
    search engine's decoy-prefix semantics.
    """
    if not selected_groups:
        raise ValueError("No FASTA groups selected")

    supplies_values = {g.supplies_decoys for g in selected_groups}
    if len(supplies_values) != 1:
        raise ValueError(
            "Selected FASTA groups mix supplied-decoy and engine-generated-decoy modes; this is not supported in one combined search"
        )
    supplies_decoys = next(iter(supplies_values))

    decoy_prefixes = {g.decoy_prefix for g in selected_groups if g.supplies_decoys}
    if len(decoy_prefixes) > 1:
        raise ValueError("Selected FASTA groups use different supplied decoy prefixes; please harmonize them")

    effective_decoy_prefix = next(iter(decoy_prefixes)) if supplies_decoys else engine_generated_decoy_prefix
    generate_decoys = not supplies_decoys

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = out_dir / "combined_search.fasta"
    manifest_path = out_dir / "combined_fasta_manifest.tsv"

    rows: List[Dict[str, object]] = []
    records: List[FastaRecord] = []

    def is_decoy_header(header: str, group: FastaGroupConfig) -> bool:
        acc = parse_accession(header)
        return group.supplies_decoys and acc.startswith(group.decoy_prefix)

    entrapment_groups = {g.name for g in selected_groups if g.is_entrapment}

    for group in selected_groups:
        n_total = 0
        n_target = 0
        n_decoy = 0
        for rec in read_fasta(group.path):
            n_total += 1
            acc = parse_accession(rec.header)
            decoy = is_decoy_header(rec.header, group)
            if decoy:
                n_decoy += 1
            else:
                n_target += 1
            parts = rec.header.split(maxsplit=1)
            parts[0] = _rewrite_accession(acc, group.name, group.decoy_prefix, group.supplies_decoys)
            new_header = parts[0] + ((" " + parts[1]) if len(parts) > 1 else "")
            records.append(FastaRecord(header=new_header, sequence=rec.sequence))

        rows.append(
            {
                "group_name": group.name,
                "path": str(group.path),
                "sha256": sha256_file(group.path),
                "supplies_decoys": group.supplies_decoys,
                "decoy_prefix": group.decoy_prefix,
                "is_entrapment": group.is_entrapment,
                "description": group.description,
                "n_total_records": n_total,
                "n_target_records": n_target,
                "n_decoy_records": n_decoy,
            }
        )
        if group.supplies_decoys and n_decoy == 0:
            raise ValueError(f"FASTA group {group.name!r} is marked as supplies_decoys=true but no decoy headers were found")
        if (not group.supplies_decoys) and n_decoy > 0:
            raise ValueError(
                f"FASTA group {group.name!r} is marked as supplies_decoys=false but {n_decoy} decoy-like headers were found"
            )

    write_fasta(records, out_fasta)
    dataframe_to_tsv(pd.DataFrame(rows), manifest_path)

    r_effective: Optional[float] = None
    if entrapment_groups and protease_cfg is not None:
        target_peps = 0
        entrap_peps = 0
        for group in selected_groups:
            include_record = None
            if group.supplies_decoys:
                include_record = lambda rec, prefix=group.decoy_prefix: not parse_accession(rec.header).startswith(prefix)
            uniq, _ = digest_fasta_unique_peptides(
                group.path,
                enzyme_cleave_at=protease_cfg.cleave_at,
                restrict=protease_cfg.restrict,
                missed_cleavages=0 if protease_cfg.forbid_miscleavages else missed_cleavages,
                min_len=protease_cfg.min_len or min_len,
                max_len=protease_cfg.max_len or max_len,
                c_terminal=protease_cfg.c_terminal,
                collapse_il=collapse_il,
                include_record=include_record,
            )
            if group.is_entrapment:
                entrap_peps += uniq
            else:
                target_peps += uniq
        if target_peps > 0:
            r_effective = entrap_peps / target_peps

    return CombinedFastaInfo(
        fasta_path=out_fasta,
        manifest_path=manifest_path,
        selected_groups=[g.name for g in selected_groups],
        supplies_decoys=supplies_decoys,
        effective_decoy_prefix=effective_decoy_prefix,
        generate_decoys=generate_decoys,
        entrapment_groups=entrapment_groups,
        r_effective=r_effective,
    )



def extract_groups_from_proteins(proteins: str, decoy_prefix: str) -> List[str]:
    groups: List[str] = []
    for token in split_proteins_field(proteins):
        token = str(token)
        if token.startswith(decoy_prefix):
            token = token[len(decoy_prefix) :]
        if token.startswith(GROUP_TOKEN):
            remainder = token[len(GROUP_TOKEN) :]
            group = remainder.split("|", 1)[0]
            if group:
                groups.append(group)
    return unique_preserve_order(groups)



def annotate_group_assignment(
    proteins: str,
    *,
    decoy_prefix: str,
    entrapment_groups: Set[str],
    entrapment_strategy: str = "unambiguous",
) -> GroupAnnotation:
    groups = extract_groups_from_proteins(proteins, decoy_prefix=decoy_prefix)
    if not groups:
        return GroupAnnotation(
            matched_groups=[],
            group_name="unassigned",
            is_entrapment=False,
            is_ambiguous_entrapment=False,
        )
    group_name = groups[0] if len(groups) == 1 else "shared:" + "|".join(sorted(groups))
    ent_flags = [g in entrapment_groups for g in groups]
    is_any_ent = any(ent_flags)
    is_all_ent = all(ent_flags)
    if entrapment_strategy == "any":
        is_ent = is_any_ent
        is_amb = is_any_ent and not is_all_ent
    else:
        is_ent = is_all_ent
        is_amb = is_any_ent and not is_all_ent
    return GroupAnnotation(
        matched_groups=groups,
        group_name=group_name,
        is_entrapment=is_ent,
        is_ambiguous_entrapment=is_amb,
    )
