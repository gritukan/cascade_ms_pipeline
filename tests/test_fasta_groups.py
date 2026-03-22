from __future__ import annotations

from pathlib import Path

from cascade_ms_pipeline.config import FastaGroupConfig
from cascade_ms_pipeline.fasta import read_fasta
from cascade_ms_pipeline.fasta_groups import build_combined_fasta
from cascade_ms_pipeline.protease import PROTEASE_CONFIGS


def test_combined_fasta_rewrites_group_tokens_and_decoys(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.fasta"
    canonical.write_text(
        ">P1 canonical protein\nMPEPTIDEK\n>P2 second protein\nMPEPTIDER\n",
        encoding="utf-8",
    )
    novel = tmp_path / "novel_decoyed.fasta"
    novel.write_text(
        ">ALT1 alt protein\nMALTPEPK\n>rev_ALT1 alt protein decoy\nKPEPTLAM\n",
        encoding="utf-8",
    )

    info = build_combined_fasta(
        [
            FastaGroupConfig(name="novel", path=novel, supplies_decoys=True, decoy_prefix="rev_", is_entrapment=False),
        ],
        tmp_path / "combined1",
        protease_cfg=PROTEASE_CONFIGS["trypsin"],
    )
    headers = [rec.header for rec in read_fasta(info.fasta_path)]
    assert headers[0].startswith("grp=novel|ALT1")
    assert headers[1].startswith("rev_grp=novel|ALT1")
    assert info.effective_decoy_prefix == "rev_"
    assert info.generate_decoys is False

    info2 = build_combined_fasta(
        [
            FastaGroupConfig(name="canonical", path=canonical, supplies_decoys=False, decoy_prefix="rev_", is_entrapment=False),
        ],
        tmp_path / "combined2",
        protease_cfg=PROTEASE_CONFIGS["trypsin"],
    )
    headers2 = [rec.header for rec in read_fasta(info2.fasta_path)]
    assert headers2[0].startswith("grp=canonical|P1")
    assert info2.generate_decoys is True
