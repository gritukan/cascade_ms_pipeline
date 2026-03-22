from __future__ import annotations

import json
from pathlib import Path

from cascade_ms_pipeline.config import load_config
from cascade_ms_pipeline.pipeline import run_pipeline


def test_pipeline_dry_run_creates_report_tree(tmp_path: Path) -> None:
    spectra = tmp_path / "run01.mzML"
    spectra.write_text("<mzML></mzML>", encoding="utf-8")
    canonical = tmp_path / "canonical.fasta"
    canonical.write_text(
        ">P1\nMPEPTIDEK\n>P2\nMPEPTIDER\n",
        encoding="utf-8",
    )
    novel = tmp_path / "novel.fasta"
    novel.write_text(
        ">ALT1\nMALTPEPK\n",
        encoding="utf-8",
    )

    config = {
        "general": {
            "report_dir": str(tmp_path / "report"),
            "spectra": [str(spectra)],
            "acquisition": "dda",
            "protease": "trypsin",
            "dry_run": True
        },
        "fasta_groups": [
            {"name": "canonical", "path": str(canonical), "supplies_decoys": False},
            {"name": "novel", "path": str(novel), "supplies_decoys": False}
        ],
        "searches": [
            {
                "name": "step1",
                "engine": "sage",
                "fasta_groups": ["canonical", "novel"],
                "fdr": {"methods": ["all_together", "per_group"]},
                "trim": {"enabled": False}
            }
        ]
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_pipeline(cfg)

    report = tmp_path / "report"
    assert (report / "config.resolved.json").exists()
    assert (report / "pipeline_summary.json").exists()
    assert (report / "steps" / "01_step1" / "combined_fasta" / "combined_search.fasta").exists()
    assert (report / "steps" / "01_step1" / "normalized" / "row_base_engine.tsv").exists()
    assert (report / "steps" / "01_step1" / "notes.txt").exists()
