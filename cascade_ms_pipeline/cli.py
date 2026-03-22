from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-ms-pipeline",
        description=(
            "Run a report-centric cascaded proteomics search workflow with optional rescoring, "
            "multiple FDR strategies, plotting, and DDA spectrum trimming."
        ),
    )
    p.add_argument("config", type=Path, help="Path to the JSON configuration file")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the configuration, print a short summary, and exit",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="After validation, print the resolved config JSON to stdout",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))

    if args.validate_only:
        print(
            f"Validated config: {args.config}\n"
            f"  acquisition: {cfg.general.acquisition}\n"
            f"  spectra: {len(cfg.general.spectra)}\n"
            f"  FASTA groups: {len(cfg.fasta_groups)}\n"
            f"  search steps: {len(cfg.searches)}\n"
            f"  report_dir: {cfg.general.report_dir}"
        )
        return 0

    run_pipeline(cfg)
    print(f"Pipeline finished. Report written to: {cfg.general.report_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
