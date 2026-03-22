from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import GeneralConfig, SearchStepConfig
from ..fasta_groups import CombinedFastaInfo


@dataclass
class SearchExecutionContext:
    general: GeneralConfig
    step: SearchStepConfig
    step_dir: Path
    spectra: List[Path]
    combined_fasta: CombinedFastaInfo
    log_path: Path


@dataclass
class SearchArtifacts:
    engine_name: str
    row_df: pd.DataFrame
    normalized_path: Path
    raw_paths: Dict[str, Path] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class SearchEngine(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: SearchExecutionContext) -> SearchArtifacts:
        raise NotImplementedError
