from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import RescorerConfig
from ..engines.base import SearchExecutionContext, SearchArtifacts
from ..results import MergeReport


@dataclass
class RescoreArtifacts:
    name: str
    merged_df: pd.DataFrame
    raw_paths: Dict[str, Path] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    merge_report: MergeReport | None = None


class Rescorer(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        cfg: RescorerConfig,
        ctx: SearchExecutionContext,
        base_artifacts: SearchArtifacts,
        current_df: pd.DataFrame,
    ) -> RescoreArtifacts:
        raise NotImplementedError
