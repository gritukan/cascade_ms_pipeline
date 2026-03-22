"""Cascade search pipeline for rare / non-canonical proteomics analyses.

This package builds reproducible, report-centric cascaded search workflows for
DDA and DIA experiments.  It wraps search engines, optional rescoring, several
FDR strategies, plotting, and (for DDA) MS2-spectrum trimming between cascade
steps.
"""

from .config import PipelineConfig, load_config
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "load_config", "run_pipeline"]
__version__ = "0.1.0"
