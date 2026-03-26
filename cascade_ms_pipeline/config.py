from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .protease import PROTEASE_CONFIGS
from .util import json_ready, read_json


DEFAULT_ALPHA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
DEFAULT_METHODS = ["all_together", "per_group", "transferred_subgroup", "group_walk"]


@dataclass
class BinaryPaths:
    sage: str = "sage"
    diann: str = "diann"
    ms2rescore: str = "ms2rescore"
    python: str = "python"

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "BinaryPaths":
        if raw is None:
            return cls()
        return cls(**{k: raw[k] for k in raw if hasattr(cls, k)})


@dataclass
class GeneralConfig:
    report_dir: Path
    spectra: List[Path]
    acquisition: str = "dda"  # dda | dia
    protease: str = "trypsin"
    fragmentation: str = "hcd"  # mainly for Sage rescoring / ion kinds
    keep_intermediate_spectra: bool = False
    plot_format: str = "png"
    dry_run: bool = False
    random_seed: int = 1
    binaries: BinaryPaths = field(default_factory=BinaryPaths)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GeneralConfig":
        if "report_dir" not in raw:
            raise ValueError("general.report_dir is required")
        spectra_raw = raw.get("spectra")
        if spectra_raw is None:
            if "spectrum" in raw:
                spectra_raw = [raw["spectrum"]]
            else:
                raise ValueError("general.spectra is required")
        spectra = [Path(str(x)) for x in spectra_raw]
        return cls(
            report_dir=Path(str(raw["report_dir"])),
            spectra=spectra,
            acquisition=str(raw.get("acquisition", "dda")).lower(),
            protease=str(raw.get("protease", "trypsin")).lower(),
            fragmentation=str(raw.get("fragmentation", "hcd")).lower(),
            keep_intermediate_spectra=bool(raw.get("keep_intermediate_spectra", False)),
            plot_format=str(raw.get("plot_format", "png")).lower(),
            dry_run=bool(raw.get("dry_run", False)),
            random_seed=int(raw.get("random_seed", 1)),
            binaries=BinaryPaths.from_dict(raw.get("binaries")),
        )


@dataclass
class FastaGroupConfig:
    name: str
    path: Path
    supplies_decoys: bool = False
    decoy_prefix: str = "rev_"
    is_entrapment: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FastaGroupConfig":
        if "name" not in raw or "path" not in raw:
            raise ValueError("Each fasta_groups entry requires name and path")
        return cls(
            name=str(raw["name"]),
            path=Path(str(raw["path"])),
            supplies_decoys=bool(raw.get("supplies_decoys", False)),
            decoy_prefix=str(raw.get("decoy_prefix", "rev_")),
            is_entrapment=bool(raw.get("is_entrapment", False)),
            description=str(raw.get("description", "")),
        )


@dataclass
class RescorerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "RescorerConfig":
        if isinstance(raw, str):
            return cls(type=raw, params={})
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid rescorer entry: {raw!r}")
        if "type" not in raw:
            raise ValueError("Rescorer entries require 'type'")
        params = raw.get("params") or {k: v for k, v in raw.items() if k != "type"}
        return cls(type=str(raw["type"]).lower(), params=dict(params))


@dataclass
class FDRConfig:
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    correction: float = 1.0
    alpha_grid: List[float] = field(default_factory=lambda: list(DEFAULT_ALPHA_GRID))
    score_source: str = "final"  # final | engine | rescorer:<name> | explicit column name
    groupwalk_k: int = 40
    groupwalk_seed: int = 1
    transferred_min_decoys: int = 20
    transferred_min_points: int = 8
    transferred_clip_min: float = 1e-6
    entrapment_strategy: str = "unambiguous"  # unambiguous | any

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "FDRConfig":
        if raw is None:
            return cls()
        return cls(
            methods=[str(x).lower() for x in raw.get("methods", DEFAULT_METHODS)],
            correction=float(raw.get("correction", 1.0)),
            alpha_grid=[float(x) for x in raw.get("alpha_grid", DEFAULT_ALPHA_GRID)],
            score_source=str(raw.get("score_source", "final")),
            groupwalk_k=int(raw.get("groupwalk_k", 40)),
            groupwalk_seed=int(raw.get("groupwalk_seed", 1)),
            transferred_min_decoys=int(raw.get("transferred_min_decoys", 20)),
            transferred_min_points=int(raw.get("transferred_min_points", 8)),
            transferred_clip_min=float(raw.get("transferred_clip_min", 1e-6)),
            entrapment_strategy=str(raw.get("entrapment_strategy", "unambiguous")).lower(),
        )


@dataclass
class TrimmingConfig:
    enabled: bool = False
    method: str = "per_group"
    level: str = "psm"
    alpha: float = 0.01
    unsupported_action: str = "skip"  # skip | error
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "TrimmingConfig":
        if raw is None:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            method=str(raw.get("method", "per_group")).lower(),
            level=str(raw.get("level", "psm")).lower(),
            alpha=float(raw.get("alpha", 0.01)),
            unsupported_action=str(raw.get("unsupported_action", "skip")).lower(),
            params=dict(raw.get("params", {})),
        )


@dataclass
class SearchStepConfig:
    name: str
    engine_type: str
    engine_params: Dict[str, Any] = field(default_factory=dict)
    fasta_groups: List[str] = field(default_factory=list)
    rescorers: List[RescorerConfig] = field(default_factory=list)
    fdr: FDRConfig = field(default_factory=FDRConfig)
    trim: TrimmingConfig = field(default_factory=TrimmingConfig)
    enabled: bool = True
    skip_engine: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchStepConfig":
        if "name" not in raw:
            raise ValueError("Each search step requires a name")
        engine = raw.get("engine")
        if engine is None:
            if "engine_type" in raw:
                engine_type = str(raw["engine_type"])
                engine_params = dict(raw.get("engine_params", {}))
            else:
                raise ValueError(f"search step {raw['name']!r} is missing engine")
        elif isinstance(engine, str):
            engine_type = engine
            engine_params = dict(raw.get("engine_params", {}))
        elif isinstance(engine, Mapping):
            if "type" not in engine:
                raise ValueError(f"search step {raw['name']!r} has engine without type")
            engine_type = str(engine["type"])
            engine_params = dict(engine.get("params", {}))
        else:
            raise ValueError(f"Invalid engine block for step {raw['name']!r}: {engine!r}")

        fasta_groups = raw.get("fasta_groups") or raw.get("input_fasta_groups")
        if not fasta_groups:
            raise ValueError(f"search step {raw['name']!r} requires fasta_groups")

        rescore_raw = raw.get("rescore", raw.get("rescorers", []))
        if isinstance(rescore_raw, Mapping) or isinstance(rescore_raw, str):
            rescore_raw = [rescore_raw]
        rescorers = [RescorerConfig.from_raw(x) for x in rescore_raw]

        return cls(
            name=str(raw["name"]),
            engine_type=str(engine_type).lower(),
            engine_params=engine_params,
            fasta_groups=[str(x) for x in fasta_groups],
            rescorers=rescorers,
            fdr=FDRConfig.from_dict(raw.get("fdr")),
            trim=TrimmingConfig.from_dict(raw.get("trim")),
            enabled=bool(raw.get("enabled", True)),
            skip_engine=bool(raw.get("skip_engine", False)),
        )


@dataclass
class PipelineConfig:
    general: GeneralConfig
    fasta_groups: List[FastaGroupConfig]
    searches: List[SearchStepConfig]

    def validate(self) -> None:
        if self.general.acquisition not in {"dda", "dia"}:
            raise ValueError("general.acquisition must be 'dda' or 'dia'")
        if self.general.protease not in PROTEASE_CONFIGS:
            raise ValueError(f"Unsupported protease {self.general.protease!r}; choose one of {sorted(PROTEASE_CONFIGS)}")
        if not self.general.spectra:
            raise ValueError("At least one spectrum file is required")

        basenames = [p.name for p in self.general.spectra]
        if len(basenames) != len(set(basenames)):
            raise ValueError(
                "Input spectra basenames must be unique; trimming and merge logic use basenames when engines do not report full paths"
            )

        names = [g.name for g in self.fasta_groups]
        if len(names) != len(set(names)):
            raise ValueError("fasta_groups names must be unique")
        group_map = {g.name: g for g in self.fasta_groups}

        if not self.searches:
            raise ValueError("At least one search step is required")
        for step in self.searches:
            if step.engine_type not in {"sage", "diann"}:
                raise ValueError(f"Unsupported engine_type {step.engine_type!r} in step {step.name!r}")
            for g in step.fasta_groups:
                if g not in group_map:
                    raise ValueError(f"Unknown fasta group {g!r} referenced by step {step.name!r}")
            if step.trim.level not in {"psm", "peptide"}:
                raise ValueError(f"trim.level must be 'psm' or 'peptide' in step {step.name!r}")
            for method in step.fdr.methods:
                if method not in set(DEFAULT_METHODS):
                    raise ValueError(f"Unsupported FDR method {method!r} in step {step.name!r}")
            if step.fdr.entrapment_strategy not in {"unambiguous", "any"}:
                raise ValueError(
                    f"Unsupported entrapment_strategy {step.fdr.entrapment_strategy!r} in step {step.name!r}"
                )

    def fasta_group_map(self) -> Dict[str, FastaGroupConfig]:
        return {g.name: g for g in self.fasta_groups}

    def to_dict(self) -> Dict[str, Any]:
        return json_ready(asdict(self))



def load_config(path: Path) -> PipelineConfig:
    raw = read_json(path)
    general = GeneralConfig.from_dict(raw.get("general", {}))
    fasta_groups = [FastaGroupConfig.from_dict(x) for x in raw.get("fasta_groups", [])]
    searches = [SearchStepConfig.from_dict(x) for x in raw.get("searches", [])]
    cfg = PipelineConfig(general=general, fasta_groups=fasta_groups, searches=searches)
    cfg.validate()
    return cfg
