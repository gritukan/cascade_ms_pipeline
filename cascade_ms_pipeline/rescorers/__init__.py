from .base import RescoreArtifacts, Rescorer
from .ms2rescore import MS2RescoreRescorer
from .oktoberfest import OktoberfestRescorer

RESCORER_REGISTRY = {
    "ms2rescore": MS2RescoreRescorer(),
    "oktoberfest": OktoberfestRescorer(),
}

__all__ = [
    "RescoreArtifacts",
    "Rescorer",
    "MS2RescoreRescorer",
    "OktoberfestRescorer",
    "RESCORER_REGISTRY",
]
