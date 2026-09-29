"""Attack detectors and P300 authenticity defenses."""

from p3a.defenses.authenticity import (
    authenticity_features,
    cross_validate_authenticity,
)

__all__ = ["authenticity_features", "cross_validate_authenticity"]
