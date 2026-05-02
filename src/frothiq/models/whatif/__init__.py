"""What-if simulator: counterfactual predictions for operator decisions."""

from .simulator import (
    WhatIfResult,
    apply_overrides_exact,
    apply_overrides_naive,
    simulate_whatif_exact,
    simulate_whatif_naive,
)

__all__ = [
    "WhatIfResult",
    "apply_overrides_exact",
    "apply_overrides_naive",
    "simulate_whatif_exact",
    "simulate_whatif_naive",
]
