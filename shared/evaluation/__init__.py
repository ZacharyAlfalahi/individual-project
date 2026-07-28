"""
shared/evaluation — evaluation-layer diagnostics shared by the Auditor and the
Scientist's Experimentalist (Scientist spec §10.2 / deviation D4). Not an agent.

Currently hosts the Layer-1 crowding survivor diagnostic (`crowding`). It
supersedes the KPP-5 factor bundle (docs/archive/kpp5_spec.md): see the module docstring
and the `crowding:` block in docs/thresholds.yaml.
"""

from .crowding import crowding_diagnostic, load_crowding_factor_bundle
from .thresholds import CrowdingConfig, CrowdingThresholdError, load_crowding_config

__all__ = [
    "crowding_diagnostic",
    "load_crowding_factor_bundle",
    "CrowdingConfig",
    "CrowdingThresholdError",
    "load_crowding_config",
]
