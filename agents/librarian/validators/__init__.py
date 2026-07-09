"""
Librarian validators -- the policy layer over the frozen schema.

  * ``tag_reason``      -- the D24 tag-reason registry (a tag outside its row is
                           a build error) + namespace grouping.
  * ``domains``         -- the D24 closed-domain declarations + value validator +
                           the ``iter_domain_cases`` G1 seam.
  * ``spec_validators`` -- ``validate_librarian_spec``: the four D8 negatives,
                           returned (not raised) so all violations surface at once.
"""

from __future__ import annotations

from .domains import (
    Domain,
    iter_domain_cases,
    load_domains,
    validate_value,
)
from .spec_validators import (
    SignalRegistryLike,
    validate_librarian_spec,
)
from .tag_reason import (
    TagReasonRegistry,
    TagReasonRow,
    assert_tag_in_registry,
    load_tag_reason_registry,
)

__all__ = [
    # tag-reason registry (D24)
    "TagReasonRow",
    "TagReasonRegistry",
    "load_tag_reason_registry",
    "assert_tag_in_registry",
    # domains (D24)
    "Domain",
    "load_domains",
    "validate_value",
    "iter_domain_cases",
    # spec validator (D8)
    "validate_librarian_spec",
    "SignalRegistryLike",
]
