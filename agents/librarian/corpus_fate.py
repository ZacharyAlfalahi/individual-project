"""
Corpus fate table -- the D21 walk made executable (D21).

D21 walked every project paper through the librarian's paper-kind gates and
decided each paper's *fate*: does it become a strategy the pipeline replicates,
or is it refused (and if refused, *why*)? This module encodes that walk as typed,
frozen data -- a **specification / fixture table**, NOT a live routing engine. The
router that decides a fresh paper's fate at runtime is a separate artifact ([P2]);
this table is the authoritative expected-answer set the corpus-walk tests assert
against, and it doubles as the RQ2 refusal-typology preview (D21-F4).

The five fates (the closed enum):

  * ``sort``                          -- a sort-family strategy the pipeline
                                         replicates (BBW 2019/2021, Momentum in
                                         Corporate Bonds).
  * ``pass_gates_curation_excluded``  -- passes every paper-kind gate but is
                                         excluded by *curation*, not by a gate:
                                         a replication study that would duplicate
                                         another paper's factor (D21-F3: gates are
                                         not curation). DRR 2023 / 2026.
  * ``refuse_family_unsupported``     -- a real strategy in an unsupported factor
                                         family (IPCA / DNN): KPP, Duraj-Giesecke.
  * ``refuse_asset_class``            -- a strategy in the wrong asset class
                                         (equity, not corporate bonds): HXZ, KPJ.
  * ``refuse_no_strategy``            -- the paper's object of study is not a
                                         portfolio strategy at all (a survey, a
                                         signal-definition source, a classification
                                         task): HLZ, GKX survey, Glasserman-Lin,
                                         AI Scientist, AlphaAgent, BPW, DPZ.

The three ``refuse_*`` fates are the RQ2 refusal typology (D21-F4); the two
non-refusal fates (``sort``, ``pass_gates_curation_excluded``) are the papers
that clear the gates.

This module imports nothing from the pipeline or the engine -- it is pure typed
data plus a tiny query API (``fate_of``, ``all_fates``) and a completeness
invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import LibrarianSchemaError

# --- the closed fate enum (D21) -------------------------------------------------

#: The three refusal fates -- these partition into the RQ2 refusal typology (D21-F4).
REFUSAL_FATES: frozenset[str] = frozenset(
    (
        "refuse_family_unsupported",
        "refuse_asset_class",
        "refuse_no_strategy",
    )
)

#: The two fates a paper reaches only by clearing every paper-kind gate.
NON_REFUSAL_FATES: frozenset[str] = frozenset(
    (
        "sort",
        "pass_gates_curation_excluded",
    )
)

#: The full closed set of legal fates (validated in ``CorpusFate.__post_init__``).
FATES: frozenset[str] = REFUSAL_FATES | NON_REFUSAL_FATES


@dataclass(frozen=True)
class CorpusFate:
    """One paper's decided fate from the D21 walk: the paper's stable id, its
    ``fate`` (one of the closed ``FATES``), and the ``reason`` D21 recorded --
    the one-line justification that makes the walk auditable and doubles as the
    RQ2 refusal-typology cell text. A specification row, not a routing result."""

    paper: str
    fate: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.paper, str) or self.paper.strip() == "":
            raise LibrarianSchemaError("CorpusFate.paper must be a non-empty string")
        if self.fate not in FATES:
            raise LibrarianSchemaError(
                f"CorpusFate.fate {self.fate!r} is not one of the closed D21 fates "
                f"{sorted(FATES)}"
            )
        if not isinstance(self.reason, str) or self.reason.strip() == "":
            raise LibrarianSchemaError(
                f"CorpusFate.reason must be a non-empty string (paper {self.paper!r})"
            )

    @property
    def is_refusal(self) -> bool:
        """Does this fate belong to the RQ2 refusal typology (a ``refuse_*``)?"""
        return self.fate in REFUSAL_FATES

    def to_dict(self) -> dict:
        return {"paper": self.paper, "fate": self.fate, "reason": self.reason}


# --- the D21 fate table ---------------------------------------------------------
# Every paper D21 names, in walk order. The paper ids are the short handles D21
# uses ("BBW 2019", "KPP", "BPW"), kept verbatim so the table reads as the log's
# walk. The reason text paraphrases D21's per-paper justification.

_FATE_ROWS: tuple[CorpusFate, ...] = (
    # -- sort family: the strategies the pipeline replicates ---------------------
    CorpusFate(
        "BBW 2019",
        "sort",
        "sort family; 4 specs + MKT auxiliary (the anchor exemplar)",
    ),
    CorpusFate(
        "Momentum in Corporate Bonds",
        "sort",
        "sort family; headline cell (six-month momentum anchor)",
    ),
    CorpusFate(
        "BBW 2021",
        "sort",
        "sort family; real test lands at binding (bespoke risk measures)",
    ),
    # -- pass gates, excluded by curation (D21-F3: gates are not curation) --------
    CorpusFate(
        "DRR 2023",
        "pass_gates_curation_excluded",
        "passes every gate; curation excludes it (would duplicate another paper's factor)",
    ),
    CorpusFate(
        "DRR 2026",
        "pass_gates_curation_excluded",
        "passes every gate; curation excludes it (would duplicate another paper's factor)",
    ),
    # -- refuse: family unsupported (IPCA / DNN, future) -------------------------
    CorpusFate(
        "KPP",
        "refuse_family_unsupported",
        "real strategy, unsupported family (IPCA); future family",
    ),
    CorpusFate(
        "Duraj-Giesecke",
        "refuse_family_unsupported",
        "real strategy, unsupported family (DNN); future family",
    ),
    # -- refuse: asset class (equity, not corporate bonds) -----------------------
    CorpusFate(
        "HXZ",
        "refuse_asset_class",
        "strategy in the wrong asset class (equity)",
    ),
    CorpusFate(
        "KPJ",
        "refuse_asset_class",
        "strategy in the wrong asset class (equity)",
    ),
    # -- refuse: no strategy (object of study is not a portfolio strategy) --------
    CorpusFate(
        "HLZ",
        "refuse_no_strategy",
        "no strategy; a multiple-testing survey, not a portfolio construction",
    ),
    CorpusFate(
        "Giglio-Kelly-Xiu survey",
        "refuse_no_strategy",
        "no strategy; a survey, not a portfolio construction",
    ),
    CorpusFate(
        "Glasserman-Lin",
        "refuse_no_strategy",
        "no strategy; not a portfolio construction as object of study",
    ),
    CorpusFate(
        "AI Scientist",
        "refuse_no_strategy",
        "no strategy; a research-automation paper, not a portfolio construction",
    ),
    CorpusFate(
        "AlphaAgent",
        "refuse_no_strategy",
        "no strategy; a research-automation paper, not a portfolio construction",
    ),
    CorpusFate(
        "BPW",
        "refuse_no_strategy",
        "no strategy; signal-definition source (the Bao-Pan-Wang gamma measure)",
    ),
    CorpusFate(
        "DPZ",
        "refuse_no_strategy",
        "no strategy; a credit-rating classification task, no portfolio construction",
    ),
)


#: The D21 fate table: paper id -> its decided ``CorpusFate``. Built from
#: ``_FATE_ROWS`` with a duplicate-paper guard so the table is a genuine mapping
#: (one fate per paper -- P5, one home per fact).
def _build_table(rows: tuple[CorpusFate, ...]) -> dict[str, CorpusFate]:
    table: dict[str, CorpusFate] = {}
    for row in rows:
        if row.paper in table:
            raise LibrarianSchemaError(
                f"duplicate paper {row.paper!r} in the D21 fate table -- "
                "one fate per paper (P5)"
            )
        table[row.paper] = row
    return table


FATE_TABLE: dict[str, CorpusFate] = _build_table(_FATE_ROWS)


# --- query API ------------------------------------------------------------------

def fate_of(paper: str) -> CorpusFate:
    """The decided fate of ``paper`` (a D21 paper handle).

    Raises ``LibrarianSchemaError`` for a paper D21 never walked -- the table is
    a *closed* specification of this corpus, not an open router; asking about an
    unlisted paper is a caller-contract error, not a fate."""
    try:
        return FATE_TABLE[paper]
    except KeyError as exc:
        raise LibrarianSchemaError(
            f"{paper!r} is not a paper in the D21 corpus walk; "
            f"known papers: {sorted(FATE_TABLE)}"
        ) from exc


def all_fates() -> tuple[CorpusFate, ...]:
    """Every decided fate, in D21 walk order."""
    return _FATE_ROWS


def papers() -> tuple[str, ...]:
    """Every paper id D21 walked, in walk order."""
    return tuple(row.paper for row in _FATE_ROWS)


def refusals() -> tuple[CorpusFate, ...]:
    """The RQ2 refusal-typology preview: every ``refuse_*`` fate, in walk order."""
    return tuple(row for row in _FATE_ROWS if row.is_refusal)
