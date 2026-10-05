"""Configuration flags: the single source of truth for what differs between
arms A-F. Spec Section 3: "All six configurations run through one graph
codebase; a configuration is a frozen dataclass of flags, and nothing else
may differ between arms." D1 is the D2 code path with the add-on vector
forced to zero, not a separate implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gcap.errors import InvalidConfigFlagsError

ConfigName = Literal["A", "B", "C", "D1", "D2", "E"]
FtpAddonMode = Literal["none", "zero", "full"]


@dataclass(frozen=True, slots=True)
class ConfigFlags:
    name: ConfigName
    multi_agent: bool
    deterministic_tools: bool
    ledger: bool
    treasury_node: bool
    ftp_addons: FtpAddonMode
    policy_engine: bool
    integrated_validator: bool
    dual_authorisation: bool

    def __post_init__(self) -> None:
        # Structural self-consistency: a flag combination that cannot occur
        # in the frozen Section-3 table is rejected at construction time,
        # not just when looked up via get_config(). This protects code paths
        # that build a ConfigFlags directly (e.g. tests, future roadmap arms)
        # from silently drifting away from the table.
        if self.treasury_node and not self.ledger:
            raise InvalidConfigFlagsError(
                f"{self.name}: treasury_node requires ledger=True (Treasury reads "
                "capacity/state before deciding)."
            )
        if self.ftp_addons == "full" and not self.treasury_node:
            raise InvalidConfigFlagsError(
                f"{self.name}: ftp_addons='full' requires a Treasury node to consume it."
            )
        if self.dual_authorisation and not (self.policy_engine and self.integrated_validator):
            raise InvalidConfigFlagsError(
                f"{self.name}: dual_authorisation requires both policy_engine and "
                "integrated_validator (execution = BUSINESS_APPROVAL AND CONTROL_CLEARANCE "
                "AND POLICY_PASS)."
            )
        if self.policy_engine and not self.multi_agent:
            raise InvalidConfigFlagsError(
                f"{self.name}: policy_engine requires multi_agent=True."
            )
        if self.deterministic_tools and not self.multi_agent:
            raise InvalidConfigFlagsError(
                f"{self.name}: deterministic_tools requires multi_agent=True."
            )


# Section 3 table, verbatim:
#
# Config  Agent nodes                   Tools  Ledger  FTP add-ons  Policy  Validator  Dual auth.
# A       single LLM                    no     no      n/a          no      no         no
# B       Research, Trade, PM           no     no      n/a          no      no         no
# C       Research, Trade, PM           yes    no      n/a          no      no         no
# D1      + Treasury                    yes    yes     forced zero  no      no         no
# D2      + Treasury                    yes    yes     full vector  no      no         no
# E       + Treasury, IC                yes    yes     full vector  yes     yes        yes
CONFIG_TABLE: dict[ConfigName, ConfigFlags] = {
    "A": ConfigFlags(
        name="A",
        multi_agent=False,
        deterministic_tools=False,
        ledger=False,
        treasury_node=False,
        ftp_addons="none",
        policy_engine=False,
        integrated_validator=False,
        dual_authorisation=False,
    ),
    "B": ConfigFlags(
        name="B",
        multi_agent=True,
        deterministic_tools=False,
        ledger=False,
        treasury_node=False,
        ftp_addons="none",
        policy_engine=False,
        integrated_validator=False,
        dual_authorisation=False,
    ),
    "C": ConfigFlags(
        name="C",
        multi_agent=True,
        deterministic_tools=True,
        ledger=False,
        treasury_node=False,
        ftp_addons="none",
        policy_engine=False,
        integrated_validator=False,
        dual_authorisation=False,
    ),
    "D1": ConfigFlags(
        name="D1",
        multi_agent=True,
        deterministic_tools=True,
        ledger=True,
        treasury_node=True,
        ftp_addons="zero",
        policy_engine=False,
        integrated_validator=False,
        dual_authorisation=False,
    ),
    "D2": ConfigFlags(
        name="D2",
        multi_agent=True,
        deterministic_tools=True,
        ledger=True,
        treasury_node=True,
        ftp_addons="full",
        policy_engine=False,
        integrated_validator=False,
        dual_authorisation=False,
    ),
    "E": ConfigFlags(
        name="E",
        multi_agent=True,
        deterministic_tools=True,
        ledger=True,
        treasury_node=True,
        ftp_addons="full",
        policy_engine=True,
        integrated_validator=True,
        dual_authorisation=True,
    ),
}


def get_config(name: str) -> ConfigFlags:
    """Look up a configuration by name. Refuses to start on any flag
    combination outside the frozen table (Section 3, last sentence)."""
    if name not in CONFIG_TABLE:
        raise InvalidConfigFlagsError(
            f"Unknown configuration {name!r}. Valid configurations: "
            f"{sorted(CONFIG_TABLE)}."
        )
    return CONFIG_TABLE[name]


def is_d1_d2_pair(a: ConfigName, b: ConfigName) -> bool:
    """True if {a, b} == {D1, D2}: the pricing-placebo pair that must share
    Research/Trade upstream calls (Section 8.2) and run on the same code
    path with only the add-on vector differing (Section 3)."""
    return {a, b} == {"D1", "D2"}
