"""Shared test-data builders for tests/store/. Kept as a plain module (not
a test_*.py file) so it is never collected as a test module and other test
files can import it directly without relying on pytest's rootless import
machinery to resolve a `tests.store.test_x` path (a fragile pattern already
worked around once in M1's tests/bench/test_sealing.py).
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.store.repository import (
    ApprovalWrite,
    DecisionPackageWrite,
    EvidenceClaimWrite,
    PolicyStateWrite,
    RunWrite,
    ToolResultWrite,
)
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents, TreasuryAction


def full_run_write(**overrides) -> RunWrite:
    defaults = dict(
        run_id=uuid.uuid4(),
        config="E",
        split="dev",
        block="E-block",
        case_id="case-1",
        replicate=0,
        family="hard_mandate_breach",
        materiality="critical",
        model_snapshot="gpt-luna-test",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="a" * 64,
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=True,
        ep_governance_components=GovernanceComponents(hard_rule=True),
    )
    defaults.update(overrides)
    rr = RunRecord(**defaults)
    return RunWrite(
        run_record=rr,
        decision_packages=(
            DecisionPackageWrite(
                proposal_id="p1",
                proposal_position=0,
                shown_margin=Decimal("0.02"),
                treasury_action=TreasuryAction.REJECT,
                approval_family=False,
                is_final=True,
            ),
        ),
        evidence_claims=(
            EvidenceClaimWrite(claim_id="c1", claim_type="fact", metric="asset_class", agent_flagged=True),
        ),
        tool_results=(
            ToolResultWrite(tool_name="policy_engine", inputs_hash="deadbeef", cache_hit=False, latency_ms=5),
        ),
        policy_state=PolicyStateWrite(
            alco_package="NORMAL", verdict="HARD_FAIL", triggering_rule_ids=("MANDATE-1",), policy_version="policy-v1"
        ),
        approval=ApprovalWrite(business_approval=True, control_clearance=False, policy_pass=False, executed=False),
        execution_events=(),
    )
