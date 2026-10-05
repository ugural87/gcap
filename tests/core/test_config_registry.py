from decimal import Decimal

import pytest

from gcap.config.flags import CONFIG_TABLE, ConfigFlags, get_config, is_d1_d2_pair
from gcap.config.registry import default_registry
from gcap.errors import InvalidConfigFlagsError, UncalibratedParameterError


def test_all_six_configs_present():
    assert set(CONFIG_TABLE) == {"A", "B", "C", "D1", "D2", "E"}


def test_get_config_rejects_unknown_name():
    with pytest.raises(InvalidConfigFlagsError):
        get_config("F")


def test_d1_is_d2_with_addons_forced_zero():
    d1 = get_config("D1")
    d2 = get_config("D2")
    assert d1.ftp_addons == "zero"
    assert d2.ftp_addons == "full"
    # every other structural flag identical between D1 and D2
    assert d1.multi_agent == d2.multi_agent
    assert d1.deterministic_tools == d2.deterministic_tools
    assert d1.ledger == d2.ledger
    assert d1.treasury_node == d2.treasury_node
    assert d1.policy_engine is d2.policy_engine is False
    assert is_d1_d2_pair("D1", "D2")
    assert not is_d1_d2_pair("D1", "E")


def test_only_e_has_dual_authorisation():
    for name, cfg in CONFIG_TABLE.items():
        if name == "E":
            assert cfg.dual_authorisation is True
        else:
            assert cfg.dual_authorisation is False


def test_invalid_flag_combination_rejected_at_construction():
    with pytest.raises(InvalidConfigFlagsError):
        ConfigFlags(
            name="D1",
            multi_agent=True,
            deterministic_tools=True,
            ledger=False,  # treasury_node requires ledger
            treasury_node=True,
            ftp_addons="zero",
            policy_engine=False,
            integrated_validator=False,
            dual_authorisation=False,
        )


def test_registry_defaults_are_placeholders_but_usable():
    reg = default_registry()
    assert reg.to_decide.day_count_basis.days == Decimal("365")
    assert reg.to_decide.numerical_error_tolerance.calibrated is False
    # dev/pilot splits tolerate placeholders
    reg.assert_ready_for_split("dev")
    reg.assert_ready_for_split("pilot")


def test_registry_blocks_confirmatory_split_with_uncalibrated_freeze_params():
    reg = default_registry()
    with pytest.raises(UncalibratedParameterError):
        reg.assert_ready_for_split("confirm")


def test_registry_has_revised_margin_recomputation_placeholder():
    reg = default_registry()
    rmr = reg.to_decide.revised_margin_recomputation
    assert rmr.calibrated is False
    assert rmr.needed_by == "M2"
    assert Decimal("0") < rmr.hedge_rp_reduction_pct < Decimal("1")
    assert Decimal("0") < rmr.limit_addon_reduction_pct < Decimal("1")


def test_registry_has_lhc_addon_bounds_placeholder():
    reg = default_registry()
    bounds = reg.to_decide.lhc_addon_bounds
    assert bounds.calibrated is False
    assert bounds.needed_by == "M3"
    param_names, param_bounds = bounds.as_param_bounds()
    assert param_names == ["lp", "tp", "cp", "rp", "cc"]
    assert len(param_bounds) == 5
    for lo, hi in param_bounds:
        assert lo == 0.0
        assert hi > lo


def test_registry_content_hash_stable_and_sensitive_to_change():
    reg1 = default_registry()
    reg2 = default_registry()
    assert reg1.content_hash() == reg2.content_hash()

    reg3 = reg1.model_copy(
        update={
            "fixed": reg1.fixed.model_copy(
                update={"marginal_band": reg1.fixed.marginal_band.model_copy(
                    update={"primary_bps": Decimal("55")}
                )}
            )
        }
    )
    assert reg3.content_hash() != reg1.content_hash()
