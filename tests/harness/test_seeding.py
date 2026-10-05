from __future__ import annotations

from gcap.harness.seeding import derive_seed, rng_for


def test_derive_seed_is_deterministic():
    a = derive_seed("manifest-1", "case-1", 0)
    b = derive_seed("manifest-1", "case-1", 0)
    assert a == b


def test_derive_seed_varies_with_each_input():
    base = derive_seed("manifest-1", "case-1", 0)
    assert derive_seed("manifest-2", "case-1", 0) != base
    assert derive_seed("manifest-1", "case-2", 0) != base
    assert derive_seed("manifest-1", "case-1", 1) != base
    assert derive_seed("manifest-1", "case-1", 0, "draw-a") != base


def test_derive_seed_fits_signed_64_bit_range():
    for i in range(50):
        seed = derive_seed("manifest-1", f"case-{i}", i)
        assert 0 <= seed <= 0x7FFFFFFFFFFFFFFF


def test_rng_for_is_reproducible_and_seed_derived():
    r1 = rng_for("manifest-1", "case-1", 0)
    r2 = rng_for("manifest-1", "case-1", 0)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]
