from __future__ import annotations

import dataclasses

from gcap.config.hashing import hash_bytes_map, hash_config_table, hash_directory, hash_run_list


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeFlags:
    name: str
    multi_agent: bool
    threshold: int = 0


def test_hash_bytes_map_is_order_independent():
    a = hash_bytes_map({"x.py": b"1", "y.py": b"2"})
    b = hash_bytes_map({"y.py": b"2", "x.py": b"1"})
    assert a == b


def test_hash_bytes_map_changes_with_content():
    a = hash_bytes_map({"x.py": b"1"})
    b = hash_bytes_map({"x.py": b"2"})
    assert a != b


def test_hash_bytes_map_changes_with_name():
    a = hash_bytes_map({"x.py": b"1"})
    b = hash_bytes_map({"z.py": b"1"})
    assert a != b


def test_hash_bytes_map_of_empty_map_is_stable():
    assert hash_bytes_map({}) == hash_bytes_map({})


def test_hash_directory_is_stable_across_absolute_path(tmp_path):
    root1 = tmp_path / "a"
    root1.mkdir()
    (root1 / "m.py").write_text("x = 1\n")
    root2 = tmp_path / "b"
    root2.mkdir()
    (root2 / "m.py").write_text("x = 1\n")
    assert hash_directory(root1) == hash_directory(root2)


def test_hash_directory_changes_when_a_file_changes(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n")
    before = hash_directory(root)
    (root / "m.py").write_text("x = 2\n")
    after = hash_directory(root)
    assert before != after


def test_hash_directory_ignores_non_matching_files(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n")
    before = hash_directory(root)
    (root / "notes.txt").write_text("irrelevant")
    after = hash_directory(root)
    assert before == after


def test_hash_directory_of_missing_root_is_the_empty_map_hash(tmp_path):
    assert hash_directory(tmp_path / "does-not-exist") == hash_bytes_map({})


def test_hash_run_list_is_order_independent():
    a = hash_run_list([("B", "case-1", 0, None), ("C", "case-1", 0, None)])
    b = hash_run_list([("C", "case-1", 0, None), ("B", "case-1", 0, None)])
    assert a == b


def test_hash_run_list_changes_with_content():
    a = hash_run_list([("B", "case-1", 0, None)])
    b = hash_run_list([("B", "case-1", 1, None)])
    assert a != b


def test_hash_run_list_preserves_field_boundaries_and_types():
    # These rows collided under the former comma-joined serialization.
    assert hash_run_list([("B", "case,1", 0, None)]) != hash_run_list(
        [("B,case", "1", 0, None)]
    )
    assert hash_run_list([("B", "1", 0, None)]) != hash_run_list(
        [("B", 1, 0, None)]
    )


def test_hash_config_table_is_order_independent():
    table_1 = {
        "B": _FakeFlags(name="B", multi_agent=False),
        "C": _FakeFlags(name="C", multi_agent=True),
    }
    table_2 = {
        "C": _FakeFlags(name="C", multi_agent=True),
        "B": _FakeFlags(name="B", multi_agent=False),
    }
    assert hash_config_table(table_1) == hash_config_table(table_2)


def test_hash_config_table_changes_when_a_flag_value_changes():
    before = {"B": _FakeFlags(name="B", multi_agent=False, threshold=1)}
    after = {"B": _FakeFlags(name="B", multi_agent=False, threshold=2)}
    assert hash_config_table(before) != hash_config_table(after)


def test_hash_config_table_changes_when_an_arm_is_added_or_removed():
    smaller = {"B": _FakeFlags(name="B", multi_agent=False)}
    larger = {
        "B": _FakeFlags(name="B", multi_agent=False),
        "C": _FakeFlags(name="C", multi_agent=True),
    }
    assert hash_config_table(smaller) != hash_config_table(larger)


def test_hash_config_table_is_deterministic_across_calls():
    table = {"B": _FakeFlags(name="B", multi_agent=False, threshold=5)}
    assert hash_config_table(table) == hash_config_table(table)


def test_hash_config_table_matches_the_real_config_table_deterministically():
    """The real CONFIG_TABLE (gcap.config.flags) hashes identically across
    two independent calls -- the property gcap.harness.manifest_binding's
    hash_config_table_default()/validate_runtime_against_manifest actually
    relies on."""
    from gcap.config.flags import CONFIG_TABLE

    assert hash_config_table(CONFIG_TABLE) == hash_config_table(CONFIG_TABLE)
