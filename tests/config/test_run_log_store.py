from __future__ import annotations

import threading
import time

from gcap.config.manifest import RunLogStore


def test_close_waits_for_in_flight_write_guard_before_validating(tmp_path):
    store = RunLogStore(tmp_path / "run-log")
    manifest_hash = "a" * 64
    writer_entered = threading.Event()
    release_writer = threading.Event()
    writer_committed = threading.Event()
    close_finished = threading.Event()

    def writer() -> None:
        with store.write_guard(manifest_hash):
            writer_entered.set()
            release_writer.wait(timeout=5)
            writer_committed.set()

    def closer() -> None:
        def validate_committed() -> None:
            assert writer_committed.is_set(), "close validated before the in-flight write committed"

        store.close(
            manifest_hash,
            closed_by="test",
            validate_before_close=validate_committed,
        )
        close_finished.set()

    writer_thread = threading.Thread(target=writer)
    close_thread = threading.Thread(target=closer)
    writer_thread.start()
    assert writer_entered.wait(timeout=5)
    close_thread.start()
    time.sleep(0.05)
    assert not close_finished.is_set()

    release_writer.set()
    writer_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert close_finished.is_set()
    assert store.is_closed(manifest_hash)
