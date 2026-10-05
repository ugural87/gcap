from __future__ import annotations

import pytest

from gcap.store.db import build_engine, get_session_factory, init_schema


@pytest.fixture()
def session_factory():
    engine = build_engine("sqlite:///:memory:")
    init_schema(engine)
    factory = get_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()
