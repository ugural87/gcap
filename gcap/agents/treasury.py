"""Re-exports TreasuryAction from store.run_record (see that module's
docstring: TreasuryAction is defined in store/ because it predates agents/
in the M1 build order and store/ sits above agents/ in the dependency
diagram). agents/ code should import TreasuryAction from here, not
directly from store.run_record, so a future reader sees at a glance that
the Treasury action vocabulary is conceptually owned by the agent layer
even though the symbol itself lives one layer down.
"""

from __future__ import annotations

from gcap.store.run_record import TreasuryAction

__all__ = ["TreasuryAction"]
