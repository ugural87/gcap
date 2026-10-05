"""Milestone 3 (proposal Section 10): "Pilot and design freeze."

This package is the M3 counterpart to M2's ``gcap.harness``: harness/ is a
generic executor (any case set, any config, any split); pilot/ is the one
consumer that decides WHAT M3's development pilot runs and what it computes
from the result — the 32-case stratified corpus (Section 9.7), its
execution across the four primary-contrast arms at the pilot replicate
count, and the descriptive/inferential glue (event rates, discordance, ICC,
the conformance gap g and its lower bound, the kmin candidate scan) that
feeds Section 9.5's design-freeze decisions.

Like harness/, no import-linter contract restricts this package: it freely
imports gcap.bench (pilot corpus generation), gcap.harness (execution),
gcap.analysis (the pure statistical machinery already built for
Milestone 4's confirmatory evaluation, reused here unchanged), gcap.store
and gcap.config.

Standing constraint carried from every other milestone: every pilot run in
this build uses ``gcap.agents.llm.FakeLLMAdapter`` exclusively (no real
API key is ever requested or used). This means the g/kmin numbers this
package's report.py produces from a run against this build's own fake,
scripted adapter are a REHEARSAL of the freeze machinery, not a real
calibration — see report.py's own docstring for exactly what that implies
and does not imply about the results.
"""
