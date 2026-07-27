#!/usr/bin/env python3
"""
ExponentialSmoothing fixtures, with statsmodels Holt-Winters as the oracle.

    python tools/fixture-gen/gen_ets.py            # regenerate
    python tools/fixture-gen/gen_ets.py --check    # diff against committed

The specification defers its forecast equations to Gardner (2005) rather than
writing them out, so there is nothing to transcribe here and a third-party
implementation of that same taxonomy is the right oracle. See oracle_ets.py
for how PMML's final smoothed states are injected into statsmodels.

Coverage:

  ETS_AddTrend           pre-existing, expectations regenerated
  ETS_SeasonalAddAdd     pre-existing, expectations regenerated
  fix_ets_damped_add     damped_additive, exercises phi
  fix_ets_mult_trend     multiplicative trend
  fix_ets_damped_mult    damped_multiplicative, phi in the exponent
  fix_ets_seasonal_mult  multiplicative seasonality
  fix_ets_phase          phase != period -- the seasonal array is indexed from
                         the season of the last known point, not from 0
  fix_ets_weights        alpha/gamma/delta set to values that would wreck the
                         forecast if they were used; the spec says they are
                         "not needed for scoring", so this must equal
                         fix_ets_damped_add row for row
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_ets
from _harness import Fixture, format_number, main, provenance

SCRIPT = "tools/fixture-gen/gen_ets.py"
HORIZON = 6


def document(name: str, note: str, *, level, trend="none", trend_value=0.0, phi=None, alpha=None,
             gamma=None, delta=None, seasonal="none", seasonal_values=None, period=1, phase=None) -> str:
    parts = ["    <ExponentialSmoothing>"]

    level_attrs = "" if alpha is None else f' alpha="{format_number(alpha)}"'
    parts.append(f'      <Level{level_attrs} smoothedValue="{format_number(level)}"/>')

    if trend != "none":
        attrs = f' trend="{trend}"'
        if gamma is not None:
            attrs += f' gamma="{format_number(gamma)}"'
        if phi is not None:
            attrs += f' phi="{format_number(phi)}"'
        parts.append(f'      <Trend_ExpoSmooth{attrs} smoothedValue="{format_number(trend_value)}"/>')

    if seasonal != "none":
        attrs = f' type="{seasonal}" period="{period}"'
        if delta is not None:
            attrs += f' delta="{format_number(delta)}"'
        if phase is not None:
            attrs += f' phase="{phase}"'
        body = " ".join(format_number(v) for v in seasonal_values)
        parts.append(f"      <Seasonality_ExpoSmooth{attrs}>")
        parts.append(f'        <Array n="{len(seasonal_values)}" type="real">{body}</Array>')
        parts.append("      </Seasonality_ExpoSmooth>")

    parts.append("    </ExponentialSmoothing>")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">\n'
        + provenance(SCRIPT, oracle_ets.VERSION, note)
        + '  <DataDictionary numberOfFields="1">\n'
        + '    <DataField name="ts" optype="continuous" dataType="double"/>\n'
        + "  </DataDictionary>\n"
        + f'  <TimeSeriesModel modelName="{name}" functionName="timeSeries" bestFit="ExponentialSmoothing">\n'
        + "    <MiningSchema>\n"
        + '      <MiningField name="ts" usageType="target"/>\n'
        + "    </MiningSchema>\n"
        + "\n".join(parts)
        + "\n  </TimeSeriesModel>\n"
        + "</PMML>\n"
    )


CASES = [
    ("ETS_AddTrend", "additive trend", dict(level=100.0, trend="additive", trend_value=5.0, alpha=0.3, gamma=0.1)),
    (
        "ETS_SeasonalAddAdd",
        "additive trend and additive seasonality",
        dict(level=100.0, trend="additive", trend_value=5.0, alpha=0.3, gamma=0.1,
             seasonal="additive", seasonal_values=[-3.0, -1.0, 2.0, 4.0], period=4, delta=0.1),
    ),
    (
        "fix_ets_damped_add",
        "damped additive trend, phi in the geometric sum",
        dict(level=100.0, trend="damped_additive", trend_value=5.0, phi=0.8),
    ),
    (
        "fix_ets_mult_trend",
        "multiplicative trend",
        dict(level=100.0, trend="multiplicative", trend_value=1.05),
    ),
    (
        "fix_ets_damped_mult",
        "damped multiplicative trend, phi in the exponent",
        dict(level=100.0, trend="damped_multiplicative", trend_value=1.05, phi=0.8),
    ),
    (
        "fix_ets_seasonal_mult",
        "multiplicative seasonality",
        dict(level=100.0, trend="additive", trend_value=5.0, seasonal="multiplicative",
             seasonal_values=[0.9, 0.95, 1.05, 1.1], period=4),
    ),
    (
        "fix_ets_phase",
        "phase=2: the season index of the last known point, so h=1 lands on array element 2",
        dict(level=100.0, trend="additive", trend_value=5.0, seasonal="additive",
             seasonal_values=[-3.0, -1.0, 2.0, 4.0], period=4, phase=2),
    ),
    (
        "fix_ets_weights",
        "alpha/gamma/delta are not needed for scoring; identical to fix_ets_damped_add",
        dict(level=100.0, trend="damped_additive", trend_value=5.0, phi=0.8, alpha=0.99, gamma=0.87),
    ),
]


def build() -> list[Fixture]:
    fixtures = []
    for name, note, kw in CASES:
        # alpha/gamma/delta are documentation only: the oracle never sees them.
        states = {k: v for k, v in kw.items() if k not in ("alpha", "gamma", "delta")}
        values = oracle_ets.ets_forecast(HORIZON, **states)
        fixtures.append(
            Fixture(
                name=name,
                # Every ETS fixture asserts variance = NaN, including the ones
                # that carry alpha/gamma/delta. An ExponentialSmoothing document
                # does not determine a forecast variance: the spec defines no
                # formula, and writes no update equations either, so the weights
                # are not tied to a parameterisation one could be rebuilt from.
                # Asserting NaN keeps a plausible-looking number from creeping
                # back in the way sigma^2*h once did.
                columns=["forecast", "variance"],
                rows=[{"forecast": format_number(v), "variance": "nan"} for v in values],
                pmml=document(name, note, **kw),
            )
        )

    by_name = {f.name: f for f in fixtures}
    # The spec states the smoothing weights are not needed for scoring. If this
    # trips, either the oracle started reading them or the states diverged.
    if by_name["fix_ets_weights"].rows != by_name["fix_ets_damped_add"].rows:
        raise AssertionError("fix_ets_weights must score identically to fix_ets_damped_add")

    return fixtures


if __name__ == "__main__":
    sys.exit(main(build, __doc__.strip().splitlines()[0]))
