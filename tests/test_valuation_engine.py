"""Checks for ValuationEngineProductFixedAccrued (fixedincomelib/yield_curve/valuation_engine.py).

    python tests/test_valuation_engine.py    # plain runner
    python -m pytest tests -q                # with pytest

The model, product and parameters are the ones in hw5_valuation_and_first_order_risk.ipynb. The
reference numbers are worked out by hand: C = 1,000,000 x 91/360 (ACT/360, unadjusted dates),
T = 273/365 and D = exp(-(0.03 x 90/365 + 0.04 x 183/365)) x exp(-0.001 x 92/365).
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from fixedincomelib import (  # noqa: E402
    qfCreateProductFixedAccruedCashflow, qfCreateValuationParameters,
    qfCreateValuationParametersCollection, qfCreateValueReport, qfDiscountFactor,
)
from fixedincomelib.valuation import RiskReport, ValuationEngineProductRegistry, ValuationRequest  # noqa: E402
from fixedincomelib.yield_curve.valuation_engine import ValuationEngineProductFixedAccrued  # noqa: E402
from test_yield_curve_gradient import (  # noqa: E402
    FUNDING_INDEX, PAYMENT_DATE, VALUE_DATE, build, central_difference, model_gradient, raises,
)

NOTIONAL = 1_000_000
CASH = NOTIONAL * 91 / 360
DISCOUNT = np.exp(-(0.03 * 90 / 365 + 0.04 * 183 / 365)) * np.exp(-0.001 * 92 / 365)
BUCKETS_SOFR = np.array([90 / 365, 183 / 365, 0.0])
BUCKETS_FUNDING = np.array([181 / 365, 92 / 365])


def make_product(payment_date=PAYMENT_DATE, notional=NOTIONAL):
    return qfCreateProductFixedAccruedCashflow(
        "2026-04-02", "2026-07-02", "USD", notional, "ACT/360",
        payment_date=payment_date, business_day_convention="NONE",
    )


def parameters(content=None):
    content = {"FUNDING INDEX": FUNDING_INDEX} if content is None else content
    return qfCreateValuationParametersCollection([qfCreateValuationParameters("FUNDING INDEX PARAMETER", content)])


def engine_for(model, product, params=None):
    engine = ValuationEngineProductFixedAccrued(model, params or parameters(), product, ValuationRequest.PV)
    engine.calculate_value()
    risk = []
    engine.calculate_risk(risk)
    return engine, risk


def test_future_payment_is_cash_times_the_funding_discount_factor():
    model = build()
    product = make_product()
    engine, _ = engine_for(model, product)
    assert abs(product.accrued - 91 / 360) < 1e-15 and abs(engine.cash_amount - CASH) < 1e-9
    assert abs(engine.value - CASH * qfDiscountFactor(model, FUNDING_INDEX, PAYMENT_DATE)) < 1e-9
    assert engine.cash == 0.0
    assert engine.funding_index.name() == FUNDING_INDEX


def test_notebook_numbers_by_hand():
    model = build()
    engine, risk = engine_for(model, make_product())
    assert abs(engine.value - CASH * DISCOUNT) < 1e-6
    assert np.allclose(risk[0], -CASH * DISCOUNT * BUCKETS_SOFR, rtol=0, atol=1e-6)
    assert np.allclose(risk[1], -CASH * DISCOUNT * BUCKETS_FUNDING, rtol=0, atol=1e-6)
    # the figures the notebook prints
    assert abs(engine.value - 245870.910580) < 5e-6
    assert np.allclose(risk[0], [-60625.703979, -123272.264756, 0.0], rtol=0, atol=5e-6)
    assert np.allclose(risk[1], [-121925.026890, -61972.941845], rtol=0, atol=5e-6)


def test_risk_is_cash_times_the_model_discount_factor_gradient():
    model = build()
    engine, risk = engine_for(model, make_product())
    for got, block in zip(risk, model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)):
        assert np.allclose(got, engine.cash_amount * block, rtol=1e-15, atol=0)


def test_risk_matches_central_bump_and_reval_of_the_pv():
    model = build()
    engine, risk = engine_for(model, make_product())

    def price():
        engine.calculate_value()
        return engine.value

    reference = price()
    for position, index in enumerate(("SOFR-1B", FUNDING_INDEX)):
        for node in range(len(risk[position])):
            finite = central_difference(model, index, node, price)
            assert abs(finite - risk[position][node]) < 1e-4, (index, node)   # PV is 2.5e5, so this is 4e-10 relative
    assert abs(price() - reference) < 1e-9


def test_same_day_payment_is_cash_with_no_risk():
    model = build()
    engine, risk = engine_for(model, make_product(payment_date=VALUE_DATE))
    assert engine.value == engine.cash == engine.cash_amount
    assert [len(block) for block in risk] == model.num_sub_components
    assert all(np.all(block == 0.0) for block in risk)
    report = engine.get_value_and_cash()
    assert report.pv == [["USD", engine.cash_amount]] and report.cash == [["USD", engine.cash_amount]]
    assert report.display()["Value"].tolist() == [0.0, engine.cash_amount]       # PV net of cash, then cash


def test_past_payment_is_worth_nothing():
    model = build()
    engine, risk = engine_for(model, make_product(payment_date="2025-12-31"))
    assert engine.value == 0.0 and engine.cash == 0.0
    assert all(np.all(block == 0.0) for block in risk)


def test_negative_notional_flips_every_sign():
    model = build()
    long_engine, long_risk = engine_for(model, make_product())
    short_engine, short_risk = engine_for(model, make_product(notional=-NOTIONAL))
    assert short_engine.value == -long_engine.value and short_engine.cash_amount == -long_engine.cash_amount
    for short, long in zip(short_risk, long_risk):
        assert np.array_equal(short, -long)


def test_risk_scaler_overwrite_and_accumulate():
    model = build()
    engine, once = engine_for(model, make_product())
    out = [np.full(3, 5.0), np.full(2, 5.0)]                       # stale content must be replaced
    engine.calculate_risk(out, scaler=2.0)
    for got, expected in zip(out, once):
        assert np.allclose(got, 2 * expected, rtol=1e-15, atol=0)
    engine.calculate_risk(out, scaler=0.5, accumulate=True)
    for got, expected in zip(out, once):
        assert np.allclose(got, 2.5 * expected, rtol=1e-15, atol=0)
    # a cash payment has no risk: accumulating leaves the blocks alone, overwriting zeroes them
    today, _ = engine_for(model, make_product(payment_date=VALUE_DATE))
    today.calculate_risk(out, accumulate=True)
    for got, expected in zip(out, once):
        assert np.allclose(got, 2.5 * expected, rtol=1e-15, atol=0)
    today.calculate_risk(out)
    assert all(np.all(block == 0.0) for block in out)


def test_engine_rejects_missing_funding_parameter_and_wrong_product():
    model = build()
    with raises(ValueError):
        ValuationEngineProductFixedAccrued(model, qfCreateValuationParametersCollection([]), make_product(),
                                           ValuationRequest.PV)
    with raises(TypeError):
        ValuationEngineProductFixedAccrued(model, parameters(), "not a product", ValuationRequest.PV)


def test_registry_and_value_report_dispatch_to_the_engine():
    model = build()
    product = make_product()
    engine = ValuationEngineProductRegistry.new_valuation_engine(model, product, parameters(), ValuationRequest.PV)
    assert isinstance(engine, ValuationEngineProductFixedAccrued)
    reference, risk = engine_for(model, product)
    assert qfCreateValueReport(model, product, parameters(), "PV") == [["USD", reference.value]]
    report = qfCreateValueReport(model, product, parameters(), "FirstOrderRisk")
    assert isinstance(report, RiskReport)
    for mine, theirs in zip(risk, report.state_gradient):
        assert np.array_equal(mine, theirs)
    table = report.display()
    sofr_name = model.retrieve_model_component("SOFR-1B").component_identifier.name()
    assert table["COMPONENT"].tolist() == [sofr_name] * 3 + [FUNDING_INDEX] * 2
    assert table["NODE_INDEX"].tolist() == [0, 1, 2, 0, 1]
    assert table["AXIS1"].tolist() == ["2026-04-02", "2027-01-02", "2028-01-02", "2026-07-02", "2027-01-02"]
    assert np.allclose(table["VALUES"].to_numpy(), np.concatenate(risk) * 1e-4, rtol=1e-15, atol=0)


def test_funding_index_can_come_from_the_currency_map():
    model = build()
    product = make_product()
    direct, _ = engine_for(model, product)
    via_map, _ = engine_for(model, product, parameters({"CURRENCIES": "USD", "FUNDING INDICES": FUNDING_INDEX}))
    assert via_map.funding_index.name() == FUNDING_INDEX and via_map.value == direct.value


def test_value_report_separates_pv_from_cash():
    model = build()
    engine, _ = engine_for(model, make_product())
    frame = engine.get_value_and_cash().display()
    assert frame["Type"].tolist() == ["PV", "CASH"] and frame["Value"].tolist() == [engine.value, 0.0]


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
