"""Checks for the discount-factor gradients in fixedincomelib/yield_curve/yield_curve_model.py.

    python tests/test_yield_curve_gradient.py    # plain runner
    python -m pytest tests -q                    # with pytest

The model is the one built in hw5_valuation_and_first_order_risk.ipynb. The expected values are
worked out by hand from the piecewise-constant forward rates: with D = exp(-integral_0^T f) and f
constant on each node's bucket, dD/df_i = -D * (length of bucket i inside [0, T]).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fixedincomelib import (  # noqa: E402
    qfCreateBuildMethod, qfCreateData1D, qfCreateDataCollection, qfCreateModel,
    qfCreateModelBuildMethodCollection, qfDiscountFactor,
)
from fixedincomelib.date import Date, accrued  # noqa: E402
from fixedincomelib.yield_curve import YieldCurve  # noqa: E402

VALUE_DATE = "2026-01-02"
PAYMENT_DATE = "2026-10-02"
FUNDING_INDEX = "SOFR-1B-FLAT"
SOFR_NODES = {"2026-04-02": 0.03, "2027-01-02": 0.04, "2028-01-02": 0.05}
FUNDING_NODES = {"2026-07-02": 0.0, "2027-01-02": 0.001}
BUMP = 1e-6


class raises:
    """Minimal stand-in for pytest.raises so the file also runs as a plain script."""

    def __init__(self, exception):
        self.exception = exception

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            raise AssertionError(f"expected {self.exception.__name__}")
        return issubclass(kind, self.exception)


def build(sofr_nodes=SOFR_NODES, funding_nodes=FUNDING_NODES):
    """The notebook's two-component model: SOFR-1B, and SOFR-1B-FLAT as an IFR spread over it."""
    sofr = pd.DataFrame({"values": list(sofr_nodes.values())},
                        index=pd.Index(list(sofr_nodes), name="Node date"))
    funding = pd.DataFrame({"values": list(funding_nodes.values())},
                           index=pd.Index(list(funding_nodes), name="Node date"))
    data = qfCreateDataCollection([
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR", sofr),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR", funding),
    ])
    methods = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod("YIELD_CURVE_INDEX",
                            {"TARGET": "SOFR-1B", "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR"}),
        qfCreateBuildMethod("YIELD_CURVE_FUNDING",
                            {"TARGET": FUNDING_INDEX, "REFERENCE INDEX": "SOFR-1B",
                             "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR"}),
    ])
    return qfCreateModel(VALUE_DATE, "YIELD_CURVE", data, methods)


def components(model):
    return model.retrieve_model_component("SOFR-1B"), model.retrieve_model_component(FUNDING_INDEX)


def curve_time(date):
    return accrued(Date(VALUE_DATE), Date(date))


def bucket_lengths(node_times, upper):
    """Length of each node's bucket inside [0, upper]: node i owns (t_{i-1}, t_i], the last node is open-ended."""
    edges = [0.0, *node_times[:-1]]
    inner = [max(0.0, min(upper, hi) - lo) for lo, hi in zip(edges[:-1], node_times[:-1])]
    return np.array(inner + [max(0.0, upper - edges[-1])])


def local_gradient_by_hand(component, date):
    lengths = bucket_lengths(component.state_data[0], curve_time(date))
    return -np.exp(-float(np.dot(component.state_data[1], lengths))) * lengths


def local_gradient(component, date, **kwargs):
    out = np.zeros(component.num_state_data)
    component.discount_factor_gradient_wrt_state(Date(date), out, **kwargs)
    return out


def model_gradient(model, index, date, **kwargs):
    out = []
    model.discount_factor_gradient_wrt_state(index, Date(date), out, **kwargs)
    return out


def central_difference(model, index, node, price):
    model.perturb_model_parameter(index, node, BUMP)
    up = price()
    model.perturb_model_parameter(index, node, -2 * BUMP)
    down = price()
    model.perturb_model_parameter(index, node, BUMP)      # back to the original state
    return (up - down) / (2 * BUMP)


# --------------------------------------------------------------------------- #
# the component's local gradient (TODO 1)
# --------------------------------------------------------------------------- #
def test_node_times_are_act_act_isda_years_from_the_value_date():
    sofr, funding = components(build())
    assert np.allclose(sofr.state_data[0][:2], [90 / 365, 1.0], rtol=0, atol=1e-15)
    assert np.allclose(funding.state_data[0], [181 / 365, 1.0], rtol=0, atol=1e-15)
    assert abs(curve_time(PAYMENT_DATE) - 273 / 365) < 1e-15


def test_local_gradient_is_minus_df_times_bucket_lengths():
    for component in components(build()):
        for date in ("2026-02-02", PAYMENT_DATE, "2027-06-02", "2031-01-06"):
            expected = local_gradient_by_hand(component, date)
            assert np.allclose(local_gradient(component, date), expected, rtol=0, atol=1e-14), (
                component.component_identifier.name(), date)


def test_local_gradient_matches_central_bump_and_reval():
    model = build()
    for index, component in zip(("SOFR-1B", FUNDING_INDEX), components(model)):
        analytic = local_gradient(component, PAYMENT_DATE)
        for node in range(component.num_state_data):
            finite = central_difference(model, index, node, lambda: component.discount_factor(Date(PAYMENT_DATE)))
            assert abs(finite - analytic[node]) < 1e-7, (index, node)


def test_nodes_after_the_date_carry_no_gradient():
    sofr, _ = components(build())
    gradient = local_gradient(sofr, PAYMENT_DATE)
    assert np.all(gradient[:2] < 0) and gradient[2] == 0.0
    first_bucket_only = local_gradient(sofr, "2026-02-02")
    assert first_bucket_only[0] < 0 and np.all(first_bucket_only[1:] == 0.0)


def test_local_gradient_on_the_value_date_is_zero():
    sofr, funding = components(build())
    assert np.all(local_gradient(sofr, VALUE_DATE) == 0.0) and np.all(local_gradient(funding, VALUE_DATE) == 0.0)


def test_local_gradient_scaler_overwrite_and_accumulate():
    sofr, _ = components(build())
    base = local_gradient(sofr, PAYMENT_DATE)
    out = np.full(sofr.num_state_data, 7.0)                       # stale content must be replaced
    sofr.discount_factor_gradient_wrt_state(Date(PAYMENT_DATE), out, scaler=2.0)
    assert np.allclose(out, 2 * base, rtol=0, atol=1e-15)
    sofr.discount_factor_gradient_wrt_state(Date(PAYMENT_DATE), out, scaler=0.5, accumulate=True)
    assert np.allclose(out, 2.5 * base, rtol=0, atol=1e-15)


def test_local_gradient_rejects_bad_containers_and_dates():
    sofr, _ = components(build())
    for bad in (np.zeros(2), np.zeros(3, dtype=int), [0.0, 0.0, 0.0], np.zeros((3, 1))):
        with raises(ValueError):
            sofr.discount_factor_gradient_wrt_state(Date(PAYMENT_DATE), bad)
    for bad_date in (Date("null date"), Date("2025-12-31")):
        with raises(ValueError):
            sofr.discount_factor_gradient_wrt_state(bad_date, np.zeros(3))


# --------------------------------------------------------------------------- #
# the model's gradient through the reference chain (TODO 2)
# --------------------------------------------------------------------------- #
def test_model_gradient_applies_the_product_rule_along_the_reference_chain():
    model = build()
    sofr, funding = components(model)
    d_sofr, d_spread = sofr.discount_factor(Date(PAYMENT_DATE)), funding.discount_factor(Date(PAYMENT_DATE))
    sofr_block, funding_block = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)
    assert np.allclose(sofr_block, d_spread * local_gradient(sofr, PAYMENT_DATE), rtol=0, atol=1e-15)
    assert np.allclose(funding_block, d_sofr * local_gradient(funding, PAYMENT_DATE), rtol=0, atol=1e-15)
    # the same thing by hand: dD_funding/df_i = -D_funding * (bucket length of node i), for both curves
    d_funding = qfDiscountFactor(model, FUNDING_INDEX, PAYMENT_DATE)
    assert abs(d_funding - d_sofr * d_spread) < 1e-15
    T = curve_time(PAYMENT_DATE)
    assert np.allclose(sofr_block, -d_funding * bucket_lengths(sofr.state_data[0], T), rtol=0, atol=1e-14)
    assert np.allclose(funding_block, -d_funding * bucket_lengths(funding.state_data[0], T), rtol=0, atol=1e-14)


def test_model_gradient_matches_bump_and_reval_of_the_full_discount_factor():
    model = build()
    blocks = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)

    def price():
        return qfDiscountFactor(model, FUNDING_INDEX, PAYMENT_DATE)

    reference = price()
    for position, index in enumerate(("SOFR-1B", FUNDING_INDEX)):
        for node in range(len(blocks[position])):
            assert abs(central_difference(model, index, node, price) - blocks[position][node]) < 1e-7, (index, node)
    assert abs(price() - reference) < 1e-15                        # the bumps were undone


def test_a_zero_spread_node_still_has_a_gradient():
    # the first funding node holds a 0.0 spread, but D_funding still moves when that node moves
    model = build()
    _, funding_block = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)
    assert funding_block[0] < 0
    assert abs(funding_block[0] / funding_block[1] - 181 / 92) < 1e-9     # ratio of the bucket lengths


def test_base_index_gradient_leaves_the_funding_block_at_zero():
    model = build()
    sofr, _ = components(model)
    sofr_block, funding_block = model_gradient(model, "SOFR-1B", PAYMENT_DATE)
    assert np.allclose(sofr_block, local_gradient(sofr, PAYMENT_DATE), rtol=0, atol=1e-15)
    assert funding_block.shape == (2,) and np.all(funding_block == 0.0)


def test_model_gradient_allocates_blocks_and_overwrites_stale_values():
    model = build()
    sofr, _ = components(model)
    fresh = model_gradient(model, "SOFR-1B", PAYMENT_DATE)
    assert [len(block) for block in fresh] == model.num_sub_components == [3, 2]
    stale = [np.full(3, 9.0), np.full(2, 9.0)]
    model.discount_factor_gradient_wrt_state("SOFR-1B", Date(PAYMENT_DATE), stale)
    assert np.allclose(stale[0], local_gradient(sofr, PAYMENT_DATE), rtol=0, atol=1e-15)
    assert np.all(stale[1] == 0.0)                                 # off the chain, so replaced by zeros
    wrong_size = [np.zeros(1)]
    model.discount_factor_gradient_wrt_state("SOFR-1B", Date(PAYMENT_DATE), wrong_size)
    assert [len(block) for block in wrong_size] == [3, 2]


def test_model_gradient_scaler_and_accumulate():
    model = build()
    once = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)
    out = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE, scaler=2.0)
    model.discount_factor_gradient_wrt_state(FUNDING_INDEX, Date(PAYMENT_DATE), out, scaler=0.5, accumulate=True)
    for got, expected in zip(out, once):
        assert np.allclose(got, 2.5 * expected, rtol=0, atol=1e-15)
    total = model_gradient(model, "SOFR-1B", PAYMENT_DATE)         # accumulate on top of another index
    model.discount_factor_gradient_wrt_state(FUNDING_INDEX, Date(PAYMENT_DATE), total, accumulate=True)
    for got, base, extra in zip(total, model_gradient(model, "SOFR-1B", PAYMENT_DATE), once):
        assert np.allclose(got, base + extra, rtol=0, atol=1e-15)


def test_model_gradient_on_the_value_date_is_zero():
    assert all(np.all(block == 0.0) for block in model_gradient(build(), FUNDING_INDEX, VALUE_DATE))


def test_model_gradient_rejects_unknown_components_and_bad_containers():
    model = build()
    with raises(KeyError):
        model.discount_factor_gradient_wrt_state("SONIA-1B", Date(PAYMENT_DATE), [])
    with raises(TypeError):
        model.discount_factor_gradient_wrt_state(FUNDING_INDEX, Date(PAYMENT_DATE), np.zeros(5))
    with raises(ValueError):
        model.discount_factor_gradient_wrt_state(FUNDING_INDEX, Date("2025-12-31"), [])


def test_gradient_survives_a_serialization_round_trip():
    model = build()
    rebuilt = YieldCurve.deserialize(model.serialize())
    for mine, theirs in zip(model_gradient(model, FUNDING_INDEX, PAYMENT_DATE),
                            model_gradient(rebuilt, FUNDING_INDEX, PAYMENT_DATE)):
        assert np.array_equal(mine, theirs)


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed")
