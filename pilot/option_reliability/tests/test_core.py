import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core import bounds, normalise, parity, verify, witness


def toy():
    rows = []
    for i, k in enumerate([.8, 1., 1.2]):
        for side, prices in [("put", [.15, .25, .35]), ("call", [.35, .25, .15])]:
            rows.append({"instrument": f"{k}-{side}", "strike": k, "type": side,
                         "bid": prices[i], "ask": prices[i]})
    return normalise(rows, 1., 1.)


@pytest.mark.parametrize("defect,top", [(0., 1.5), (.1, 1.3)])
def test_exact_toy_and_solver(defect, top):
    assert verify([.5, top], [.5, .5], toy(), defect)["valid"]
    assert witness(toy(), defect)["status"] == "admitted_by_checked_witness"


def test_discount_units_and_parity():
    # Dollar prices generated independently from a discounted two-atom law.
    spot, discount, strike = 200., .95, 210.
    put = discount * (.4*max(strike-100, 0)+.6*max(strike-250, 0))
    defect_dollars = spot-discount*(.4*100+.6*250)
    call = discount*(.4*max(100-strike, 0)+.6*max(250-strike, 0))+defect_dollars
    q = normalise([{"instrument": side, "type": side, "strike": strike, "bid": price, "ask": price}
                   for side, price in [("put", put), ("call", call)]], spot, discount)
    b = bounds(q)
    assert b["U_put_raw"] == pytest.approx(b["U_call_raw"])
    assert all(r["compatible"] for r in parity(q))
    assert verify([discount*100/spot, discount*250/spot], [.4, .6], q, defect_dollars/spot)["valid"]


def test_relaxation_preserves_witness_and_bounds():
    q = toy()
    w = witness(q, .1)
    loose = [{**r, "hi": r["hi"]+.03, "lo": max(0., r["lo"]-.03)} for r in q]
    assert verify(w["support"], w["probabilities"], loose, .1)["valid"]
    assert verify(w["support"], w["probabilities"], q[:-2], .1)["valid"]
    assert bounds(loose)["U_combined"] >= bounds(q)["U_combined"]
    assert bounds(q[:-2])["U_combined"] >= bounds(q)["U_combined"]


def test_equal_call_control_excludes_zero():
    q = normalise([{"instrument": str(k), "type": "call", "strike": k, "bid": .1, "ask": .1}
                   for k in [1., 2.]], 1., 1.)
    assert verify([.9], [1.], q, .1)["valid"]
    assert witness(q, .1)["status"] == "admitted_by_checked_witness"
    assert witness(q, 0.)["status"] == "unresolved_on_searched_supports"
    # Analytic exclusion is in the report; solver failure itself is not a certificate.


def test_negative_bound_not_clipped_and_minimum_not_highest_strike():
    q = normalise([{"instrument": "negative", "type": "put", "strike": 2., "ask": .5, "bid": None},
                   {"instrument": "lower-strike", "type": "call", "strike": 1., "ask": .2, "bid": 0.},
                   {"instrument": "highest-strike", "type": "call", "strike": 2., "ask": .4, "bid": None}], 1., 1.)
    b = bounds(q)
    assert b["U_put_raw"] == -.5
    assert b["negative_put_inconsistency"]
    assert b["call_instrument"] == "lower-strike"


def test_tail_expansion_can_restore_zero():
    q = normalise([{"instrument": "p", "type": "put", "strike": 1., "bid": .99, "ask": .99}], 1., 1.)
    assert witness(q, 0., "A", ceilings=[2])["status"] == "unresolved_on_searched_supports"
    w = witness(q, 0., "A", ceilings=[2, 100, 1000])
    assert w["status"] == "admitted_by_checked_witness"
    assert max(w["support"]) >= 100


def test_missing_sides_and_bad_witness():
    q = [{**r, "lo": None} for r in toy()]
    assert verify([.5, 1.3], [.5, .5], q, .1)["valid"]
    assert not verify([.5, 1.3], [.5, .4], q, .1)["valid"]
    with pytest.raises(ValueError):
        normalise([], 0., 1.)
