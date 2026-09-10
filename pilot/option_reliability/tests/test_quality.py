from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse import clean_batch, discount_diagnostic
from core import normalise, verify
from collect import choose_expiries


def test_missing_six_month_expiry_is_not_replaced_by_short_expiry(config):
    unit = 86400000
    selected, missing = choose_expiries([18*unit, 53*unit, 109*unit], 0, config)
    assert selected == [(30, 18*unit), (90, 109*unit)]
    assert [m["target_days"] for m in missing] == [180]


@pytest.fixture
def batch():
    now = datetime.now(timezone.utc)
    spec = {"instrument_name": "example", "option_type": "put", "strike": 100.,
            "contract_size": 1, "price_index": "btc_usdc", "instrument_type": "linear",
            "settlement_currency": "USDC", "quote_currency": "USDC", "base_currency": "BTC",
            "expiration_timestamp": (now.timestamp()+86400)*1000}
    return {"reference_before": {"result": {"index_price": 100.}},
            "reference_after": {"result": {"index_price": 100.}}, "elapsed_seconds": 1.,
            "books": [{"spec": spec, "meta": {"retrieval_finished": now.isoformat()},
                       "result": {"timestamp": now.timestamp()*1000, "state": "open", "index_price": 100.,
                                  "bids": [[0., 1.]], "asks": [[2., 1.]]}}]}


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())


def test_zero_bid_and_missing_side(batch, config):
    qs, _, quality = clean_batch(batch, config)
    assert quality["usable"] and qs[0]["bid"] == 0.
    batch["books"][0]["result"]["bids"] = []
    qs, issues, _ = clean_batch(batch, config)
    assert qs[0]["bid"] is None and qs[0]["constraint"] == "one-sided"
    assert any(x["reason"] == "missing quoted side" for x in issues)


@pytest.mark.parametrize("mutation", ["crossed", "expired", "stale", "inverse", "zero_ask"])
def test_unusable_records_are_rejected(batch, config, mutation):
    r = batch["books"][0]
    if mutation == "crossed": r["result"]["bids"] = [[3., 1.]]
    if mutation == "expired": r["spec"]["expiration_timestamp"] = 1
    if mutation == "stale": r["result"]["timestamp"] -= 60000
    if mutation == "inverse": r["spec"]["instrument_type"] = "reversed"
    if mutation == "zero_ask": r["result"]["asks"] = [[0., 1.]]
    qs, issues, _ = clean_batch(batch, config)
    assert not qs and issues


def test_nonpositive_size_is_not_used(batch, config):
    batch["books"][0]["result"]["asks"] = [[2., 0.]]
    qs, _, _ = clean_batch(batch, config)
    assert qs[0]["ask"] is None and qs[0]["bid"] == 0.


@pytest.mark.parametrize("mutation", ["slow", "moving"])
def test_batch_quality_gate(batch, config, mutation):
    if mutation == "slow": batch["elapsed_seconds"] = 31
    if mutation == "moving": batch["reference_after"]["result"]["index_price"] = 101
    qs, issues, quality = clean_batch(batch, config)
    assert not qs and issues and not quality["usable"]


def test_discount_diagnostic_recovers_exact_parity():
    qs = [{"type": "call", "strike": 100., "bid": 10., "ask": 10.},
          {"type": "put", "strike": 100., "bid": 5., "ask": 5.}]
    result = discount_diagnostic(qs, 100., 1.)
    assert result["D_lower"] == pytest.approx(.95)
    assert result["D_upper"] == pytest.approx(.95)


def test_nonfinite_or_mismatched_witness_rejected():
    for xs, ps in [([float("nan")], [1.]), ([1., 2.], [1.]), ([], [])]:
        with pytest.raises(ValueError):
            verify(xs, ps, [], 0.)
