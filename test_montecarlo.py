import pytest

from montecarlo import bs_call_price, call_mc_price


def test_bs_call_price_known_value():
    price = bs_call_price(100.0, 100.0, 0.01, 0.2, 1.0)
    assert abs(price - 8.433318690109608) < 1e-9


def test_call_mc_price_requires_numpy():
    with pytest.raises(ImportError):
        call_mc_price(100.0, 100.0, 0.01, 0.2, 1.0, nsim=10)
