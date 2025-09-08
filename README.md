# Option Pricing Demo

This repository contains small Python functions that price a European call option.
It aims to be easy to read for people who are new to programming and finance.

## Files

- `montecarlo.py` – core functions.
- `test_montecarlo.py` – simple tests that show how the functions work.

## Two ways to price the option

### 1. Black–Scholes formula

`bs_call_price(S0, K, r, sigma, T)` plugs numbers into the classic
Black–Scholes equation.  You give:

- `S0`: current stock price.
- `K`: strike price (the price at which you can buy the stock in future).
- `r`: risk‑free interest rate.
- `sigma`: volatility, a measure of how much the stock moves.
- `T`: time to expiry in years.

If either `sigma` or `T` is zero the function simply returns the discounted
intrinsic value `max(S0 − K, 0)`.

### 2. Monte Carlo simulation

`call_mc_price(...)` uses random numbers to imitate future stock prices.
For each simulated future price we compute the option payoff and then take the
average.  The function returns an `MCResult` object with:

- `price`: the estimated option value.
- `se`: standard error of the estimate.
- `ci95`: a 95% confidence interval.
- `n_effective`: number of effective simulations used.

Setting `antithetic=True` pairs each random number with its negative.
This **antithetic variate** technique makes the estimate more stable.

## Example

```python
from montecarlo import bs_call_price, call_mc_price

S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
mc = call_mc_price(S0, K, r, sigma, T, nsim=100_000, seed=42)
bs = bs_call_price(S0, K, r, sigma, T)
print("MC price", mc.price)
print("Black–Scholes price", bs)
```

## Requirements

- Python 3.11+
- NumPy (needed only for the Monte Carlo function)

## Running tests

```bash
pytest -q
```
