import itertools
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from scipy.stats import linregress, spearmanr


HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_COMMODITY_DEX = "xyz"
HL_CANDLE_INTERVAL = "1h"
HL_REQUEST_START = "2024-12-01"
DAILY_START = "2024-11-01"

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

MIN_WEEKEND_BARS = 40
FIRST_OUT_OF_SAMPLE_WEEKEND = 10
TIMING_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

ASSETS = ["brent", "wti", "gold", "silver", "platinum"]

SESSIONS = {
    "brent": {
        "tickers": ["BRENTOIL", "BRENT"],
        "stooq": "cb.f",
        "yahoo": "BZ=F",
        "timezone": "Europe/London",
        "friday_close": "23:00",
        "sunday_open": "23:00",
    },
    "wti": {
        "tickers": ["CL", "WTI", "WTIOIL"],
        "stooq": "cl.f",
        "yahoo": "CL=F",
        "timezone": "America/New_York",
        "friday_close": "17:00",
        "sunday_open": "18:00",
    },
    "gold": {
        "tickers": ["GOLD", "XAU"],
        "stooq": "gc.f",
        "yahoo": "GC=F",
        "timezone": "America/New_York",
        "friday_close": "17:00",
        "sunday_open": "18:00",
    },
    "silver": {
        "tickers": ["SILVER", "XAG"],
        "stooq": "si.f",
        "yahoo": "SI=F",
        "timezone": "America/New_York",
        "friday_close": "17:00",
        "sunday_open": "18:00",
    },
    "platinum": {
        "tickers": ["PLATINUM", "XPT"],
        "stooq": "pl.f",
        "yahoo": "PL=F",
        "timezone": "America/New_York",
        "friday_close": "17:00",
        "sunday_open": "18:00",
    },
}


def to_utc(moment):
    moment = pd.Timestamp(moment)
    if moment.tzinfo is None:
        return moment.tz_localize("UTC")
    return moment.tz_convert("UTC")


def cache_path(name):
    safe_name = name.replace(":", "_").replace("=", "_")
    return DATA_DIR / f"{safe_name}.csv"


def load_cached(name):
    path = cache_path(name)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def save_cached(name, frame):
    DATA_DIR.mkdir(exist_ok=True)
    frame.to_csv(cache_path(name))


def fetch_hl_coin_names():
    response = requests.post(HL_INFO_URL, json={"type": "meta", "dex": HL_COMMODITY_DEX}, timeout=30)
    response.raise_for_status()
    names = []
    for market in response.json()["universe"]:
        names.append(market["name"])
    return names


def resolve_coin(names, candidates):
    by_ticker = {}
    for name in names:
        ticker = name.split(":")[-1].upper()
        by_ticker[ticker] = name
    for candidate in candidates:
        if candidate in by_ticker:
            return by_ticker[candidate]
    raise KeyError(f"none of {candidates} listed on Hyperliquid dex '{HL_COMMODITY_DEX}'")


def fetch_hl_candles(coin):
    start_ms = int(to_utc(HL_REQUEST_START).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    request = {"coin": coin, "interval": HL_CANDLE_INTERVAL, "startTime": start_ms, "endTime": end_ms}
    response = requests.post(HL_INFO_URL, json={"type": "candleSnapshot", "req": request}, timeout=30)
    response.raise_for_status()
    rows = response.json()
    if len(rows) == 0:
        raise ValueError(f"no candles returned for {coin}")
    bars = pd.DataFrame(rows)
    bars = bars.rename(columns={"t": "open_time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for column in ["open", "high", "low", "close", "volume"]:
        bars[column] = bars[column].astype(float)
    bars.index = pd.to_datetime(bars["open_time"], unit="ms", utc=True)
    bars = bars[["open", "high", "low", "close", "volume"]].sort_index()
    time.sleep(0.3)
    return bars


def get_hl_bars(name, coin):
    cached = load_cached(f"hl_{name}")
    if cached is not None:
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached
    bars = fetch_hl_candles(coin)
    save_cached(f"hl_{name}", bars)
    return bars


def fetch_daily_from_stooq(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    bars = pd.read_csv(url, parse_dates=["Date"])
    bars = bars.set_index("Date")
    bars = bars.rename(columns={"Open": "open", "Close": "close"})
    return bars[["open", "close"]].dropna()


def fetch_daily_from_yahoo(symbol):
    bars = yf.download(symbol, start=DAILY_START, interval="1d", auto_adjust=False, progress=False)
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars = bars.rename(columns={"Open": "open", "Close": "close"})
    return bars[["open", "close"]].dropna()


def get_daily_bars(name, session):
    cached = load_cached(f"daily_{name}")
    if cached is not None:
        cached.index = pd.to_datetime(cached.index).normalize()
        return cached, "cache"
    try:
        bars = fetch_daily_from_stooq(session["stooq"])
        source = "stooq"
    except Exception:
        bars = pd.DataFrame()
        source = ""
    if len(bars) == 0:
        bars = fetch_daily_from_yahoo(session["yahoo"])
        source = "yahoo"
    bars.index = pd.to_datetime(bars.index).normalize()
    save_cached(f"daily_{name}", bars)
    return bars, source


def weekend_bounds(session, friday):
    exchange_timezone = ZoneInfo(session["timezone"])
    sunday = friday + pd.Timedelta(days=2)
    close_local = pd.Timestamp(f"{friday.date()} {session['friday_close']}", tz=exchange_timezone)
    open_local = pd.Timestamp(f"{sunday.date()} {session['sunday_open']}", tz=exchange_timezone)
    return close_local.tz_convert("UTC"), open_local.tz_convert("UTC")


def price_at(close_series, moment):
    earlier = close_series[close_series.index < moment]
    return earlier.iloc[-1]


def safe_ratio(numerator, denominator):
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def build_weekend_table(session, hl_bars, daily_bars, btc_close):
    hl_close = hl_bars["close"]
    absolute_returns = np.log(hl_close).diff().abs()
    fridays = daily_bars.index[daily_bars.index.weekday == 4]
    rows = []

    for friday in fridays:
        monday = friday + pd.Timedelta(days=3)
        if monday not in daily_bars.index:
            continue

        weekend_start, weekend_end = weekend_bounds(session, friday)
        if hl_bars.index[0] >= weekend_start:
            continue
        if hl_bars.index[-1] < weekend_end:
            continue

        window = hl_bars.loc[weekend_start:weekend_end]
        if len(window) < MIN_WEEKEND_BARS:
            continue

        friday_close = daily_bars.loc[friday, "close"]
        monday_open = daily_bars.loc[monday, "open"]
        hl_start_price = price_at(hl_close, weekend_start)
        hl_end_price = price_at(hl_close, weekend_end)
        btc_start_price = price_at(btc_close, weekend_start)
        btc_end_price = price_at(btc_close, weekend_end)

        row = {}
        row["friday"] = friday
        row["gap"] = np.log(monday_open / friday_close)
        row["hl_move"] = np.log(hl_end_price / hl_start_price)
        row["btc_move"] = np.log(btc_end_price / btc_start_price)
        row["basis_at_open"] = np.log(hl_end_price / monday_open)

        for fraction in TIMING_FRACTIONS:
            moment = weekend_start + fraction * (weekend_end - weekend_start)
            row[f"hl_move_{fraction}"] = np.log(price_at(hl_close, moment) / hl_start_price)

        row["weekend_dollar_volume"] = (window["volume"] * window["close"]).sum()

        inside = hl_bars.loc[weekend_start:weekend_end - pd.Timedelta(hours=1)]
        last_hour = inside.index[-1]
        other_hours = inside.index[:-1]
        typical_move = absolute_returns.loc[other_hours].median()
        typical_volume = inside["volume"].iloc[:-1].median()
        row["final_hour_move_ratio"] = safe_ratio(absolute_returns.loc[last_hour], typical_move)
        row["final_hour_volume_ratio"] = safe_ratio(inside["volume"].iloc[-1], typical_volume)

        rows.append(row)

    table = pd.DataFrame(rows).set_index("friday")
    table["gap_up"] = (table["gap"] > 0).astype(int)
    return table


def test_gap_fit(table):
    moves = table["hl_move"]
    gaps = table["gap"]

    fit = linregress(moves, gaps)
    residuals = gaps - fit.intercept - fit.slope * moves

    huber = sm.RLM(gaps, sm.add_constant(moves), M=sm.robust.norms.HuberT()).fit()

    jackknife_slopes = []
    for friday in table.index:
        jackknife_slopes.append(linregress(moves.drop(friday), gaps.drop(friday)).slope)

    biggest_two = moves.abs().nlargest(2).index
    trimmed = linregress(moves.drop(biggest_two), gaps.drop(biggest_two))

    return {
        "weekends": len(table),
        "slope": fit.slope,
        "slope_se": fit.stderr,
        "r2": fit.rvalue ** 2,
        "residual_sd": residuals.std(),
        "huber_slope": huber.params.iloc[1],
        "jackknife_min": min(jackknife_slopes),
        "jackknife_max": max(jackknife_slopes),
        "slope_without_top_two": trimmed.slope,
        "median_abs_basis_at_open": table["basis_at_open"].abs().median(),
    }


def timing_r2(table):
    result = {}
    for fraction in TIMING_FRACTIONS:
        column = f"hl_move_{fraction}"
        result[f"{int(fraction * 100)}%"] = linregress(table[column], table["gap"]).rvalue ** 2
    return result


def test_information_timing(table):
    biggest_two = table["gap"].abs().nlargest(2).index
    return timing_r2(table), timing_r2(table.drop(biggest_two))


def test_move_size(table):
    size = table["hl_move"].abs()
    is_big = size > size.median()
    big = table[is_big]
    small = table[~is_big]

    big_fit = linregress(big["hl_move"], big["gap"])
    small_fit = linregress(small["hl_move"], small["gap"])

    return {
        "small_slope": small_fit.slope,
        "small_slope_se": small_fit.stderr,
        "small_r2": small_fit.rvalue ** 2,
        "big_slope": big_fit.slope,
        "big_slope_se": big_fit.stderr,
        "big_r2": big_fit.rvalue ** 2,
    }


def test_btc_control(table):
    regressors = sm.add_constant(table[["hl_move", "btc_move"]])
    model = sm.OLS(table["gap"], regressors).fit(cov_type="HC1")
    return {
        "beta_hl": model.params["hl_move"],
        "t_hl": model.tvalues["hl_move"],
        "beta_btc": model.params["btc_move"],
        "t_btc": model.tvalues["btc_move"],
        "r2": model.rsquared,
    }


def test_out_of_sample_hedge(table):
    gaps = table["gap"].to_numpy()
    moves = table["hl_move"].to_numpy()

    unhedged = []
    fitted_hedged = []
    one_to_one_hedged = []

    for i in range(FIRST_OUT_OF_SAMPLE_WEEKEND, len(table)):
        past_fit = linregress(moves[:i], gaps[:i])
        hedge_ratio = past_fit.slope
        unhedged.append(gaps[i])
        fitted_hedged.append(gaps[i] - hedge_ratio * moves[i])
        one_to_one_hedged.append(gaps[i] - moves[i])

    unhedged_variance = np.var(unhedged)

    return {
        "oos_weekends": len(unhedged),
        "hedge_ratio_full_sample": linregress(moves, gaps).slope,
        "variance_removed_fitted": 1 - np.var(fitted_hedged) / unhedged_variance,
        "variance_removed_one_to_one": 1 - np.var(one_to_one_hedged) / unhedged_variance,
        "unhedged_gap_sd": np.std(unhedged),
        "hedged_gap_sd": np.std(fitted_hedged),
    }


def test_final_hour(table):
    return {
        "final_hour_move_ratio": table["final_hour_move_ratio"].median(),
        "final_hour_volume_ratio": table["final_hour_volume_ratio"].median(),
    }


def test_liquidity(table):
    return {
        "weekend_dollar_volume_m": table["weekend_dollar_volume"].median() / 1e6,
        "r2": linregress(table["hl_move"], table["gap"]).rvalue ** 2,
    }


def exact_spearman(x, y):
    observed = spearmanr(x, y)[0]
    at_least_as_strong = 0
    total = 0
    for order in itertools.permutations(y):
        total += 1
        if spearmanr(x, order)[0] >= observed - 1e-12:
            at_least_as_strong += 1
    return observed, at_least_as_strong / total


def test_tokenised_gold(gold_session, gold_table, paxg_close):
    table = gold_table.copy()
    paxg_moves = []
    for friday in table.index:
        weekend_start, weekend_end = weekend_bounds(gold_session, friday)
        paxg_moves.append(np.log(price_at(paxg_close, weekend_end) / price_at(paxg_close, weekend_start)))
    table["paxg_move"] = paxg_moves

    hl_only = sm.OLS(table["gap"], sm.add_constant(table[["hl_move"]])).fit(cov_type="HC1")
    both = sm.OLS(table["gap"], sm.add_constant(table[["hl_move", "paxg_move"]])).fit(cov_type="HC1")

    return {
        "r2_hl_only": hl_only.rsquared,
        "r2_hl_and_paxg": both.rsquared,
        "t_hl": both.tvalues["hl_move"],
        "t_paxg": both.tvalues["paxg_move"],
        "corr_hl_paxg": table["hl_move"].corr(table["paxg_move"]),
    }


def report(title, slug, rows):
    frame = pd.DataFrame(rows).set_index("asset").round(3)
    frame.to_csv(RESULTS_DIR / f"{slug}.csv")
    print()
    print(title)
    print(frame.to_string())
    return frame


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    coin_names = fetch_hl_coin_names()
    btc_close = get_hl_bars("btc", "BTC")["close"]
    paxg_close = get_hl_bars("paxg", "PAXG")["close"]

    tables = {}
    data_rows = []

    for asset in ASSETS:
        session = SESSIONS[asset]
        coin = resolve_coin(coin_names, session["tickers"])
        hl_bars = get_hl_bars(asset, coin)
        daily_bars, daily_source = get_daily_bars(asset, session)
        table = build_weekend_table(session, hl_bars, daily_bars, btc_close)
        table.to_csv(RESULTS_DIR / f"weekends_{asset}.csv")
        tables[asset] = table

        data_rows.append({
            "asset": asset,
            "hl_coin": coin,
            "daily_source": daily_source,
            "hl_first_bar": hl_bars.index[0].strftime("%Y-%m-%d"),
            "first_weekend": table.index[0].strftime("%Y-%m-%d"),
            "last_weekend": table.index[-1].strftime("%Y-%m-%d"),
            "weekends": len(table),
        })

    report("Data coverage", "data_coverage", data_rows)

    fit_rows = []
    timing_rows = []
    timing_trimmed_rows = []
    size_rows = []
    control_rows = []
    hedge_rows = []
    final_hour_rows = []
    liquidity_rows = []

    for asset in ASSETS:
        table = tables[asset]

        fit_rows.append({"asset": asset, **test_gap_fit(table)})

        full_timing, trimmed_timing = test_information_timing(table)
        timing_rows.append({"asset": asset, **full_timing})
        timing_trimmed_rows.append({"asset": asset, **trimmed_timing})

        size_rows.append({"asset": asset, **test_move_size(table)})
        control_rows.append({"asset": asset, **test_btc_control(table)})
        hedge_rows.append({"asset": asset, **test_out_of_sample_hedge(table)})
        final_hour_rows.append({"asset": asset, **test_final_hour(table)})
        liquidity_rows.append({"asset": asset, **test_liquidity(table)})

    report("1. Monday gap on Hyperliquid weekend move", "gap_fit", fit_rows)
    report("2. Information timing: R2 of gap on HL move up to each point of the weekend", "timing", timing_rows)
    report("2b. Information timing without the two largest gaps", "timing_trimmed", timing_trimmed_rows)
    report("3. Small vs big weekends", "move_size", size_rows)
    report("4. BTC control (HC1 t-stats)", "btc_control", control_rows)
    report("5. Walk-forward hedge", "hedge_oos", hedge_rows)
    report("6. Final hour before reopen vs a typical weekend hour", "final_hour", final_hour_rows)
    liquidity = report("7. Weekend liquidity vs informativeness", "liquidity", liquidity_rows)

    rho, p_value = exact_spearman(liquidity["weekend_dollar_volume_m"].to_numpy(), liquidity["r2"].to_numpy())
    print(f"\nSpearman rho = {rho:.2f}, exact one-sided p = {p_value:.3f} (n = {len(liquidity)} assets)")

    gold_result = test_tokenised_gold(SESSIONS["gold"], tables["gold"], paxg_close)
    report("8. Gold: Hyperliquid gold vs tokenised gold (PAXG)", "tokenised_gold", [{"asset": "gold", **gold_result}])


if __name__ == "__main__":
    main()
