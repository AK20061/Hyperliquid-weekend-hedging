# Weekend price discovery on a 24/7 DEX: can Hyperliquid hedge commodity gap risk?

When CME and ICE shut for the weekend, commodity perpetuals on Hyperliquid keep trading. During the Strait of Hormuz escalation, oil traders went there to see where crude was heading while the official market was dark. This project asks what that weekend price is actually worth.

The first question was whether Hyperliquid's weekend move predicts the direction of Monday's gap. It does, but that turns out to be the wrong question: the futures market opens already reflecting the same information, so there is nothing to trade. The useful question is whether the weekend price lets you **hedge** gap risk while the exchange is closed, how well, at what ratio, when the information arrives, and why it works for some commodities and not others.

## Data

Hyperliquid hourly candles for five commodity perps on the `xyz` dex (BRENTOIL, CL, GOLD, SILVER, PLATINUM), plus BTC and PAXG as controls, from the public `info` API. Daily futures open and close come from Stooq with a yfinance fallback. Everything is free, and raw pulls are cached to `data/` so a run is reproducible.

The unit of observation is one weekend: Friday's futures close to the Sunday-evening reopen. Weekday maintenance breaks are an hour or two long and carry almost nothing, so they are excluded. The Hyperliquid candle endpoint only serves the most recent 5,000 bars, which at 1h gives about 24 usable weekends per asset. That sample size is the main limitation of everything below.

## Method

For each weekend, the gap is `log(Monday open / Friday close)` and the predictor is the Hyperliquid log move over the same dark window, measured in exchange-local time (London for Brent, New York for the rest). Eight tests were fixed in advance and run once across all assets:

1. Gap on Hyperliquid move: slope, R², Huber slope, jackknife range, slope without the two largest moves
2. Information timing: R² using the Hyperliquid move up to 10%, 25%, ... 100% of the weekend, with and without the two largest gaps
3. Small versus big weekends, split at the median absolute move
4. A BTC weekend return as a control, since Hyperliquid traders are crypto-native and the usual DXY and equity-index controls are also closed
5. A walk-forward hedge: ratio fitted only on past weekends, applied to the next one, compared with a naive 1:1 hedge
6. Activity in the final hour before the reopen relative to a typical weekend hour
7. Weekend dollar volume against informativeness across assets, with an exact permutation Spearman test
8. For gold, whether tokenised gold (PAXG) explains Hyperliquid gold's weekend information

Brent and gold were explored first. WTI, silver and platinum were added afterwards, with predictions written down before running: WTI should behave like Brent, the metals like gold.

## Results

| Asset | Weekend $ volume | R² | β_HL (t) | β_BTC (t) | OOS variance removed (fitted / 1:1) |
|---|---|---|---|---|---|
| WTI | $207m | 0.94 | 0.77 (22.0) | −0.07 (−0.7) | 88% / 79% |
| Brent | $118m | 0.83 | 0.65 (5.3) | −0.04 (−0.2) | 80% / 77% |
| Gold | $26m | 0.62 | 0.77 (3.2) | 0.05 (0.5) | 17% / 15% |
| Silver | $50m | 0.12 | −0.20 (−0.3) | 0.95 (1.6) | −10% / −11% |
| Platinum | $1m | 0.01 | −1.17 (−1.7) | 0.72 (2.3) | −4% / −25% |

**Oil is hedgeable from Hyperliquid.** A weekend position sized from past weekends removes 80% of Brent and 88% of WTI Monday-gap variance out of sample, cutting the standard deviation of the gap from 2.5% to 1.1% for Brent and from 2.2% to 0.8% for WTI. The fitted ratios of 0.66 and 0.79 beat a 1:1 hedge in both, so the sub-1 ratio is real and worth using. Both are stable: the jackknife range is 0.57–0.73 for Brent and 0.78–0.80 for WTI.

**It isn't crypto beta.** The Hyperliquid move survives a BTC control in both oil contracts with t-stats of 5.3 and 22, while BTC's coefficient is indistinguishable from zero.

**It works when it is needed.** On big-move weekends the Hyperliquid move explains 90% of the Brent gap and 98% of WTI. On quiet weekends it explains 12% and 35%, where the hedge costs more in spread and funding than the risk it removes.

**Information arrives Saturday night to Sunday morning.** Brent's R² is 0.24 at the halfway point of the weekend and 0.77 by Sunday morning London time, and still rises from 0.11 to 0.57 with the two largest weekends removed. WTI follows the same path, 0.20 to 0.72, and 0.15 to 0.59 without its two largest weekends. The final pre-open hour adds little for oil, so hedging early on Saturday captures little.

**Liquidity explains which commodities work.** Informativeness ranks with weekend dollar volume across all five assets (Spearman ρ = 0.90, exact one-sided p = 0.042 at n = 5). Silver and platinum see bursts of volume before the reopen but on a thin base, and their weekend prices carry no information about Monday. Gold is the exception: more informative than its volume suggests, with nearly all of it arriving in the final hours before the reopen. Tokenised gold does not explain it. Measured over the same window, Hyperliquid gold and PAXG move together at a correlation of 0.99, and adding PAXG lifts R² only from 0.622 to 0.624. Over the weekend they are the same market, so the gold exception remains open.

## What did not work

Predicting the sign of the gap is close to trivial over a 49-hour weekend and was never distinguishable from the base rate at n = 24. A signed-volume proxy for order-flow imbalance added nothing, so full OFI reconstruction was not pursued. A post-reopen fade on Hyperliquid and a price-discovery share test (does CME converge to Hyperliquid, or the reverse) were both null and mutually inconsistent across assets. An early explanation that Hyperliquid gold was merely echoing spot gold's reopen was rejected when silver and platinum, which share that reopen, showed no such effect. An early result that PAXG carried information independent of Hyperliquid gold disappeared once both were measured over an identical window.

Roughly twenty tests were run during exploration before the final set was fixed. At that count one or two results near |t| ≈ 2.5 are expected by chance, which is why only the pre-registered set is reported as findings. The same applies inside the control table: on simulated data with no true relationship, the script produced |t| ≈ 3 on both coefficients for one asset. Platinum's β_BTC at t = 2.3 is read as noise for that reason.

## Caveats

About 24 weekends per asset, limited by the candle API's 5,000-bar window. Daily open and close from free sources rather than exchange-grade settlement and reopen prints. Brent shows the largest median basis at the reopen (1.6%) and the lowest ratio (0.66), consistent with Hyperliquid's Brent referencing a different contract month than the continuous series. Hedge costs (weekend spreads, funding) are not modelled. Holiday weekends are skipped.

## Running it

```
pip install requests pandas numpy scipy statsmodels yfinance
python weekend_gap_study.py
```

Raw data is cached in `data/` on first run. Delete it to refresh. Every table above is written to `results/` as CSV, along with the per-weekend panel for each asset.


See [FINDINGS.md](FINDINGS.md) for the reopen-convergence and IOP tradeability results.
