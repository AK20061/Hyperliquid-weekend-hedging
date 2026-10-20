## Findings (XYZ:CL, 23 weekends, Apr–Oct 2026)

Hyperliquid's weekend oil perp explains most of the CME Sunday reopen gap. Regressing the CME gap on the HL weekend return (HL at 21:59 UTC) gives b = 0.84 (se 0.11), R² = 0.91, cutting gap volatility from 2.74% to a residual 0.83%. HL is a better predictor of the open than CME's own pre-open indicative price (IOP: b = 0.59, R² = 0.68, n = 22).

At the reopen HL jumps on average 43 ± 16 bp toward the CME print, then is fully converged within 30 seconds (later lags indistinguishable from zero).

**The convergence is not tradeable ex ante.** A rule that trades HL toward the IOP 60s before the open and exits 30s after nets 9.6 ± 18.0 bp per weekend after a 10 bp cost assumption (sign-flip p = 0.14), using snapshot prices, so this is an upper bound. HL closes only 28% of the IOP–HL gap (k = 0.28 ± 0.13). At a 30-minute lead the IOP carries no information (k ≈ 0). Distinguishing a ~10 bp edge from zero would take roughly 270 weekends. Reproduce with `python iop_tradeability.py`.

**Implication:** HL is an effective weekend hedge venue, but not a source of directional alpha. Resting orders should be pulled before the reopen.

**Caveats:** one market (gold excluded, n = 2); b < 1 may partly reflect noise in the 21:59 snapshot rather than true overshoot; 2 weekends dropped for missing data; open time is 22:00 UTC during US daylight saving and 23:00 UTC otherwise.
