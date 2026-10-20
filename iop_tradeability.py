"""
Ex-ante tradeability test: does the CME pre-open indicative price (IOP) let you
trade Hyperliquid's convergence to the CME Sunday reopen?

Rule: at `lead` seconds before the CME open, compute s = IOP - HL (log-return space).
Take sign(s) on HL, exit at HL open+30s. Net P&L = sign(s) * y - fee.

Input: the weekend table built by weekend_gap_study.py, saved as
data/weekend_table.parquet (gitignored). One row per sym-weekend, with columns
  sym, cme_gap, hl-<lead>, iop-<lead> (one pair per lead, seconds), hl30.
Values are returns vs the Friday reference, as produced by the study.

Entry uses snapshot prices, so results are an upper bound (no spread paid at entry).
"""
import argparse
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


def signflip_p(x, n=20000, seed=0):
    """One-sided sign-flip permutation p-value for mean(x) > 0."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    sims = (rng.choice([-1, 1], (n, len(x))) * x).mean(axis=1)
    return float((sims >= x.mean()).mean())


def run(L, sym="XYZ:CL", fee_bp=10.0, exit_col="hl30", min_n=5):
    leads = sorted(
        {int(m[1]) for c in L.columns
         if (m := re.fullmatch(r"iop-(\d+)", c)) and f"hl-{m[1]}" in L.columns},
        reverse=True,
    )
    if exit_col not in L.columns:
        raise KeyError(f"{exit_col} not in table")

    is_px = L[f"hl-{leads[0]}"].abs().median() > 20  # prices vs returns
    scale = 1e4 if (is_px or L["cme_gap"].abs().std() < 0.5) else 100  # -> bp
    lr = (lambda a, b: np.log(a / b)) if is_px else (lambda a, b: a - b)

    df = L[L.sym == sym]
    rows = []
    for lead in leads:
        d = df.dropna(subset=[f"iop-{lead}", f"hl-{lead}", exit_col])
        s = lr(d[f"iop-{lead}"], d[f"hl-{lead}"])
        y = lr(d[exit_col], d[f"hl-{lead}"])
        fit = sm.OLS(y, sm.add_constant(s)).fit(cov_type="HC1")
        for c_bp in (0.0, fee_bp, 2 * fee_bp):
            m = (s.abs() * scale) > c_bp
            if m.sum() < min_n:
                continue
            gross = np.sign(s[m]) * y[m] * scale
            net = gross - fee_bp
            rows.append(dict(
                lead_s=lead, threshold_bp=c_bp,
                k=fit.params.iloc[1], k_se=fit.bse.iloc[1],
                n=int(m.sum()), net_bp=net.mean(),
                net_se_bp=net.std(ddof=1) / np.sqrt(m.sum()),
                p_signflip=signflip_p(gross),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/weekend_table.parquet")
    ap.add_argument("--sym", default="XYZ:CL")
    ap.add_argument("--fee-bp", type=float, default=10.0)
    a = ap.parse_args()
    res = run(pd.read_parquet(a.table), sym=a.sym, fee_bp=a.fee_bp)
    print(res.round(3).to_string(index=False))
    print("\nPre-registered config: lead 60s, threshold = fee. Other rows are robustness only.")
