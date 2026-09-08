"""Analisi obbligazionaria: curva dei rendimenti BCE + ETF obbligazionari quotati.

Nota metodologica: i dati sui singoli titoli di Stato (ISIN, cedola, YTM) non sono
disponibili gratuitamente in modo affidabile. Questo modulo lavora su due livelli:
  1. la curva dei rendimenti area euro (BCE) per leggere il contesto macro;
  2. gli ETF obbligazionari come strumenti effettivamente investibili.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from .data_sources import fetch_ecb_yield_curve
from .technicals import compute_snapshot

log = logging.getLogger(__name__)

_MAT_YEARS = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5,
              "7Y": 7, "10Y": 10, "15Y": 15, "20Y": 20, "30Y": 30}


def curve_analysis(maturities: list[str]) -> dict:
    """Scarica la curva e calcola pendenza, inversione e variazioni recenti."""
    aaa = fetch_ecb_yield_curve(maturities, rating="AAA")
    allb = fetch_ecb_yield_curve(maturities, rating="ALL")

    if aaa.empty:
        return {"curve_aaa": pd.DataFrame(), "curve_all": pd.DataFrame(), "metrics": {}}

    latest = aaa.iloc[-1]
    metrics: dict = {"date": aaa.index[-1].date().isoformat()}

    def spread(a, b):
        if a in aaa.columns and b in aaa.columns:
            return float(latest[a] - latest[b])
        return np.nan

    metrics["slope_10y_2y"] = spread("10Y", "2Y")
    metrics["slope_30y_10y"] = spread("30Y", "10Y")
    metrics["slope_10y_3m"] = spread("10Y", "3M")
    metrics["inverted"] = bool(metrics["slope_10y_2y"] < 0) if not np.isnan(metrics["slope_10y_2y"]) else None

    # variazione dei rendimenti negli ultimi 1 e 3 mesi (in punti base)
    for label, lag in [("1m", 21), ("3m", 63)]:
        if len(aaa) > lag:
            prev = aaa.iloc[-lag - 1]
            for m in ["2Y", "10Y"]:
                if m in aaa.columns:
                    metrics[f"chg_{m}_{label}_bps"] = float((latest[m] - prev[m]) * 100)

    # premio al rischio: tutti gli emittenti vs solo AAA
    if not allb.empty and "10Y" in allb.columns and "10Y" in aaa.columns:
        metrics["credit_spread_10y"] = float(allb.iloc[-1]["10Y"] - latest["10Y"])

    return {"curve_aaa": aaa, "curve_all": allb, "metrics": metrics}


def analyze_bond_etfs(etf_df: pd.DataFrame, period: str = "3y") -> pd.DataFrame:
    """Metriche di rischio/rendimento e tecniche per ogni ETF obbligazionario."""
    rows = []
    for _, row in etf_df.iterrows():
        t = row["ticker"]
        rec = {"ticker": t, "name": row.get("name", t), "bucket": row.get("bucket", "")}
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period=period, auto_adjust=True)
            if len(hist) < 60:
                continue

            snap = compute_snapshot(hist)
            rec.update({k: snap.get(k) for k in
                        ["price", "rsi14", "px_vs_sma200", "mom_3m", "mom_6m",
                         "mom_12m", "volatility_ann", "max_drawdown_1y",
                         "pct_from_52w_high", "pos_in_52w_range"]})

            info = tk.info or {}
            y = info.get("yield") or info.get("trailingAnnualDividendYield")
            if y is not None:
                rec["distribution_yield"] = y * 100 if y <= 1 else y
            rec["ter"] = info.get("annualReportExpenseRatio")
            rec["aum"] = info.get("totalAssets")
            rec["currency"] = info.get("currency")

            # rendimento corretto per il rischio
            ret = hist["Close"].pct_change().dropna()
            if len(ret) > 250 and ret.std() > 0:
                ann_ret = (1 + ret.mean()) ** 252 - 1
                rec["sharpe_proxy"] = float(ann_ret / (ret.std() * np.sqrt(252)))
                rec["return_ann_3y"] = float(ann_ret * 100)
        except Exception as exc:
            log.warning("ETF %s: %s", t, exc)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # score semplice: carry + trend - rischio
    def z(c, sign=1):
        if c not in df:
            return pd.Series(0.0, index=df.index)
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 3 or not s.std():
            return pd.Series(0.0, index=df.index)
        return ((s - s.mean()) / s.std()).fillna(0) * sign

    df["score_bond"] = (
        50
        + 16.7 * (0.40 * z("distribution_yield")
                  + 0.25 * z("mom_6m")
                  + 0.20 * z("sharpe_proxy")
                  - 0.15 * z("volatility_ann"))
    ).round(1).clip(0, 100)

    return df.sort_values("score_bond", ascending=False)
