"""Motore di scoring: normalizza le metriche e costruisce punteggi compositi.

Filosofia: nessuna metrica singola decide. Ogni titolo riceve quattro punteggi
(Value, Quality, Momentum, Health) normalizzati per settore, più uno score
finale ponderato. I pesi sono configurabili dalla dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# metrica -> (verso, peso) ; verso=-1 significa "più basso è meglio"
VALUE_METRICS = {
    "trailingPE":                   (-1, 1.0),
    "forwardPE":                    (-1, 1.2),
    "priceToBook":                  (-1, 0.8),
    "enterpriseToEbitda":           (-1, 1.2),
    "priceToSalesTrailing12Months": (-1, 0.6),
    "fcf_yield":                    (+1, 1.5),
    "earnings_yield":               (+1, 1.0),
    "dividendYield":                (+1, 0.8),
}

QUALITY_METRICS = {
    "returnOnEquity":   (+1, 1.2),
    "roic":             (+1, 1.5),
    "operatingMargins": (+1, 1.0),
    "grossMargins":     (+1, 0.6),
    "profitMargins":    (+1, 0.8),
    "returnOnAssets":   (+1, 0.8),
}

MOMENTUM_METRICS = {
    "mom_12_1":     (+1, 1.3),
    "mom_6m":       (+1, 1.0),
    "mom_3m":       (+1, 0.8),
    "px_vs_sma200": (+1, 1.0),
    "rsi14":        (+1, 0.4),
    "adx14":        (+1, 0.4),
}

HEALTH_METRICS = {
    "net_debt_to_ebitda": (-1, 1.5),
    "debtToEquity":       (-1, 1.0),
    "currentRatio":       (+1, 0.8),
    "revenue_cagr_3y":    (+1, 1.0),
    "revenueGrowth":      (+1, 0.8),
    "share_count_change": (-1, 0.6),   # premia i buyback
    "volatility_ann":     (-1, 0.5),
    "max_drawdown_1y":    (+1, 0.5),   # meno negativo è meglio
}

DEFAULT_WEIGHTS = {"value": 0.30, "quality": 0.30, "momentum": 0.25, "health": 0.15}


# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le metriche derivate che servono allo scoring."""
    df = df.copy()

    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)

    mcap = num("marketCap")
    fcf = num("freeCashflow").fillna(num("fcf_calc"))
    df["fcf_yield"] = np.where(mcap > 0, fcf / mcap * 100, np.nan)

    pe = num("trailingPE")
    df["earnings_yield"] = np.where(pe > 0, 100 / pe, np.nan)

    # percentuali coerenti: yfinance restituisce frazioni per i margini
    for c in ["returnOnEquity", "returnOnAssets", "profitMargins",
              "operatingMargins", "grossMargins", "ebitdaMargins",
              "revenueGrowth", "earningsGrowth", "payoutRatio"]:
        if c in df:
            s = num(c)
            df[c] = np.where(s.abs() <= 5, s * 100, s)

    # upside vs target medio analisti
    tgt, px = num("targetMeanPrice"), num("price")
    df["analyst_upside"] = np.where((tgt > 0) & (px > 0), (tgt / px - 1) * 100, np.nan)

    # filtri di sanità: multipli assurdi vengono neutralizzati
    for c, lo, hi in [("trailingPE", 0, 150), ("forwardPE", 0, 150),
                      ("priceToBook", 0, 40), ("enterpriseToEbitda", 0, 80),
                      ("priceToSalesTrailing12Months", 0, 50),
                      ("net_debt_to_ebitda", -20, 25), ("debtToEquity", 0, 2000)]:
        if c in df:
            s = num(c)
            df[c] = s.where((s > lo) & (s < hi))

    return df


def _zscore(s: pd.Series, winsor: float = 0.05) -> pd.Series:
    """Z-score robusto: taglia le code prima di normalizzare."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(winsor), s.quantile(1 - winsor)
    s = s.clip(lo, hi)
    sd = s.std()
    if not sd or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def _block_score(df: pd.DataFrame, spec: dict, by_sector: bool = True) -> pd.Series:
    """Media pesata degli z-score di un blocco di metriche."""
    total = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)

    group = df["sector"].fillna("N/D") if (by_sector and "sector" in df) else pd.Series("ALL", index=df.index)
    # settori troppo piccoli vengono confrontati con l'intero universo
    counts = group.value_counts()
    group = group.where(group.map(counts) >= 6, "ALL")

    for metric, (direction, weight) in spec.items():
        if metric not in df.columns:
            continue
        z = df.groupby(group)[metric].transform(_zscore) * direction
        mask = z.notna()
        total[mask] += z[mask] * weight
        wsum[mask] += weight

    return (total / wsum.replace(0, np.nan)).clip(-3, 3)


def compute_scores(df: pd.DataFrame, weights: dict | None = None,
                   by_sector: bool = True) -> pd.DataFrame:
    """Calcola i quattro punteggi e lo score composito (scala 0-100)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = enrich(df)

    df["score_value"] = _block_score(df, VALUE_METRICS, by_sector)
    df["score_quality"] = _block_score(df, QUALITY_METRICS, by_sector)
    df["score_momentum"] = _block_score(df, MOMENTUM_METRICS, by_sector)
    df["score_health"] = _block_score(df, HEALTH_METRICS, by_sector)

    blocks = ["value", "quality", "momentum", "health"]
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for b in blocks:
        col = df[f"score_{b}"]
        w = weights.get(b, 0)
        num = num.add(col.fillna(0) * w, fill_value=0)
        den = den.add(col.notna().astype(float) * w, fill_value=0)

    raw = num / den.replace(0, np.nan)
    # copertura dati: penalizza i titoli con troppe metriche mancanti
    df["data_coverage"] = df[[f"score_{b}" for b in blocks]].notna().mean(axis=1) * 100
    df["score_total"] = (50 + raw * 16.7).clip(0, 100)   # z ~ ±3 -> 0-100
    df.loc[df["data_coverage"] < 50, "score_total"] = np.nan

    # etichette leggibili
    for b in blocks + ["total"]:
        c = f"score_{b}"
        if b != "total":
            df[c] = (50 + df[c] * 16.7).clip(0, 100)
        df[c] = df[c].round(1)

    df["rank"] = df["score_total"].rank(ascending=False, method="min")
    return df.sort_values("score_total", ascending=False)


# --------------------------------------------------------------------------- #
# Segnali qualitativi
# --------------------------------------------------------------------------- #
def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Etichette leggibili che spiegano *perché* un titolo emerge."""
    df = df.copy()

    def flags(r):
        f = []
        if pd.notna(r.get("fcf_yield")) and r["fcf_yield"] > 8:
            f.append("FCF yield elevato")
        if pd.notna(r.get("forwardPE")) and 0 < r["forwardPE"] < 12:
            f.append("P/E forward basso")
        if pd.notna(r.get("roic")) and r["roic"] > 15:
            f.append("ROIC forte")
        if pd.notna(r.get("net_debt_to_ebitda")) and r["net_debt_to_ebitda"] < 1:
            f.append("Bilancio solido")
        if pd.notna(r.get("dividendYield")) and r["dividendYield"] > 4 \
           and pd.notna(r.get("payoutRatio")) and 0 < r["payoutRatio"] < 75:
            f.append("Dividendo sostenibile")
        if r.get("golden_cross") and r.get("trend_up"):
            f.append("Trend rialzista")
        if pd.notna(r.get("rsi14")) and r["rsi14"] < 32:
            f.append("Ipervenduto")
        if pd.notna(r.get("rsi14")) and r["rsi14"] > 72:
            f.append("Ipercomprato")
        if pd.notna(r.get("pct_from_52w_high")) and r["pct_from_52w_high"] < -30:
            f.append("Lontano dai massimi")
        if pd.notna(r.get("share_count_change")) and r["share_count_change"] < -2:
            f.append("Buyback in corso")
        if pd.notna(r.get("analyst_upside")) and r["analyst_upside"] > 25:
            f.append("Upside analisti")
        # campanelli d'allarme
        if pd.notna(r.get("net_debt_to_ebitda")) and r["net_debt_to_ebitda"] > 4:
            f.append("Debito elevato")
        if pd.notna(r.get("revenueGrowth")) and r["revenueGrowth"] < -10:
            f.append("Ricavi in calo")
        return " | ".join(f)

    df["flags"] = df.apply(flags, axis=1)
    return df
