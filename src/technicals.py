"""Indicatori di analisi tecnica calcolati in pandas puro (nessuna TA-Lib)."""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Indicatori base
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return pd.DataFrame({"bb_mid": ma, "bb_up": ma + k * sd, "bb_low": ma - k * sd})


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Forza del trend, indipendente dalla direzione."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, window)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Sintesi per titolo
# --------------------------------------------------------------------------- #
def compute_snapshot(df: pd.DataFrame) -> dict:
    """Riduce lo storico OHLCV a un dizionario di metriche puntuali."""
    if df is None or len(df) < 60:
        return {}

    df = df.dropna(subset=["Close"]).copy()
    close = df["Close"]
    last = float(close.iloc[-1])

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()

    r = rsi(close)
    m = macd(close)
    bb = bollinger(close)
    a = atr(df)
    ad = adx(df)

    hi52 = close.tail(252).max()
    lo52 = close.tail(252).min()
    rng = (hi52 - lo52) or np.nan

    ret = close.pct_change()
    vol_ann = ret.tail(252).std() * np.sqrt(252) * 100
    dd = (close / close.cummax() - 1)

    def mom(days: int):
        if len(close) <= days:
            return np.nan
        return (last / float(close.iloc[-days - 1]) - 1) * 100

    vol_ratio = np.nan
    if "Volume" in df and df["Volume"].tail(60).sum() > 0:
        v20 = df["Volume"].tail(20).mean()
        v60 = df["Volume"].tail(60).mean()
        vol_ratio = v20 / v60 if v60 else np.nan

    out = {
        "price": last,
        "sma20": _f(sma20), "sma50": _f(sma50), "sma200": _f(sma200),
        "px_vs_sma50": (last / _f(sma50) - 1) * 100 if _f(sma50) else np.nan,
        "px_vs_sma200": (last / _f(sma200) - 1) * 100 if _f(sma200) else np.nan,
        "golden_cross": bool(_f(sma50) and _f(sma200) and _f(sma50) > _f(sma200)),
        "rsi14": _f(r),
        "macd_hist": _f(m["macd_hist"]),
        "macd_bullish": bool(_f(m["macd"]) is not None and _f(m["macd"]) > _f(m["macd_signal"])),
        "adx14": _f(ad),
        "atr_pct": (_f(a) / last * 100) if _f(a) else np.nan,
        "bb_position": ((last - _f(bb["bb_low"])) / (_f(bb["bb_up"]) - _f(bb["bb_low"])) * 100)
                       if _f(bb["bb_up"]) and (_f(bb["bb_up"]) - _f(bb["bb_low"])) else np.nan,
        "high_52w": float(hi52), "low_52w": float(lo52),
        "pct_from_52w_high": (last / hi52 - 1) * 100,
        "pos_in_52w_range": (last - lo52) / rng * 100 if rng and not np.isnan(rng) else np.nan,
        "mom_1m": mom(21), "mom_3m": mom(63), "mom_6m": mom(126), "mom_12m": mom(252),
        "volatility_ann": vol_ann,
        "max_drawdown_1y": float(dd.tail(252).min() * 100),
        "volume_trend": vol_ratio,
        "trend_up": bool(_f(sma20) and _f(sma50) and _f(sma20) > _f(sma50) and last > _f(sma50)),
    }
    # momentum 12-1 (esclude l'ultimo mese: standard accademico)
    if not np.isnan(out["mom_12m"]) and not np.isnan(out["mom_1m"]):
        out["mom_12_1"] = out["mom_12m"] - out["mom_1m"]
    return out


def _f(s: pd.Series):
    """Ultimo valore valido di una serie, come float."""
    if s is None or s.dropna().empty:
        return None
    return float(s.dropna().iloc[-1])


def technical_table(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in prices.items():
        snap = compute_snapshot(df)
        if snap:
            rows.append({"ticker": t, **snap})
    return pd.DataFrame(rows)
