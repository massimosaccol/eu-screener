"""Livello di accesso ai dati: prezzi e fondamentali da Yahoo Finance,
curve dei rendimenti dal Data Portal della BCE."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
import yaml
import yfinance as yf

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Su Windows yfinance non riesce a creare la propria cache dei fusi orari in
# AppData e stampa un warning a ogni avvio. La spostiamo dentro il progetto.
try:
    yf.set_tz_cache_location(str(CACHE_DIR))
except Exception:  # versioni di yfinance che non espongono la funzione
    pass

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #
@dataclass
class Universe:
    markets: dict = field(default_factory=dict)
    bond_etfs: dict = field(default_factory=dict)
    ecb_curve_maturities: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Universe":
        path = Path(path) if path else ROOT / "config" / "universe.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(
            markets=raw.get("markets", {}),
            bond_etfs=raw.get("bond_etfs", {}),
            ecb_curve_maturities=raw.get("ecb_curve_maturities", []),
        )

    def all_tickers(self) -> list[str]:
        out: list[str] = []
        for cfg in self.markets.values():
            out.extend(cfg.get("tickers", []))
        # dedup mantenendo l'ordine
        return list(dict.fromkeys(out))

    def market_of(self, ticker: str) -> str:
        for name, cfg in self.markets.items():
            if ticker in cfg.get("tickers", []):
                return name
        return "N/D"

    def all_bond_etfs(self) -> pd.DataFrame:
        rows = []
        for bucket, items in self.bond_etfs.items():
            for it in items:
                rows.append({"bucket": bucket, **it})
        return pd.DataFrame(rows).drop_duplicates(subset=["ticker"])


# --------------------------------------------------------------------------- #
# Prezzi
# --------------------------------------------------------------------------- #
def download_prices(
    tickers: list[str],
    period: str = "3y",
    interval: str = "1d",
    batch_size: int = 40,
    pause: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Scarica gli storici OHLCV a lotti. Restituisce {ticker: DataFrame}."""
    result: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("Prezzi: lotto %d/%d (%d titoli)",
                 i // batch_size + 1,
                 (len(tickers) - 1) // batch_size + 1, len(batch))
        try:
            raw = yf.download(
                batch, period=period, interval=interval,
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception as exc:  # rete instabile, rate limit, ecc.
            log.warning("Lotto fallito: %s", exc)
            time.sleep(pause * 3)
            continue

        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")
                if len(df) >= 60:  # storico minimo utile
                    result[t] = df
            except (KeyError, TypeError):
                log.debug("Nessun dato per %s", t)
        time.sleep(pause)

    log.info("Prezzi scaricati per %d/%d titoli", len(result), len(tickers))
    return result


# --------------------------------------------------------------------------- #
# Fondamentali
# --------------------------------------------------------------------------- #
_INFO_FIELDS = [
    "shortName", "sector", "industry", "currency", "country", "marketCap",
    "enterpriseValue", "trailingPE", "forwardPE", "pegRatio", "priceToBook",
    "priceToSalesTrailing12Months", "enterpriseToEbitda", "enterpriseToRevenue",
    "returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins",
    "grossMargins", "ebitdaMargins", "debtToEquity", "currentRatio", "quickRatio",
    "totalDebt", "totalCash", "ebitda", "freeCashflow", "operatingCashflow",
    "dividendYield", "payoutRatio", "fiveYearAvgDividendYield",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "beta", "trailingEps", "forwardEps", "bookValue",
    "targetMeanPrice", "recommendationMean", "numberOfAnalystOpinions",
    "sharesOutstanding", "floatShares", "heldPercentInsiders",
]


def fetch_fundamentals(tickers: list[str], pause: float = 0.25) -> pd.DataFrame:
    """Estrae i dati fondamentali titolo per titolo. Robusto ai campi mancanti."""
    rows = []
    for n, t in enumerate(tickers, 1):
        if n % 25 == 0:
            log.info("Fondamentali: %d/%d", n, len(tickers))
        row = {"ticker": t}
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            for f in _INFO_FIELDS:
                row[f] = info.get(f)
            row.update(_derive_from_statements(tk))
        except Exception as exc:
            log.debug("Fondamentali KO per %s: %s", t, exc)
        rows.append(row)
        time.sleep(pause)

    df = pd.DataFrame(rows)
    # yfinance a volte restituisce il dividend yield in percentuale, a volte in frazione
    if "dividendYield" in df:
        dy = pd.to_numeric(df["dividendYield"], errors="coerce")
        df["dividendYield"] = dy.where(dy <= 1, dy / 100) * 100  # sempre in %
    return df


def _derive_from_statements(tk) -> dict:
    """Metriche che richiedono i bilanci: ROIC, net debt/EBITDA, FCF yield,
    trend dei margini, variazione del numero di azioni."""
    out: dict = {}
    try:
        fin = tk.financials
        bal = tk.balance_sheet
        cfs = tk.cashflow
    except Exception:
        return out

    def pick(df, *names, col=0):
        if df is None or df.empty or col >= df.shape[1]:
            return None
        for nm in names:
            if nm in df.index:
                v = df.loc[nm].iloc[col]
                return float(v) if pd.notna(v) else None
        return None

    ebit = pick(fin, "EBIT", "Operating Income")
    tax = pick(fin, "Tax Provision")
    pretax = pick(fin, "Pretax Income")
    equity = pick(bal, "Stockholders Equity", "Total Stockholder Equity")
    debt = pick(bal, "Total Debt")
    cash = pick(bal, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")

    if ebit and equity and debt is not None:
        tax_rate = (tax / pretax) if (tax and pretax and pretax > 0) else 0.25
        tax_rate = min(max(tax_rate, 0.0), 0.6)
        invested = equity + debt - (cash or 0)
        if invested and invested > 0:
            out["roic"] = ebit * (1 - tax_rate) / invested * 100

    ebitda_now = pick(fin, "EBITDA")
    if debt is not None and ebitda_now and ebitda_now > 0:
        out["net_debt_to_ebitda"] = (debt - (cash or 0)) / ebitda_now

    ocf = pick(cfs, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = pick(cfs, "Capital Expenditure")
    if ocf is not None and capex is not None:
        out["fcf_calc"] = ocf + capex  # capex è negativo

    # crescita dei ricavi a 3 anni (CAGR)
    rev_now = pick(fin, "Total Revenue", col=0)
    rev_old = pick(fin, "Total Revenue", col=min(3, (fin.shape[1] - 1) if fin is not None and not fin.empty else 0))
    if rev_now and rev_old and rev_old > 0:
        yrs = min(3, fin.shape[1] - 1) or 1
        out["revenue_cagr_3y"] = ((rev_now / rev_old) ** (1 / yrs) - 1) * 100

    # buyback: azioni in circolazione oggi vs 1 anno fa
    sh_now = pick(bal, "Ordinary Shares Number", col=0)
    sh_old = pick(bal, "Ordinary Shares Number", col=1)
    if sh_now and sh_old and sh_old > 0:
        out["share_count_change"] = (sh_now / sh_old - 1) * 100

    return out


# --------------------------------------------------------------------------- #
# Obbligazionario: curve BCE
# --------------------------------------------------------------------------- #
def fetch_ecb_yield_curve(maturities: list[str], rating: str = "AAA") -> pd.DataFrame:
    """Curva dei rendimenti dei titoli di Stato area euro (BCE, dataset YC).

    rating='AAA' -> solo emittenti AAA;  rating='ALL' -> tutti gli emittenti.
    """
    grp = "G_N_A" if rating.upper() == "AAA" else "G_N_C"
    frames = []
    for m in maturities:
        key = f"B.U2.EUR.4F.{grp}.SV_C_YM.SR_{m}"
        url = f"{ECB_BASE}/YC/{key}"
        for attempt in range(3):
            try:
                r = requests.get(
                    url,
                    params={"format": "csvdata", "lastNObservations": 400},
                    timeout=90,
                )
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                df = df[["TIME_PERIOD", "OBS_VALUE"]].rename(
                    columns={"TIME_PERIOD": "date", "OBS_VALUE": m})
                df["date"] = pd.to_datetime(df["date"])
                frames.append(df.set_index("date"))
                break
            except Exception as exc:
                if attempt == 2:
                    log.warning("Curva BCE %s non disponibile dopo 3 tentativi: %s", m, exc)
                else:
                    log.debug("Curva BCE %s, tentativo %d fallito", m, attempt + 1)
                    time.sleep(3)
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def fetch_ecb_series(key: str, dataset: str, n: int = 400) -> pd.DataFrame:
    """Serie generica dal Data Portal BCE (es. tassi ufficiali, inflazione)."""
    url = f"{ECB_BASE}/{dataset}/{key}"
    r = requests.get(url, params={"format": "csvdata", "lastNObservations": n}, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")
