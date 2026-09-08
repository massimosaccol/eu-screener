"""Pipeline principale: scarica, calcola, salva.

Uso:
    python -m src.pipeline                 # universo completo
    python -m src.pipeline --markets Italia Francia
    python -m src.pipeline --skip-bonds --period 5y
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bonds as bonds_mod
from . import scoring, technicals
from .data_sources import Universe, download_prices, fetch_fundamentals

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def run(markets: list[str] | None = None, period: str = "3y",
        skip_bonds: bool = False, batch_size: int = 40) -> pd.DataFrame:
    uni = Universe.load()

    if markets:
        uni.markets = {k: v for k, v in uni.markets.items() if k in markets}
        if not uni.markets:
            raise SystemExit(f"Nessun mercato corrispondente. Disponibili: "
                             f"{list(Universe.load().markets)}")

    tickers = uni.all_tickers()
    log.info("Universo: %d titoli su %d mercati", len(tickers), len(uni.markets))

    # 1. prezzi e tecnica
    prices = download_prices(tickers, period=period, batch_size=batch_size)
    tech = technicals.technical_table(prices)
    log.info("Tecnica calcolata su %d titoli", len(tech))

    # 2. fondamentali (solo per i titoli con prezzi validi)
    valid = tech["ticker"].tolist() if not tech.empty else []
    fund = fetch_fundamentals(valid)
    log.info("Fondamentali raccolti per %d titoli", len(fund))

    # 3. merge e scoring
    df = fund.merge(tech, on="ticker", how="outer")
    df["market"] = df["ticker"].map(uni.market_of)
    df = scoring.compute_scores(df)
    df = scoring.add_flags(df)
    df["updated_at"] = datetime.now(timezone.utc).isoformat()

    _save(df, "equities")
    log.info("Top 10: %s", ", ".join(df.head(10)["ticker"].tolist()))

    # 4. obbligazionario
    if not skip_bonds:
        try:
            etfs = bonds_mod.analyze_bond_etfs(uni.all_bond_etfs(), period=period)
            if not etfs.empty:
                _save(etfs, "bonds")
            curve = bonds_mod.curve_analysis(uni.ecb_curve_maturities)
            if not curve["curve_aaa"].empty:
                curve["curve_aaa"].to_parquet(OUT / "yield_curve_aaa.parquet")
                if not curve["curve_all"].empty:
                    curve["curve_all"].to_parquet(OUT / "yield_curve_all.parquet")
                (OUT / "curve_metrics.json").write_text(
                    json.dumps(curve["metrics"], indent=2), encoding="utf-8")
                log.info("Curva BCE aggiornata: %s", curve["metrics"].get("date"))
        except Exception as exc:
            log.error("Modulo obbligazionario fallito: %s", exc)

    return df


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    df.to_csv(OUT / f"{name}.csv", index=False)
    log.info("Salvato data/%s.parquet (%d righe)", name, len(df))


def main() -> None:
    p = argparse.ArgumentParser(description="Screener azioni e obbligazioni europee")
    p.add_argument("--markets", nargs="*", help="Filtra per mercato (nomi da universe.yaml)")
    p.add_argument("--period", default="3y", help="Storico prezzi (1y, 3y, 5y, max)")
    p.add_argument("--skip-bonds", action="store_true")
    p.add_argument("--batch-size", type=int, default=40)
    a = p.parse_args()
    run(markets=a.markets, period=a.period, skip_bonds=a.skip_bonds, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
