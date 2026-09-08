# Screener azioni e obbligazioni europee

Automazione Python che analizza i principali mercati azionari europei con metriche
fondamentali e tecniche, produce un punteggio composito per ogni titolo e presenta
i risultati in una dashboard Streamlit interattiva.

## Cosa copre

**Azionario** — Italia, Francia, Germania, Paesi Bassi, Spagna, Belgio/Portogallo,
Svizzera, Regno Unito, area nordica, Irlanda/Austria. Circa 200 titoli nella
configurazione di partenza, estendibili modificando `config/universe.yaml`.

**Obbligazionario** — curva dei rendimenti dell'area euro dal Data Portal della BCE
(pendenza, inversione, spread di credito) e una selezione di ETF obbligazionari
quotati analizzati per carry, rendimento corretto per il rischio e trend.

## Struttura

```
├── config/universe.yaml        # universo dei titoli e degli ETF
├── src/
│   ├── data_sources.py         # download prezzi, fondamentali, curve BCE
│   ├── technicals.py           # RSI, MACD, ADX, ATR, Bollinger, momentum
│   ├── scoring.py              # z-score settoriali e punteggi compositi
│   ├── bonds.py                # analisi obbligazionaria
│   └── pipeline.py             # orchestratore
├── app/streamlit_app.py        # dashboard
├── .github/workflows/          # aggiornamento automatico giornaliero
└── data/                       # output (parquet + csv)
```

## Avvio rapido

```bash
git clone https://github.com/<tuo-utente>/eu-screener.git
cd eu-screener
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# primo giro: parti da pochi mercati per capire i tempi
python -m src.pipeline --markets Italia Francia

# universo completo (15-25 minuti circa)
python -m src.pipeline

streamlit run app/streamlit_app.py
```

## Le metriche

**Valutazione** — P/E, P/E forward, P/B, EV/EBITDA, P/S, FCF yield, earnings yield,
dividend yield.

**Qualità** — ROE, ROIC (calcolato dai bilanci), margine lordo, operativo e netto, ROA.

**Momentum** — rendimento 12-1 mesi, 6m, 3m, 1m, prezzo contro SMA 50 e 200,
RSI a 14 periodi, ADX, golden cross, posizione nel range a 52 settimane.

**Solidità** — debito netto/EBITDA, debt/equity, current ratio, crescita dei ricavi
(CAGR a 3 anni), variazione del numero di azioni (buyback), volatilità annualizzata,
massimo drawdown a un anno.

Ogni metrica viene convertita in z-score robusto **all'interno del settore di
appartenenza**: confrontare il P/B di una banca con quello di un produttore di
software genererebbe classifiche prive di significato. I quattro pilastri vengono
poi combinati con pesi che puoi modificare dal vivo nella sidebar della dashboard.

## Automazione su GitHub

Il workflow incluso gira alle 18:00 UTC nei giorni feriali, esegue la pipeline e
committa i dati aggiornati nel repository. Va abilitato in **Settings → Actions →
General → Workflow permissions → Read and write**.

Per pubblicare la dashboard gratuitamente: collega il repository a
[share.streamlit.io](https://share.streamlit.io), indica `app/streamlit_app.py`
come file principale. La dashboard leggerà i parquet committati dal workflow.

## Personalizzazione

Aggiungere titoli: inserisci i ticker in `config/universe.yaml` usando i suffissi
Yahoo Finance (`.MI` Milano, `.PA` Parigi, `.DE` Xetra, `.AS` Amsterdam, `.MC` Madrid,
`.SW` Zurigo, `.L` Londra, `.ST` Stoccolma, e così via).

Modificare la logica di punteggio: i dizionari `VALUE_METRICS`, `QUALITY_METRICS`,
`MOMENTUM_METRICS` e `HEALTH_METRICS` in `src/scoring.py` definiscono metriche,
direzione (`+1` alto è meglio, `-1` basso è meglio) e peso relativo.

## Limiti

Yahoo Finance è gratuito ma non impeccabile: sui titoli europei minori diversi campi
risultano assenti o datati di qualche trimestre. Il ROIC è una stima derivata dai
bilanci, non un dato certificato. Per i singoli titoli di Stato non esiste una fonte
gratuita affidabile su ISIN, cedole e YTM: il modulo obbligazionario lavora sulla
curva BCE e su ETF quotati.

Se in futuro vorrai dati di qualità istituzionale, le alternative da valutare sono
Financial Modeling Prep, EODHD o Borsa Italiana per il segmento MOT.

## Avvertenza

Questo progetto produce materiale informativo e didattico. Uno screener genera
**candidati da approfondire**, non raccomandazioni: un punteggio elevato può riflettere
un settore in declino strutturale o una classica value trap. Non sono un consulente
finanziario e questo strumento non sostituisce un'analisi personale né il parere di un
professionista abilitato.
