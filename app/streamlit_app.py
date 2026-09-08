"""Dashboard Streamlit per lo screener europeo.

Avvio:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from src import scoring  # noqa: E402

st.set_page_config(page_title="Screener Europa", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Spiegazioni mostrate al passaggio del cursore
# --------------------------------------------------------------------------- #
HELP = {
    "score_total": (
        "Media pesata dei quattro pilastri, secondo i pesi impostati nella barra "
        "laterale. Scala 0-100 dove **50 è la media del settore**, non un giudizio "
        "assoluto. I titoli con meno del 50% delle metriche disponibili sono esclusi."
    ),
    "value": (
        "**Valutazione — quanto costa rispetto a quello che produce.**\n\n"
        "Basata su P/E, P/E forward, P/B, EV/EBITDA, P/S, FCF yield, earnings yield "
        "e dividend yield. Il peso maggiore va al FCF yield: la cassa generata è più "
        "difficile da manipolare contabilmente degli utili.\n\n"
        "Punteggio alto significa *costa poco rispetto ai suoi pari*, non "
        "*conviene comprarlo*: il mercato può avere buone ragioni per pagarlo poco."
    ),
    "quality": (
        "**Qualità — quanto bene l'azienda usa il capitale che ha.**\n\n"
        "Il perno è il ROIC, il rendimento sul capitale investito. Un'azienda al 20% "
        "crea valore ogni volta che reinveste; una al 5% lo distrugge, perché il "
        "capitale le costa più di quanto renda. Accanto: ROE, ROA e i margini lordo, "
        "operativo e netto.\n\n"
        "È il pilastro che separa le buone aziende dalle mediocri, a prescindere dal prezzo."
    ),
    "momentum": (
        "**Momentum — cosa sta facendo il prezzo.**\n\n"
        "L'unico pilastro che ignora i bilanci. Metrica principale: rendimento a 12 mesi "
        "escludendo l'ultimo (l'ultimo mese tende a invertire per ragioni tecniche). "
        "Poi prezzo vs media a 200 giorni, RSI e ADX.\n\n"
        "Serve da contrappeso: il classico errore del value investing è comprare "
        "qualcosa di economico che continua a scendere per anni."
    ),
    "health": (
        "**Solidità — quanto è fragile.**\n\n"
        "Debito netto/EBITDA (quanti anni di margine servirebbero per estinguere il "
        "debito), debito/patrimonio, current ratio, crescita dei ricavi, variazione del "
        "numero di azioni (i buyback riducono le azioni e aumentano la quota di ciascun "
        "azionista), volatilità e drawdown.\n\n"
        "Dice cosa succede se le cose vanno male. È il segnale da prendere più sul serio: "
        "è ciò che trasforma un investimento sbagliato in una perdita permanente."
    ),
    "trailingPE": "Prezzo diviso utile per azione degli ultimi 12 mesi. Non confrontabile tra settori diversi.",
    "forwardPE": "Come il P/E ma sugli utili attesi dagli analisti. Riflette le aspettative, non i fatti.",
    "enterpriseToEbitda": "Valore d'impresa (capitalizzazione + debito − cassa) diviso EBITDA. Più onesto del P/E per confrontare aziende con strutture finanziarie diverse.",
    "fcf_yield": "Flusso di cassa libero diviso capitalizzazione. Quanta cassa genera l'azienda per ogni euro di prezzo pagato.",
    "dividendYield": "Dividendo annuo in percentuale del prezzo. Da leggere insieme al payout: uno yield alto con payout oltre il 100% non è sostenibile.",
    "roic": "Rendimento sul capitale investito, stimato come EBIT·(1−aliquota) / capitale investito. Sopra il 15% indica un vantaggio competitivo.",
    "net_debt_to_ebitda": "Debito netto diviso EBITDA. Sotto 1 è solido, sopra 4 è fragile in caso di recessione o rialzo dei tassi.",
    "rsi14": "Indice di forza relativa a 14 giorni. Sotto 30 è considerato ipervenduto, sopra 70 ipercomprato. Segnale debole se preso da solo.",
    "mom_6m": "Variazione percentuale del prezzo negli ultimi sei mesi.",
    "pct_from_52w_high": "Distanza dal massimo delle ultime 52 settimane. Un valore molto negativo può indicare un'occasione o un declino in corso.",
    "analyst_upside": "Scostamento tra il prezzo attuale e il target medio degli analisti. I target sono storicamente ottimistici: usalo come indizio, non come previsione.",
    "flags": "Etichette generate automaticamente per spiegare perché un titolo emerge, inclusi i campanelli d'allarme.",
}


# --------------------------------------------------------------------------- #
# Caricamento dati
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800)
def load(name: str) -> pd.DataFrame:
    for ext, reader in [("parquet", pd.read_parquet), ("csv", pd.read_csv)]:
        f = DATA / f"{name}.{ext}"
        if f.exists():
            return reader(f)
    return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_curve():
    f = DATA / "yield_curve_aaa.parquet"
    curve = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    m = DATA / "curve_metrics.json"
    metrics = json.loads(m.read_text()) if m.exists() else {}
    return curve, metrics


equities = load("equities")

if equities.empty:
    st.error("Nessun dato disponibile. Esegui prima la pipeline:")
    st.code("python -m src.pipeline", language="bash")
    st.stop()

st.title("📊 Screener azioni e obbligazioni europee")
updated = equities["updated_at"].iloc[0][:16].replace("T", " ") if "updated_at" in equities else "n/d"
st.caption(f"Ultimo aggiornamento dati: {updated} UTC · {len(equities)} titoli analizzati")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Pesi dello score")
    st.caption("Sposta i cursori per riflettere la tua strategia.")
    w_val = st.slider("Valutazione", 0.0, 1.0, 0.30, 0.05, help=HELP["value"])
    w_qua = st.slider("Qualità", 0.0, 1.0, 0.30, 0.05, help=HELP["quality"])
    w_mom = st.slider("Momentum", 0.0, 1.0, 0.25, 0.05, help=HELP["momentum"])
    w_hea = st.slider("Solidità", 0.0, 1.0, 0.15, 0.05, help=HELP["health"])
    by_sector = st.checkbox("Normalizza per settore", value=True,
                            help="Confronta ogni titolo con i pari settore invece che con tutto il mercato.")

    st.divider()
    st.header("Filtri")
    mkts = sorted(equities["market"].dropna().unique())
    sel_mkt = st.multiselect("Mercati", mkts, default=mkts)
    secs = sorted(equities["sector"].dropna().unique()) if "sector" in equities else []
    sel_sec = st.multiselect("Settori", secs, default=secs)

    mc = pd.to_numeric(equities.get("marketCap"), errors="coerce")
    min_mc = st.number_input("Capitalizzazione minima (mld €)", 0.0, 500.0, 1.0, 0.5)

    max_pe = st.slider("P/E massimo", 0, 100, 100)
    min_dy = st.slider("Dividend yield minimo (%)", 0.0, 12.0, 0.0, 0.5)
    max_debt = st.slider("Debito netto / EBITDA massimo", 0.0, 10.0, 10.0, 0.5)
    rsi_rng = st.slider("Intervallo RSI", 0, 100, (0, 100))

# ricalcolo con i pesi scelti
weights = {"value": w_val, "quality": w_qua, "momentum": w_mom, "health": w_hea}
df = scoring.compute_scores(equities.copy(), weights=weights, by_sector=by_sector)
df = scoring.add_flags(df)

# applicazione filtri
f = df.copy()
f = f[f["market"].isin(sel_mkt)]
if sel_sec and "sector" in f:
    f = f[f["sector"].isin(sel_sec) | f["sector"].isna()]
f = f[pd.to_numeric(f["marketCap"], errors="coerce").fillna(0) >= min_mc * 1e9]
if max_pe < 100:
    pe = pd.to_numeric(f["trailingPE"], errors="coerce")
    f = f[(pe <= max_pe) | pe.isna()]
if min_dy > 0:
    f = f[pd.to_numeric(f["dividendYield"], errors="coerce").fillna(0) >= min_dy]
if max_debt < 10:
    nd = pd.to_numeric(f["net_debt_to_ebitda"], errors="coerce")
    f = f[(nd <= max_debt) | nd.isna()]
r = pd.to_numeric(f["rsi14"], errors="coerce")
f = f[((r >= rsi_rng[0]) & (r <= rsi_rng[1])) | r.isna()]
f = f.dropna(subset=["score_total"])


# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏆 Classifica", "🔍 Scheda titolo", "🗺️ Mappa", "🏛️ Obbligazioni", "ℹ️ Metodo"])


# --- Tab 1: classifica ------------------------------------------------------ #
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titoli filtrati", len(f))
    c2.metric("Score medio", f"{f['score_total'].mean():.1f}" if len(f) else "–")
    c3.metric("P/E mediano", f"{pd.to_numeric(f['trailingPE'], errors='coerce').median():.1f}" if len(f) else "–")
    c4.metric("Yield mediano", f"{pd.to_numeric(f['dividendYield'], errors='coerce').median():.2f}%" if len(f) else "–")

    st.subheader("Migliori opportunità secondo i pesi impostati")

    cols = ["ticker", "shortName", "market", "sector", "score_total", "score_value",
            "score_quality", "score_momentum", "score_health", "price",
            "trailingPE", "forwardPE", "enterpriseToEbitda", "fcf_yield",
            "dividendYield", "roic", "net_debt_to_ebitda", "rsi14",
            "mom_6m", "pct_from_52w_high", "analyst_upside", "flags"]
    cols = [c for c in cols if c in f.columns]

    labels = {
        "ticker": "Ticker", "shortName": "Nome", "market": "Mercato", "sector": "Settore",
        "score_total": "Score", "score_value": "Valore", "score_quality": "Qualità",
        "score_momentum": "Momentum", "score_health": "Solidità", "price": "Prezzo",
        "trailingPE": "P/E", "forwardPE": "P/E fwd", "enterpriseToEbitda": "EV/EBITDA",
        "fcf_yield": "FCF yield %", "dividendYield": "Div %", "roic": "ROIC %",
        "net_debt_to_ebitda": "ND/EBITDA", "rsi14": "RSI", "mom_6m": "Mom 6m %",
        "pct_from_52w_high": "Da max 52s %", "analyst_upside": "Upside %", "flags": "Segnali",
    }

    n = st.slider("Numero di titoli da mostrare", 10, 200, 40, 10)
    view = f[cols].head(n).rename(columns=labels)

    # tooltip sulle intestazioni: passa il cursore sul nome della colonna
    help_by_original = {
        "score_total": HELP["score_total"], "score_value": HELP["value"],
        "score_quality": HELP["quality"], "score_momentum": HELP["momentum"],
        "score_health": HELP["health"],
    }
    help_by_original.update({k: v for k, v in HELP.items() if k in labels})

    col_config = {
        labels[orig]: st.column_config.Column(labels[orig], help=txt)
        for orig, txt in help_by_original.items() if orig in labels
    }

    st.caption("Passa il cursore sul nome di una colonna per leggerne la spiegazione.")
    st.dataframe(
        view.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100)
                  .format(precision=2, na_rep="–"),
        column_config=col_config,
        width='stretch', height=620, hide_index=True,
    )

    st.download_button("⬇️ Scarica CSV", f[cols].to_csv(index=False).encode(),
                       "screener_europa.csv", "text/csv")


# --- Tab 2: scheda titolo --------------------------------------------------- #
with tab2:
    if len(f) == 0:
        st.info("Nessun titolo soddisfa i filtri attuali.")
    else:
        opts = f["ticker"].tolist()
        sel = st.selectbox("Seleziona un titolo", opts,
                           format_func=lambda t: f"{t} — {f.loc[f.ticker == t, 'shortName'].iloc[0]}")
        row = f[f["ticker"] == sel].iloc[0]

        st.subheader(f"{row.get('shortName', sel)}  ·  {sel}")
        if row.get("flags"):
            st.info(row["flags"])

        c = st.columns(5)
        c[0].metric("Score totale", f"{row['score_total']:.0f}/100", help=HELP["score_total"])
        c[1].metric("Valore", f"{row['score_value']:.0f}", help=HELP["value"])
        c[2].metric("Qualità", f"{row['score_quality']:.0f}", help=HELP["quality"])
        c[3].metric("Momentum", f"{row['score_momentum']:.0f}", help=HELP["momentum"])
        c[4].metric("Solidità", f"{row['score_health']:.0f}", help=HELP["health"])

        # radar dei quattro pilastri
        cats = ["Valore", "Qualità", "Momentum", "Solidità"]
        vals = [row.get(f"score_{k}", 50) for k in ["value", "quality", "momentum", "health"]]
        fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                        fill="toself", name=sel))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                          height=340, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')

        left, right = st.columns(2)
        with left:
            st.markdown("**Fondamentali**")
            fm = {"Capitalizzazione": row.get("marketCap"), "P/E": row.get("trailingPE"),
                  "P/E forward": row.get("forwardPE"), "P/B": row.get("priceToBook"),
                  "EV/EBITDA": row.get("enterpriseToEbitda"), "ROE %": row.get("returnOnEquity"),
                  "ROIC %": row.get("roic"), "Margine operativo %": row.get("operatingMargins"),
                  "FCF yield %": row.get("fcf_yield"), "Dividendo %": row.get("dividendYield"),
                  "Payout %": row.get("payoutRatio"), "ND/EBITDA": row.get("net_debt_to_ebitda"),
                  "Crescita ricavi %": row.get("revenueGrowth")}
            st.table(pd.Series(fm, name="Valore").to_frame().style.format(precision=2, na_rep="–"))
        with right:
            st.markdown("**Tecnica**")
            tm = {"Prezzo": row.get("price"), "SMA 50": row.get("sma50"), "SMA 200": row.get("sma200"),
                  "vs SMA 200 %": row.get("px_vs_sma200"), "RSI 14": row.get("rsi14"),
                  "ADX 14": row.get("adx14"), "ATR %": row.get("atr_pct"),
                  "Momentum 3m %": row.get("mom_3m"), "Momentum 12m %": row.get("mom_12m"),
                  "Volatilità ann. %": row.get("volatility_ann"),
                  "Max drawdown 1a %": row.get("max_drawdown_1y"),
                  "Dal max 52s %": row.get("pct_from_52w_high"),
                  "Posizione range 52s %": row.get("pos_in_52w_range")}
            st.table(pd.Series(tm, name="Valore").to_frame().style.format(precision=2, na_rep="–"))


# --- Tab 3: mappa ----------------------------------------------------------- #
with tab3:
    st.subheader("Valutazione contro qualità")
    plot = f.dropna(subset=["score_value", "score_quality"]).copy()
    plot["size"] = pd.to_numeric(plot["marketCap"], errors="coerce").fillna(1e9)
    if len(plot):
        fig = px.scatter(plot, x="score_value", y="score_quality", size="size",
                         color="market", hover_name="shortName",
                         hover_data={"ticker": True, "score_total": ":.0f", "size": False},
                         labels={"score_value": "Score valutazione",
                                 "score_quality": "Score qualità"},
                         height=560, size_max=45)
        fig.add_hline(y=50, line_dash="dot", opacity=0.4)
        fig.add_vline(x=50, line_dash="dot", opacity=0.4)
        fig.add_annotation(x=85, y=90, text="Economiche e di qualità", showarrow=False)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Score medio per settore")
    if "sector" in f:
        agg = (f.groupby("sector")["score_total"]
                 .agg(["mean", "count"]).reset_index()
                 .query("count >= 3").sort_values("mean"))
        if len(agg):
            st.plotly_chart(px.bar(agg, x="mean", y="sector", orientation="h",
                                   labels={"mean": "Score medio", "sector": ""},
                                   height=450), width='stretch')


# --- Tab 4: obbligazioni ---------------------------------------------------- #
with tab4:
    curve, cm = load_curve()

    if cm:
        st.subheader("Contesto macro: curva dei rendimenti area euro (BCE)")
        c = st.columns(4)
        c[0].metric("Pendenza 10a-2a", f"{cm.get('slope_10y_2y', float('nan')):.2f} pp")
        c[1].metric("Pendenza 30a-10a", f"{cm.get('slope_30y_10y', float('nan')):.2f} pp")
        c[2].metric("Var. 10a (3 mesi)", f"{cm.get('chg_10Y_3m_bps', float('nan')):.0f} pb")
        c[3].metric("Spread di credito 10a", f"{cm.get('credit_spread_10y', float('nan')):.2f} pp")
        if cm.get("inverted"):
            st.warning("Curva invertita sul tratto 2-10 anni: storicamente un segnale di rallentamento atteso.")

    if not curve.empty:
        mats = [c for c in curve.columns]
        latest = curve.iloc[-1]
        past = curve.iloc[-63] if len(curve) > 63 else curve.iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mats, y=latest.values, name="Oggi", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=mats, y=past.values, name="3 mesi fa",
                                 mode="lines+markers", line=dict(dash="dash")))
        fig.update_layout(height=400, yaxis_title="Rendimento %", xaxis_title="Scadenza")
        st.plotly_chart(fig, width='stretch')

    bdf = load("bonds")
    st.subheader("ETF obbligazionari")
    if bdf.empty:
        st.info("Dati obbligazionari non disponibili. Esegui la pipeline senza --skip-bonds.")
    else:
        bcols = [c for c in ["ticker", "name", "bucket", "score_bond", "distribution_yield",
                             "return_ann_3y", "sharpe_proxy", "volatility_ann",
                             "max_drawdown_1y", "mom_6m", "rsi14", "px_vs_sma200"] if c in bdf]
        st.dataframe(
            bdf[bcols].rename(columns={
                "ticker": "Ticker", "name": "Nome", "bucket": "Categoria",
                "score_bond": "Score", "distribution_yield": "Yield %",
                "return_ann_3y": "Rend. ann. %", "sharpe_proxy": "Sharpe",
                "volatility_ann": "Volatilità %", "max_drawdown_1y": "Max DD %",
                "mom_6m": "Mom 6m %", "rsi14": "RSI", "px_vs_sma200": "vs SMA200 %"})
              .style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100)
              .format(precision=2, na_rep="–"),
            width='stretch', hide_index=True)


# --- Tab 5: metodo ---------------------------------------------------------- #
with tab5:
    st.markdown("""
### Come funziona lo score

Ogni titolo riceve quattro punteggi, ciascuno costruito come media pesata di
**z-score robusti** (code tagliate al 5%) calcolati **all'interno del proprio settore**.
Confrontare il P/E di una banca con quello di una società di software non ha senso;
la normalizzazione settoriale evita questo errore.

| Pilastro | Metriche principali |
|---|---|
| **Valutazione** | P/E, P/E forward, P/B, EV/EBITDA, P/S, FCF yield, earnings yield, dividend yield |
| **Qualità** | ROE, ROIC, margine operativo, margine lordo, margine netto, ROA |
| **Momentum** | rendimento 12-1 mesi, 6m, 3m, prezzo vs SMA200, RSI, ADX |
| **Solidità** | debito netto/EBITDA, debt/equity, current ratio, crescita ricavi, buyback, volatilità, drawdown |

Lo score finale è la media pesata dei quattro, riportata su scala 0-100 (50 = media
del settore). I titoli con meno del 50% delle metriche disponibili vengono esclusi.

### Limiti da conoscere

- I dati provengono da Yahoo Finance: gratuiti ma non sempre puntuali. Campi come
  `pegRatio` o `freeCashflow` sono spesso assenti sui titoli europei minori.
- Il ROIC è approssimato da EBIT·(1−aliquota) / capitale investito, non è un dato di bilancio.
- Per le obbligazioni non esiste una fonte gratuita affidabile sui singoli ISIN:
  il modulo lavora sulla curva BCE e su ETF quotati come proxy investibili.
- Uno screener produce **candidati da approfondire**, non decisioni. Un punteggio
  alto può nascere da un settore in crisi strutturale o da una value trap.

Questo strumento è materiale informativo, non una consulenza finanziaria personalizzata.
""")
