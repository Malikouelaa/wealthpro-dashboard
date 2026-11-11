import datetime as dt
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import wealth_core as core

DB_PATH = "wealthpro.db"
TODAY = dt.date.today()
RECENT = (TODAY - dt.timedelta(days=14)).isoformat()

st.set_page_config(page_title="WealthPro Ultimate", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #00E676; }
    div[data-testid="stMetric"] { background-color: #1E222D; padding: 14px; border-radius: 12px; border: 1px solid #2A2E39; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1E222D; border-radius: 8px; padding: 8px 16px; color: #B2B5BE; }
    .stTabs [aria-selected="true"] { background-color: #2962FF !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Base de données : journal de transactions (source unique de vérité)
# ==========================================================================
def migrate_legacy(conn):
    """Reprend les anciennes tables cash/portfolio (v1) si le journal est vide."""
    if conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]:
        return
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"portfolio", "cash"} <= tables:
        return
    row = conn.execute("SELECT amount FROM cash WHERE id = 1").fetchone()
    positions = conn.execute("SELECT ticker, shares, buy_price FROM portfolio").fetchall()
    deposit = (row[0] if row else 0.0) + sum(s * p for _, s, p in positions)
    today = TODAY.isoformat()
    if deposit > 0:
        conn.execute(
            "INSERT INTO transactions (date, type, quantity, price) VALUES (?, 'DEPOSIT', 1, ?)", (today, deposit)
        )
    for ticker, shares, buy_price in positions:
        conn.execute(
            "INSERT INTO transactions (date, type, ticker, quantity, price) VALUES (?, 'BUY', ?, ?, ?)",
            (today, ticker, shares, buy_price),
        )
    conn.commit()


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            ticker TEXT,
            quantity REAL DEFAULT 0,
            price REAL DEFAULT 0,
            fees REAL DEFAULT 0,
            currency TEXT DEFAULT 'USD',
            fx_rate REAL DEFAULT 1.0)"""
    )
    conn.commit()
    migrate_legacy(conn)
    return conn


conn = get_conn()


def add_tx(date, type_, ticker=None, quantity=1.0, price=0.0, fees=0.0, currency="USD", fx_rate=1.0):
    conn.execute(
        "INSERT INTO transactions (date, type, ticker, quantity, price, fees, currency, fx_rate) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pd.Timestamp(date).date().isoformat(), type_, ticker, quantity, price, fees, currency, fx_rate),
    )
    conn.commit()


def load_tx() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date, id", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ==========================================================================
# Données de marché (mises en cache)
# ==========================================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_history(ticker: str, start: str, adjusted: bool = False) -> pd.Series:
    """Clôtures quotidiennes. adjusted=True : rendement total (dividendes inclus), pour les benchmarks."""
    try:
        df = yf.Ticker(ticker).history(start=start, auto_adjust=adjusted)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker: str) -> dict:
    out = {"name": ticker, "sector": "Autre / Inconnu", "currency": "USD"}
    try:
        info = yf.Ticker(ticker).info
        out["name"] = info.get("longName") or info.get("shortName") or ticker
        out["sector"] = info.get("sector") or info.get("quoteType") or "Autre / Inconnu"
        out["currency"] = info.get("currency") or "USD"
    except Exception:
        pass
    return out


def fx_on(cur: str, date) -> float:
    """Taux devise -> USD à une date donnée (dernier cours connu avant cette date)."""
    if cur == "USD":
        return 1.0
    date = pd.Timestamp(date).normalize()
    s = get_history(f"{cur}USD=X", (date - pd.Timedelta(days=7)).date().isoformat())
    s = s[s.index <= date]
    return float(s.iloc[-1]) if not s.empty else 1.0


def fx_now(cur: str) -> float:
    return fx_on(cur, TODAY)


def import_dataframe(imp: pd.DataFrame):
    """Insère les lignes d'un DataFrame de transactions. Retourne (nb_ok, erreurs)."""
    missing = {"date", "type", "ticker", "quantity", "price"} - set(imp.columns)
    if missing:
        return 0, [f"Colonnes manquantes : {', '.join(sorted(missing))}. Colonnes lues : {', '.join(imp.columns)}"]
    ok, errors = 0, []
    for i, row in imp.iterrows():
        try:
            typ = str(row["type"]).strip().upper()
            if typ not in ("DEPOSIT", "WITHDRAW", "BUY", "SELL", "DIVIDEND"):
                raise ValueError(f"type inconnu '{typ}'")
            date = pd.to_datetime(row["date"], dayfirst="/" in str(row["date"]))
            fees = float(row["fees"]) if "fees" in imp.columns and pd.notna(row["fees"]) else 0.0
            if typ in ("DEPOSIT", "WITHDRAW"):
                add_tx(date, typ, None, 1.0, float(row["price"]), 0.0, "USD", 1.0)
            else:
                tk = str(row["ticker"]).strip().upper()
                if not tk or tk == "NAN":
                    raise ValueError("ticker manquant")
                c_ = get_info(tk)["currency"]
                qty = 1.0 if typ == "DIVIDEND" else float(row["quantity"])
                add_tx(date, typ, tk, qty, float(row["price"]), fees, c_, fx_on(c_, date))
            ok += 1
        except Exception as e:
            errors.append(f"Ligne {i + 2} : {e}")
    return ok, errors


# ==========================================================================
# État courant
# ==========================================================================
tx = load_tx()
state = core.compute_positions(tx) if not tx.empty else {
    "positions": {}, "cash": 0.0, "realized": 0.0, "dividends": 0.0, "realized_log": []
}

# ==========================================================================
# Barre latérale
# ==========================================================================
st.sidebar.header("⚙️ Paramètres")
currency_choice = st.sidebar.selectbox("Devise d'affichage", ["USD ($)", "EUR (€)"])
sym = "€" if "EUR" in currency_choice else "$"
eur_usd = fx_on("EUR", TODAY) if "EUR" in currency_choice else 1.0
fx_mult = (1 / eur_usd) if "EUR" in currency_choice else 1.0


def money(x_usd: float) -> str:
    return f"{x_usd * fx_mult:,.2f} {sym}"


st.sidebar.divider()
st.sidebar.header("💳 Cash")
with st.sidebar.expander("📥 / 📤 Dépôt & retrait"):
    cash_date = st.date_input("Date", value=TODAY, max_value=TODAY, key="cash_date")
    cash_input = st.number_input("Montant ($)", min_value=10.0, value=500.0, step=50.0)
    c_dep, c_ret = st.columns(2)
    if c_dep.button("Déposer"):
        add_tx(cash_date, "DEPOSIT", quantity=1.0, price=cash_input)
        st.rerun()
    if c_ret.button("Retirer"):
        if state["cash"] >= cash_input:
            add_tx(cash_date, "WITHDRAW", quantity=1.0, price=cash_input)
            st.rerun()
        else:
            st.error("Solde cash insuffisant.")

st.sidebar.divider()
st.sidebar.header("📈 Ordres")
t_buy, t_sell, t_div = st.sidebar.tabs(["Acheter", "Vendre", "Dividende"])

with t_buy:
    b_ticker = st.text_input("Symbole boursier", "AAPL", key="b_ticker").upper().strip()
    last, cur = 0.0, "USD"
    if b_ticker:
        h = get_history(b_ticker, RECENT)
        if not h.empty:
            info = get_info(b_ticker)
            last, cur = float(h.iloc[-1]), info["currency"]
            st.success(f"📌 **{info['name']}**\n\n💵 Dernier cours : `{last:.2f} {cur}`")
            st.line_chart(get_history(b_ticker, (TODAY - dt.timedelta(days=90)).isoformat()))
        else:
            st.warning("⚠️ Symbole introuvable (ex : AAPL, MSFT, MC.PA).")
    with st.form("buy_form"):
        b_date = st.date_input("Date de l'ordre", value=TODAY, max_value=TODAY)
        b_qty = st.number_input("Unités", min_value=0.001, value=1.0, step=1.0)
        b_price = st.number_input(f"Prix unitaire ({cur})", min_value=0.01, value=last or 100.0, key=f"bp_{b_ticker}")
        b_fees = st.number_input(f"Frais ({cur})", min_value=0.0, value=0.0, step=0.5)
        if st.form_submit_button("Exécuter l'achat"):
            if last <= 0:
                st.error("Impossible d'acheter : symbole invalide.")
            else:
                fx = fx_on(cur, b_date)
                if (b_qty * b_price + b_fees) * fx > state["cash"] + 1e-9:
                    st.error("Fonds insuffisants en cash.")
                else:
                    add_tx(b_date, "BUY", b_ticker, b_qty, b_price, b_fees, cur, fx)
                    st.rerun()

with t_sell:
    positions = state["positions"]
    if positions:
        s_ticker = st.selectbox("Actif à vendre", list(positions), key="s_ticker")
        p = positions[s_ticker]
        h = get_history(s_ticker, RECENT)
        s_last = float(h.iloc[-1]) if not h.empty else 1.0
        st.info(f"📌 **{s_ticker}** : {p['qty']:g} parts, PRU {p['cost_native'] / p['qty']:.2f} {p['currency']}")
        with st.form("sell_form"):
            s_date = st.date_input("Date de l'ordre", value=TODAY, max_value=TODAY)
            s_qty = st.number_input(
                "Unités à vendre", min_value=0.001, max_value=float(p["qty"]), value=float(p["qty"]), key=f"sq_{s_ticker}"
            )
            s_price = st.number_input(f"Prix de vente ({p['currency']})", min_value=0.01, value=s_last, key=f"sp_{s_ticker}")
            s_fees = st.number_input(f"Frais ({p['currency']})", min_value=0.0, value=0.0, step=0.5, key=f"sf_{s_ticker}")
            if st.form_submit_button("Exécuter la vente"):
                add_tx(s_date, "SELL", s_ticker, s_qty, s_price, s_fees, p["currency"], fx_on(p["currency"], s_date))
                st.rerun()
    else:
        st.write("Aucun actif à vendre.")

with t_div:
    if state["positions"]:
        d_ticker = st.selectbox("Actif", list(state["positions"]), key="d_ticker")
        d_cur = state["positions"][d_ticker]["currency"]
        with st.form("div_form"):
            d_date = st.date_input("Date de paiement", value=TODAY, max_value=TODAY)
            d_amount = st.number_input(f"Montant brut ({d_cur})", min_value=0.01, value=10.0, step=1.0)
            d_tax = st.number_input(f"Retenue / frais ({d_cur})", min_value=0.0, value=0.0, step=0.5)
            if st.form_submit_button("Enregistrer le dividende"):
                add_tx(d_date, "DIVIDEND", d_ticker, 1.0, d_amount, d_tax, d_cur, fx_on(d_cur, d_date))
                st.rerun()
    else:
        st.write("Aucune position.")

st.sidebar.divider()
with st.sidebar.expander("🗑️ Réinitialiser le compte"):
    confirm = st.checkbox("Je confirme la suppression de toutes les transactions")
    if st.button("Réinitialiser", disabled=not confirm):
        conn.execute("DELETE FROM transactions")
        conn.commit()
        st.rerun()

# ==========================================================================
# Interface principale
# ==========================================================================
st.title("💎 WealthPro Ultimate - Dashboard & Wealth Analytics")
st.caption("Journal de transactions, prix de revient moyen, performance pondérée par le temps (TWR), XIRR et comparaison PME.")

if tx.empty:
    st.info("Aucune transaction. Commence par un dépôt de cash dans la barre latérale, ou importe un CSV (onglet Carnet).")

# --- Valorisation actuelle des positions ---
rows, stocks_value, cost_basis = [], 0.0, 0.0
for t, p in state["positions"].items():
    info = get_info(t)
    h = get_history(t, RECENT)
    price = float(h.iloc[-1]) if not h.empty else None
    value = p["qty"] * price * fx_now(p["currency"]) if price is not None else p["cost"]
    pnl = value - p["cost"]
    stocks_value += value
    cost_basis += p["cost"]
    rows.append(
        {
            "Actif": t,
            "Nom": info["name"],
            "Secteur": info["sector"],
            "Parts": p["qty"],
            "PRU (devise actif)": p["cost_native"] / p["qty"],
            "Cours (devise actif)": price,
            "Devise": p["currency"],
            "Coût": p["cost"] * fx_mult,
            "Valeur": value * fx_mult,
            "Gain/Perte latent": pnl * fx_mult,
            "Performance (%)": pnl / p["cost"] * 100 if p["cost"] > 0 else 0.0,
        }
    )
df_pos = pd.DataFrame(rows)
if not df_pos.empty:
    df_pos["Poids (%)"] = df_pos["Valeur"] / df_pos["Valeur"].sum() * 100

net_worth = stocks_value + state["cash"]
unrealized = stocks_value - cost_basis

st.subheader("🏛️ Synthèse du patrimoine")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Patrimoine total", money(net_worth))
m2.metric("Portefeuille actions", money(stocks_value))
m3.metric("Cash disponible", money(state["cash"]))
m4.metric("Plus-value latente", money(unrealized), f"{unrealized / cost_basis * 100:.2f}%" if cost_basis > 0 else None)
m5, m6, _, _ = st.columns(4)
m5.metric("Plus-values réalisées", money(state["realized"]))
m6.metric("Dividendes encaissés (nets)", money(state["dividends"]))
st.divider()

# --- Séries historiques (valeur, flux, rendements) ---
series = assets = r_daily = None
if not tx.empty:
    start_str = tx["date"].min().date().isoformat()
    traded = tx[tx["type"].isin(["BUY", "SELL"])]
    with st.spinner("Reconstitution de l'historique du patrimoine..."):
        prices = {t: get_history(t, start_str) for t in traded["ticker"].dropna().unique()}
        fxs = {c: get_history(f"{c}USD=X", start_str) for c in traded["currency"].unique() if c != "USD"}
        series, assets = core.build_series(tx, prices, fxs)
        r_daily = core.daily_returns(series["value"], series["flow"])

tab_perf, tab_risk, tab_alloc, tab_markets, tab_news, tab_book = st.tabs(
    ["📊 Performance", "⚠️ Risque", "🏢 Allocation", "📉 Marchés & VIX", "📰 Actualités", "📄 Carnet & Import/Export"]
)

# --------------------------------------------------------------------------
with tab_perf:
    if series is None:
        st.info("Pas encore de données.")
    else:
        bc1, bc2 = st.columns([1, 2])
        bench_pick = bc1.selectbox("Benchmark (en USD)", ["SPY", "QQQ", "VT", "Autre..."])
        bench = bc2.text_input("Symbole du benchmark", "SPY").upper().strip() if bench_pick == "Autre..." else bench_pick
        bench_px = get_history(bench, start_str, adjusted=True)

        n_days = (series.index[-1] - series.index[0]).days
        twr_cum = (1 + r_daily).cumprod() - 1
        x_port = core.xirr(core.investor_cashflows(series))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TWR cumulé (hors apports)", f"{twr_cum.iloc[-1] * 100:.2f} %")
        k2.metric("XIRR portefeuille (annualisé)", f"{x_port * 100:.2f} %" if x_port is not None else "n/a")

        if bench_px.empty:
            st.warning(f"Données indisponibles pour {bench}.")
        else:
            pm = core.pme(series, bench_px)
            x_pme = core.xirr(core.investor_cashflows(series, pm["value"].iloc[-1]))
            k3.metric(f"XIRR {bench} (PME)", f"{x_pme * 100:.2f} %" if x_pme is not None else "n/a")
            k4.metric("Valeur PME vs portefeuille", money(series["value"].iloc[-1] - pm["value"].iloc[-1]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series["value"] * fx_mult, name="Mon portefeuille"))
            fig.add_trace(go.Scatter(x=pm.index, y=pm["value"] * fx_mult, name=f"{bench} avec mes mêmes flux (PME)"))
            fig.update_layout(template="plotly_dark", title="Valeur du patrimoine vs benchmark à flux identiques", yaxis_title=sym)
            st.plotly_chart(fig, width="stretch")

            b_ret = (bench_px.reindex(series.index.union(bench_px.index)).ffill().reindex(series.index).bfill())
            b_cum = b_ret / b_ret.iloc[0] - 1
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=twr_cum.index, y=twr_cum * 100, name="TWR portefeuille"))
            fig2.add_trace(go.Scatter(x=b_cum.index, y=b_cum * 100, name=f"{bench} (rendement total)"))
            fig2.update_layout(template="plotly_dark", title="Performance cumulée (%)", yaxis_title="%")
            st.plotly_chart(fig2, width="stretch")

        if n_days < 90:
            st.caption("⚠️ Historique court (< 3 mois) : le XIRR annualisé est peu significatif, privilégie le TWR.")
        st.caption(
            "TWR : rendement qui neutralise les apports/retraits. XIRR : rendement annualisé tenant compte des dates de flux. "
            "PME : mêmes apports/retraits investis dans l'indice à chaque date."
        )

        if not df_pos.empty:
            st.subheader("Gains / pertes latents par actif")
            st.plotly_chart(
                px.bar(df_pos, x="Actif", y="Gain/Perte latent", template="plotly_dark", color="Gain/Perte latent",
                       color_continuous_scale="RdYlGn", color_continuous_midpoint=0),
                width="stretch",
            )

# --------------------------------------------------------------------------
with tab_risk:
    if series is None:
        st.info("Pas encore de données.")
    else:
        rf = st.number_input("Taux sans risque annuel (%)", min_value=0.0, value=2.0, step=0.25) / 100
        bench_px_r = get_history("SPY", start_str, adjusted=True)
        bench_r = bench_px_r.pct_change().fillna(0.0) if not bench_px_r.empty else None
        rm = core.risk_metrics(r_daily, bench_r, rf)
        if rm is None:
            st.info("Il faut au moins 20 jours ouvrés d'historique pour calculer les indicateurs de risque.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Volatilité annualisée", f"{rm['vol'] * 100:.2f} %")
            b.metric("Ratio de Sharpe", f"{rm['sharpe']:.2f}")
            c.metric("Drawdown maximal", f"{rm['max_dd'] * 100:.2f} %")
            d.metric("Bêta vs SPY", f"{rm['beta']:.2f}" if rm["beta"] is not None else "n/a")
            fig = px.area(rm["drawdown"] * 100, template="plotly_dark", title="Drawdown du portefeuille (%)")
            fig.update_layout(showlegend=False, yaxis_title="%")
            st.plotly_chart(fig, width="stretch")

        held = [t for t in state["positions"] if assets is not None and t in assets.columns]
        if len(held) >= 2:
            corr = assets[held].pct_change().dropna().corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                          template="plotly_dark", title="Corrélation des rendements quotidiens"),
                width="stretch",
            )
        else:
            st.caption("La matrice de corrélation apparaît dès 2 positions.")

# --------------------------------------------------------------------------
with tab_alloc:
    if df_pos.empty:
        st.info("Ajoutez des actions pour voir la répartition.")
    else:
        top3 = df_pos.nlargest(3, "Poids (%)")["Poids (%)"].sum()
        cash_pct = state["cash"] / net_worth * 100 if net_worth > 0 else 0
        a, b = st.columns(2)
        a.metric("Concentration top 3 positions", f"{top3:.1f} %")
        b.metric("Cash / patrimoine", f"{cash_pct:.1f} %")
        st.plotly_chart(
            px.treemap(df_pos, path=["Secteur", "Actif"], values="Valeur", template="plotly_dark",
                       title="Répartition par secteur et par actif"),
            width="stretch",
        )
        sec = df_pos.groupby("Secteur", as_index=False)["Valeur"].sum()
        st.plotly_chart(px.pie(sec, names="Secteur", values="Valeur", hole=0.45, template="plotly_dark"), width="stretch")

# --------------------------------------------------------------------------
with tab_markets:
    st.subheader("📉 Indicateurs macro & volatilité (VIX)")
    macro_choice = st.selectbox(
        "Indice ou indicateur",
        [
            "VIX (Indice de volatilité) - ^VIX",
            "S&P 500 - ^GSPC",
            "Nasdaq 100 (Tech) - QQQ",
            "Or (valeur refuge) - GC=F",
            "Bitcoin (sentiment risque) - BTC-USD",
        ],
    )
    macro_ticker = macro_choice.split(" - ")[-1]
    m_close = get_history(macro_ticker, (TODAY - dt.timedelta(days=183)).isoformat())
    if m_close.empty:
        st.warning("Impossible de récupérer les données pour cet indicateur.")
    else:
        cur_v = float(m_close.iloc[-1])
        prev_v = float(m_close.iloc[-2]) if len(m_close) > 1 else cur_v
        st.metric(f"Cours actuel ({macro_ticker})", f"{cur_v:,.2f}", f"{cur_v - prev_v:+.2f} ({(cur_v / prev_v - 1) * 100:+.2f}%)")
        st.line_chart(m_close)

# --------------------------------------------------------------------------
with tab_news:
    st.subheader("📰 Actualités financières")
    tickers_news = list(state["positions"]) + [t for t in ["SPY", "AAPL"] if t not in state["positions"]]
    pick = st.selectbox("Actif", tickers_news)

    @st.cache_data(ttl=900, show_spinner=False)
    def get_news(ticker):
        try:
            return getattr(yf.Ticker(ticker), "news", []) or []
        except Exception:
            return []

    news = get_news(pick)
    if not news:
        st.info(f"Aucune actualité trouvée pour {pick}.")
    for item in news[:6]:
        content = item.get("content", {}) or {}
        title = item.get("title") or content.get("title", "Titre indisponible")
        publisher = item.get("publisher") or (content.get("provider") or {}).get("displayName", "Source")
        link = item.get("link") or (content.get("clickThroughUrl") or {}).get("url", "#")
        st.markdown(f"### [{title}]({link})")
        st.caption(f"Source : **{publisher}** | Actif : `{pick}`")
        st.divider()

# --------------------------------------------------------------------------
with tab_book:
    st.subheader("Positions")
    if df_pos.empty:
        st.write("Aucune position active.")
    else:
        st.dataframe(df_pos.set_index("Actif"), width="stretch")
        st.download_button("📥 Rapport positions (CSV)", df_pos.to_csv(index=False).encode("utf-8"),
                           "wealthpro_positions.csv", "text/csv")

    if state["realized_log"]:
        st.subheader("Plus-values réalisées")
        st.dataframe(pd.DataFrame(state["realized_log"]), width="stretch")

    st.subheader("Journal des transactions")
    if not tx.empty:
        st.dataframe(tx.sort_values(["date", "id"], ascending=False), width="stretch", hide_index=True)
        st.download_button("📥 Exporter le journal (CSV)", tx.to_csv(index=False).encode("utf-8"),
                           "wealthpro_transactions.csv", "text/csv")
        dcol1, dcol2 = st.columns([2, 1])
        del_id = dcol1.selectbox(
            "Supprimer une transaction (correction d'erreur)",
            tx["id"].tolist()[::-1],
            format_func=lambda i: "{} | {} | {}".format(
                i, tx.loc[tx["id"] == i, "date"].iloc[0].date(), tx.loc[tx["id"] == i, "type"].iloc[0]
            ),
        )
        if dcol2.button("Supprimer"):
            conn.execute("DELETE FROM transactions WHERE id = ?", (int(del_id),))
            conn.commit()
            st.rerun()

    st.subheader("Importer des transactions (CSV)")
    st.caption(
        "Colonnes : date (AAAA-MM-JJ), type (DEPOSIT, WITHDRAW, BUY, SELL, DIVIDEND), ticker, quantity, price, fees. "
        "Pour DEPOSIT/WITHDRAW, mets le montant en USD dans `price`."
    )
    template = pd.DataFrame(
        [["2026-01-05", "DEPOSIT", "", 1, 5000, 0], ["2026-01-06", "BUY", "AAPL", 5, 190.5, 1]],
        columns=["date", "type", "ticker", "quantity", "price", "fees"],
    )
    st.download_button("Modèle CSV", template.to_csv(index=False).encode("utf-8"), "modele_transactions.csv", "text/csv")
    if "import_msg" in st.session_state:
        n_ok, errs = st.session_state.pop("import_msg")
        (st.success if n_ok else st.error)(f"{n_ok} transaction(s) importée(s).")
        if errs:
            st.warning("\n".join(errs))

    st.markdown("**Option A : coller le contenu du CSV** (sans passer par l'envoi de fichier)")
    pasted = st.text_area("Contenu CSV (avec la ligne d'en-tête)", height=180, key="paste_csv",
                          placeholder="date,type,ticker,quantity,price,fees\n2026-05-04,DEPOSIT,,1,10000,0")
    if st.button("Importer le texte collé"):
        try:
            res = import_dataframe(core.read_transactions_csv(pasted))
        except Exception as e:
            res = (0, [f"Lecture impossible : {e}"])
        st.session_state["import_msg"] = res
        st.rerun()

    st.markdown("**Option B : envoyer un fichier CSV**")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None and st.button("Lancer l'import"):
        try:
            res = import_dataframe(core.read_transactions_csv(up))
        except Exception as e:
            res = (0, [f"Lecture impossible : {e}"])
        st.session_state["import_msg"] = res
        st.rerun()
