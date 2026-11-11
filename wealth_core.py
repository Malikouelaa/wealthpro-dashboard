"""Logique financière pure de WealthPro (sans Streamlit ni accès réseau).

Conventions
-----------
- Table `transactions` : date, type, ticker, quantity, price, fees, currency, fx_rate
- DEPOSIT / WITHDRAW : montant en USD dans `price`, `quantity` = 1
- BUY / SELL        : prix et frais dans la devise de l'actif, `fx_rate` = taux devise -> USD
- DIVIDEND          : montant brut dans `price` (devise de l'actif), retenue/frais dans `fees`
- Tous les calculs internes sont en USD ; la conversion d'affichage se fait dans l'app.
"""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Flux de trésorerie par transaction
# --------------------------------------------------------------------------
def cash_delta(row) -> float:
    """Variation du cash (USD) provoquée par une transaction."""
    fx = row.fx_rate if row.fx_rate else 1.0
    if row.type == "DEPOSIT":
        return float(row.price)
    if row.type == "WITHDRAW":
        return -float(row.price)
    if row.type == "BUY":
        return -(row.quantity * row.price + row.fees) * fx
    if row.type == "SELL":
        return (row.quantity * row.price - row.fees) * fx
    if row.type == "DIVIDEND":
        return (row.price - row.fees) * fx
    return 0.0


def external_flow(row) -> float:
    """Flux externe (apport / retrait) : seuls ceux-ci sont exclus de la performance."""
    if row.type == "DEPOSIT":
        return float(row.price)
    if row.type == "WITHDRAW":
        return -float(row.price)
    return 0.0


# --------------------------------------------------------------------------
# Positions, prix de revient moyen pondéré (PRU), plus-values réalisées
# --------------------------------------------------------------------------
def compute_positions(tx: pd.DataFrame) -> dict:
    positions: dict[str, dict] = {}
    cash = realized = dividends = 0.0
    realized_log: list[dict] = []

    for r in tx.sort_values(["date", "id"]).itertuples(index=False):
        cash += cash_delta(r)
        fx = r.fx_rate if r.fx_rate else 1.0

        if r.type == "BUY":
            p = positions.setdefault(
                r.ticker, {"qty": 0.0, "cost": 0.0, "cost_native": 0.0, "currency": r.currency}
            )
            p["qty"] += r.quantity
            p["cost"] += (r.quantity * r.price + r.fees) * fx
            p["cost_native"] += r.quantity * r.price + r.fees
            p["currency"] = r.currency

        elif r.type == "SELL" and r.ticker in positions:
            p = positions[r.ticker]
            qty = min(r.quantity, p["qty"])
            if qty <= 0:
                continue
            ratio = qty / p["qty"]
            cost_out = p["cost"] * ratio
            proceeds = (qty * r.price - r.fees) * fx
            pnl = proceeds - cost_out
            realized += pnl
            realized_log.append(
                {"Date": r.date.date(), "Actif": r.ticker, "Quantité": qty, "Plus-value (USD)": pnl}
            )
            p["cost"] -= cost_out
            p["cost_native"] -= p["cost_native"] * ratio
            p["qty"] -= qty
            if p["qty"] < 1e-9:
                del positions[r.ticker]

        elif r.type == "DIVIDEND":
            dividends += (r.price - r.fees) * fx

    return {
        "positions": positions,
        "cash": cash,
        "realized": realized,
        "dividends": dividends,
        "realized_log": realized_log,
    }


# --------------------------------------------------------------------------
# Série de valeur du patrimoine (cash + titres valorisés au cours historique)
# --------------------------------------------------------------------------
def _fill(s: pd.Series | None, idx: pd.DatetimeIndex, fallback: pd.Series | None = None) -> pd.Series:
    """Aligne une série sur idx (ffill puis bfill). Utilise `fallback` si s est vide."""
    if s is None or len(s) == 0:
        s = fallback
    if s is None or len(s) == 0:
        return pd.Series(np.nan, index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(idx.union(s.index)).ffill().reindex(idx).bfill()


def build_series(tx, prices: dict, fxs: dict, end=None):
    """Retourne (DataFrame[value, flow], DataFrame des prix en USD par actif).

    value : patrimoine total (cash + titres) en USD, jours ouvrés
    flow  : apports (+) / retraits (-) externes en USD, rattachés au jour ouvré suivant
    """
    end = pd.Timestamp(end or dt.date.today()).normalize()
    start = tx["date"].min().normalize()
    idx = pd.bdate_range(start, max(end, start))
    if len(idx) == 0:
        idx = pd.DatetimeIndex([start])

    tx = tx.copy()
    tx["snap"] = idx[np.clip(idx.searchsorted(pd.DatetimeIndex(tx["date"].dt.normalize())), 0, len(idx) - 1)]
    tx["cash_d"] = [cash_delta(r) for r in tx.itertuples(index=False)]
    tx["flow"] = [external_flow(r) for r in tx.itertuples(index=False)]

    cash = tx.groupby("snap")["cash_d"].sum().reindex(idx, fill_value=0.0).cumsum()
    flow = tx.groupby("snap")["flow"].sum().reindex(idx, fill_value=0.0)

    value = cash.copy()
    assets = {}
    for t, g in tx[tx["type"].isin(["BUY", "SELL"])].groupby("ticker"):
        sign = np.where(g["type"] == "BUY", 1.0, -1.0)
        dq = pd.Series(sign * g["quantity"].values, index=g["snap"].values).groupby(level=0).sum()
        qty = dq.reindex(idx, fill_value=0.0).cumsum()

        trade_px = pd.Series(g["price"].values, index=g["snap"].values).groupby(level=0).last()
        px = _fill(prices.get(t), idx, fallback=trade_px)

        cur = g["currency"].iloc[-1]
        if cur == "USD":
            fx = pd.Series(1.0, index=idx)
        else:
            trade_fx = pd.Series(g["fx_rate"].values, index=g["snap"].values).groupby(level=0).last()
            fx = _fill(fxs.get(cur), idx, fallback=trade_fx)

        px_usd = px * fx
        assets[t] = px_usd
        value = value + (qty * px_usd).fillna(0.0)

    return pd.DataFrame({"value": value, "flow": flow}), pd.DataFrame(assets, index=idx)


# --------------------------------------------------------------------------
# Rendements : TWR, XIRR, PME, risque
# --------------------------------------------------------------------------
def daily_returns(value: pd.Series, flow: pd.Series) -> pd.Series:
    """Rendement quotidien pondéré par le temps, neutralisant apports et retraits."""
    prev = value.shift(1)
    r = (value - flow) / prev - 1
    return r.where(prev > 0, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def xirr(cashflows) -> float | None:
    """Taux de rendement interne annualisé (flux datés). None si non calculable."""
    cfs = [(pd.Timestamp(d), float(a)) for d, a in cashflows if a != 0]
    if len(cfs) < 2:
        return None
    if not (any(a > 0 for _, a in cfs) and any(a < 0 for _, a in cfs)):
        return None
    t0 = min(d for d, _ in cfs)
    ts = np.array([(d - t0).days / 365.0 for d, _ in cfs])
    am = np.array([a for _, a in cfs])

    def f(rate: float) -> float:
        return float(np.sum(am / (1.0 + rate) ** ts))

    lo, hi = -0.999, 100.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def investor_cashflows(series: pd.DataFrame, final_value: float | None = None):
    """Flux vus par l'investisseur : apports négatifs, retraits positifs, valeur finale positive."""
    cfs = [(d, -f) for d, f in series["flow"].items() if f != 0]
    cfs.append((series.index[-1], series["value"].iloc[-1] if final_value is None else final_value))
    return cfs


def pme(series: pd.DataFrame, bench_px: pd.Series) -> pd.DataFrame:
    """Public Market Equivalent : mêmes apports/retraits investis dans l'indice à la date de chaque flux."""
    px = _fill(bench_px, series.index)
    units = (series["flow"] / px).cumsum()
    return pd.DataFrame({"value": units * px, "px": px})


def risk_metrics(r: pd.Series, bench_r: pd.Series | None = None, rf: float = 0.0) -> dict | None:
    r = r.iloc[1:]
    n = len(r)
    if n < 20:
        return None
    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    ann_ret = (1 + total) ** (TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = cum / cum.cummax() - 1
    beta = None
    if bench_r is not None:
        b = bench_r.reindex(r.index).fillna(0.0)
        if b.var() > 0:
            beta = r.cov(b) / b.var()
    return {
        "twr": total,
        "ann_return": ann_ret,
        "vol": vol,
        "sharpe": (ann_ret - rf) / vol if vol > 0 else np.nan,
        "max_dd": dd.min(),
        "drawdown": dd,
        "beta": beta,
    }


# --------------------------------------------------------------------------
# Lecture robuste d'un CSV de transactions (Excel/Numbers FR, BOM, ; ou ,)
# --------------------------------------------------------------------------
def read_transactions_csv(file) -> pd.DataFrame:
    raw = file.read() if hasattr(file, "read") else file
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    else:
        text = str(raw)
    lines = text.splitlines()
    first = lines[0] if lines else ""
    sep = max([",", ";", "\t"], key=first.count)
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()
    if sep != ",":
        for col in ("quantity", "price", "fees"):
            if col in df.columns:
                df[col] = df[col].str.replace(",", ".", regex=False)
    return df.dropna(how="all")
