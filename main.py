import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# USER CONFIG
# ---------------------------
ETF_UNIVERSE = {
    'US Equity': 'SPY',
    'Intl Equity': 'EFA',
    'Bonds': 'BND',
    'REITs': 'VNQ',
    'Commodities': 'GSG'
}
# Example stock tickers per asset class (extend this from your own RWA/stocks universe)
STOCKS_BY_CLASS = {
    'US Equity': ['AAPL','MSFT','AMZN','GOOGL','TSLA'],
    'Intl Equity': ['SAP.DE','TM','RIO.L','BHP.AX','NVS'],
    'REITs': ['SPG','PLD','EQIX','AMT','O'],
    'Commodities': ['GLD','SLV','USO'],  # note: commodity ETFs
    'Bonds': ['TLT','LQD','BND']  # bond ETFs / proxies
}

# Parameters
MOM_LOOKBACK_MONTHS = 12
REBALANCE_FREQ = 'M'   # 'M' monthly
TOP_N_ETFS = 3
TOP_K_STOCKS = 5

# ---------------------------
# Helper functions
# ---------------------------
def fetch_price_df(tickers, period='3y', interval='1d'):
    # Use auto_adjust to get adjusted close prices directly when possible.
    data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
    # yf.download may return several shapes depending on number of tickers and options:
    # - DataFrame with tickers as columns (ideal)
    # - Series for a single ticker
    # - DataFrame with MultiIndex columns (if auto_adjust=False or older yfinance)
    if data is None:
        return pd.DataFrame()

    # If MultiIndex columns, try to extract adjusted/close prices levels
    if isinstance(data.columns, pd.MultiIndex):
        top_levels = list(data.columns.get_level_values(0))
        if 'Adj Close' in top_levels:
            data = data['Adj Close']
        elif 'Close' in top_levels:
            data = data['Close']
        else:
            # fallback: try to pick the last level (usually the ticker level)
            try:
                # select numeric/price-like first column level if present
                data = data.xs(data.columns.levels[0][0], axis=1, level=0)
            except Exception:
                # give up and return empty
                return pd.DataFrame()

    # If a Series (single ticker), convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Drop columns that are all-NaN (tickers with no data)
    data = data.dropna(how='all', axis=1)
    return data

def momentum_score(price_series, months=MOM_LOOKBACK_MONTHS):
    # approximate months -> trading days ~ 21 * months
    days = int(21 * months)
    if len(price_series) < days: return np.nan
    return price_series.iloc[-1] / price_series.iloc[-days] - 1

def hist_vol(price_series, window=63):  # ~3 months
    returns = price_series.pct_change().dropna()
    return returns.rolling(window).std().iloc[-1] * np.sqrt(252)

def avg_daily_volume(ticker):
    # placeholder: using yfinance info (may be slower)
    info = yf.Ticker(ticker).info
    return info.get('averageDailyVolume10Day', np.nan)

def fetch_basic_fundamentals(ticker):
    t = yf.Ticker(ticker)
    info = t.info
    pe = info.get('trailingPE', np.nan)
    roe = info.get('returnOnEquity', np.nan)
    eps_growth = info.get('earningsQuarterlyGrowth', np.nan)
    return {'pe': pe, 'roe': roe, 'eps_growth': eps_growth}

# ---------------------------
# Asset-class picking (Quantpedia-style momentum)
# ---------------------------
print("Fetching ETF prices...")
etf_prices = fetch_price_df(list(ETF_UNIVERSE.values()), period='3y')
etf_scores = []
for name, etf in ETF_UNIVERSE.items():
    s = etf_prices[etf].dropna()
    mom = momentum_score(s)
    vol = hist_vol(s)
    etf_scores.append({'class': name, 'etf': etf, 'momentum_12m': mom, 'vol_3m': vol})

etf_df = pd.DataFrame(etf_scores).sort_values('momentum_12m', ascending=False).reset_index(drop=True)
print("\nETF momentum ranking (12m):")
print(etf_df[['class','etf','momentum_12m']])

top_etfs = etf_df.head(TOP_N_ETFS)
print(f"\nTop {TOP_N_ETFS} asset classes selected: ")
print(top_etfs[['class','etf']])

# ---------------------------
# Quantamental scoring inside each chosen class
# ---------------------------
def score_universe(tickers):
    # fetch price matrix
    price_df = fetch_price_df(tickers, period='2y')
    # build features per ticker
    rows = []
    for t in tickers:
        if t not in price_df.columns:
            continue
        p = price_df[t].dropna()
        if p.empty: continue
        mom1 = momentum_score(p, months=3)    # 3m
        mom12 = momentum_score(p, months=12)  # 12m
        vol = hist_vol(p, window=42)          # ~2 months
        # fundamentals (best effort)
        f = fetch_basic_fundamentals(t)
        avgvol = avg_daily_volume(t)
        rows.append({
            'ticker': t,
            'mom_3m': mom1,
            'mom_12m': mom12,
            'vol_2m': vol,
            'avg10d_vol': avgvol,
            'pe': f['pe'],
            'roe': f['roe'],
            'eps_g': f['eps_growth']
        })
    df = pd.DataFrame(rows).set_index('ticker')
    # Clean & standardize: fill nans with medians then scale
    df_filled = df.fillna(df.median(numeric_only=True))
    scaler = StandardScaler()
    numeric = df_filled.select_dtypes(include=[np.number])
    scaled = pd.DataFrame(scaler.fit_transform(numeric), index=numeric.index, columns=numeric.columns)
    # TerraNovaAlpha: weighted sum (simple example)
    # We reward momentum, roe, eps growth; penalize vol and high PE
    weights = {
        'mom_12m': 0.30,
        'mom_3m': 0.20,
        'roe': 0.20,
        'eps_g': 0.15,
        'vol_2m': -0.10,
        'pe': -0.05
    }
    # ensure all fields present in scaled; if not, drop weights for missing
    for col in list(weights.keys()):
        if col not in scaled.columns:
            weights.pop(col, None)
    alpha = sum(scaled[c] * w for c,w in weights.items())
    df_out = scaled.copy()
    df_out['terranova_alpha'] = alpha
    df_out = df_out.sort_values('terranova_alpha', ascending=False)
    return df_out

# iterate chosen classes
final_recommendations = {}
for idx,row in top_etfs.iterrows():
    cls = row['class']
    tickers = STOCKS_BY_CLASS.get(cls, [])
    if not tickers:
        continue
    print(f"\nScoring universe for class: {cls} ...")
    scored = score_universe(tickers)
    print(scored[['terranova_alpha']].head(TOP_K_STOCKS))
    final_recommendations[cls] = scored

# ---------------------------
# Simple allocation suggestion
# ---------------------------
print("\n--- Suggested Allocation ---")
alloc = {}
equal_alloc = 1.0 / len(top_etfs)
for idx,row in top_etfs.iterrows():
    cls = row['class']
    alloc[cls] = equal_alloc
print(alloc)

# print top picks summary
print("\nTop stock picks per chosen class (top 3):")
for cls, df in final_recommendations.items():
    print(f"\n{cls}:")
    # print rounded values for clarity
    display_df = df[['terranova_alpha']].round(3).head(3)
    print(display_df)

# save results
for cls, df in final_recommendations.items():
    # save rounded values to CSV for clearer numbers
    df_to_save = df[['terranova_alpha']].round(3)
    df_to_save.to_csv(f"terranova_{cls.replace(' ','_')}_scores.csv")

print("\nDone. CSVs saved for each class.")

