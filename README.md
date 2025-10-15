# Asset-class-picking-
Asset picking model based on features like moving avarage ,volatilty e.t.c


 Fetches price data for a small ETF universe (SPY, EFA, BND, VNQ, GSG) using yfinance.
Ranks the ETFs using 12-month momentum and selects the top N (configurable; default 3).
For each selected asset class, fetches prices for a small list of representative tickers (stocks or ETFs).
Computes per-ticker features:
Momentum (3m and 12m), historical volatility (~2 months), average daily volume, and a few fundamental fields fetched via yfinance (PE, ROE, EPS growth).
Fills missing numeric values with column medians, standardizes numeric features with StandardScaler, and computes a simple weighted score ("terranova_alpha") as a linear combination of selected scaled features (positive weights for momentum/ROE/eps growth, negative weights for volatility/PE).
Prints top picks and saves CSVs (now with rounded alpha values).
