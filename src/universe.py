import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from loguru import logger
from src.config import UNIVERSE_SIZE, TRAILING_DAYS_VOL, TRAILING_DAYS_BETA

def get_sp500_tickers() -> list[str]:
    logger.info('Fetching S&P 500 tickers...')
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    logger.info(f'Fetched {len(tickers)} S&P 500 tickers.')
    return tickers

def fetch_market_data(tickers: list[str]) -> pd.DataFrame:
    target_tickers = tickers + ['SPY']
    logger.info(f'Downloading historical data for {len(target_tickers)} symbols...')
    
    data = yf.download(target_tickers, period="2y", interval="1d", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
        
    return closes.ffill()

def calculate_quant_features(closes: pd.DataFrame) -> pd.DataFrame:
    logger.info('Calculating log returns, beta, and volatility features...')
    
    log_returns = np.log(closes / closes.shift(1)).dropna(how='all')
    returns_1y = log_returns.tail(TRAILING_DAYS_BETA)
    
    cov_matrix = returns_1y.cov()
    market_variance = returns_1y['SPY'].var()
    
    cov_with_market = cov_matrix['SPY']
    
    betas = cov_with_market / market_variance
    
    returns_21d = log_returns.tail(TRAILING_DAYS_VOL)
    volatility_21d = returns_21d.std(ddof=1)
    
    latest_returns = log_returns.iloc[-1]
    
    features = pd.DataFrame({
        'beta_1y': betas,
        'volatility_21d': volatility_21d,
        'daily_return': latest_returns,
        'last_close_price': closes.iloc[-1]
    })
    
    features = features.drop(index=['SPY'], errors='ignore').dropna()
    features.index.name = 'ticker'
    
    return features.reset_index()

def get_high_beta_universe() -> pd.DataFrame:
    tickers = get_sp500_tickers()
    closes = fetch_market_data(tickers)
    features = calculate_quant_features(closes)
    
    top_universe = features.nlargest(UNIVERSE_SIZE, 'beta_1y')
    logger.info(f"Filtered down to Top {UNIVERSE_SIZE} high-beta S&P 500 equities.")
    
    return top_universe

if __name__ == '__main__':
    logger.info('Starting Universe Generation...')
    universe_df = get_high_beta_universe()
    print("\n--- Top 5 High Beta Stocks ---")
    print(universe_df.head())
    