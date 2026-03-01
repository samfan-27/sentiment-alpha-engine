import requests
import pandas as pd
from datetime import datetime, timezone
import time
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt

from src.config import FINNHUB_API_KEY

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def _make_finnhub_request(endpoint: str, params: dict) -> list:
    if not FINNHUB_API_KEY:
        raise ValueError('FINNHUB_API_KEY is not set in environment variables.')
        
    headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
    url = f"{FINNHUB_BASE_URL}{endpoint}"
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 429:
        logger.warning('Rate limit hit. Tenacity will trigger backoff...')
        response.raise_for_status()
        
    response.raise_for_status()
    return response.json()

def fetch_market_news() -> pd.DataFrame:
    logger.info("Fetching general market news...")
    try:
        data = _make_finnhub_request("/news", {"category": "general"})
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No market news found.")
            return pd.DataFrame()
            
        df['ticker'] = 'MACRO'
        return _clean_news_dataframe(df)
    except Exception as e:
        logger.error(f"Failed to fetch market news: {e}")
        return pd.DataFrame()

def fetch_company_news(ticker: str, target_date: str) -> pd.DataFrame:
    try:
        params = {
            "symbol": ticker,
            "from": target_date,
            "to": target_date
        }
        data = _make_finnhub_request("/company-news", params)
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame()
            
        df['ticker'] = ticker
        return _clean_news_dataframe(df)
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return pd.DataFrame()

def _clean_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = ['ticker', 'id', 'datetime', 'headline', 'summary', 'source', 'url']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
        
    return df

def get_daily_news(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logger.info(f'Fetching news for {target_date}...')
    
    macro_news_df = fetch_market_news()
    
    company_news_list = []
    logger.info(f"Fetching company news for {len(tickers)} tickers. Pacing requests...")
    
    for i, ticker in enumerate(tickers):
        df = fetch_company_news(ticker, target_date)
        if not df.empty:
            company_news_list.append(df)
            
        # Pacing: Max 60 calls per minute = ~1 call per second
        # If we batch 100 tickers, a 0.5s sleep + request latency usually stays safely under the limit.
        time.sleep(0.5)
        
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i + 1}/{len(tickers)} tickers...")
            
    company_news_df = pd.concat(company_news_list, ignore_index=True) if company_news_list else pd.DataFrame()
    
    logger.info(f"Retrieved {len(macro_news_df)} macro headlines and {len(company_news_df)} company headlines.")
    return macro_news_df, company_news_df

if __name__ == "__main__":
    # local execution test
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    logger.info(f'Testing News Fetcher with {test_tickers}...')
    
    macro_df, company_df = get_daily_news(test_tickers)
    
    print('\n--- Macro News Sample ---')
    print(macro_df[['datetime', 'headline']].head(3))
    
    print('\n--- Company News Sample ---')
    print(company_df[['ticker', 'datetime', 'headline']].head(3))
    