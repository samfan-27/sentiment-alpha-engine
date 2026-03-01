import requests
import pandas as pd
import numpy as np
import concurrent.futures
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.config import HF_TOKEN

HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"

class ModelLoadingError(Exception):
    pass

def _query_hf_api(payload: dict) -> list:
    headers = {'Authorization': f'Bearer {HF_TOKEN}'}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=30), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((ModelLoadingError, requests.exceptions.RequestException))
)
def calculate_finbert_score(text_batch: list[str]) -> list[float]:
    if not text_batch:
        return []

    def score_single_text(text: str) -> float:
        try:
            prediction = _query_hf_api({"inputs": text})
            
            if isinstance(prediction, list) and len(prediction) > 0:
                if isinstance(prediction[0], list):
                    preds = prediction[0] 
                else:
                    preds = prediction 
            else:
                preds = []

            pos_score = 0.0
            neg_score = 0.0
            for pred in preds:
                if isinstance(pred, dict):
                    label = pred.get('label', '').lower()
                    if label == 'positive':
                        pos_score = pred.get('score', 0.0)
                    elif label == 'negative':
                        neg_score = pred.get('score', 0.0)
            
            return pos_score - neg_score
            
        except Exception as e:
            logger.error(f'Failed to score text: "{text[:30]}..." - Error: {e}')
            return 0.0 
        
    logger.info(f'Scoring {len(text_batch)} texts concurrently...')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(score_single_text, text_batch)
        
    return list(results)

def process_macro_sentiment(macro_news_df: pd.DataFrame) -> float:
    if macro_news_df.empty or 'headline' not in macro_news_df.columns:
        logger.info('No macro news found. Defaulting M_t to 0.0 (Neutral).')
        return 0.0

    headlines = macro_news_df['headline'].tolist()
    logger.info(f'Processing sentiment for {len(headlines)} macro headlines...')
    
    scores = calculate_finbert_score(headlines)
    m_t = float(np.mean(scores))
    logger.info(f'Daily Macro Sentiment (M_t) calculated: {m_t:.4f}')
    return m_t

def process_company_sentiment(company_news_df: pd.DataFrame) -> pd.DataFrame:
    if company_news_df.empty or 'headline' not in company_news_df.columns:
        logger.info('No company news found. Returning empty sentiment dataframe.')
        return pd.DataFrame(columns=['ticker', 'finbert_score'])

    logger.info(f'Processing sentiment for {len(company_news_df)} company headlines...')
    headlines = company_news_df['headline'].tolist()
    scores = calculate_finbert_score(headlines)
    
    company_news_df['article_score'] = scores
    
    daily_sentiment = company_news_df.groupby('ticker', as_index=False)['article_score'].mean()
    daily_sentiment.rename(columns={'article_score': 'finbert_score'}, inplace=True)
    
    logger.info(f'Processed daily sentiment for {len(daily_sentiment)} unique tickers.')
    return daily_sentiment

if __name__ == "__main__":
    # test code
    logger.info('Testing FinBERT Inference Pipeline...')
    
    test_macro = pd.DataFrame({
        'headline': ['Federal Reserve cuts interest rates, markets rally', 
                     'Inflation remains stubbornly high, sparking recession fears']
    })
    
    test_company = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'TSLA'],
        'headline': [
            'Apple reports record-breaking iPhone 15 sales.',
            'Apple hit with massive European antitrust fine.',
            'Tesla deliveries miss expectations by 20%.'
        ]
    })
    
    m_t = process_macro_sentiment(test_macro)
    s_it = process_company_sentiment(test_company)
    
    print(f'\nMacro Sentiment (M_t): {m_t:.4f}')
    print('\nCompany Sentiment (S_i,t):')
    print(s_it)
    