import pandas as pd
from loguru import logger
from datetime import datetime, timezone

S_THRESH = -0.05
M_THRESH = 0.0
SHOCK_MULTIPLIER = -1.5

def merge_features_and_sentiment(features_df: pd.DataFrame, sentiment_df: pd.DataFrame, m_t: float) -> pd.DataFrame:
    logger.info('Merging quantitative features with FinBERT sentiment scores...')
    
    if sentiment_df.empty:
        df_merged = features_df.copy()
        df_merged['finbert_score'] = 0.0
    else:
        df_merged = pd.merge(features_df, sentiment_df, on='ticker', how='left')
        df_merged['finbert_score'] = df_merged['finbert_score'].fillna(0.0)
    
    df_merged['macro_sentiment'] = m_t
    df_merged['date'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    return df_merged

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Evaluating mathematical signal conditions...')
    
    signals = []
    
    for _, row in df.iterrows():
        ticker = row['ticker']
        r_t = row['daily_return']
        sigma = row['volatility_21d']
        s_it = row['finbert_score']
        m_t = row['macro_sentiment']
        
        macro_pass = m_t > M_THRESH
        is_shock = r_t < (SHOCK_MULTIPLIER * sigma)
        sentiment_pass = s_it > S_THRESH
        
        signal_triggered = bool(macro_pass and is_shock and sentiment_pass)
        signals.append(signal_triggered)
        
        if signal_triggered:
            logger.success(f'Signal triggered for {ticker}: Return={r_t:.4f}, Vol={sigma:.4f}, S_it={s_it:.4f}')

    df['signal_triggered'] = signals
    
    triggered_count = sum(signals)
    logger.info(f'Signal generation complete. {triggered_count} trades triggered today.')
    
    return df

if __name__ == "__main__":
    # test code
    test_features = pd.DataFrame({
        'ticker': ['AAPL', 'TSLA', 'NVDA'],
        'beta_1y': [1.2, 2.1, 1.8],
        'volatility_21d': [0.015, 0.035, 0.025],
        'daily_return': [-0.030, -0.060, 0.010],
        'last_close_price': [150.0, 180.0, 450.0]
    })
    
    test_sentiment = pd.DataFrame({
        'ticker': ['AAPL', 'TSLA'],
        'finbert_score': [0.10, -0.80]
    })
    
    macro_score = 0.20
    
    merged = merge_features_and_sentiment(test_features, test_sentiment, macro_score)
    final_signals = generate_signals(merged)
    
    print("\n--- Final Signal Output ---")
    print(final_signals[['ticker', 'daily_return', 'finbert_score', 'signal_triggered']])
    