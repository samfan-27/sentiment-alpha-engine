import pandas as pd
import pytest
from src.signals import generate_signals, merge_features_and_sentiment, S_THRESH, M_THRESH, SHOCK_MULTIPLIER

def test_signal_triggers_on_valid_conditions():
    df = pd.DataFrame({
        'ticker': ['TEST_A'],
        'daily_return': [-0.05],
        'volatility_21d': [0.02],
        'finbert_score': [0.1],
        'macro_sentiment': [0.5]
    })
    result = generate_signals(df)
    assert result['signal_triggered'].iloc[0]

def test_signal_fails_on_bad_macro():
    df = pd.DataFrame({
        'ticker': ['TEST_B'],
        'daily_return': [-0.05],
        'volatility_21d': [0.02],
        'finbert_score': [0.1],
        'macro_sentiment': [-0.5]
    })
    result = generate_signals(df)
    assert not result['signal_triggered'].iloc[0]

def test_signal_fails_on_bad_idiosyncratic_news():
    df = pd.DataFrame({
        'ticker': ['TEST_C'],
        'daily_return': [-0.05],
        'volatility_21d': [0.02],
        'finbert_score': [-0.8],
        'macro_sentiment': [0.5]
    })
    result = generate_signals(df)
    assert not result['signal_triggered'].iloc[0]

def test_no_news_defaults_to_neutral_and_triggers():
    features = pd.DataFrame({
        'ticker': ['TEST_D'],
        'daily_return': [-0.05],
        'volatility_21d': [0.02],
        'last_close_price': [100.0]
    })
    empty_sentiment = pd.DataFrame()
    
    merged = merge_features_and_sentiment(features, empty_sentiment, 0.5)
    
    result = generate_signals(merged)
    assert result['finbert_score'].iloc[0] == 0.0
    assert result['signal_triggered'].iloc[0]
    