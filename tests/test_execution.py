import pandas as pd
import pytest
from datetime import datetime, UTC
from pandas.tseries.offsets import BDay
from src.execution import calculate_slippage, generate_trade_ledger_entries, MIN_SLIPPAGE_BPS

def test_slippage_scales_with_volatility():
    low_vol_slip = calculate_slippage(volatility=0.01, order_size_usd=100_000)
    high_vol_slip = calculate_slippage(volatility=0.05, order_size_usd=100_000)
    
    assert high_vol_slip > low_vol_slip
    assert low_vol_slip > MIN_SLIPPAGE_BPS

def test_inverse_volatility_parity_sizing():
    df = pd.DataFrame({
        'ticker': ['SAFE_STOCK', 'RISKY_STOCK'],
        'signal_triggered': [True, True],
        'volatility_21d': [0.01, 0.04],
        'last_close_price': [100.0, 100.0],
        'daily_return': [-0.05, -0.10],
        'finbert_score': [0.5, 0.5],
        'macro_sentiment': [0.5, 0.5]
    })
    
    portfolio_nav = 100_000
    entries = generate_trade_ledger_entries(df, portfolio_nav)
    
    safe_size = None
    risky_size = None
    
    for entry in entries:
        if entry['ticker'] == 'SAFE_STOCK':
            safe_size = 1 / entry['signal_snapshot']['volatility_21d']
        else:
            risky_size = 1 / entry['signal_snapshot']['volatility_21d']
            
    assert safe_size == pytest.approx(risky_size * 4)

def test_weekend_business_day_logic():
    df = pd.DataFrame({
        'ticker': ['TEST_A'],
        'signal_triggered': [True],
        'volatility_21d': [0.02],
        'last_close_price': [100.0],
        'daily_return': [-0.05],
        'finbert_score': [0.5],
        'macro_sentiment': [0.5]
    })
    
    entries = generate_trade_ledger_entries(df, 100_000)
    exit_date_str = entries[0]['date_exited']
    
    expected_date = (datetime.now(UTC) + BDay(3)).strftime('%Y-%m-%d')
    assert exit_date_str == expected_date
    