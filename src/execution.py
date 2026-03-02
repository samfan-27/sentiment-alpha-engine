import pandas as pd
from loguru import logger
from datetime import datetime, timezone
from pandas.tseries.offsets import BDay

PORTFOLIO_MAX_EXPOSURE = 0.10
MIN_SLIPPAGE_BPS = 0.0005
VOL_PENALTY_FACTOR = 0.05

def calculate_slippage(volatility: float, order_size_usd: float, adv_usd: float = 10_000_000) -> float:
    s_base = MIN_SLIPPAGE_BPS
    s_volatility = volatility * VOL_PENALTY_FACTOR
    
    s_volume = VOL_PENALTY_FACTOR * (order_size_usd / adv_usd)
    
    return s_base + s_volatility + s_volume

def generate_trade_ledger_entries(signals_df: pd.DataFrame, portfolio_nav: float) -> list[dict]:
    logger.info('Generating trade execution ledger...')
    triggered = signals_df[signals_df['signal_triggered'] == True].copy()
    
    if triggered.empty:
        logger.info('No trades triggered today. Ledger remains unchanged.')
        return []

    # Position Sizing: Inverse Volatility Parity
    total_capital_to_deploy = portfolio_nav * PORTFOLIO_MAX_EXPOSURE
    
    inv_vol = 1.0 / triggered['volatility_21d']
    inv_vol_sum = inv_vol.sum()
    
    triggered['target_weight'] = inv_vol / inv_vol_sum
    triggered['position_size_usd'] = triggered['target_weight'] * total_capital_to_deploy
    
    ledger_entries = []
    current_date = datetime.now(timezone.utc)
    date_str = current_date.strftime('%Y-%m-%d')
    
    exit_date_str = (current_date + BDay(3)).strftime('%Y-%m-%d')
    for _, row in triggered.iterrows():
        ticker = row['ticker']
        volatility = row['volatility_21d']
        pos_size = row['position_size_usd']
        expected_price = row['last_close_price'] 
        
        slippage_pct = calculate_slippage(volatility, pos_size)
        
        fill_price = expected_price * (1 + slippage_pct)
        
        # State Snapshot for Post-Trade Attribution
        signal_snapshot = {
            "daily_return": row['daily_return'],
            "volatility_21d": volatility,
            "finbert_score": row['finbert_score'],
            "macro_sentiment": row['macro_sentiment'],
            "slippage_applied_pct": slippage_pct
        }
        
        entry = {
            "ticker": ticker,
            "status": "OPEN",
            "date_entered": date_str,
            "entry_price": fill_price,
            "date_exited": exit_date_str, 
            "hold_duration_days": 3,
            "signal_snapshot": signal_snapshot
        }
        
        ledger_entries.append(entry)
        logger.success(f"Trade generated for {ticker}: Size ${pos_size:.2f}, Slippage {slippage_pct*10000:.1f} bps")
        
    return ledger_entries
