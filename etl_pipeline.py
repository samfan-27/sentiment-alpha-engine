import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from loguru import logger

from src.universe import get_high_beta_universe
from src.news_fetcher import get_daily_news
from src.sentiment import process_macro_sentiment, process_company_sentiment
from src.signals import merge_features_and_sentiment, generate_signals
from src.execution import generate_trade_ledger_entries
from src.llm_explainer import generate_trade_explanation
from src.db import (
    upsert_market_data_signals, 
    insert_trade, 
    update_trade_exit, 
    db
)

PAPER_NAV = 100_000.0

def manage_trade_exits():
    logger.info('Checking for expired T+3 trades to close...')
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    try:
        response = db.table('trade_ledger').select('*').eq('status', 'OPEN').lte('date_exited', today_str).execute()
        open_trades = response.data
        
        if not open_trades:
            logger.info('No trades require closing today.')
            return

        logger.info(f'Found {len(open_trades)} trades ready to be closed.')
        tickers_to_close = list(set([t['ticker'] for t in open_trades]))
        
        current_data = yf.download(tickers_to_close, period="1d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(current_data.columns, pd.MultiIndex):
            closes = current_data['Close']
        else:
            closes = pd.DataFrame({tickers_to_close[0]: current_data['Close']})

        for trade in open_trades:
            ticker = trade['ticker']
            entry_price = trade['entry_price']
            trade_id = trade['trade_id']
            
            exit_price = float(closes[ticker].iloc[-1])
                
            pnl_percent = (exit_price - entry_price) / entry_price
            
            exit_data = {
                "status": "CLOSED",
                "exit_price": exit_price,
                "pnl_percent": pnl_percent
            }
            
            update_trade_exit(trade_id, exit_data)
            logger.success(f'Closed trade {trade_id} ({ticker}) at {exit_price:.2f}. PnL: {pnl_percent*100:.2f}%')
            
    except Exception as e:
        logger.error(f'Error managing trade exits: {e}')

def run_pipeline():
    logger.info('Daily Algo Pipeline')
    
    try:
        universe_df = get_high_beta_universe()
        tickers = universe_df['ticker'].tolist()
        
        macro_news, company_news = get_daily_news(tickers)
        
        m_t = process_macro_sentiment(macro_news)
        s_it_df = process_company_sentiment(company_news)
        
        merged_df = merge_features_and_sentiment(universe_df, s_it_df, m_t)
        signals_df = generate_signals(merged_df)
        
        db_records = signals_df.replace({np.nan: None}).to_dict(orient='records')
        for record in db_records:
            record['signal_triggered'] = bool(record['signal_triggered'])
        upsert_market_data_signals(db_records)
        
        trades = generate_trade_ledger_entries(signals_df, PAPER_NAV)
        for trade in trades:
            explanation = generate_trade_explanation(trade['ticker'], trade['signal_snapshot'])
            trade['ai_explanation'] = explanation
            insert_trade(trade)
            
        manage_trade_exits()
        
        logger.info("Pipline Execution Completed")
        
    except Exception as e:
        logger.error(f"Pipeline Execution Failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
    