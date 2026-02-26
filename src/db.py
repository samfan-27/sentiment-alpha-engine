from supabase import create_client, Client
from loguru import logger
from src.config import SUPABASE_URL, SUPABASE_KEY

def get_db_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

db = get_db_client()

def upsert_market_data_signals(records: list[dict]):
    try:
        response = db.table('market_data_signals').upsert(records).execute()
        logger.info(f'Upserted {len(records)} records into market_data_signals.')
        return response.data
    except Exception as e:
        logger.error(f'Failed to upsert market data signals: {e}')
        raise
    
def insert_trade(trade_record: dict):
    try:
        response = db.table('trade_ledger').insert(trade_record).execute()
        logger.info(f'Inserted new trade for {trade_record.get('ticker')}.')
        return response.data
    except Exception as e:
        logger.error(f'Failed to insert trade record: {e}')
        raise
    
def update_trade_exit(trade_id: str, exit_data: dict):
    try:
        response = db.table('trade_ledger').update(exit_data).eq('trade_id', trade_id).execute()
        logger.info(f'Updated trade {trade_id} with exit data.')
        return response.data
    except Exception as e:
        logger.error(f'Failed to update trade {trade_id}: {e}')
        raise
    
def upsert_portfolio_metrics(metrics_record: dict):
    try:
        response = db.table('portfolio_metrics').upsert(metrics_record).execute()
        logger.info(f'Upserted portfolio metrics for {metrics_record.get('date')}.')
        return response.data
    except Exception as e:
        logger.error(f'Failed to upsert portfolio metrics: {e}')
        raise
    