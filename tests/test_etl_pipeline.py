import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from etl_pipeline import run_pipeline, manage_trade_exits

@patch('etl_pipeline.get_high_beta_universe')
@patch('etl_pipeline.get_daily_news')
@patch('etl_pipeline.process_macro_sentiment')
@patch('etl_pipeline.process_company_sentiment')
@patch('etl_pipeline.generate_signals')
@patch('etl_pipeline.upsert_market_data_signals')
@patch('etl_pipeline.generate_trade_ledger_entries')
@patch('etl_pipeline.generate_trade_explanation')
@patch('etl_pipeline.insert_trade')
@patch('etl_pipeline.manage_trade_exits')
def test_etl_pipeline_orchestration(
    mock_manage_exits, mock_insert_trade, mock_generate_explanation, 
    mock_generate_ledger, mock_upsert_signals, mock_generate_signals, 
    mock_process_company_sentiment, mock_process_macro_sentiment, 
    mock_get_daily_news, mock_get_universe
):
    
    mock_get_universe.return_value = pd.DataFrame({'ticker': ['AAPL']})
    
    mock_get_daily_news.return_value = (pd.DataFrame(), pd.DataFrame())
    
    mock_process_macro_sentiment.return_value = 0.5
    mock_process_company_sentiment.return_value = pd.DataFrame({'ticker': ['AAPL'], 'finbert_score': [0.1]})
    
    mock_generate_signals.return_value = pd.DataFrame({
        'ticker': ['AAPL'], 
        'signal_triggered': [True],
        'date': ['2026-03-03']
    })
    
    mock_trade = {
        'ticker': 'AAPL', 
        'signal_snapshot': {'volatility_21d': 0.02}
    }
    mock_generate_ledger.return_value = [mock_trade]
    
    mock_generate_explanation.return_value = "Mocked explanation."
    
    run_pipeline()
    mock_get_universe.assert_called_once()
    mock_get_daily_news.assert_called_once_with(['AAPL'])
    mock_upsert_signals.assert_called_once()
    mock_generate_ledger.assert_called_once()
    mock_generate_explanation.assert_called_once_with('AAPL', {'volatility_21d': 0.02})
    mock_insert_trade.assert_called_once()
    mock_manage_exits.assert_called_once()
    
@patch('etl_pipeline.get_high_beta_universe')
def test_etl_pipeline_halts_on_error(mock_get_universe):
    mock_get_universe.side_effect = Exception('Database connection failed')
    
    with pytest.raises(Exception, match='Database connection failed'):
        run_pipeline()


@patch('etl_pipeline.db')
@patch('etl_pipeline.yf.download')
@patch('etl_pipeline.update_trade_exit')
def test_manage_trade_exits_success(mock_update_exit, mock_yf_download, mock_db):
    # Mock DB response to return one open trade
    mock_db_response = MagicMock()
    mock_db_response.data = [{'trade_id': 'txn_123', 'ticker': 'AAPL', 'entry_price': 150.0}]
    mock_db.table.return_value.select.return_value.eq.return_value.lte.return_value.execute.return_value = mock_db_response
    
    mock_yf_download.return_value = pd.DataFrame({'Close': [165.0]})
    
    manage_trade_exits()
    
    mock_yf_download.assert_called_once_with(['AAPL'], period="1d", interval="1d", auto_adjust=True, progress=False)
    
    expected_exit_data = {
        'status': 'CLOSED',
        'exit_price': 165.0,
        'pnl_percent': 0.10 # (165 - 150) / 150
    }
    mock_update_exit.assert_called_once_with('txn_123', expected_exit_data)

@patch('etl_pipeline.db')
@patch('etl_pipeline.yf.download')
@patch('etl_pipeline.update_trade_exit')
def test_manage_trade_exits_no_open_trades(mock_update_exit, mock_yf_download, mock_db):
    mock_db_response = MagicMock()
    mock_db_response.data = []
    mock_db.table.return_value.select.return_value.eq.return_value.lte.return_value.execute.return_value = mock_db_response
    
    manage_trade_exits()
    
    mock_yf_download.assert_not_called()
    mock_update_exit.assert_not_called()

@patch('etl_pipeline.db')
def test_manage_trade_exits_handles_exception(mock_db):
    mock_db.table.side_effect = Exception('Supabase timeout')
    
    try:
        manage_trade_exits()
    except Exception:
        pytest.fail('manage_trade_exits raised an exception instead of catching it.')
    