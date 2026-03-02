import pytest
from unittest.mock import patch, MagicMock
from src.llm_explainer import generate_trade_explanation, generate_news_summary

@pytest.fixture
def sample_snapshot():
    return {
        'daily_return': -0.05,
        'volatility_21d': 0.02,
        'finbert_score': 0.15,
        'macro_sentiment': 0.30,
        'slippage_applied_pct': 0.001
    }

@patch('src.llm_explainer.client')
def test_generate_trade_explanation_success(mock_client, sample_snapshot):
    mock_message = MagicMock()
    mock_message.content = 'This is a mocked AI explanation of the trade.'
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_completion

    result = generate_trade_explanation("AAPL", sample_snapshot)
    
    assert result == 'This is a mocked AI explanation of the trade.'
    mock_client.chat.completions.create.assert_called_once()

@patch('src.llm_explainer.client')
def test_generate_trade_explanation_api_failure(mock_client, sample_snapshot):
    mock_client.chat.completions.create.side_effect = Exception('Groq API is down')

    result = generate_trade_explanation("TSLA", sample_snapshot)
    
    assert result == 'AI explanation failed due to an API error.'
    
@patch('src.llm_explainer.client')
def test_generate_news_summary_success(mock_client):
    mock_message = MagicMock()
    mock_message.content = 'This news is bearish because of a massive lawsuit.'
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_completion

    result = generate_news_summary('AAPL', 'Apple sued', 'Apple faces a $2B fine in the EU.')
    
    assert result == 'This news is bearish because of a massive lawsuit.'
    mock_client.chat.completions.create.assert_called_once()
    