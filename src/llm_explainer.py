import json
from loguru import logger
from groq import Groq
from src.config import GROQ_API_KEY

GROQ_MODEL = "llama3-8b-8192"

def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)

client = get_groq_client()

def generate_trade_explanation(ticker: str, signal_snapshot: dict) -> str:
    if not client:
        return 'AI explanation unavailable: Missing Groq API Key.'
        
    system_prompt = (
        'You are a quantitative trading reporting assistant. Your job is to translate '
        'algorithmic trade data into a concise, 2-sentence human-readable explanation. '
        'Do NOT invent any facts or provide financial advice. Base your explanation strictly '
        'on the provided signal snapshot.'
    )
    
    user_prompt = (
        f'We just executed a mean-reversion BUY order for {ticker}.\n'
        f'Here is the mathematical snapshot at the time of execution:\n'
        f'{json.dumps(signal_snapshot, indent=2)}\n\n'
        'Explain this trade. Mention the price shock relative to volatility, '
        'the FinBERT score, and the macro sentiment.'
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=150,
        )
        explanation = chat_completion.choices[0].message.content.strip()
        logger.info(f"Generated AI explanation for {ticker}.")
        return explanation
        
    except Exception as e:
        logger.error(f'Failed to generate LLM explanation for {ticker}: {e}')
        return 'AI explanation failed due to an API error.'
    
def generate_news_summary(ticker: str, headline: str, summary: str) -> str:   
    system_prompt = (
        'You are a strict financial news summarizer. Your job is to read a headline and summary, '
        'and explain in 1 to 2 sentences why this news is bullish, bearish, or neutral for the company. '
        'You must NOT hallucinate outside information, earnings numbers, or dates that are not provided in the text.'
    )
    
    user_prompt = (
        f'Ticker: {ticker}\n'
        f'Headline: {headline}\n'
        f'Summary: {summary}\n\n'
        'Provide a brief sentiment analysis of this news.'
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=150,
        )
        explanation = chat_completion.choices[0].message.content.strip()
        logger.info(f'Generated AI news summary for {ticker}.')
        return explanation
        
    except Exception as e:
        logger.error(f'Failed to generate LLM news summary for {ticker}: {e}')
        return 'News summary failed due to an API error.'
    