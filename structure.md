# Project Structure

```text
sentiment-alpha-engine/
├── .github/
│   └── workflows/
│       └── daily_run.yml     # The cron job for the daily ETL and trading pipeline
├── data/                     # Local cache for parquet/csv files (ignored in .gitignore)
├── notebooks/                # Jupyter notebooks for EDA, hyperparameter tuning, and backtesting
├── src/                      # Core engine source code
│   ├── __init__.py
│   ├── config.py             # Centralized configuration and environment variable loading
│   ├── db.py                 # Supabase connection and CRUD operations
│   ├── universe.py           # Logic for S&P 500 fetching, Beta, and Volatility calculations
│   ├── news_fetcher.py       # Finnhub API integration
│   ├── sentiment.py          # FinBERT HF inference logic
│   ├── signals.py            # Mathematical implementation of the Overreaction Arbitrage rule
│   ├── execution.py          # Position sizing, slippage modeling, and trade ledger logic
│   └── llm_explainer.py      # Groq integration for trade explanations
├── tests/                    # Unit and integration tests (pytest)
├── .gitignore
├── app.py                    # Streamlit dashboard entry point
├── etl_pipeline.py           # The main executable script run by GitHub Actions
├── LICENSE
├── README.md 
├── structure.md            
└── requirement.txt           # Dependencies
