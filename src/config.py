import os
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

UNIVERSE_SIZE = 100
TRAILING_DAYS_VOL = 21
TRAILING_DAYS_BETA = 252
