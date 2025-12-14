"""
Fichier de configuration centralisé
Paramètres par défaut, chemins, constantes
"""

import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DB_PATH = BASE_DIR / 'finance_app.db'

# Créer les répertoires s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Paramètres de l'application
APP_TITLE = "Market Predictor Pro"
APP_ICON = "📈"
INITIAL_CASH = 100000.0

# Paramètres des indicateurs techniques
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Paramètres de trading
DEFAULT_FEES_PCT = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%

# Paramètres ML
ML_TRAIN_WINDOW = 252  # 1 an
ML_TEST_WINDOW = 20  # 20 jours
ML_FEATURES = ['RSI', 'MACD', 'BB_Pct', 'ROC', 'ATR']
ML_RANDOM_STATE = 42

# Paramètres de backtesting
BACKTEST_TRAIN_WINDOW = 252
BACKTEST_TEST_WINDOW = 20

# Paramètres de news
NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_ARTICLES = 20

# Paramètres de sentiment
SENTIMENT_WINDOW_DAYS = 7

# Paramètres d'alertes
ALERT_CHECK_INTERVAL = 300  # 5 minutes en secondes

# API Keys (chargées depuis .env)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

# SMTP Configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
ALERT_EMAIL = os.getenv('ALERT_EMAIL', SMTP_USER)

# Secteurs et ETFs
SECTEURS_ETF = {
    # Actions / Secteurs
    "Technologie": "XLK",
    "Santé": "XLV",
    "Finance": "XLF",
    "Immobilier": "VNQ",
    "Automobile": "CARZ",
    "Énergie": "XLE",
    # Matières Premières
    "Or (Gold)": "GC=F",
    "Argent (Silver)": "SI=F",
    "Pétrole (WTI)": "CL=F",
    "Gaz Naturel": "NG=F",
    "Blé (Wheat)": "ZW=F",
    "Maïs (Corn)": "ZC=F",
    # Crypto
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    # Forex
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CHF": "CHF=X"
}

# Catégories
CATEGORIES = {
    "Actions": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
    "Secteurs/Indices": ["Technologie", "Finance", "Immobilier", "Énergie", "Automobile"],
    "Matières Premières": ["Or (Gold)", "Argent (Silver)", "Pétrole (WTI)", "Gaz Naturel", "Blé (Wheat)", "Maïs (Corn)"],
    "Forex (Devises)": ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF"],
    "Cryptos": ["Bitcoin", "Ethereum", "Solana"]
}
