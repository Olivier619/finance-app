"""
Module pour récupérer les données de marché depuis yfinance, Alpha Vantage et Finnhub
Compatible avec Streamlit et Flask
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Récupération des clés API depuis les variables d'environnement
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


def fetch_from_yfinance(symbol, period="1y", interval="1d"):
    """
    Récupère les données depuis Yahoo Finance (yfinance)
    GRATUIT - Aucune clé API nécessaire
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"Aucune donnée trouvée pour {symbol}")
        
        df = df.reset_index().rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    except Exception as e:
        raise RuntimeError(f"Erreur yfinance pour {symbol}: {str(e)}")


def fetch_from_alphavantage(symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    """
    Récupère les données depuis Alpha Vantage
    GRATUIT - 25 appels/jour - Inscription: https://www.alphavantage.co/support/#api-key
    """
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY non définie. Ajoutez-la dans .env")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage erreur: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(f"Limite API atteinte: {data['Note']}")
        
        ts_key = [k for k in data.keys() if "Time Series" in k]
        if not ts_key:
            raise RuntimeError(f"Réponse inattendue: {data}")
        
        ts = data[ts_key[0]]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        rename_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
        df = df.rename(columns=rename_map)
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["timestamp"] = df.index
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    except Exception as e:
        raise RuntimeError(f"Erreur Alpha Vantage pour {symbol}: {str(e)}")


def fetch_from_finnhub(symbol, resolution="D", start=None, end=None):
    """
    Récupère les données depuis Finnhub
    GRATUIT - 60 appels/minute - Inscription: https://finnhub.io/register
    """
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY non définie. Ajoutez-la dans .env")

    url = "https://finnhub.io/api/v1/stock/candle"
    now = int(time.time())
    
    if end is None:
        end = now
    if start is None:
        start = now - 365 * 24 * 60 * 60

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": int(start),
        "to": int(end),
        "token": FINNHUB_API_KEY,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("s") != "ok":
            raise RuntimeError(f"Finnhub erreur: {data}")
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"],
        })
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    except Exception as e:
        raise RuntimeError(f"Erreur Finnhub pour {symbol}: {str(e)}")


def fetch_market_data(source, symbol, **kwargs):
    """
    Fonction principale pour récupérer les données depuis n'importe quelle source
    """
    source = source.lower()
    
    if source == "yfinance":
        return fetch_from_yfinance(symbol, **kwargs)
    elif source == "alpha_vantage":
        return fetch_from_alphavantage(symbol, **kwargs)
    elif source == "finnhub":
        return fetch_from_finnhub(symbol, **kwargs)
    else:
        raise ValueError(f"Source inconnue: {source}")


# Mapping des secteurs vers des ETFs
SECTEURS_ETF = {
    "immobilier": "VNQ",
    "automobiles": "CARZ",
    "banques": "KBE",
    "assurances": "KIE",
    "technologie": "XLK",
    "energie": "XLE",
    "sante": "XLV",
}

def fetch_secteur_data(secteur, source="yfinance", **kwargs):
    """Récupère les données d'un secteur via un ETF"""
    secteur = secteur.lower()
    if secteur not in SECTEURS_ETF:
        raise ValueError(f"Secteur inconnu: {secteur}")
    return fetch_market_data(source, SECTEURS_ETF[secteur], **kwargs)
