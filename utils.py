"""
Fonctions utilitaires
Formatage, validation, cache
"""

import pandas as pd
import streamlit as st
from datetime import datetime


def format_number(num, suffix=''):
    """
    Formate un nombre avec K, M, B
    
    Args:
        num: Nombre à formater
        suffix: Suffixe optionnel (ex: '$')
    
    Returns:
        str: Nombre formaté
    """
    if num is None:
        return 'N/A'
    
    if abs(num) >= 1_000_000_000:
        return f'{suffix}{num/1_000_000_000:.2f}B'
    elif abs(num) >= 1_000_000:
        return f'{suffix}{num/1_000_000:.2f}M'
    elif abs(num) >= 1_000:
        return f'{suffix}{num/1_000:.2f}K'
    else:
        return f'{suffix}{num:.2f}'


def format_percentage(value, decimals=2):
    """
    Formate un pourcentage
    
    Args:
        value: Valeur (0.15 = 15%)
        decimals: Nombre de décimales
    
    Returns:
        str: Pourcentage formaté
    """
    if value is None:
        return 'N/A'
    
    return f'{value * 100:.{decimals}f}%'


def format_currency(value, currency='$', decimals=None):
    """
    Formate une valeur monétaire avec précision intelligente
    
    Args:
        value: Valeur
        currency: Symbole de devise
        decimals: Nombre de décimales forcées (None = auto)
    
    Returns:
        str: Valeur formatée
    """
    if value is None:
        return 'N/A'
    
    if decimals is None:
        # Détection intelligente pour le Forex et petits montants
        abs_val = abs(value)
        if abs_val < 5:  # Forex (EURUSD) ou penny stocks
            decimals = 4
        else:
            decimals = 2
    
    return f'{currency}{value:,.{decimals}f}'


def validate_symbol(symbol):
    """
    Valide un symbole boursier
    
    Args:
        symbol: Symbole à valider
    
    Returns:
        bool: True si valide
    """
    if not symbol:
        return False
    
    # Symbole doit contenir uniquement des lettres, chiffres, et certains caractères spéciaux
    import re
    pattern = r'^[A-Z0-9\-=\.]+$'
    
    return bool(re.match(pattern, symbol.upper()))


def convert_date(date_str):
    """
    Convertit une chaîne de date en datetime
    
    Args:
        date_str: Chaîne de date
    
    Returns:
        datetime: Date convertie
    """
    try:
        return pd.to_datetime(date_str)
    except:
        return None


@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def cached_fetch_data(symbol, period='1y'):
    """
    Récupère des données avec cache
    
    Args:
        symbol: Symbole de l'actif
        period: Période
    
    Returns:
        DataFrame: Données
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df.reset_index(inplace=True)
        
        # Renommer les colonnes
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Supprimer timezone
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=86400) # Cache 24h
def get_asset_name(symbol):
    """
    Récupère le nom complet de l'actif
    """
    import yfinance as yf
    if not symbol or not isinstance(symbol, str):
        return ""
    try:
        ticker = yf.Ticker(symbol)
        # Utiliser fast_info si disponible (plus rapide que info)
        if hasattr(ticker, 'fast_info') and 'longName' in ticker.fast_info:
            return ticker.fast_info['longName']
        
        # Fallback sur info (plus lent)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or symbol
        return name
    except:
        return symbol


def get_color_for_change(change):
    """
    Retourne une couleur basée sur le changement
    
    Args:
        change: Valeur du changement
    
    Returns:
        str: Code couleur
    """
    if change > 0:
        return 'green'
    elif change < 0:
        return 'red'
    else:
        return 'gray'


def truncate_text(text, max_length=100):
    """
    Tronque un texte
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
    
    Returns:
        str: Texte tronqué
    """
    if not text:
        return ''
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + '...'


def calculate_days_ago(date_str):
    """
    Calcule le nombre de jours depuis une date
    
    Args:
        date_str: Chaîne de date
    
    Returns:
        int: Nombre de jours
    """
    try:
        date = pd.to_datetime(date_str)
        now = datetime.now()
        delta = now - date
        return delta.days
    except:
        return None


def safe_divide(numerator, denominator, default=0):
    """
    Division sécurisée
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si division par zéro
    
    Returns:
        float: Résultat
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default
