"""
Module d'indicateurs techniques avancés
Stochastique, VWAP, Ichimoku Cloud, Fibonacci Retracements
"""

import pandas as pd
import numpy as np


def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """
    Calcule l'oscillateur stochastique
    
    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        k_period: Période pour %K (défaut: 14)
        d_period: Période pour %D (défaut: 3)
        smooth_k: Période de lissage pour %K (défaut: 3)
    
    Returns:
        DataFrame avec colonnes 'stoch_k' et 'stoch_d'
    """
    data = df.copy()
    
    # Calcul du %K brut
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    data['stoch_k_raw'] = 100 * (data['close'] - low_min) / (high_max - low_min)
    
    # Lissage du %K
    data['stoch_k'] = data['stoch_k_raw'].rolling(window=smooth_k).mean()
    
    # Calcul du %D (moyenne mobile du %K)
    data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
    
    return data[['stoch_k', 'stoch_d']]


def calculate_vwap(df):
    """
    Calcule le VWAP (Volume Weighted Average Price)
    
    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close', 'volume', 'timestamp'
    
    Returns:
        Series avec le VWAP
    """
    data = df.copy()
    
    # Prix typique
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # VWAP cumulatif par jour
    if 'timestamp' in data.columns:
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        data['cumul_tp_vol'] = (data['typical_price'] * data['volume']).groupby(data['date']).cumsum()
        data['cumul_vol'] = data['volume'].groupby(data['date']).cumsum()
    else:
        data['cumul_tp_vol'] = (data['typical_price'] * data['volume']).cumsum()
        data['cumul_vol'] = data['volume'].cumsum()
    
    data['vwap'] = data['cumul_tp_vol'] / data['cumul_vol']
    
    return data['vwap']


def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    Calcule l'Ichimoku Cloud
    
    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        tenkan_period: Période Tenkan-sen (défaut: 9)
        kijun_period: Période Kijun-sen (défaut: 26)
        senkou_b_period: Période Senkou Span B (défaut: 52)
        displacement: Décalage pour Senkou Spans (défaut: 26)
    
    Returns:
        DataFrame avec colonnes ichimoku
    """
    data = df.copy()
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = data['high'].rolling(window=tenkan_period).max()
    tenkan_low = data['low'].rolling(window=tenkan_period).min()
    data['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = data['high'].rolling(window=kijun_period).max()
    kijun_low = data['low'].rolling(window=kijun_period).min()
    data['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    senkou_b_high = data['high'].rolling(window=senkou_b_period).max()
    senkou_b_low = data['low'].rolling(window=senkou_b_period).min()
    data['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    data['chikou_span'] = data['close'].shift(-displacement)
    
    return data[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]


def calculate_fibonacci_levels(df, lookback_period=100):
    """
    Calcule les niveaux de retracement de Fibonacci
    
    Args:
        df: DataFrame avec colonnes 'high', 'low'
        lookback_period: Période pour identifier le swing high/low (défaut: 100)
    
    Returns:
        Dict avec les niveaux de Fibonacci
    """
    data = df.tail(lookback_period)
    
    # Identifier le plus haut et le plus bas
    swing_high = data['high'].max()
    swing_low = data['low'].min()
    
    # Calculer la différence
    diff = swing_high - swing_low
    
    # Niveaux de Fibonacci (retracement)
    levels = {
        '0.0%': swing_high,
        '23.6%': swing_high - 0.236 * diff,
        '38.2%': swing_high - 0.382 * diff,
        '50.0%': swing_high - 0.500 * diff,
        '61.8%': swing_high - 0.618 * diff,
        '78.6%': swing_high - 0.786 * diff,
        '100.0%': swing_low,
        # Extensions
        '161.8%': swing_high + 0.618 * diff,
        '261.8%': swing_high + 1.618 * diff,
    }
    
    return levels


def calculate_all_advanced_indicators(df):
    """
    Calcule tous les indicateurs avancés en une seule fois
    
    Args:
        df: DataFrame avec colonnes OHLCV
    
    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    data = df.copy()
    
    # Stochastique
    stoch = calculate_stochastic(data)
    data = pd.concat([data, stoch], axis=1)
    
    # VWAP
    data['vwap'] = calculate_vwap(data)
    
    # Ichimoku
    ichimoku = calculate_ichimoku(data)
    data = pd.concat([data, ichimoku], axis=1)
    
    # Fibonacci (retourne un dict, pas ajouté au DataFrame)
    fib_levels = calculate_fibonacci_levels(data)
    
    return data, fib_levels


def get_stochastic_signal(stoch_k, stoch_d, oversold=20, overbought=80):
    """
    Génère un signal de trading basé sur le stochastique
    
    Args:
        stoch_k: Valeur actuelle de %K
        stoch_d: Valeur actuelle de %D
        oversold: Seuil de survente (défaut: 20)
        overbought: Seuil de surachat (défaut: 80)
    
    Returns:
        str: 'BUY', 'SELL', ou 'NEUTRAL'
    """
    if pd.isna(stoch_k) or pd.isna(stoch_d):
        return 'NEUTRAL'
    
    # Signal d'achat : %K croise %D vers le haut en zone de survente
    if stoch_k > stoch_d and stoch_k < oversold:
        return 'BUY'
    
    # Signal de vente : %K croise %D vers le bas en zone de surachat
    if stoch_k < stoch_d and stoch_k > overbought:
        return 'SELL'
    
    return 'NEUTRAL'


def get_ichimoku_signal(close, tenkan, kijun, senkou_a, senkou_b):
    """
    Génère un signal de trading basé sur Ichimoku
    
    Args:
        close: Prix de clôture actuel
        tenkan: Tenkan-sen actuel
        kijun: Kijun-sen actuel
        senkou_a: Senkou Span A actuel
        senkou_b: Senkou Span B actuel
    
    Returns:
        str: 'BULLISH', 'BEARISH', ou 'NEUTRAL'
    """
    if any(pd.isna(x) for x in [close, tenkan, kijun, senkou_a, senkou_b]):
        return 'NEUTRAL'
    
    # Prix au-dessus du nuage
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    
    if close > cloud_top and tenkan > kijun:
        return 'BULLISH'
    elif close < cloud_bottom and tenkan < kijun:
        return 'BEARISH'
    else:
        return 'NEUTRAL'
