"""
Module de détection de patterns de chandeliers
Détecte automatiquement les patterns haussiers, baissiers et neutres
"""

import pandas as pd
import numpy as np


def is_doji(open_price, close, high, low, threshold=0.1):
    """
    Détecte un pattern Doji
    Corps très petit par rapport à la mèche
    """
    body = abs(close - open_price)
    range_price = high - low
    
    if range_price == 0:
        return False
    
    return (body / range_price) < threshold


def is_hammer(open_price, close, high, low, body_ratio=0.3, wick_ratio=2.0):
    """
    Détecte un Hammer (pattern haussier)
    Petit corps en haut, longue mèche basse
    """
    body = abs(close - open_price)
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    range_price = high - low
    
    if range_price == 0:
        return False
    
    # Corps petit, mèche basse longue
    body_small = (body / range_price) < body_ratio
    lower_wick_long = lower_wick > (body * wick_ratio)
    upper_wick_small = upper_wick < body
    
    return body_small and lower_wick_long and upper_wick_small


def is_inverted_hammer(open_price, close, high, low, body_ratio=0.3, wick_ratio=2.0):
    """
    Détecte un Inverted Hammer (pattern haussier)
    Petit corps en bas, longue mèche haute
    """
    body = abs(close - open_price)
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    range_price = high - low
    
    if range_price == 0:
        return False
    
    # Corps petit, mèche haute longue
    body_small = (body / range_price) < body_ratio
    upper_wick_long = upper_wick > (body * wick_ratio)
    lower_wick_small = lower_wick < body
    
    return body_small and upper_wick_long and lower_wick_small


def is_shooting_star(open_price, close, high, low, body_ratio=0.3, wick_ratio=2.0):
    """
    Détecte un Shooting Star (pattern baissier)
    Identique à Inverted Hammer mais en contexte baissier
    """
    return is_inverted_hammer(open_price, close, high, low, body_ratio, wick_ratio)


def is_bullish_engulfing(df, idx):
    """
    Détecte un Bullish Engulfing (pattern haussier)
    Chandelier haussier englobe complètement le précédent baissier
    """
    if idx < 1:
        return False
    
    # Chandelier actuel
    curr_open = df.iloc[idx]['open']
    curr_close = df.iloc[idx]['close']
    
    # Chandelier précédent
    prev_open = df.iloc[idx - 1]['open']
    prev_close = df.iloc[idx - 1]['close']
    
    # Précédent baissier, actuel haussier
    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    
    # Englobement complet
    engulfs = curr_open < prev_close and curr_close > prev_open
    
    return prev_bearish and curr_bullish and engulfs


def is_bearish_engulfing(df, idx):
    """
    Détecte un Bearish Engulfing (pattern baissier)
    Chandelier baissier englobe complètement le précédent haussier
    """
    if idx < 1:
        return False
    
    # Chandelier actuel
    curr_open = df.iloc[idx]['open']
    curr_close = df.iloc[idx]['close']
    
    # Chandelier précédent
    prev_open = df.iloc[idx - 1]['open']
    prev_close = df.iloc[idx - 1]['close']
    
    # Précédent haussier, actuel baissier
    prev_bullish = prev_close > prev_open
    curr_bearish = curr_close < curr_open
    
    # Englobement complet
    engulfs = curr_open > prev_close and curr_close < prev_open
    
    return prev_bullish and curr_bearish and engulfs


def is_morning_star(df, idx):
    """
    Détecte un Morning Star (pattern haussier à 3 chandeliers)
    1. Chandelier baissier
    2. Petit corps (gap down)
    3. Chandelier haussier (gap up)
    """
    if idx < 2:
        return False
    
    # Chandeliers
    first = df.iloc[idx - 2]
    second = df.iloc[idx - 1]
    third = df.iloc[idx]
    
    # Premier : baissier
    first_bearish = first['close'] < first['open']
    
    # Deuxième : petit corps
    second_body = abs(second['close'] - second['open'])
    second_small = second_body < abs(first['close'] - first['open']) * 0.3
    
    # Troisième : haussier
    third_bullish = third['close'] > third['open']
    
    # Gap down puis gap up
    gap_down = second['open'] < first['close']
    gap_up = third['close'] > (first['open'] + first['close']) / 2
    
    return first_bearish and second_small and third_bullish and gap_down and gap_up


def is_evening_star(df, idx):
    """
    Détecte un Evening Star (pattern baissier à 3 chandeliers)
    1. Chandelier haussier
    2. Petit corps (gap up)
    3. Chandelier baissier (gap down)
    """
    if idx < 2:
        return False
    
    # Chandeliers
    first = df.iloc[idx - 2]
    second = df.iloc[idx - 1]
    third = df.iloc[idx]
    
    # Premier : haussier
    first_bullish = first['close'] > first['open']
    
    # Deuxième : petit corps
    second_body = abs(second['close'] - second['open'])
    second_small = second_body < abs(first['close'] - first['open']) * 0.3
    
    # Troisième : baissier
    third_bearish = third['close'] < third['open']
    
    # Gap up puis gap down
    gap_up = second['open'] > first['close']
    gap_down = third['close'] < (first['open'] + first['close']) / 2
    
    return first_bullish and second_small and third_bearish and gap_up and gap_down


def is_piercing_line(df, idx):
    """
    Détecte un Piercing Line (pattern haussier)
    Chandelier haussier pénètre plus de 50% du précédent baissier
    """
    if idx < 1:
        return False
    
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    
    # Précédent baissier, actuel haussier
    prev_bearish = prev['close'] < prev['open']
    curr_bullish = curr['close'] > curr['open']
    
    # Pénétration > 50%
    prev_midpoint = (prev['open'] + prev['close']) / 2
    penetrates = curr['close'] > prev_midpoint
    
    return prev_bearish and curr_bullish and penetrates


def is_dark_cloud_cover(df, idx):
    """
    Détecte un Dark Cloud Cover (pattern baissier)
    Chandelier baissier pénètre plus de 50% du précédent haussier
    """
    if idx < 1:
        return False
    
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    
    # Précédent haussier, actuel baissier
    prev_bullish = prev['close'] > prev['open']
    curr_bearish = curr['close'] < curr['open']
    
    # Pénétration > 50%
    prev_midpoint = (prev['open'] + prev['close']) / 2
    penetrates = curr['close'] < prev_midpoint
    
    return prev_bullish and curr_bearish and penetrates


def detect_all_patterns(df):
    """
    Détecte tous les patterns de chandeliers dans le DataFrame
    
    Args:
        df: DataFrame avec colonnes OHLC
    
    Returns:
        List de dicts avec les patterns détectés
    """
    patterns = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        timestamp = row.get('timestamp', idx)
        
        # Patterns à 1 chandelier
        if is_doji(row['open'], row['close'], row['high'], row['low']):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Doji',
                'type': 'Neutral',
                'confidence': 0.6
            })
        
        if is_hammer(row['open'], row['close'], row['high'], row['low']):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Hammer',
                'type': 'Bullish',
                'confidence': 0.7
            })
        
        if is_inverted_hammer(row['open'], row['close'], row['high'], row['low']):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Inverted Hammer',
                'type': 'Bullish',
                'confidence': 0.7
            })
        
        if is_shooting_star(row['open'], row['close'], row['high'], row['low']):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Shooting Star',
                'type': 'Bearish',
                'confidence': 0.7
            })
        
        # Patterns à 2 chandeliers
        if is_bullish_engulfing(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Bullish Engulfing',
                'type': 'Bullish',
                'confidence': 0.8
            })
        
        if is_bearish_engulfing(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Bearish Engulfing',
                'type': 'Bearish',
                'confidence': 0.8
            })
        
        if is_piercing_line(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Piercing Line',
                'type': 'Bullish',
                'confidence': 0.75
            })
        
        if is_dark_cloud_cover(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Dark Cloud Cover',
                'type': 'Bearish',
                'confidence': 0.75
            })
        
        # Patterns à 3 chandeliers
        if is_morning_star(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Morning Star',
                'type': 'Bullish',
                'confidence': 0.85
            })
        
        if is_evening_star(df, idx):
            patterns.append({
                'timestamp': timestamp,
                'pattern': 'Evening Star',
                'type': 'Bearish',
                'confidence': 0.85
            })
    
    return patterns


def get_recent_patterns(df, lookback_days=30):
    """
    Retourne uniquement les patterns récents
    
    Args:
        df: DataFrame avec colonnes OHLC et timestamp
        lookback_days: Nombre de jours à analyser (défaut: 30)
    
    Returns:
        List de patterns récents
    """
    all_patterns = detect_all_patterns(df)
    
    if not all_patterns or 'timestamp' not in df.columns:
        return all_patterns
    
    # Filtrer les patterns récents
    latest_date = pd.to_datetime(df['timestamp'].iloc[-1])
    cutoff_date = latest_date - pd.Timedelta(days=lookback_days)
    
    recent_patterns = [
        p for p in all_patterns 
        if pd.to_datetime(p['timestamp']) >= cutoff_date
    ]
    
    return recent_patterns
