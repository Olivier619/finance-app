"""
Module d'identification des niveaux de support et résistance
Utilise les pivots locaux et le clustering
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def find_local_extrema(df, order=5):
    """
    Trouve les maxima et minima locaux
    
    Args:
        df: DataFrame avec colonnes 'high' et 'low'
        order: Nombre de points de chaque côté pour comparer (défaut: 5)
    
    Returns:
        Tuple (resistance_levels, support_levels)
    """
    # Maxima locaux (résistances potentielles)
    high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
    resistance_levels = df.iloc[high_indices][['timestamp', 'high']].copy()
    resistance_levels.rename(columns={'high': 'level'}, inplace=True)
    
    # Minima locaux (supports potentiels)
    low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
    support_levels = df.iloc[low_indices][['timestamp', 'low']].copy()
    support_levels.rename(columns={'low': 'level'}, inplace=True)
    
    return resistance_levels, support_levels


def cluster_levels(levels, tolerance=0.02):
    """
    Regroupe les niveaux proches en clusters
    
    Args:
        levels: DataFrame avec colonne 'level'
        tolerance: Tolérance en % pour regrouper (défaut: 2%)
    
    Returns:
        List de niveaux consolidés avec leur force
    """
    if levels.empty:
        return []
    
    # Trier les niveaux
    sorted_levels = sorted(levels['level'].values)
    
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        # Si le niveau est proche du cluster actuel
        if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) <= tolerance:
            current_cluster.append(level)
        else:
            # Sauvegarder le cluster et en commencer un nouveau
            clusters.append({
                'level': np.mean(current_cluster),
                'strength': len(current_cluster),
                'touches': len(current_cluster)
            })
            current_cluster = [level]
    
    # Ajouter le dernier cluster
    if current_cluster:
        clusters.append({
            'level': np.mean(current_cluster),
            'strength': len(current_cluster),
            'touches': len(current_cluster)
        })
    
    return clusters


def identify_support_resistance(df, order=5, tolerance=0.02, min_strength=2):
    """
    Identifie les niveaux de support et résistance significatifs
    
    Args:
        df: DataFrame avec colonnes OHLC
        order: Ordre pour la détection des extrema (défaut: 5)
        tolerance: Tolérance pour le clustering (défaut: 2%)
        min_strength: Force minimale pour considérer un niveau (défaut: 2)
    
    Returns:
        Dict avec 'support' et 'resistance' lists
    """
    # Prix actuel pour filtrer les niveaux pertinents
    current_price = df.iloc[-1]['close']
    
    # Trouver les extrema locaux
    resistance_levels, support_levels = find_local_extrema(df, order)
    
    # Filtrer les niveaux pour ne garder que ceux proches du prix actuel (±20%)
    price_range_min = current_price * 0.80  # -20%
    price_range_max = current_price * 1.20  # +20%
    
    if not resistance_levels.empty:
        resistance_levels = resistance_levels[
            (resistance_levels['level'] >= price_range_min) & 
            (resistance_levels['level'] <= price_range_max)
        ]
    
    if not support_levels.empty:
        support_levels = support_levels[
            (support_levels['level'] >= price_range_min) & 
            (support_levels['level'] <= price_range_max)
        ]
    
    # Si pas assez de niveaux, élargir à ±30%
    if (resistance_levels.empty or len(resistance_levels) < 2) or (support_levels.empty or len(support_levels) < 2):
        price_range_min = current_price * 0.70
        price_range_max = current_price * 1.30
        
        resistance_levels, support_levels = find_local_extrema(df, order)
        
        if not resistance_levels.empty:
            resistance_levels = resistance_levels[
                (resistance_levels['level'] >= price_range_min) & 
                (resistance_levels['level'] <= price_range_max)
            ]
        
        if not support_levels.empty:
            support_levels = support_levels[
                (support_levels['level'] >= price_range_min) & 
                (support_levels['level'] <= price_range_max)
            ]
    
    # Regrouper en clusters
    resistance_clusters = cluster_levels(resistance_levels, tolerance)
    support_clusters = cluster_levels(support_levels, tolerance)
    
    # Filtrer par force minimale
    significant_resistance = [
        r for r in resistance_clusters 
        if r['strength'] >= min_strength
    ]
    
    significant_support = [
        s for s in support_clusters 
        if s['strength'] >= min_strength
    ]
    
    # Trier par force (décroissant)
    significant_resistance.sort(key=lambda x: x['strength'], reverse=True)
    significant_support.sort(key=lambda x: x['strength'], reverse=True)
    
    return {
        'resistance': significant_resistance[:5],  # Top 5
        'support': significant_support[:5]  # Top 5
    }


def calculate_level_strength(df, level, tolerance=0.01):
    """
    Calcule la force d'un niveau basé sur le nombre de touches
    
    Args:
        df: DataFrame avec colonnes 'high' et 'low'
        level: Niveau de prix à analyser
        tolerance: Tolérance en % (défaut: 1%)
    
    Returns:
        int: Nombre de touches du niveau
    """
    touches = 0
    
    for idx in range(len(df)):
        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']
        
        # Vérifier si le niveau est touché
        level_min = level * (1 - tolerance)
        level_max = level * (1 + tolerance)
        
        if low <= level_max and high >= level_min:
            touches += 1
    
    return touches


def is_level_broken(df, level, level_type='support', lookback=10):
    """
    Vérifie si un niveau a été cassé récemment
    
    Args:
        df: DataFrame avec colonnes OHLC
        level: Niveau de prix
        level_type: 'support' ou 'resistance'
        lookback: Nombre de périodes à analyser (défaut: 10)
    
    Returns:
        bool: True si le niveau est cassé
    """
    recent_data = df.tail(lookback)
    
    if level_type == 'support':
        # Support cassé si close < level
        return any(recent_data['close'] < level)
    else:
        # Résistance cassée si close > level
        return any(recent_data['close'] > level)


def get_nearest_levels(df, current_price, levels_dict, max_distance=0.05):
    """
    Retourne les niveaux les plus proches du prix actuel
    
    Args:
        df: DataFrame avec colonnes OHLC
        current_price: Prix actuel
        levels_dict: Dict avec 'support' et 'resistance'
        max_distance: Distance maximale en % (défaut: 5%)
    
    Returns:
        Dict avec nearest_support et nearest_resistance
    """
    nearest_support = None
    nearest_resistance = None
    
    # Support le plus proche (en dessous du prix)
    for s in levels_dict['support']:
        if s['level'] < current_price:
            distance = abs(current_price - s['level']) / current_price
            if distance <= max_distance:
                nearest_support = s
                break
    
    # Résistance la plus proche (au-dessus du prix)
    for r in levels_dict['resistance']:
        if r['level'] > current_price:
            distance = abs(r['level'] - current_price) / current_price
            if distance <= max_distance:
                nearest_resistance = r
                break
    
    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance
    }


def analyze_price_action(df, levels_dict):
    """
    Analyse l'action du prix par rapport aux niveaux
    
    Args:
        df: DataFrame avec colonnes OHLC
        levels_dict: Dict avec support/resistance
    
    Returns:
        Dict avec analyse
    """
    current_price = df.iloc[-1]['close']
    nearest = get_nearest_levels(df, current_price, levels_dict)
    
    analysis = {
        'current_price': current_price,
        'nearest_support': nearest['nearest_support'],
        'nearest_resistance': nearest['nearest_resistance'],
        'trend': 'NEUTRAL'
    }
    
    # Déterminer la tendance
    if nearest['nearest_support'] and nearest['nearest_resistance']:
        support_level = nearest['nearest_support']['level']
        resistance_level = nearest['nearest_resistance']['level']
        
        # Position dans le range
        range_size = resistance_level - support_level
        position = (current_price - support_level) / range_size
        
        if position > 0.7:
            analysis['trend'] = 'BULLISH'
        elif position < 0.3:
            analysis['trend'] = 'BEARISH'
    
    return analysis
