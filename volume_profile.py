"""
Module d'analyse du Volume Profile
Calcule la distribution du volume par niveau de prix
"""

import pandas as pd
import numpy as np


def calculate_volume_profile(df, num_bins=50):
    """
    Calcule le profil de volume
    
    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close', 'volume'
        num_bins: Nombre de niveaux de prix (défaut: 50)
    
    Returns:
        DataFrame avec les niveaux de prix et le volume associé
    """
    # Déterminer la plage de prix
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    # Créer les bins de prix
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    # Initialiser le volume par bin
    volume_by_bin = np.zeros(num_bins)
    
    # Distribuer le volume
    for idx in range(len(df)):
        row = df.iloc[idx]
        candle_low = row['low']
        candle_high = row['high']
        candle_volume = row['volume']
        
        # Trouver les bins touchés par cette bougie
        bins_touched = []
        for i, (bin_low, bin_high) in enumerate(zip(price_bins[:-1], price_bins[1:])):
            # Vérifier si le bin est dans la plage de la bougie
            if not (candle_high < bin_low or candle_low > bin_high):
                bins_touched.append(i)
        
        # Distribuer le volume uniformément sur les bins touchés
        if bins_touched:
            volume_per_bin = candle_volume / len(bins_touched)
            for bin_idx in bins_touched:
                volume_by_bin[bin_idx] += volume_per_bin
    
    # Créer le DataFrame de résultat
    profile_df = pd.DataFrame({
        'price_level': bin_centers,
        'volume': volume_by_bin
    })
    
    return profile_df


def find_point_of_control(profile_df):
    """
    Trouve le Point of Control (POC) - niveau avec le plus de volume
    
    Args:
        profile_df: DataFrame du profil de volume
    
    Returns:
        float: Prix du POC
    """
    poc_idx = profile_df['volume'].idxmax()
    poc_price = profile_df.loc[poc_idx, 'price_level']
    
    return poc_price


def calculate_value_area(profile_df, value_area_pct=0.70):
    """
    Calcule la Value Area (zone contenant X% du volume)
    
    Args:
        profile_df: DataFrame du profil de volume
        value_area_pct: Pourcentage du volume (défaut: 70%)
    
    Returns:
        Dict avec 'value_area_high', 'value_area_low', 'poc'
    """
    # Trouver le POC
    poc_idx = profile_df['volume'].idxmax()
    poc_price = profile_df.loc[poc_idx, 'price_level']
    
    # Volume total
    total_volume = profile_df['volume'].sum()
    target_volume = total_volume * value_area_pct
    
    # Initialiser avec le POC
    included_indices = {poc_idx}
    current_volume = profile_df.loc[poc_idx, 'volume']
    
    # Étendre la zone de valeur
    while current_volume < target_volume:
        # Trouver les indices adjacents
        min_idx = min(included_indices)
        max_idx = max(included_indices)
        
        # Volume des bins adjacents
        volume_below = profile_df.loc[min_idx - 1, 'volume'] if min_idx > 0 else 0
        volume_above = profile_df.loc[max_idx + 1, 'volume'] if max_idx < len(profile_df) - 1 else 0
        
        # Ajouter le bin avec le plus de volume
        if volume_below > volume_above and min_idx > 0:
            included_indices.add(min_idx - 1)
            current_volume += volume_below
        elif max_idx < len(profile_df) - 1:
            included_indices.add(max_idx + 1)
            current_volume += volume_above
        else:
            break
    
    # Calculer les limites
    value_area_indices = sorted(list(included_indices))
    value_area_low = profile_df.loc[value_area_indices[0], 'price_level']
    value_area_high = profile_df.loc[value_area_indices[-1], 'price_level']
    
    return {
        'poc': poc_price,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'value_area_volume_pct': (current_volume / total_volume) * 100
    }


def analyze_volume_profile(df, num_bins=50, lookback_days=None):
    """
    Analyse complète du profil de volume
    
    Args:
        df: DataFrame avec colonnes OHLCV
        num_bins: Nombre de niveaux de prix (défaut: 50)
        lookback_days: Nombre de jours à analyser (None = tout)
    
    Returns:
        Dict avec profil, POC, Value Area, et analyse
    """
    # Filtrer par période si nécessaire
    data = df.copy()
    if lookback_days and 'timestamp' in data.columns:
        cutoff_date = pd.to_datetime(data['timestamp'].iloc[-1]) - pd.Timedelta(days=lookback_days)
        data = data[pd.to_datetime(data['timestamp']) >= cutoff_date]
    
    # Calculer le profil
    profile_df = calculate_volume_profile(data, num_bins)
    
    # Calculer POC et Value Area
    value_area = calculate_value_area(profile_df)
    
    # Prix actuel
    current_price = data.iloc[-1]['close']
    
    # Analyse de position
    position_analysis = 'NEUTRAL'
    if current_price > value_area['value_area_high']:
        position_analysis = 'ABOVE_VALUE'
    elif current_price < value_area['value_area_low']:
        position_analysis = 'BELOW_VALUE'
    elif abs(current_price - value_area['poc']) / value_area['poc'] < 0.005:
        position_analysis = 'AT_POC'
    else:
        position_analysis = 'WITHIN_VALUE'
    
    return {
        'profile': profile_df,
        'poc': value_area['poc'],
        'value_area_high': value_area['value_area_high'],
        'value_area_low': value_area['value_area_low'],
        'current_price': current_price,
        'position': position_analysis,
        'value_area_volume_pct': value_area['value_area_volume_pct']
    }


def get_high_volume_nodes(profile_df, threshold_percentile=80):
    """
    Identifie les HVN (High Volume Nodes)
    
    Args:
        profile_df: DataFrame du profil de volume
        threshold_percentile: Percentile pour considérer un HVN (défaut: 80)
    
    Returns:
        List de prix des HVN
    """
    threshold = np.percentile(profile_df['volume'], threshold_percentile)
    hvn = profile_df[profile_df['volume'] >= threshold]['price_level'].tolist()
    
    return hvn


def get_low_volume_nodes(profile_df, threshold_percentile=20):
    """
    Identifie les LVN (Low Volume Nodes)
    
    Args:
        profile_df: DataFrame du profil de volume
        threshold_percentile: Percentile pour considérer un LVN (défaut: 20)
    
    Returns:
        List de prix des LVN
    """
    threshold = np.percentile(profile_df['volume'], threshold_percentile)
    lvn = profile_df[profile_df['volume'] <= threshold]['price_level'].tolist()
    
    return lvn


def compare_price_to_profile(current_price, profile_analysis):
    """
    Compare le prix actuel au profil de volume
    
    Args:
        current_price: Prix actuel
        profile_analysis: Dict retourné par analyze_volume_profile
    
    Returns:
        Dict avec interprétation
    """
    poc = profile_analysis['poc']
    va_high = profile_analysis['value_area_high']
    va_low = profile_analysis['value_area_low']
    
    interpretation = {
        'price': current_price,
        'distance_from_poc_pct': ((current_price - poc) / poc) * 100,
        'position': profile_analysis['position'],
        'signal': 'NEUTRAL'
    }
    
    # Signaux de trading
    if current_price < va_low:
        interpretation['signal'] = 'POTENTIAL_BUY'
        interpretation['reason'] = 'Prix en dessous de la Value Area (zone de valeur)'
    elif current_price > va_high:
        interpretation['signal'] = 'POTENTIAL_SELL'
        interpretation['reason'] = 'Prix au-dessus de la Value Area (zone de valeur)'
    elif abs(current_price - poc) / poc < 0.01:
        interpretation['signal'] = 'AT_EQUILIBRIUM'
        interpretation['reason'] = 'Prix proche du POC (Point of Control)'
    
    return interpretation
