"""
Fixtures partagées pour les tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Génère des données OHLCV de test"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    # Générer des prix avec une tendance
    base_price = 100
    trend = np.linspace(0, 20, 252)
    noise = np.random.randn(252) * 2
    close_prices = base_price + trend + noise
    
    # Générer OHLC basé sur close
    data = {
        'timestamp': dates,
        'open': close_prices + np.random.randn(252) * 0.5,
        'high': close_prices + abs(np.random.randn(252)) * 1.5,
        'low': close_prices - abs(np.random.randn(252)) * 1.5,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 252)
    }
    
    df = pd.DataFrame(data)
    
    # S'assurer que high >= close >= low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def trending_up_data():
    """Génère des données avec tendance haussière claire"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = np.linspace(100, 150, 100) + np.random.randn(100) * 0.5
    
    data = {
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + abs(np.random.randn(100)) * 0.8,
        'low': close_prices - abs(np.random.randn(100)) * 0.8,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }
    
    df = pd.DataFrame(data)
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def trending_down_data():
    """Génère des données avec tendance baissière claire"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = np.linspace(150, 100, 100) + np.random.randn(100) * 0.5
    
    data = {
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + abs(np.random.randn(100)) * 0.8,
        'low': close_prices - abs(np.random.randn(100)) * 0.8,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }
    
    df = pd.DataFrame(data)
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sideways_data():
    """Génère des données sans tendance (sideways)"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.random.randn(100) * 2
    
    data = {
        'timestamp': dates,
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + abs(np.random.randn(100)) * 0.8,
        'low': close_prices - abs(np.random.randn(100)) * 0.8,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }
    
    df = pd.DataFrame(data)
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    
    return df
