"""
Tests de performance pour les calculs intensifs
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_strategies import MACrossover, RSIStrategy
from parallel_computing import ParallelComputing

class TestPerformance:
    """Benchmarks de performance"""
    
    def test_strategy_execution_speed(self, sample_ohlcv_data):
        """Vérifie que l'exécution d'une stratégie est rapide (< 0.1s pour 252 jours)"""
        strategy = MACrossover()
        
        start_time = time.time()
        # Exécuter 10 fois pour avoir une moyenne fiable
        for _ in range(10):
            strategy.backtest(sample_ohlcv_data.copy())
        duration = (time.time() - start_time) / 10
        
        # Le seuil dépend de la machine, mais 0.1s est large pour 252 lignes
        assert duration < 0.1, f"L'exécution est trop lente: {duration:.4f}s"
        
    def test_parallel_vs_serial_execution(self):
        """Compare l'exécution parallèle vs série (devrait être plus rapide sur de gros volumes)"""
        # Créer un gros jeu de données factice ou simuler beaucoup de symboles
        # Pour ce test, on va simuler le traitement de plusieurs stratégies
        
        # Simulation de données
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(1000).cumsum() + 100,
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Créer 20 stratégies avec des paramètres différents
        strategies = {}
        for i in range(5, 25):
            strategies[f'MA_{i}'] = MACrossover(short_period=i, long_period=i+10)
            
        pc = ParallelComputing()
        
        # Test Série
        start_serial = time.time()
        results_serial = []
        for name, strat in strategies.items():
            results_serial.append(strat.backtest(df.copy()))
        duration_serial = time.time() - start_serial
        
        # Test Parallèle
        start_parallel = time.time()
        results_parallel = pc.backtest_strategies_parallel(strategies, df)
        duration_parallel = time.time() - start_parallel
        
        # Note: Sur de très petits calculs, le surcoût du parallélisme peut rendre le test parallèle plus lent.
        # Ce test est informatif. On vérifie juste que ça s'exécute sans erreur.
        assert len(results_parallel) == len(strategies)
        print(f"Serial: {duration_serial:.4f}s, Parallel: {duration_parallel:.4f}s")

    def test_memory_usage_large_dataset(self):
        """Vérifie que le traitement d'un gros dataset ne fait pas exploser la mémoire"""
        # Générer 10 ans de données quotidiennes (~2500 lignes)
        idx = pd.date_range(start='2010-01-01', periods=2500, freq='D')
        df_large = pd.DataFrame(np.random.randn(2500, 5), columns=['open', 'high', 'low', 'close', 'volume'], index=idx)
        df_large['timestamp'] = idx
        
        strategy = RSIStrategy()
        
        # Juste vérifier que ça passe sans MemoryError
        try:
            strategy.backtest(df_large)
        except MemoryError:
            pytest.fail("MemoryError encountered with large dataset")
