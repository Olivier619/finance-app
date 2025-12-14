"""
Tests d'intégration pour le module de backtesting avancé
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy_backtester import StrategyBacktester
from trading_strategies import MACrossover

class TestStrategyBacktester:
    """Tests pour le backtester avancé"""
    
    def test_calculate_advanced_metrics(self, sample_ohlcv_data):
        """Test le calcul des métriques avancées"""
        # Créer une stratégie et obtenir des résultats basiques
        strategy = MACrossover(short_period=10, long_period=20)
        basic_result = strategy.backtest(sample_ohlcv_data.copy())
        
        # Initialiser le backtester avancé
        backtester = StrategyBacktester(initial_capital=100000)
        
        # Calculer les métriques avancées
        metrics = backtester.calculate_advanced_metrics(
            basic_result['equity_curve'], 
            basic_result['trades']
        )
        
        # Vérifier la présence des métriques
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'avg_trade_duration' in metrics
        
        # Vérifier la cohérence des valeurs
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert metrics['max_drawdown_pct'] <= 0  # Drawdown devrait être négatif ou 0 (selon implémentation, ici souvent positif en valeur absolue mais vérifions)
        # Note: Dans l'implémentation actuelle, drawdown est négatif (equity - max) / max
        
    def test_create_equity_chart(self, sample_ohlcv_data):
        """Test la création du graphique d'équité"""
        strategy = MACrossover(short_period=10, long_period=20)
        result = strategy.backtest(sample_ohlcv_data.copy())
        
        backtester = StrategyBacktester()
        fig = backtester.create_equity_chart([result])
        
        assert fig is not None
        # Vérifier que c'est bien une figure plotly
        assert hasattr(fig, 'layout')
        assert len(fig.data) >= 2  # Au moins la courbe de stratégie et le capital initial
        
    def test_create_drawdown_chart(self, sample_ohlcv_data):
        """Test la création du graphique de drawdown"""
        strategy = MACrossover(short_period=10, long_period=20)
        result = strategy.backtest(sample_ohlcv_data.copy())
        
        backtester = StrategyBacktester()
        fig = backtester.create_drawdown_chart(result['equity_curve'])
        
        assert fig is not None
        assert len(fig.data) > 0
        
    def test_generate_report(self, sample_ohlcv_data):
        """Test la génération du rapport textuel"""
        strategy = MACrossover(short_period=10, long_period=20)
        result = strategy.backtest(sample_ohlcv_data.copy())
        
        backtester = StrategyBacktester()
        metrics = backtester.calculate_advanced_metrics(result['equity_curve'], result['trades'])
        
        report = backtester.generate_report(result, metrics)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert "RAPPORT DE BACKTESTING" in report
        assert "PERFORMANCE GLOBALE" in report

    def test_metrics_consistency_no_trades(self):
        """Test les métriques quand il n'y a aucun trade"""
        # Créer des données qui ne génèrent pas de trades (prix constants)
        dates = pd.date_range(start='2023-01-01', periods=100)
        df_flat = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 100,
            'high': [100] * 100,
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        strategy = MACrossover()
        result = strategy.backtest(df_flat)
        
        backtester = StrategyBacktester()
        metrics = backtester.calculate_advanced_metrics(result['equity_curve'], result['trades'])
        
        assert metrics['avg_trade_duration'] == 0
        assert metrics['max_consecutive_wins'] == 0
