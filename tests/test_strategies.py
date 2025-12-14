"""
Tests unitaires pour les stratégies de trading
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')

from trading_strategies import (
    MACrossover, RSIStrategy, BollingerBandsStrategy, MACDStrategy,
    get_all_strategies, compare_strategies
)


class TestMACrossover:
    """Tests pour la stratégie Moving Average Crossover"""
    
    def test_signal_generation(self, sample_ohlcv_data):
        """Test que les signaux sont générés correctement"""
        strategy = MACrossover(short_period=10, long_period=20)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        assert df_signals is not None
        assert 'signal' in df_signals.columns
        assert 'ma_short' in df_signals.columns
        assert 'ma_long' in df_signals.columns
        
        # Vérifier que les signaux sont valides (-1, 0, 1)
        assert df_signals['signal'].isin([-1, 0, 1]).all()
    
    def test_buy_signal_on_crossover_up(self):
        """Test qu'un signal d'achat est généré lors d'un croisement haussier"""
        # Créer des données avec un croisement : Prix bas -> Prix haut
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 50 jours constants à 100, puis 50 jours à 120
        # MA Short (période 5) va monter vite, MA Long (période 10) va monter doucement
        # Donc Short va croiser Long vers le haut
        prices = [100.0] * 50 + [120.0] * 50
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        })
        
        strategy = MACrossover(short_period=5, long_period=10)
        df_signals = strategy.generate_signals(df)
        
        # Il devrait y avoir au moins un signal d'achat
        assert (df_signals['signal'] == 1).any()
    
    def test_sell_signal_on_crossover_down(self):
        """Test qu'un signal de vente est généré lors d'un croisement baissier"""
        # Créer des données avec un croisement : Prix haut -> Prix bas
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 50 jours constants à 120, puis 50 jours à 100
        # MA Short va descendre vite, MA Long va descendre doucement
        # Donc Short va croiser Long vers le bas
        prices = [120.0] * 50 + [100.0] * 50
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        })
        
        strategy = MACrossover(short_period=5, long_period=10)
        df_signals = strategy.generate_signals(df)
        
        # Il devrait y avoir au moins un signal de vente
        assert (df_signals['signal'] == -1).any()
    
    def test_backtest_returns_valid_result(self, sample_ohlcv_data):
        """Test que le backtesting retourne un résultat valide"""
        strategy = MACrossover(short_period=10, long_period=20)
        result = strategy.backtest(sample_ohlcv_data.copy(), initial_capital=10000)
        
        assert result is not None
        assert 'strategy' in result
        assert 'final_capital' in result
        assert 'total_return' in result
        assert 'total_trades' in result
        assert result['final_capital'] > 0


class TestRSIStrategy:
    """Tests pour la stratégie RSI"""
    
    def test_signal_generation(self, sample_ohlcv_data):
        """Test que les signaux RSI sont générés correctement"""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        assert df_signals is not None
        assert 'signal' in df_signals.columns
        assert 'rsi' in df_signals.columns
        
        # Vérifier que le RSI est dans la bonne plage
        rsi_values = df_signals['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_buy_signal_on_oversold(self):
        """Test qu'un signal d'achat est généré en zone de survente"""
        # Créer des données qui vont générer un RSI bas puis une remontée
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Prix qui baissent fortement (RSI < 30) puis remontent
        prices = [100.0]
        for i in range(1, 60):
            prices.append(prices[-1] * 0.98) # Baisse continue -> RSI très bas
        
        for i in range(60, 100):
            prices.append(prices[-1] * 1.05) # Remontée brusque -> RSI remonte > 30
            
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        })
        
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        df_signals = strategy.generate_signals(df)
        
        # Devrait y avoir un signal d'achat quand le RSI remonte de la zone de survente
        assert (df_signals['signal'] == 1).any()
    
    def test_parameters_are_configurable(self):
        """Test que les paramètres sont configurables"""
        strategy = RSIStrategy(period=10, oversold=25, overbought=75)
        
        assert strategy.period == 10
        assert strategy.oversold == 25
        assert strategy.overbought == 75


class TestBollingerBandsStrategy:
    """Tests pour la stratégie Bollinger Bands"""
    
    def test_signal_generation(self, sample_ohlcv_data):
        """Test que les signaux Bollinger Bands sont générés"""
        strategy = BollingerBandsStrategy(period=20, std=2)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        assert df_signals is not None
        assert 'signal' in df_signals.columns
        assert 'bb_lower' in df_signals.columns
        assert 'bb_upper' in df_signals.columns
        assert 'bb_middle' in df_signals.columns
    
    def test_bands_are_ordered_correctly(self, sample_ohlcv_data):
        """Test que les bandes sont dans le bon ordre (lower < middle < upper)"""
        strategy = BollingerBandsStrategy(period=20, std=2)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        # Supprimer les NaN
        df_clean = df_signals.dropna(subset=['bb_lower', 'bb_middle', 'bb_upper'])
        
        # Vérifier l'ordre
        assert (df_clean['bb_lower'] <= df_clean['bb_middle']).all()
        assert (df_clean['bb_middle'] <= df_clean['bb_upper']).all()


class TestMACDStrategy:
    """Tests pour la stratégie MACD"""
    
    def test_signal_generation(self, sample_ohlcv_data):
        """Test que les signaux MACD sont générés"""
        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        assert df_signals is not None
        assert 'signal' in df_signals.columns
        assert 'macd' in df_signals.columns
        assert 'macd_signal' in df_signals.columns
        assert 'macd_hist' in df_signals.columns
    
    def test_histogram_calculation(self, sample_ohlcv_data):
        """Test que l'histogramme MACD est calculé correctement"""
        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        df_signals = strategy.generate_signals(sample_ohlcv_data.copy())
        
        # Supprimer les NaN
        df_clean = df_signals.dropna(subset=['macd', 'macd_signal', 'macd_hist'])
        
        # L'histogramme devrait être la différence entre MACD et signal
        expected_hist = df_clean['macd'] - df_clean['macd_signal']
        
        # Vérifier avec une tolérance pour les erreurs d'arrondi
        assert np.allclose(df_clean['macd_hist'], expected_hist, rtol=1e-5)


class TestStrategyComparison:
    """Tests pour la comparaison de stratégies"""
    
    def test_compare_all_strategies(self, sample_ohlcv_data):
        """Test que toutes les stratégies peuvent être comparées"""
        comparison = compare_strategies(sample_ohlcv_data.copy(), initial_capital=10000)
        
        assert not comparison.empty
        assert len(comparison) == 4  # 4 stratégies
        assert 'strategy' in comparison.columns
        assert 'total_return' in comparison.columns
    
    def test_comparison_is_sorted_by_return(self, sample_ohlcv_data):
        """Test que la comparaison est triée par rendement"""
        comparison = compare_strategies(sample_ohlcv_data.copy())
        
        # Vérifier que c'est trié par ordre décroissant de rendement
        returns = comparison['total_return'].values
        assert all(returns[i] >= returns[i+1] for i in range(len(returns)-1))
    
    def test_get_all_strategies_returns_dict(self):
        """Test que get_all_strategies retourne un dictionnaire"""
        strategies = get_all_strategies()
        
        assert isinstance(strategies, dict)
        assert len(strategies) == 4
        assert 'MA Crossover' in strategies
        assert 'RSI Strategy' in strategies
        assert 'Bollinger Bands' in strategies
        assert 'MACD Strategy' in strategies


class TestEdgeCases:
    """Tests des cas limites"""
    
    def test_empty_dataframe(self):
        """Test avec un DataFrame vide"""
        df_empty = pd.DataFrame()
        strategy = MACrossover()
        
        # Ne devrait pas crasher
        result = strategy.backtest(df_empty)
        # Le résultat peut être None ou un dict avec des valeurs par défaut
    
    def test_insufficient_data(self):
        """Test avec trop peu de données"""
        df_small = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        })
        
        strategy = MACrossover(short_period=50, long_period=200)
        df_signals = strategy.generate_signals(df_small)
        
        # Devrait retourner un DataFrame même si les MA ne peuvent pas être calculées
        assert df_signals is not None
    
    def test_constant_prices(self):
        """Test avec des prix constants (pas de volatilité)"""
        df_constant = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100),
            'open': [100] * 100,
            'high': [100] * 100,
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100
        })
        
        strategy = RSIStrategy()
        result = strategy.backtest(df_constant)
        
        # Devrait retourner un résultat sans trades
        if result:
            assert result['total_trades'] == 0
