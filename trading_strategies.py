"""
Module de stratégies de trading
Implémente 4 stratégies classiques avec backtesting
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from abc import ABC, abstractmethod


class TradingStrategy(ABC):
    """Classe de base pour toutes les stratégies de trading"""
    
    def __init__(self, name):
        self.name = name
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, df):
        """
        Génère les signaux de trading
        
        Args:
            df: DataFrame avec colonnes OHLCV
        
        Returns:
            DataFrame avec colonne 'signal' (1=BUY, -1=SELL, 0=HOLD)
        """
        pass
    
    def backtest(self, df, initial_capital=100000, fees_pct=0.001):
        """
        Effectue le backtesting de la stratégie
        
        Args:
            df: DataFrame avec données OHLCV
            initial_capital: Capital initial
            fees_pct: Frais de transaction en %
        
        Returns:
            Dict avec résultats du backtesting
        """
        # Générer les signaux
        df_signals = self.generate_signals(df.copy())
        
        if df_signals is None or 'signal' not in df_signals.columns:
            return None
        
        # Initialisation
        capital = initial_capital
        position = 0  # 0 = pas de position, 1 = long
        shares = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df_signals)):
            row = df_signals.iloc[i]
            signal = row['signal']
            price = row['close']
            date = row['timestamp'] if 'timestamp' in row else i
            
            # Enregistrer l'équité
            current_equity = capital + (shares * price if position == 1 else 0)
            equity_curve.append({
                'date': date,
                'equity': current_equity,
                'price': price
            })
            
            # Signal d'achat
            if signal == 1 and position == 0:
                # Acheter
                shares = capital / price
                fees = capital * fees_pct
                capital = 0
                position = 1
                
                trades.append({
                    'type': 'BUY',
                    'date': date,
                    'price': price,
                    'shares': shares,
                    'fees': fees
                })
            
            # Signal de vente
            elif signal == -1 and position == 1:
                # Vendre
                capital = shares * price
                fees = capital * fees_pct
                capital -= fees
                
                trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'shares': shares,
                    'fees': fees,
                    'capital': capital
                })
                
                shares = 0
                position = 0
        
        # Clôturer la position finale si nécessaire
        if position == 1:
            final_price = df_signals.iloc[-1]['close']
            capital = shares * price
            fees = capital * fees_pct
            capital -= fees
        
        # Calculer les métriques
        final_equity = capital
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Compter les trades gagnants/perdants
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades) and trades[i]['type'] == 'BUY' and trades[i+1]['type'] == 'SELL':
                buy_price = trades[i]['price']
                sell_price = trades[i+1]['price']
                pnl = (sell_price - buy_price) * trades[i]['shares']
                
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                else:
                    losing_trades += 1
                    total_loss += abs(pnl)
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Calculer le max drawdown
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        return {
            'strategy': self.name,
            'initial_capital': initial_capital,
            'final_capital': final_equity,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_df,
            'trades': trades
        }


class MACrossover(TradingStrategy):
    """Stratégie Moving Average Crossover"""
    
    def __init__(self, short_period=50, long_period=200):
        super().__init__(f"MA Crossover ({short_period}/{long_period})")
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, df):
        """Génère les signaux basés sur le croisement des MA"""
        if df.empty or 'close' not in df.columns:
            return df
            
        # Calculer les moyennes mobiles
        df['ma_short'] = df['close'].rolling(window=self.short_period).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_period).mean()
        
        # Générer les signaux
        df['signal'] = 0
        
        # Signal continu : 1 si tendance haussière (MA Short > MA Long), -1 sinon
        df['signal'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
        
        # Gestion des NaN (début de l'historique)
        df.loc[df['ma_long'].isna(), 'signal'] = 0
        
        return df


class RSIStrategy(TradingStrategy):
    """Stratégie RSI Overbought/Oversold"""
    
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__(f"RSI Strategy ({oversold}/{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df):
        """Génère les signaux basés sur le RSI"""
        if df.empty or 'close' not in df.columns:
            return df
            
        # Calculer le RSI
        df['rsi'] = ta.rsi(df['close'], length=self.period)
        
        # Générer les signaux
        df['signal'] = 0
        
        # Signal d'achat : RSI passe en dessous de oversold puis remonte
        df.loc[(df['rsi'] > self.oversold) & 
               (df['rsi'].shift(1) <= self.oversold), 'signal'] = 1
        
        # Signal de vente : RSI passe au-dessus de overbought puis redescend
        df.loc[(df['rsi'] < self.overbought) & 
               (df['rsi'].shift(1) >= self.overbought), 'signal'] = -1
        
        return df


class BollingerBandsStrategy(TradingStrategy):
    """Stratégie Bollinger Bands Breakout"""
    
    def __init__(self, period=20, std=2):
        super().__init__(f"Bollinger Bands ({period}, {std}σ)")
        self.period = period
        self.std = std
    
    def generate_signals(self, df):
        """Génère les signaux basés sur les Bollinger Bands"""
        if df.empty or 'close' not in df.columns:
            return df
            
        # Calculer les Bollinger Bands
        bb = ta.bbands(df['close'], length=self.period, std=self.std)
        
        if bb is not None:
            df['bb_lower'] = bb[bb.columns[0]]  # BBL
            df['bb_middle'] = bb[bb.columns[1]]  # BBM
            df['bb_upper'] = bb[bb.columns[2]]  # BBU
        else:
            return None
        
        # Générer les signaux
        df['signal'] = 0
        
        # Signal d'achat : Prix touche ou passe en dessous de la bande inférieure
        df.loc[(df['close'] <= df['bb_lower']) & 
               (df['close'].shift(1) > df['bb_lower'].shift(1)), 'signal'] = 1
        
        # Signal de vente : Prix touche ou passe au-dessus de la bande supérieure
        df.loc[(df['close'] >= df['bb_upper']) & 
               (df['close'].shift(1) < df['bb_upper'].shift(1)), 'signal'] = -1
        
        return df


class MACDStrategy(TradingStrategy):
    """Stratégie MACD Signal"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(f"MACD Strategy ({fast}/{slow}/{signal})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, df):
        """Génère les signaux basés sur le MACD"""
        if df.empty or 'close' not in df.columns:
            return df
            
        # Calculer le MACD
        macd = ta.macd(df['close'], fast=self.fast, slow=self.slow, signal=self.signal_period)
        
        if macd is not None:
            df['macd'] = macd[macd.columns[0]]  # MACD line
            df['macd_signal'] = macd[macd.columns[1]]  # Signal line
            df['macd_hist'] = macd[macd.columns[2]]  # Histogram
        else:
            return None
        
        # Générer les signaux
        df['signal'] = 0
        
        # Signal continu : 1 si Momentum haussier (MACD > Signal), -1 sinon
        df['signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Gestion des NaN
        df.loc[df['macd_signal'].isna(), 'signal'] = 0
        
        return df


class CoupledStrategy(TradingStrategy):
    """Stratégie couplée combinant plusieurs stratégies"""
    
    def __init__(self, strategies):
        # Créer un nom composé
        names = "+".join([s.name.split(' ')[0] for s in strategies])
        if len(names) > 30:
            names = f"Coupled ({len(strategies)} strategies)"
        super().__init__(names)
        self.strategies = strategies
        
    def generate_signals(self, df):
        """Génère les signaux basés sur le consensus des stratégies"""
        if df.empty or 'close' not in df.columns:
            return df
        
        # Initialiser le signal global
        df_consensus = df.copy()
        df_consensus['signal_sum'] = 0
        
        valid_strategies = 0
        
        for strategy in self.strategies:
            # Générer les signaux pour chaque stratégie
            # On passe une copie pour éviter les effets de bord
            df_temp = strategy.generate_signals(df.copy())
            
            if df_temp is not None and 'signal' in df_temp.columns:
                df_consensus['signal_sum'] += df_temp['signal'].fillna(0)
                valid_strategies += 1
        
        if valid_strategies == 0:
            return df
            
        # Règle de consensus : 
        # Si somme > 0 => ACHAT (plus d'acheteurs que de vendeurs)
        # Si somme < 0 => VENTE (plus de vendeurs que d'acheteurs)
        df_consensus['signal'] = 0
        df_consensus.loc[df_consensus['signal_sum'] > 0, 'signal'] = 1
        df_consensus.loc[df_consensus['signal_sum'] < 0, 'signal'] = -1
        
        return df_consensus



def get_all_strategies():
    """Retourne toutes les stratégies disponibles"""
    return {
        'MA Crossover': MACrossover(),
        'RSI Strategy': RSIStrategy(),
        'Bollinger Bands': BollingerBandsStrategy(),
        'MACD Strategy': MACDStrategy()
    }


def compare_strategies(df, strategies=None, initial_capital=100000):
    """
    Compare plusieurs stratégies
    
    Args:
        df: DataFrame avec données OHLCV
        strategies: Dict de stratégies (si None, utilise toutes)
        initial_capital: Capital initial
    
    Returns:
        DataFrame avec comparaison des résultats
    """
    if strategies is None:
        strategies = get_all_strategies()
    
    results = []
    
    for name, strategy in strategies.items():
        result = strategy.backtest(df, initial_capital)
        if result:
            results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    # Créer un DataFrame de comparaison
    comparison = pd.DataFrame(results)
    
    # Trier par rendement total
    comparison = comparison.sort_values('total_return', ascending=False)
    
    return comparison
