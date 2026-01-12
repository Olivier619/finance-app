"""
Module de calcul des métriques de performance du portfolio
ROI, Sharpe Ratio, Max Drawdown, Win Rate, etc.
"""

import pandas as pd
import numpy as np
from database import Database
import yfinance as yf


class PerformanceMetrics:
    def __init__(self):
        self.db = Database()
    
    def calculate_roi(self, initial_value, current_value):
        """
        Calcule le Return on Investment
        
        Args:
            initial_value: Valeur initiale
            current_value: Valeur actuelle
        
        Returns:
            float: ROI en %
        """
        if initial_value == 0:
            return 0.0
        
        return ((current_value - initial_value) / initial_value) * 100
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calcule le Sharpe Ratio
        
        Args:
            returns: Series de rendements
            risk_free_rate: Taux sans risque annualisé (défaut: 2%)
        
        Returns:
            float: Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Taux journalier
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calcule le Maximum Drawdown
        
        Args:
            equity_curve: Series de valeurs du portfolio
        
        Returns:
            Dict avec max_drawdown, max_drawdown_pct, peak, trough
        """
        if len(equity_curve) == 0:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0}
        
        # Si une seule valeur, pas de drawdown
        if len(equity_curve) == 1:
            return {
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'peak_value': equity_curve.iloc[0],
                'trough_value': equity_curve.iloc[0],
                'peak_date': equity_curve.index[0] if hasattr(equity_curve, 'index') else 0,
                'trough_date': equity_curve.index[0] if hasattr(equity_curve, 'index') else 0
            }
        
        # Calculer le pic cumulatif
        cumulative_max = equity_curve.cummax()
        
        # Calculer le drawdown
        drawdown = equity_curve - cumulative_max
        drawdown_pct = (drawdown / cumulative_max) * 100
        
        # Trouver le drawdown maximum
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown[max_dd_idx]
        max_dd_pct = drawdown_pct[max_dd_idx]
        
        # Trouver le pic correspondant (vérifier qu'il y a des données avant max_dd_idx)
        cumulative_before = cumulative_max[:max_dd_idx]
        if len(cumulative_before) > 0:
            peak_idx = cumulative_before.idxmax()
            peak_value = cumulative_max[peak_idx]
        else:
            peak_idx = equity_curve.index[0] if hasattr(equity_curve, 'index') else 0
            peak_value = equity_curve.iloc[0]
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd_pct),
            'peak_value': peak_value,
            'trough_value': equity_curve[max_dd_idx],
            'peak_date': peak_idx,
            'trough_date': max_dd_idx
        }
    
    def calculate_win_rate(self, transactions):
        """
        Calcule le Win Rate (% de trades gagnants)
        
        Args:
            transactions: List de transactions
        
        Returns:
            Dict avec win_rate, total_trades, winning_trades, losing_trades
        """
        if not transactions:
            return {'win_rate': 0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        # Filtrer uniquement les ventes
        sells = [t for t in transactions if t['type'] == 'SELL']
        
        if not sells:
            return {'win_rate': 0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        # Calculer le P&L pour chaque vente
        winning_trades = 0
        losing_trades = 0
        
        for sell in sells:
            # Extraire le P&L depuis les notes (format: "P&L: $X.XX")
            notes = sell.get('notes', '')
            if 'P&L:' in notes:
                try:
                    pnl_str = notes.split('P&L:')[1].split(')')[0].strip().replace('$', '')
                    pnl = float(pnl_str)
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                except:
                    continue
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def calculate_profit_factor(self, transactions):
        """
        Calcule le Profit Factor (gains totaux / pertes totales)
        
        Args:
            transactions: List de transactions
        
        Returns:
            float: Profit Factor
        """
        sells = [t for t in transactions if t['type'] == 'SELL']
        
        if not sells:
            return 0.0
        
        total_gains = 0
        total_losses = 0
        
        for sell in sells:
            notes = sell.get('notes', '')
            if 'P&L:' in notes:
                try:
                    pnl_str = notes.split('P&L:')[1].split(')')[0].strip().replace('$', '')
                    pnl = float(pnl_str)
                    
                    if pnl > 0:
                        total_gains += pnl
                    else:
                        total_losses += abs(pnl)
                except:
                    continue
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0.0
        
        return total_gains / total_losses
    
    def get_equity_curve(self, initial_cash=100000.0):
        """
        Génère la courbe d'équité du portfolio
        
        Returns:
            DataFrame avec dates et valeurs du portfolio
        """
        transactions = self.db.get_transaction_history(limit=10000)
        
        if not transactions:
            return pd.DataFrame({'date': [], 'equity': []})
        
        # Créer un DataFrame des transactions
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculer l'équité au fil du temps
        equity = initial_cash
        equity_curve = []
        
        for _, row in df.iterrows():
            if row['type'] == 'BUY':
                equity -= row['total']
            else:  # SELL
                equity += row['total']
            
            equity_curve.append({
                'date': row['date'],
                'equity': equity
            })
        
        return pd.DataFrame(equity_curve)
    
    def get_portfolio_metrics(self, initial_cash=100000.0):
        """
        Calcule toutes les métriques de performance
        
        Returns:
            Dict avec toutes les métriques
        """
        # Récupérer les données
        transactions = self.db.get_transaction_history(limit=10000)
        
        # Si aucune transaction, retourner des valeurs par défaut
        if not transactions:
            return {
                'initial_value': initial_cash,
                'current_value': initial_cash,
                'roi': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'profit_factor': 0.0
            }
        
        equity_curve_df = self.get_equity_curve(initial_cash)
        
        # Valeur actuelle du portfolio
        holdings = self.db.get_holdings()
        cash = self.db.get_cash_balance()
        
        current_value = cash
        for holding in holdings:
            try:
                ticker = yf.Ticker(holding['symbol'])
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_value += holding['quantity'] * current_price
            except:
                continue
        
        # Calculer les métriques
        roi = self.calculate_roi(initial_cash, current_value)
        
        # Sharpe Ratio (basé sur les rendements quotidiens)
        if len(equity_curve_df) > 1:
            equity_curve_df['returns'] = equity_curve_df['equity'].pct_change()
            sharpe = self.calculate_sharpe_ratio(equity_curve_df['returns'].dropna())
        else:
            sharpe = 0.0
        
        # Max Drawdown
        if len(equity_curve_df) > 0:
            max_dd = self.calculate_max_drawdown(equity_curve_df['equity'])
        else:
            max_dd = {'max_drawdown': 0, 'max_drawdown_pct': 0}
        
        # Win Rate
        win_rate_data = self.calculate_win_rate(transactions)
        
        # Profit Factor
        profit_factor = self.calculate_profit_factor(transactions)
        
        return {
            'initial_value': initial_cash,
            'current_value': current_value,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd['max_drawdown'],
            'max_drawdown_pct': max_dd['max_drawdown_pct'],
            'win_rate': win_rate_data['win_rate'],
            'total_trades': win_rate_data['total_trades'],
            'winning_trades': win_rate_data['winning_trades'],
            'losing_trades': win_rate_data['losing_trades'],
            'profit_factor': profit_factor
        }
