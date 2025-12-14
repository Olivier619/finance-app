"""
Module de backtesting avancé pour stratégies de trading
Métriques avancées et visualisations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StrategyBacktester:
    """Backtester avancé avec métriques détaillées"""
    
    def __init__(self, initial_capital=100000, fees_pct=0.001):
        self.initial_capital = initial_capital
        self.fees_pct = fees_pct
    
    def calculate_advanced_metrics(self, equity_curve, trades, risk_free_rate=0.02):
        """
        Calcule des métriques avancées
        
        Args:
            equity_curve: DataFrame avec l'évolution de l'équité
            trades: Liste des trades
            risk_free_rate: Taux sans risque annualisé
        
        Returns:
            Dict avec métriques avancées
        """
        if equity_curve.empty:
            return {}
        
        # Calculer les rendements
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        # Sharpe Ratio
        excess_returns = equity_curve['returns'] - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino Ratio (ne pénalise que la volatilité négative)
        downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Max Drawdown
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
        max_drawdown = equity_curve['drawdown'].min() * 100
        max_drawdown_abs = equity_curve['drawdown'].min() * self.initial_capital
        
        # Calmar Ratio (rendement annualisé / max drawdown)
        total_return = (equity_curve['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return  # Simplification, devrait être annualisé selon la période
        calmar_ratio = abs(annual_return / (max_drawdown / 100)) if max_drawdown != 0 else 0
        
        # Recovery Factor (profit net / max drawdown absolu)
        net_profit = equity_curve['equity'].iloc[-1] - self.initial_capital
        recovery_factor = abs(net_profit / max_drawdown_abs) if max_drawdown_abs != 0 else 0
        
        # Analyse des trades
        if trades and len(trades) > 1:
            # Durée moyenne des trades
            trade_durations = []
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_date = trades[i]['date']
                    sell_date = trades[i+1]['date']
                    if isinstance(buy_date, (int, float)) and isinstance(sell_date, (int, float)):
                        duration = sell_date - buy_date
                    else:
                        try:
                            duration = (pd.to_datetime(sell_date) - pd.to_datetime(buy_date)).days
                        except:
                            duration = 0
                    trade_durations.append(duration)
            
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
            
            # Séries de gains/pertes consécutifs
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades) and trades[i]['type'] == 'BUY':
                    buy_price = trades[i]['price']
                    sell_price = trades[i+1]['price']
                    pnl = sell_price - buy_price
                    
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            avg_trade_duration = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'max_drawdown_pct': max_drawdown,
            'avg_trade_duration': avg_trade_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def create_equity_chart(self, results_list):
        """
        Crée un graphique comparatif des courbes d'équité
        
        Args:
            results_list: Liste de résultats de backtesting
        
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        for result in results_list:
            if 'equity_curve' in result and not result['equity_curve'].empty:
                equity_df = result['equity_curve']
                fig.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['equity'],
                    mode='lines',
                    name=result['strategy'],
                    hovertemplate='%{y:,.2f}$<extra></extra>'
                ))
        
        # Ligne de référence (capital initial)
        if results_list and 'equity_curve' in results_list[0]:
            equity_df = results_list[0]['equity_curve']
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=[self.initial_capital] * len(equity_df),
                mode='lines',
                name='Capital Initial',
                line=dict(dash='dash', color='gray'),
                hovertemplate='%{y:,.2f}$<extra></extra>'
            ))
        
        fig.update_layout(
            title='Comparaison des Courbes d\'Équité',
            xaxis_title='Date',
            yaxis_title='Équité ($)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_drawdown_chart(self, equity_curve):
        """
        Crée un graphique de drawdown
        
        Args:
            equity_curve: DataFrame avec l'évolution de l'équité
        
        Returns:
            Figure Plotly
        """
        if equity_curve.empty:
            return go.Figure()
        
        equity_curve = equity_curve.copy()
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_curve['date'],
            y=equity_curve['drawdown'],
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Drawdown au Fil du Temps',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_returns_distribution(self, equity_curve):
        """
        Crée un histogramme de distribution des rendements
        
        Args:
            equity_curve: DataFrame avec l'évolution de l'équité
        
        Returns:
            Figure Plotly
        """
        if equity_curve.empty:
            return go.Figure()
        
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change() * 100
        returns = equity_curve['returns'].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Rendements',
            marker=dict(color='blue', opacity=0.7)
        ))
        
        fig.update_layout(
            title='Distribution des Rendements Quotidiens',
            xaxis_title='Rendement (%)',
            yaxis_title='Fréquence',
            height=400
        )
        
        return fig
    
    def generate_report(self, result, advanced_metrics):
        """
        Génère un rapport textuel détaillé
        
        Args:
            result: Résultats du backtesting
            advanced_metrics: Métriques avancées
        
        Returns:
            str: Rapport formaté
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           RAPPORT DE BACKTESTING - {result['strategy']}
╚══════════════════════════════════════════════════════════════╝

📊 PERFORMANCE GLOBALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Capital Initial      : ${result['initial_capital']:,.2f}
  Capital Final        : ${result['final_capital']:,.2f}
  Rendement Total      : {result['total_return']:.2f}%
  Profit Net           : ${result['final_capital'] - result['initial_capital']:,.2f}

📈 MÉTRIQUES DE RISQUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sharpe Ratio         : {advanced_metrics.get('sharpe_ratio', 0):.3f}
  Sortino Ratio        : {advanced_metrics.get('sortino_ratio', 0):.3f}
  Calmar Ratio         : {advanced_metrics.get('calmar_ratio', 0):.3f}
  Recovery Factor      : {advanced_metrics.get('recovery_factor', 0):.3f}
  Max Drawdown         : {advanced_metrics.get('max_drawdown_pct', 0):.2f}%

💼 STATISTIQUES DE TRADING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Trades         : {result['total_trades']}
  Trades Gagnants      : {result['winning_trades']}
  Trades Perdants      : {result['losing_trades']}
  Win Rate             : {result['win_rate']:.2f}%
  Profit Factor        : {result['profit_factor']:.2f}
  Durée Moy. Trade     : {advanced_metrics.get('avg_trade_duration', 0):.1f} jours
  Max Gains Consécutifs: {advanced_metrics.get('max_consecutive_wins', 0)}
  Max Pertes Consécutives: {advanced_metrics.get('max_consecutive_losses', 0)}

💡 INTERPRÉTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        

        report += "\n📚 GUIDE D'INTERPRÉTATION DES MÉTRIQUES\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 1. Sharpe Ratio
        report += "1. Sharpe Ratio (Rendement ajusté au risque)\n"
        report += "   - C'est quoi ? Mesure le rendement par unité de risque.\n"
        report += "   - Formule : (Rendement - Taux sans risque) / Volatilité\n"
        report += f"   - Votre score : {advanced_metrics.get('sharpe_ratio', 0):.2f}\n"
        if advanced_metrics.get('sharpe_ratio', 0) > 1:
            report += "   ✅ Bon (>1) : Le rendement justifie le risque pris.\n"
        else:
            report += "   ⚠️ Faible (<1) : Le rendement est faible par rapport au risque.\n"
            
        report += "\n2. Sortino Ratio (Risque de perte uniquement)\n"
        report += "   - C'est quoi ? Comme le Sharpe, mais ne penalise que la volatilité à la baisse.\n"
        report += "   - Pourquoi ? La volatilité à la hausse (gains rapides) est positive !\n"
        report += f"   - Votre score : {advanced_metrics.get('sortino_ratio', 0):.2f}\n"

        # 3. Max Drawdown
        max_dd = abs(advanced_metrics.get('max_drawdown_pct', 0))
        report += "\n3. Max Drawdown (Perte Maximale)\n"
        report += "   - C'est quoi ? La plus grosse chute du capital depuis un sommet historique.\n"
        report += "   - Exemple : Si vous passez de 100k$ à 80k$, le DD est de 20%.\n"
        report += f"   - Votre score : -{max_dd:.2f}%\n"
        if max_dd < 15:
            report += "   ✅ Faible (<15%) : Stratégie conservatrice et sûre.\n"
        elif max_dd < 30:
            report += "   ⚠️ Modéré (15-30%) : Risque standard pour une stratégie active.\n"
        else:
            report += "   ❌ Élevé (>30%) : Risque important de perte de capital.\n"
            
        # 4. Win Rate vs Profit Factor
        report += "\n4. Win Rate & Profit Factor\n"
        report += f"   - Win Rate ({result['win_rate']:.1f}%) : Pourcentage de trades gagnants.\n"
        report += f"   - Profit Factor ({result['profit_factor']:.2f}) : Somme des gains / Somme des pertes.\n"
        
        if result['profit_factor'] > 1.5:
            report += "   ✅ Excellent Profit Factor (>1.5) : Vos gains couvrent largement vos pertes.\n"
        elif result['profit_factor'] > 1:
            report += "   ⚠️ Profit Factor juste (>1.0) : Stratégie profitable mais fragile.\n"
        else:
            report += "   ❌ Profit Factor négatif (<1.0) : Vous perdez plus d'argent que vous n'en gagnez.\n"
            
        report += "\n💡 CONSEIL : Une stratégie avec un Win Rate faible (ex: 40%) peut être très rentable\n"
        report += "   si le Profit Factor est élevé (elle gagne peu souvent mais gagne gros).\n"
        
        report += "\n╚══════════════════════════════════════════════════════════════╝\n"
        
        return report
