"""
Module de backtesting
Teste les prédictions de l'IA sur données historiques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class Backtesting:
    def __init__(self):
        pass
    
    def walk_forward_analysis(self, df, model, features, target_col='Target', 
                              train_window=252, test_window=20, horizon=1):
        """
        Effectue une analyse walk-forward
        
        Args:
            df: DataFrame avec features et target
            model: Modèle sklearn (non entraîné)
            features: List des colonnes de features
            target_col: Nom de la colonne target
            train_window: Taille de la fenêtre d'entraînement (défaut: 252 jours)
            test_window: Taille de la fenêtre de test (défaut: 20 jours)
            horizon: Horizon de prédiction en jours (défaut: 1)
        
        Returns:
            Dict avec résultats du backtesting
        """
        # Préparer les données
        data = df.copy()
        data[target_col] = (data['close'].shift(-horizon) > data['close']).astype(int)
        data = data.dropna(subset=features + [target_col])
        
        predictions = []
        actuals = []
        dates = []
        
        # Walk-forward
        for i in range(train_window, len(data) - test_window, test_window):
            # Fenêtre d'entraînement
            train_start = max(0, i - train_window)
            train_end = i
            
            train_data = data.iloc[train_start:train_end]
            X_train = train_data[features]
            y_train = train_data[target_col]
            
            # Fenêtre de test
            test_data = data.iloc[i:i + test_window]
            X_test = test_data[features]
            y_test = test_data[target_col]
            
            # Vérifier qu'il y a au moins 2 classes
            if len(y_train.unique()) < 2:
                continue
            
            # Entraîner le modèle
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                predictions.extend(y_pred)
                actuals.extend(y_test.values)
                dates.extend(test_data['timestamp'].values if 'timestamp' in test_data.columns else range(len(y_test)))
            except:
                continue
        
        if not predictions:
            return None
        
        # Calculer les métriques
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        f1 = f1_score(actuals, predictions, zero_division=0)
        
        # Matrice de confusion
        cm = confusion_matrix(actuals, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'total_predictions': len(predictions)
        }
    
    def calculate_trading_performance(self, predictions, actuals, prices):
        """
        Calcule la performance de trading basée sur les prédictions
        
        Args:
            predictions: List de prédictions (0 ou 1)
            actuals: List de valeurs réelles (0 ou 1)
            prices: List de prix
        
        Returns:
            Dict avec métriques de trading
        """
        if len(predictions) != len(actuals) or len(predictions) != len(prices):
            return None
        
        # Simuler le trading
        initial_capital = 10000
        capital = initial_capital
        position = 0  # 0 = pas de position, 1 = long
        trades = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            price = prices[i]
            
            # Signal d'achat
            if pred == 1 and position == 0:
                position = 1
                shares = capital / price
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'index': i
                })
            
            # Signal de vente
            elif pred == 0 and position == 1:
                position = 0
                capital = shares * price
                trades.append({
                    'type': 'SELL',
                    'price': price,
                    'capital': capital,
                    'index': i
                })
        
        # Clôturer la position finale
        if position == 1:
            capital = shares * prices[-1]
        
        # Calculer les métriques
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        # Compter les trades gagnants/perdants
        winning_trades = 0
        losing_trades = 0
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_price = trades[i]['price']
                sell_price = trades[i + 1]['price']
                
                if sell_price > buy_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate
        }
    
    def generate_backtest_report(self, backtest_results, trading_performance=None):
        """
        Génère un rapport de backtesting
        
        Args:
            backtest_results: Dict retourné par walk_forward_analysis
            trading_performance: Dict retourné par calculate_trading_performance (optionnel)
        
        Returns:
            str: Rapport formaté
        """
        if not backtest_results:
            return "Aucun résultat de backtesting disponible."
        
        report = []
        report.append("=" * 50)
        report.append("RAPPORT DE BACKTESTING")
        report.append("=" * 50)
        report.append("")
        
        # Métriques de prédiction
        report.append("📊 MÉTRIQUES DE PRÉDICTION")
        report.append(f"  Accuracy:  {backtest_results['accuracy']:.2%}")
        report.append(f"  Precision: {backtest_results['precision']:.2%}")
        report.append(f"  Recall:    {backtest_results['recall']:.2%}")
        report.append(f"  F1-Score:  {backtest_results['f1_score']:.2%}")
        report.append(f"  Total Predictions: {backtest_results['total_predictions']}")
        report.append("")
        
        # Matrice de confusion
        cm = backtest_results['confusion_matrix']
        report.append("📈 MATRICE DE CONFUSION")
        report.append(f"  True Negatives:  {cm[0][0]}")
        report.append(f"  False Positives: {cm[0][1]}")
        report.append(f"  False Negatives: {cm[1][0]}")
        report.append(f"  True Positives:  {cm[1][1]}")
        report.append("")
        
        # Performance de trading
        if trading_performance:
            report.append("💰 PERFORMANCE DE TRADING")
            report.append(f"  Capital Initial:  ${trading_performance['initial_capital']:,.2f}")
            report.append(f"  Capital Final:    ${trading_performance['final_capital']:,.2f}")
            report.append(f"  Rendement Total:  {trading_performance['total_return']:.2f}%")
            report.append(f"  Nombre de Trades: {trading_performance['total_trades']}")
            report.append(f"  Trades Gagnants:  {trading_performance['winning_trades']}")
            report.append(f"  Trades Perdants:  {trading_performance['losing_trades']}")
            report.append(f"  Win Rate:         {trading_performance['win_rate']:.2f}%")
            report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def compare_predictions_vs_reality(self, backtest_results):
        """
        Compare les prédictions avec la réalité
        
        Args:
            backtest_results: Dict retourné par walk_forward_analysis
        
        Returns:
            DataFrame avec comparaison
        """
        if not backtest_results:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'date': backtest_results['dates'],
            'prediction': backtest_results['predictions'],
            'actual': backtest_results['actuals'],
            'correct': [p == a for p, a in zip(backtest_results['predictions'], backtest_results['actuals'])]
        })
        
        return df
