"""
Module de simulation de trading (Paper Trading)
Gestion du portefeuille virtuel
"""

import yfinance as yf
from database import Database
from datetime import datetime


class PaperTrading:
    def __init__(self, initial_cash=100000.0):
        self.db = Database()
        
        # Vérifier si le portfolio est vide et initialiser si nécessaire
        current_cash = self.db.get_cash_balance()
        if current_cash == 0:
            self.db.reset_portfolio(initial_cash)
    
    def get_current_price(self, symbol):
        """Récupère le prix actuel d'un symbole"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                return hist['Close'].iloc[-1]
            else:
                return None
        except:
            return None
    
    def buy(self, symbol, quantity, price=None, fees_pct=0.001):
        """
        Achète des actions
        
        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à acheter
            price: Prix d'achat (None = prix actuel)
            fees_pct: Frais en % (défaut: 0.1%)
        
        Returns:
            Dict avec le résultat de la transaction
        """
        # Récupérer le prix actuel si non spécifié
        if price is None:
            price = self.get_current_price(symbol)
            if price is None:
                return {'success': False, 'error': 'Impossible de récupérer le prix'}
        
        # Calculer le coût total
        cost = quantity * price
        fees = cost * fees_pct
        total_cost = cost + fees
        
        # Vérifier le solde
        cash_balance = self.db.get_cash_balance()
        
        if total_cost > cash_balance:
            return {
                'success': False,
                'error': f'Solde insuffisant. Requis: ${total_cost:.2f}, Disponible: ${cash_balance:.2f}'
            }
        
        # Exécuter la transaction
        transaction_id = self.db.add_transaction(
            symbol=symbol,
            transaction_type='BUY',
            quantity=quantity,
            price=price,
            fees=fees,
            notes=f'Achat de {quantity} actions à ${price:.2f}'
        )
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'total_cost': total_cost,
            'new_balance': self.db.get_cash_balance()
        }
    
    def sell(self, symbol, quantity, price=None, fees_pct=0.001):
        """
        Vend des actions
        
        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à vendre
            price: Prix de vente (None = prix actuel)
            fees_pct: Frais en % (défaut: 0.1%)
        
        Returns:
            Dict avec le résultat de la transaction
        """
        # Vérifier les holdings
        holdings = self.db.get_holdings()
        holding = next((h for h in holdings if h['symbol'] == symbol.upper()), None)
        
        if not holding:
            return {'success': False, 'error': f'Aucune position pour {symbol}'}
        
        if holding['quantity'] < quantity:
            return {
                'success': False,
                'error': f'Quantité insuffisante. Disponible: {holding["quantity"]}, Demandé: {quantity}'
            }
        
        # Récupérer le prix actuel si non spécifié
        if price is None:
            price = self.get_current_price(symbol)
            if price is None:
                return {'success': False, 'error': 'Impossible de récupérer le prix'}
        
        # Calculer le montant de la vente
        revenue = quantity * price
        fees = revenue * fees_pct
        net_revenue = revenue - fees
        
        # Calculer le P&L
        avg_cost = holding['avg_price'] * quantity
        pnl = net_revenue - avg_cost
        pnl_pct = (pnl / avg_cost) * 100 if avg_cost != 0 else 0
        
        # Exécuter la transaction
        transaction_id = self.db.add_transaction(
            symbol=symbol,
            transaction_type='SELL',
            quantity=quantity,
            price=price,
            fees=fees,
            notes=f'Vente de {quantity} actions à ${price:.2f} (P&L: ${pnl:.2f})'
        )
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'revenue': revenue,
            'net_revenue': net_revenue,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'new_balance': self.db.get_cash_balance()
        }
    
    def get_portfolio_summary(self):
        """Retourne un résumé du portfolio"""
        cash = self.db.get_cash_balance()
        holdings = self.db.get_holdings()
        
        total_invested = 0
        total_current_value = 0
        positions = []
        
        for holding in holdings:
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_price = holding['avg_price']
            
            # Récupérer le prix actuel
            current_price = self.get_current_price(symbol)
            
            if current_price:
                invested = quantity * avg_price
                current_value = quantity * current_price
                pnl = current_value - invested
                pnl_pct = (pnl / invested) * 100 if invested != 0 else 0
                
                total_invested += invested
                total_current_value += current_value
                
                positions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'invested': invested,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
        
        total_portfolio_value = cash + total_current_value
        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested != 0 else 0
        
        return {
            'cash': cash,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_portfolio_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions': positions
        }
    
    def get_transaction_history(self, symbol=None, limit=100):
        """Retourne l'historique des transactions"""
        return self.db.get_transaction_history(symbol, limit)
    
    def reset_portfolio(self, initial_cash=100000.0):
        """Réinitialise le portfolio"""
        self.db.reset_portfolio(initial_cash)
        return {'success': True, 'message': f'Portfolio réinitialisé avec ${initial_cash:.2f}'}
