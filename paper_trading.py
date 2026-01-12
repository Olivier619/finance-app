"""
Module de simulation de trading (Paper Trading)
Gestion du portefeuille virtuel
"""

import yfinance as yf
from database import Database
from datetime import datetime


class PaperTrading:
    VERSION = "1.1.0"
    
    def __init__(self, initial_cash=1000000.0):
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
        holdings = self.db.get_holdings()
        holding = next((h for h in holdings if h['symbol'] == symbol.upper()), None)
        
        trans_type = 'BUY'
        notes = f'Achat de {quantity} actions à ${price:.2f}'
        
        if holding and holding['quantity'] < 0:
            notes = f'Rachat (Cover) de {quantity} actions à ${price:.2f}'
        
        transaction_id = self.db.add_transaction(
            symbol=symbol,
            transaction_type=trans_type,
            quantity=quantity,
            price=price,
            fees=fees,
            notes=notes
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
        Vend des actions (permet la vente à découvert)
        """
        # Vérifier les holdings (pour distinction long/short)
        holdings = self.db.get_holdings()
        holding = next((h for h in holdings if h['symbol'] == symbol.upper()), None)
        
        is_shorting = True
        if holding and holding['quantity'] > 0:
            is_shorting = False
        
        # Récupérer le prix actuel si non spécifié
        if price is None:
            price = self.get_current_price(symbol)
            if price is None:
                return {'success': False, 'error': 'Impossible de récupérer le prix'}
        
        # Calculer le montant de la vente
        revenue = quantity * price
        fees = revenue * fees_pct
        net_revenue = revenue - fees
        
        # Calculer le P&L si on vend une position longue
        pnl = 0
        pnl_pct = 0
        notes = f'Vente à découvert de {quantity} actions à ${price:.2f}'
        
        if not is_shorting and holding:
            sell_qty = min(holding['quantity'], quantity)
            avg_cost = holding['avg_price'] * sell_qty
            revenue_pnl = sell_qty * price
            pnl = revenue_pnl - avg_cost - fees * (sell_qty/quantity)
            pnl_pct = (pnl / avg_cost) * 100 if avg_cost != 0 else 0
            notes = f'Vente de {quantity} actions à ${price:.2f} (P&L: ${pnl:.2f})'
            if quantity > holding['quantity']:
                notes += f' (dont {quantity - holding["quantity"]} en short)'

        # Exécuter la transaction
        transaction_id = self.db.add_transaction(
            symbol=symbol,
            transaction_type='SELL',
            quantity=quantity,
            price=price,
            fees=fees,
            notes=notes
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
                
                # Pour le P&L, la formule (current - invested) fonctionne pour les shorts
                pnl = current_value - invested
                pnl_pct = (pnl / abs(invested)) * 100 if invested != 0 else 0
                
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
        total_pnl_pct = (total_pnl / abs(total_invested)) * 100 if total_invested != 0 else 0
        
        # Ajouter les dividendes au résumé
        total_dividends = self.db.get_total_dividends()
        
        return {
            'cash': cash,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_portfolio_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_dividends': total_dividends,
            'positions': positions
        }
    
    def get_transaction_history(self, symbol=None, limit=100):
        """Retourne l'historique des transactions"""
        return self.db.get_transaction_history(symbol, limit)
    
    def reset_portfolio(self, initial_cash=1000000.0):
        """Réinitialise le portfolio"""
        self.db.reset_portfolio(initial_cash)
        return {'success': True, 'message': f'Portfolio réinitialisé avec ${initial_cash:.2f}'}

    def process_dividends(self):
        """Vérifie et crédite les dividendes (ou débite pour les positions shorts)"""
        history = self.db.get_transaction_history(limit=5000)
        symbols = list(set([t['symbol'] for t in history]))
        
        if not symbols:
            return 0
            
        count = 0
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                divs = ticker.dividends
                
                if divs.empty:
                    continue
                    
                for date, amount in divs.items():
                    date_str = date.strftime('%Y-%m-%d')
                    ex_date_dt = datetime.strptime(date_str + " 23:59:59", '%Y-%m-%d %H:%M:%S')
                    
                    if self.db.is_dividend_processed(symbol, date_str):
                        continue
                        
                    # Quantité détenue à l'ex-date
                    qty_at_date = 0
                    for t in history:
                        t_date = datetime.strptime(t['date'], '%Y-%m-%d %H:%M:%S')
                        if t['symbol'] == symbol and t_date < ex_date_dt:
                            if t['type'] == 'BUY':
                                qty_at_date += t['quantity']
                            else:
                                qty_at_date -= t['quantity']
                    
                    if qty_at_date != 0:
                        total_amount = qty_at_date * amount
                        self.db.add_dividend_payment(
                            symbol=symbol,
                            ex_date=date_str,
                            amount_per_share=amount,
                            quantity=qty_at_date,
                            total_amount=total_amount
                        )
                        count += 1
            except Exception as e:
                print(f"Erreur dividendes pour {symbol}: {e}")
                continue
        return count

    def get_dividend_history(self, limit=50):
        """Retourne l'historique des dividendes perçus"""
        return self.db.get_dividend_history(limit)

    def export_portfolio(self):
        """Exporte les données pour sauvegarde"""
        return self.db.export_data()

    def import_portfolio(self, data):
        """Importe des données de sauvegarde"""
        return self.db.import_data(data)