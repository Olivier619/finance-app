"""
Module de gestion de la watchlist personnalisée
"""

import yfinance as yf
import pandas as pd
from database import Database


class Watchlist:
    def __init__(self):
        self.db = Database()
    
    def add_symbol(self, symbol, category='', notes=''):
        """Ajoute un symbole à la watchlist"""
        return self.db.add_to_watchlist(symbol, category, notes)
    
    def remove_symbol(self, symbol):
        """Retire un symbole de la watchlist"""
        self.db.remove_from_watchlist(symbol)
    
    def get_all_symbols(self):
        """Retourne tous les symboles de la watchlist"""
        return self.db.get_watchlist()
    
    def get_current_prices(self):
        """Récupère les prix actuels de tous les symboles"""
        watchlist = self.db.get_watchlist()
        
        if not watchlist:
            return []
        
        results = []
        
        for item in watchlist:
            symbol = item['symbol']
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    results.append({
                        'symbol': symbol,
                        'category': item['category'],
                        # Ne pas arrondir ici pour laisser le formatage intelligent faire son travail
                        'current_price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'added_date': item['added_date'],
                        'notes': item['notes']
                    })
                else:
                    results.append({
                        'symbol': symbol,
                        'category': item['category'],
                        'current_price': 'N/A',
                        'change': 0,
                        'change_pct': 0,
                        'added_date': item['added_date'],
                        'notes': item['notes']
                    })
            
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'category': item['category'],
                    'current_price': 'Error',
                    'change': 0,
                    'change_pct': 0,
                    'added_date': item['added_date'],
                    'notes': item['notes']
                })
        
        return results
    
    def export_to_csv(self, filename='watchlist.csv'):
        """Exporte la watchlist en CSV"""
        data = self.get_current_prices()
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            return True
        
        return False
    
    def import_from_csv(self, filename='watchlist.csv'):
        """Importe une watchlist depuis un CSV"""
        try:
            df = pd.read_csv(filename)
            
            for _, row in df.iterrows():
                symbol = row.get('symbol', '')
                category = row.get('category', '')
                notes = row.get('notes', '')
                
                if symbol:
                    self.add_symbol(symbol, category, notes)
            
            return True
        except Exception as e:
            return False
