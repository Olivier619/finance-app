"""
Module de gestion de la base de données SQLite
Tables: watchlist, alerts, portfolio_transactions, portfolio_holdings, portfolio_dividends, portfolio_cash
"""

import sqlite3
import os
from datetime import datetime
import json


class Database:
    def __init__(self, db_path='finance_app.db'):
        """Initialise la connexion à la base de données"""
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Retourne une connexion à la base de données"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Crée les tables si elles n'existent pas"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Table watchlist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    category TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')
            
            # Table alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    threshold_price REAL NOT NULL,
                    current_price REAL,
                    is_active INTEGER DEFAULT 1,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_date TIMESTAMP,
                    email_sent INTEGER DEFAULT 0
                )
            ''')
            
            # Table portfolio_transactions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    total_amount REAL NOT NULL,
                    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')
            
            # Table portfolio_holdings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table portfolio_dividends
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_dividends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    amount_per_share REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_amount REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_date)
                )
            ''')
            
            # Table portfolio_cash
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_cash (
                    id INTEGER PRIMARY KEY,
                    balance REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Initialiser le cash si vide
            cursor.execute('SELECT COUNT(*) FROM portfolio_cash')
            if cursor.fetchone()[0] == 0:
                cursor.execute('INSERT INTO portfolio_cash (id, balance) VALUES (1, 1000000.0)')
            
            conn.commit()
        finally:
            conn.close()
    
    # --- WATCHLIST ---
    
    def add_to_watchlist(self, symbol, category='', notes=''):
        """Ajoute un symbole à la watchlist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO watchlist (symbol, category, notes)
                VALUES (?, ?, ?)
            ''', (symbol.upper(), category, notes))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def remove_from_watchlist(self, symbol):
        """Retire un symbole de la watchlist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM watchlist WHERE symbol = ?', (symbol.upper(),))
            conn.commit()
        finally:
            conn.close()
    
    def get_watchlist(self):
        """Retourne tous les symboles de la watchlist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT symbol, category, added_date, notes FROM watchlist ORDER BY added_date DESC')
            results = cursor.fetchall()
            return [
                {'symbol': r[0], 'category': r[1], 'added_date': r[2], 'notes': r[3]}
                for r in results
            ]
        finally:
            conn.close()
    
    # --- ALERTS ---
    
    def create_alert(self, symbol, alert_type, threshold_price, current_price=None):
        """Crée une nouvelle alerte de prix"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (symbol, alert_type, threshold_price, current_price)
                VALUES (?, ?, ?, ?)
            ''', (symbol.upper(), alert_type, threshold_price, current_price))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_active_alerts(self):
        """Retourne toutes les alertes actives"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, symbol, alert_type, threshold_price, current_price, created_date
                FROM alerts
                WHERE is_active = 1
                ORDER BY created_date DESC
            ''')
            results = cursor.fetchall()
            return [
                {
                    'id': r[0], 'symbol': r[1], 'alert_type': r[2],
                    'threshold_price': r[3], 'current_price': r[4], 'created_date': r[5]
                }
                for r in results
            ]
        finally:
            conn.close()
    
    def trigger_alert(self, alert_id):
        """Marque une alerte comme déclenchée"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alerts
                SET is_active = 0, triggered_date = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (alert_id,))
            conn.commit()
        finally:
            conn.close()
    
    def delete_alert(self, alert_id):
        """Supprime une alerte"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM alerts WHERE id = ?', (alert_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_alert_history(self, limit=50):
        """Retourne l'historique des alertes déclenchées"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, alert_type, threshold_price, triggered_date
                FROM alerts
                WHERE is_active = 0
                ORDER BY triggered_date DESC
                LIMIT ?
            ''', (limit,))
            results = cursor.fetchall()
            return [
                {'symbol': r[0], 'alert_type': r[1], 'threshold': r[2], 'triggered': r[3]}
                for r in results
            ]
        finally:
            conn.close()
    
    # --- PORTFOLIO ---
    
    def add_transaction(self, symbol, transaction_type, quantity, price, fees=0, notes=''):
        """Ajoute une transaction au portfolio"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            total_amount = (quantity * price) + fees
            cursor.execute('''
                INSERT INTO portfolio_transactions
                (symbol, transaction_type, quantity, price, fees, total_amount, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol.upper(), transaction_type, quantity, price, fees, total_amount, notes))
            
            if transaction_type == 'BUY':
                self._update_holdings_buy(cursor, symbol.upper(), quantity, price)
                self._update_cash(cursor, -total_amount)
            elif transaction_type == 'SELL':
                self._update_holdings_sell(cursor, symbol.upper(), quantity, price)
                self._update_cash(cursor, total_amount - fees)
            
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def _update_holdings_buy(self, cursor, symbol, quantity, price):
        """Met à jour les holdings après un achat (ou rachat de short)"""
        cursor.execute('SELECT quantity, average_price FROM portfolio_holdings WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        
        if result:
            current_qty, current_avg = result
            new_qty = current_qty + quantity
            if current_qty < 0:
                if new_qty > 0:
                    new_avg = price
                elif new_qty == 0:
                    new_avg = 0
                else:
                    new_avg = current_avg
            else:
                new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty if new_qty != 0 else 0
            
            if new_qty == 0:
                cursor.execute('DELETE FROM portfolio_holdings WHERE symbol = ?', (symbol,))
            else:
                cursor.execute('''
                    UPDATE portfolio_holdings
                    SET quantity = ?, average_price = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (new_qty, new_avg, symbol))
        else:
            cursor.execute('''
                INSERT INTO portfolio_holdings (symbol, quantity, average_price)
                VALUES (?, ?, ?)
            ''', (symbol, quantity, price))

    def _update_holdings_sell(self, cursor, symbol, quantity, price):
        """Met à jour les holdings après une vente (ou vente à découvert)"""
        cursor.execute('SELECT quantity, average_price FROM portfolio_holdings WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        
        if result:
            current_qty, current_avg = result
            new_qty = current_qty - quantity
            if current_qty > 0:
                if new_qty < 0:
                    new_avg = price
                elif new_qty == 0:
                    new_avg = 0
                else:
                    new_avg = current_avg
            else:
                new_qty_abs = abs(current_qty)
                new_avg = ((new_qty_abs * current_avg) + (quantity * price)) / (new_qty_abs + quantity)
            
            if new_qty == 0:
                cursor.execute('DELETE FROM portfolio_holdings WHERE symbol = ?', (symbol,))
            else:
                cursor.execute('''
                    UPDATE portfolio_holdings
                    SET quantity = ?, average_price = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (new_qty, new_avg, symbol))
        else:
            cursor.execute('''
                INSERT INTO portfolio_holdings (symbol, quantity, average_price)
                VALUES (?, ?, ?)
            ''', (symbol, -quantity, price))
    
    def _update_cash(self, cursor, amount):
        """Met à jour le solde de cash"""
        cursor.execute('''
            UPDATE portfolio_cash
            SET balance = balance + ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = 1
        ''', (amount,))
    
    def get_cash_balance(self):
        """Retourne le solde de cash"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT balance FROM portfolio_cash WHERE id = 1')
            result = cursor.fetchone()
            return result[0] if result else 0.0
        finally:
            conn.close()
    
    def get_holdings(self):
        """Retourne tous les holdings"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, quantity, average_price, last_updated
                FROM portfolio_holdings
                ORDER BY symbol
            ''')
            results = cursor.fetchall()
            return [
                {'symbol': r[0], 'quantity': r[1], 'avg_price': r[2], 'last_updated': r[3]}
                for r in results
            ]
        finally:
            conn.close()
    
    def get_transaction_history(self, symbol=None, limit=100):
        """Retourne l'historique des transactions"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute('''
                    SELECT symbol, transaction_type, quantity, price, fees, total_amount, transaction_date
                    FROM portfolio_transactions
                    WHERE symbol = ?
                    ORDER BY transaction_date DESC
                    LIMIT ?
                ''', (symbol.upper(), limit))
            else:
                cursor.execute('''
                    SELECT symbol, transaction_type, quantity, price, fees, total_amount, transaction_date
                    FROM portfolio_transactions
                    ORDER BY transaction_date DESC
                    LIMIT ?
                ''', (limit,))
            results = cursor.fetchall()
            return [
                {
                    'symbol': r[0], 'type': r[1], 'quantity': r[2],
                    'price': r[3], 'fees': r[4], 'total': r[5], 'date': r[6]
                }
                for r in results
            ]
        finally:
            conn.close()
    
    def reset_portfolio(self, initial_cash=1000000.0):
        """Réinitialise le portfolio"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM portfolio_transactions')
            cursor.execute('DELETE FROM portfolio_holdings')
            cursor.execute('DELETE FROM portfolio_dividends')
            cursor.execute('UPDATE portfolio_cash SET balance = ?, last_updated = CURRENT_TIMESTAMP WHERE id = 1', (initial_cash,))
            conn.commit()
        finally:
            conn.close()

    # --- DIVIDENDS ---
    
    def add_dividend_payment(self, symbol, ex_date, amount_per_share, quantity, total_amount):
        """Enregistre un paiement de dividende"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_dividends (symbol, ex_date, amount_per_share, quantity, total_amount)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol.upper(), ex_date, amount_per_share, quantity, total_amount))
            self._update_cash(cursor, total_amount)
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
            
    def is_dividend_processed(self, symbol, ex_date):
        """Vérifie si un dividende a déjà été traité"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM portfolio_dividends WHERE symbol = ? AND ex_date = ?', (symbol.upper(), ex_date))
            result = cursor.fetchone()
            return result is not None
        finally:
            conn.close()
        
    def get_dividend_history(self, limit=50):
        """Retourne l'historique des dividendes"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, ex_date, amount_per_share, quantity, total_amount, timestamp
                FROM portfolio_dividends
                ORDER BY ex_date DESC
                LIMIT ?
            ''', (limit,))
            results = cursor.fetchall()
            return [
                {
                    'symbol': r[0], 'ex_date': r[1], 'amount_per_share': r[2], 
                    'quantity': r[3], 'total_amount': r[4], 'timestamp': r[5]
                }
                for r in results
            ]
        finally:
            conn.close()

    def get_total_dividends(self):
        """Retourne le montant total des dividendes perçus"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT SUM(total_amount) FROM portfolio_dividends')
            result = cursor.fetchone()
            return result[0] if result[0] else 0.0
        finally:
            conn.close()

    # --- BACKUP / RESTORE ---

    def export_data(self):
        """Exporte toutes les données du portfolio en dict"""
        return {
            'transactions': self.get_transaction_history(limit=1000),
            'holdings': self.get_holdings(),
            'cash': self.get_cash_balance(),
            'dividends': self.get_dividend_history(limit=1000)
        }

    def import_data(self, data):
        """Importe et écrase les données du portfolio"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM portfolio_transactions')
            cursor.execute('DELETE FROM portfolio_holdings')
            cursor.execute('DELETE FROM portfolio_dividends')
            cursor.execute('UPDATE portfolio_cash SET balance = ?, last_updated = CURRENT_TIMESTAMP WHERE id = 1', (data.get('cash', 1000000.0),))
            for h in data.get('holdings', []):
                cursor.execute('INSERT INTO portfolio_holdings (symbol, quantity, average_price, last_updated) VALUES (?, ?, ?, ?)', (h['symbol'], h['quantity'], h['avg_price'], h['last_updated']))
            for t in data.get('transactions', []):
                cursor.execute('INSERT INTO portfolio_transactions (symbol, transaction_type, quantity, price, fees, total_amount, transaction_date) VALUES (?, ?, ?, ?, ?, ?, ?)', (t['symbol'], t['type'], t['quantity'], t['price'], t['fees'], t['total'], t['date']))
            for d in data.get('dividends', []):
                cursor.execute('INSERT INTO portfolio_dividends (symbol, ex_date, amount_per_share, quantity, total_amount, timestamp) VALUES (?, ?, ?, ?, ?, ?)', (d['symbol'], d['ex_date'], d['amount_per_share'], d['quantity'], d['total_amount'], d['timestamp']))
            conn.commit()
            return True
        except Exception as e:
            print(f"Erreur lors de l'import : {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
