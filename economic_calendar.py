"""
Module de calendrier économique
Événements Fed, BCE, earnings, dividendes
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class EconomicCalendar:
    def __init__(self):
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
    
    def get_earnings_calendar(self, symbol):
        """
        Récupère le calendrier des earnings pour un symbole
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les dates d'earnings
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Récupérer les informations
            info = ticker.info
            
            earnings_data = {
                'symbol': symbol,
                'next_earnings_date': None,
                'last_earnings_date': None,
                'earnings_history': []
            }
            
            # Date du prochain earnings (si disponible)
            if 'earningsDate' in info and info['earningsDate']:
                earnings_dates = info['earningsDate']
                if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    earnings_data['next_earnings_date'] = earnings_dates[0].strftime('%Y-%m-%d')
            
            # Historique des earnings
            try:
                earnings_history = ticker.earnings_dates
                if earnings_history is not None and not earnings_history.empty:
                    earnings_data['earnings_history'] = [
                        {
                            'date': idx.strftime('%Y-%m-%d'),
                            'eps_estimate': row.get('Reported EPS', None),
                            'eps_actual': row.get('Surprise(%)', None)
                        }
                        for idx, row in earnings_history.head(10).iterrows()
                    ]
            except:
                pass
            
            return earnings_data
        
        except Exception as e:
            print(f"Erreur lors de la récupération des earnings pour {symbol}: {e}")
            return None
    
    def get_dividend_calendar(self, symbol):
        """
        Récupère le calendrier des dividendes pour un symbole
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les dates de dividendes
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            dividend_data = {
                'symbol': symbol,
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'ex_dividend_date': None,
                'dividend_history': []
            }
            
            # Date ex-dividende
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_date = datetime.fromtimestamp(info['exDividendDate'])
                dividend_data['ex_dividend_date'] = ex_date.strftime('%Y-%m-%d')
            
            # Historique des dividendes
            try:
                dividends = ticker.dividends
                if dividends is not None and not dividends.empty:
                    dividend_data['dividend_history'] = [
                        {
                            'date': idx.strftime('%Y-%m-%d'),
                            'amount': value
                        }
                        for idx, value in dividends.tail(10).items()
                    ]
            except:
                pass
            
            return dividend_data
        
        except Exception as e:
            print(f"Erreur lors de la récupération des dividendes pour {symbol}: {e}")
            return None
    
    def get_economic_events(self, days_ahead=30):
        """
        Récupère les événements économiques importants (Fed, BCE, etc.)
        Note: Nécessite une API externe ou scraping
        
        Args:
            days_ahead: Nombre de jours à l'avance (défaut: 30)
        
        Returns:
            List d'événements économiques
        """
        # Liste statique des événements récurrents (à titre d'exemple)
        # Dans une version production, utiliser une API comme TradingEconomics ou scraper un calendrier
        
        events = []
        
        # Exemple d'événements récurrents
        today = datetime.now()
        
        # FOMC Meetings (environ tous les 45 jours)
        fomc_dates = [
            datetime(2025, 1, 29),
            datetime(2025, 3, 19),
            datetime(2025, 5, 7),
            datetime(2025, 6, 18),
            datetime(2025, 7, 30),
            datetime(2025, 9, 17),
            datetime(2025, 11, 5),
            datetime(2025, 12, 17),
        ]
        
        for date in fomc_dates:
            if today <= date <= today + timedelta(days=days_ahead):
                events.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'event': 'FOMC Meeting',
                    'importance': 'HIGH',
                    'description': 'Federal Reserve FOMC Meeting - Interest Rate Decision'
                })
        
        # ECB Meetings
        ecb_dates = [
            datetime(2025, 1, 30),
            datetime(2025, 3, 13),
            datetime(2025, 4, 17),
            datetime(2025, 6, 5),
            datetime(2025, 7, 24),
            datetime(2025, 9, 11),
            datetime(2025, 10, 30),
            datetime(2025, 12, 18),
        ]
        
        for date in ecb_dates:
            if today <= date <= today + timedelta(days=days_ahead):
                events.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'event': 'ECB Meeting',
                    'importance': 'HIGH',
                    'description': 'European Central Bank Governing Council Meeting'
                })
        
        # Trier par date
        events.sort(key=lambda x: x['date'])
        
        return events
    
    def get_upcoming_events(self, symbol, days_ahead=30):
        """
        Récupère tous les événements à venir pour un symbole
        
        Args:
            symbol: Symbole de l'actif
            days_ahead: Nombre de jours à l'avance (défaut: 30)
        
        Returns:
            Dict avec tous les événements
        """
        upcoming = {
            'symbol': symbol,
            'earnings': None,
            'dividends': None,
            'economic_events': []
        }
        
        # Earnings
        earnings = self.get_earnings_calendar(symbol)
        if earnings and earnings.get('next_earnings_date'):
            next_date = datetime.strptime(earnings['next_earnings_date'], '%Y-%m-%d')
            if next_date <= datetime.now() + timedelta(days=days_ahead):
                upcoming['earnings'] = earnings
        
        # Dividendes
        dividends = self.get_dividend_calendar(symbol)
        if dividends and dividends.get('ex_dividend_date'):
            ex_date = datetime.strptime(dividends['ex_dividend_date'], '%Y-%m-%d')
            if ex_date <= datetime.now() + timedelta(days=days_ahead):
                upcoming['dividends'] = dividends
        
        # Événements économiques
        upcoming['economic_events'] = self.get_economic_events(days_ahead)
        
        return upcoming
    
    def get_events_summary(self, symbol, days_ahead=30):
        """
        Retourne un résumé textuel des événements à venir
        
        Args:
            symbol: Symbole de l'actif
            days_ahead: Nombre de jours à l'avance
        
        Returns:
            str: Résumé textuel
        """
        events = self.get_upcoming_events(symbol, days_ahead)
        
        summary = f"Événements à venir pour {symbol} ({days_ahead} prochains jours):\n\n"
        
        # Earnings
        if events['earnings']:
            next_date = events['earnings'].get('next_earnings_date')
            if next_date:
                summary += f"📊 Earnings: {next_date}\n"
        
        # Dividendes
        if events['dividends']:
            ex_date = events['dividends'].get('ex_dividend_date')
            dividend_rate = events['dividends'].get('dividend_rate', 0)
            if ex_date:
                summary += f"💰 Ex-Dividend Date: {ex_date} (${dividend_rate:.2f})\n"
        
        # Événements économiques
        if events['economic_events']:
            summary += f"\n🌍 Événements Économiques:\n"
            for event in events['economic_events'][:5]:
                summary += f"  - {event['date']}: {event['event']} ({event['importance']})\n"
        
        return summary
