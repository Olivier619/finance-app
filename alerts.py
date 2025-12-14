"""
Module de gestion des alertes de prix
Envoi d'emails via SMTP
"""

import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import Database
import os
from dotenv import load_dotenv

load_dotenv()


class AlertSystem:
    def __init__(self):
        self.db = Database()
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alert_email = os.getenv('ALERT_EMAIL', self.smtp_user)
    
    def create_alert(self, symbol, alert_type, threshold_price):
        """
        Crée une nouvelle alerte
        
        Args:
            symbol: Symbole de l'actif
            alert_type: 'ABOVE' ou 'BELOW'
            threshold_price: Prix seuil
        """
        # Récupérer le prix actuel
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
        except:
            current_price = None
        
        return self.db.create_alert(symbol, alert_type, threshold_price, current_price)
    
    def check_alerts(self):
        """Vérifie toutes les alertes actives et déclenche si nécessaire"""
        active_alerts = self.db.get_active_alerts()
        triggered_alerts = []
        
        for alert in active_alerts:
            symbol = alert['symbol']
            alert_type = alert['alert_type']
            threshold = alert['threshold_price']
            alert_id = alert['id']
            
            try:
                # Récupérer le prix actuel
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                
                if hist.empty:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Vérifier si l'alerte doit être déclenchée
                should_trigger = False
                
                if alert_type == 'ABOVE' and current_price >= threshold:
                    should_trigger = True
                elif alert_type == 'BELOW' and current_price <= threshold:
                    should_trigger = True
                
                if should_trigger:
                    # Déclencher l'alerte
                    self.db.trigger_alert(alert_id)
                    
                    # Envoyer l'email
                    self.send_alert_email(symbol, alert_type, threshold, current_price)
                    
                    triggered_alerts.append({
                        'symbol': symbol,
                        'type': alert_type,
                        'threshold': threshold,
                        'current_price': current_price
                    })
            
            except Exception as e:
                print(f"Erreur lors de la vérification de l'alerte pour {symbol}: {e}")
                continue
        
        return triggered_alerts
    
    def send_alert_email(self, symbol, alert_type, threshold, current_price):
        """Envoie un email d'alerte"""
        if not self.smtp_user or not self.smtp_password:
            print("Configuration SMTP manquante. Email non envoyé.")
            return False
        
        try:
            # Créer le message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            msg['Subject'] = f"🔔 Alerte Prix: {symbol}"
            
            # Corps du message
            direction = "au-dessus" if alert_type == 'ABOVE' else "en dessous"
            body = f"""
            Alerte de prix déclenchée !
            
            Symbole: {symbol}
            Type: Prix {direction} du seuil
            Seuil configuré: ${threshold:.2f}
            Prix actuel: ${current_price:.2f}
            
            Cette alerte a été automatiquement désactivée.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Envoyer l'email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"Email d'alerte envoyé pour {symbol}")
            return True
        
        except Exception as e:
            print(f"Erreur lors de l'envoi de l'email: {e}")
            return False
    
    def get_active_alerts(self):
        """Retourne toutes les alertes actives"""
        return self.db.get_active_alerts()
    
    def delete_alert(self, alert_id):
        """Supprime une alerte"""
        self.db.delete_alert(alert_id)
    
    def get_alert_history(self, limit=50):
        """Retourne l'historique des alertes déclenchées"""
        return self.db.get_alert_history(limit)
