# 🚀 Guide de Démarrage Rapide - Market Predictor Pro V5

## ✅ L'application est maintenant lancée !

**URL locale** : http://localhost:8502

---

## 📱 Interface de l'Application

L'application V5 dispose de **6 onglets principaux** :

### 1. 📊 **Analyse**
- Prédictions IA (4 horizons : 1j, 5j, 20j, 252j)
- Indicateurs techniques avancés (Stochastique, VWAP, Ichimoku)
- Patterns de chandeliers détectés automatiquement
- Niveaux de support/résistance
- Graphique interactif avec annotations

### 2. ⭐ **Watchlist**
- Ajouter/supprimer des symboles favoris
- Prix en temps réel
- Variations % journalières
- Catégorisation (Actions, Crypto, Forex, Commodities)

### 3. 🔔 **Alertes**
- Créer des alertes de prix (au-dessus/en-dessous d'un seuil)
- Notifications par email (si SMTP configuré)
- Historique des alertes déclenchées
- Gestion des alertes actives

### 4. 💼 **Portfolio**
- Paper trading avec 100 000$ virtuels
- Acheter/vendre des actions
- Suivi des positions en temps réel
- Métriques de performance :
  - ROI (Return on Investment)
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor

### 5. 📰 **News & Sentiment**
- Actualités en temps réel (NewsAPI + Finnhub)
- Analyse de sentiment NLP (score 0-100)
- Calendrier économique (earnings, dividendes)
- Événements Fed/BCE

### 6. 🧪 **Backtesting**
- Tester les prédictions IA sur historique
- Métriques : Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Rapport détaillé

---

## 🎯 Utilisation Rapide

### Analyser un Actif
1. Dans la **sidebar** : sélectionner une catégorie
2. Choisir un symbole (ex: NVDA, Bitcoin, EUR/USD)
3. Cliquer sur **"⚡ Analyser"**
4. Explorer les résultats dans l'onglet **"Analyse"**

### Créer une Watchlist
1. Aller dans l'onglet **"Watchlist"**
2. Entrer un symbole (ex: AAPL)
3. Choisir une catégorie
4. Cliquer sur **"➕ Ajouter"**
5. Les prix se mettent à jour automatiquement

### Configurer une Alerte
1. Onglet **"Alertes"**
2. Entrer le symbole (ex: TSLA)
3. Choisir le type (ABOVE ou BELOW)
4. Définir le seuil de prix
5. Cliquer sur **"➕ Créer"**
6. Recevoir un email quand le prix atteint le seuil (si SMTP configuré)

### Paper Trading
1. Onglet **"Portfolio"**
2. Entrer un symbole
3. Choisir BUY ou SELL
4. Définir la quantité
5. Cliquer sur **"📊 Exécuter"**
6. Suivre vos performances en temps réel

---

## ⚙️ Configuration Optionnelle

### API Keys (pour fonctionnalités avancées)

Créer un fichier `.env` à la racine du projet :

```env
# News
NEWSAPI_KEY=votre_cle_newsapi

# Données de marché supplémentaires
ALPHAVANTAGE_API_KEY=votre_cle_alphavantage
FINNHUB_API_KEY=votre_cle_finnhub

# Alertes email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=votre_email@gmail.com
SMTP_PASSWORD=votre_mot_de_passe_app
ALERT_EMAIL=votre_email@gmail.com
```

**Sans ces clés, l'application fonctionne toujours** mais certaines fonctionnalités seront limitées :
- ❌ News en temps réel
- ❌ Alertes email
- ✅ Analyse technique (fonctionne)
- ✅ Prédictions IA (fonctionne)
- ✅ Paper trading (fonctionne)
- ✅ Watchlist (fonctionne)

---

## 🔧 Commandes Utiles

### Lancer l'application V5
```bash
.\venv\Scripts\python.exe -m streamlit run app_streamlit_v5.py
```

### Lancer l'ancienne version (V4)
```bash
.\venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

### Installer les nouvelles dépendances
```bash
pip install -r requirements.txt
```

### Réinitialiser la base de données
```bash
# Supprimer le fichier
rm finance_app.db
# Relancer l'app, la DB sera recréée automatiquement
```

---

## 📊 Modules Disponibles

Tous les modules peuvent être utilisés **indépendamment** dans vos propres scripts :

```python
# Exemple : Utiliser le module de patterns
from pattern_detection import detect_all_patterns
import yfinance as yf

df = yf.Ticker("AAPL").history(period="1y")
patterns = detect_all_patterns(df)
print(patterns)
```

```python
# Exemple : Paper trading
from paper_trading import PaperTrading

pt = PaperTrading(initial_cash=50000)
result = pt.buy("TSLA", quantity=10)
print(result)
```

---

## 🐛 Dépannage

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### Port 8502 déjà utilisé
L'application se lancera automatiquement sur le prochain port disponible (8503, 8504, etc.)

### Données non disponibles
- Vérifier la connexion Internet
- Vérifier que le symbole est correct (ex: AAPL, BTC-USD, EURUSD=X)
- Certains symboles nécessitent des suffixes spéciaux

### Erreur SMTP
- Vérifier les identifiants dans `.env`
- Pour Gmail : utiliser un "Mot de passe d'application" (pas le mot de passe normal)
- Activer "Autoriser les applications moins sécurisées" ou utiliser OAuth2

---

## 📈 Prochaines Étapes Recommandées

1. **Tester toutes les fonctionnalités** dans chaque onglet
2. **Configurer les API keys** pour débloquer les fonctionnalités avancées
3. **Créer une watchlist** avec vos actifs favoris
4. **Simuler du trading** pour tester vos stratégies
5. **Analyser le sentiment** des news pour vos actifs

---

## 💡 Astuces

- **Shift + R** : Recharger l'application
- **Ctrl + C** dans le terminal : Arrêter l'application
- Les données sont **mises en cache** pendant 5 minutes pour améliorer les performances
- La **base de données SQLite** stocke votre watchlist, alertes, et portfolio
- Tous les **modèles ML** peuvent être sauvegardés et rechargés

---

## 🎉 Profitez de l'Application !

Vous avez maintenant accès à une plateforme complète d'analyse financière avec :
- ✅ 17 modules professionnels
- ✅ Analyse technique avancée
- ✅ IA/ML de pointe
- ✅ Gestion de portefeuille
- ✅ News et sentiment
- ✅ Backtesting

**Bon trading ! 📈**
