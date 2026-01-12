# Finance App - Enhanced Version

Application complète d'analyse financière avec IA, gestion de portefeuille, et données fondamentales.

## 🚀 Nouvelles Fonctionnalités

### Phase 1: Analyse Technique Avancée ✅
- **Indicateurs supplémentaires** : Stochastique, VWAP, Ichimoku Cloud, Fibonacci
- **Détection de patterns** : 10+ patterns de chandeliers (Doji, Hammer, Engulfing, etc.)
- **Support/Résistance** : Identification automatique des niveaux clés
- **Volume Profile** : Analyse du volume par niveau de prix

### Phase 2: Gestion de Portefeuille ✅
- **Watchlist personnalisée** : Sauvegarde de vos actifs favoris
- **Alertes de prix** : Notifications par email
- **Paper Trading** : Simulation de trading avec portefeuille virtuel
- **Métriques de performance** : ROI, Sharpe Ratio, Max Drawdown, Win Rate

### Phase 3: Données Fondamentales ✅
- **News en temps réel** : Intégration NewsAPI et Finnhub
- **Calendrier économique** : Earnings, dividendes, événements Fed/BCE
- **Ratios financiers** : P/E, P/B, ROE, ROA, etc.
- **Analyse de sentiment** : Score de sentiment basé sur les news

### Phase 4: IA et ML Avancé ✅
- **Backtesting** : Test des prédictions sur historique
- **Modèles multiples** : Random Forest, XGBoost, LSTM
- **Explainability** : SHAP values pour comprendre les prédictions
- **Optimisation** : Hyperparameter tuning automatique

## 📦 Installation

```bash
# Cloner le repository
git clone <your-repo>
cd finance-app

# Installer les dépendances
pip install -r requirements.txt

# Configurer les API keys
cp .env.example .env
# Éditer .env et ajouter vos clés API
```

## 🔑 API Keys Requises

### Gratuites
- **NewsAPI** : https://newsapi.org/ (500 requêtes/jour)
- **Alpha Vantage** : https://www.alphavantage.co/ (25 requêtes/jour)
- **Finnhub** : https://finnhub.io/ (60 requêtes/minute)

### SMTP (pour les alertes email)
- Gmail, Outlook, ou autre service SMTP
- Pour Gmail : activer "Mots de passe d'application"

## 🚀 Lancement

### Version Streamlit (Recommandée)
```bash
streamlit run app_streamlit_v5.py
```

### Version Flask (API)
```bash
python app_flask.py
```

## 📁 Structure du Projet

```
finance-app/
├── app_streamlit_v5.py          # Application principale (nouvelle version)
├── app_streamlit.py             # Ancienne version
├── config.py                    # Configuration centralisée
├── utils.py                     # Fonctions utilitaires
├── database.py                  # Gestion SQLite
│
├── # Analyse Technique
├── technical_indicators.py      # Stochastique, VWAP, Ichimoku, Fibonacci
├── pattern_detection.py         # Détection de patterns de chandeliers
├── support_resistance.py        # Support/Résistance
├── volume_profile.py            # Volume Profile
│
├── # Gestion de Portefeuille
├── watchlist.py                 # Watchlist personnalisée
├── alerts.py                    # Système d'alertes
├── paper_trading.py             # Simulation de trading
├── performance_metrics.py       # Métriques de performance
│
├── # Données Fondamentales
├── news_fetcher.py              # Récupération de news
├── economic_calendar.py         # Calendrier économique
├── fundamental_data.py          # Ratios financiers
├── sentiment_analysis.py        # Analyse de sentiment
│
├── # IA et ML
├── backtesting.py               # Backtesting
├── ml_models.py                 # Random Forest, XGBoost, LSTM
├── model_explainability.py      # SHAP explainability
├── hyperparameter_tuning.py     # Optimisation des hyperparamètres
│
└── # Données
    ├── data/                    # Données sauvegardées
    ├── models/                  # Modèles ML sauvegardés
    └── finance_app.db           # Base de données SQLite
```

## 🎯 Utilisation

### 1. Analyse d'un Actif
- Sélectionner une catégorie (Actions, Secteurs, Commodities, Forex, Cryptos)
- Choisir un symbole
- Cliquer sur "Analyser"
- Voir les prédictions IA, indicateurs techniques, patterns, etc.

### 2. Watchlist
- Aller dans l'onglet "Watchlist"
- Ajouter des symboles
- Voir les prix en temps réel et variations

### 3. Alertes de Prix
- Onglet "Alertes"
- Configurer un seuil (au-dessus/en-dessous)
- Recevoir un email quand le prix atteint le seuil

### 4. Paper Trading
- Onglet "Portfolio"
- Acheter/Vendre des actifs avec de l'argent virtuel
- Suivre vos performances (ROI, Sharpe Ratio, etc.)

### 5. News et Sentiment
- Onglet "News"
- Voir les actualités récentes
- Score de sentiment global

### 6. Backtesting
- Onglet "Backtesting"
- Tester les prédictions sur données historiques
- Voir les métriques de performance

## ⚙️ Configuration Avancée

### Paramètres ML (config.py)
```python
ML_TRAIN_WINDOW = 252  # Fenêtre d'entraînement (jours)
ML_TEST_WINDOW = 20    # Fenêtre de test (jours)
ML_FEATURES = ['RSI', 'MACD', 'BB_Pct', 'ROC', 'ATR']
```

### Frais de Trading (config.py)
```python
DEFAULT_FEES_PCT = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
```

## 📊 Métriques de Performance

- **ROI** : Return on Investment
- **Sharpe Ratio** : Rendement ajusté au risque
- **Max Drawdown** : Perte maximale depuis un pic
- **Win Rate** : % de trades gagnants
- **Profit Factor** : Ratio gains/pertes

## 🤖 Modèles ML Disponibles

1. **Random Forest** : Rapide, robuste, bon par défaut
2. **XGBoost** : Plus performant, nécessite plus de ressources
3. **LSTM** : Deep Learning pour séries temporelles (expérimental)

## 🔧 Dépannage

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### Erreur SMTP
- Vérifier les identifiants dans `.env`
- Pour Gmail : utiliser un "Mot de passe d'application"

### Base de données corrompue
```bash
rm finance_app.db
# Relancer l'app, la DB sera recréée
```

## 📝 TODO / Améliorations Futures

- [ ] Intégration TradingView charts
- [ ] Support pour plus de cryptos
- [ ] Backtesting avec stratégies personnalisées
- [ ] Export des rapports en PDF
- [ ] Mode multi-utilisateurs
- [ ] API REST pour accès externe

## 📄 Licence

MIT License

## 👤 Auteur

Votre nom

## 🙏 Remerciements

- yfinance pour les données de marché
- Streamlit pour l'interface
- scikit-learn, XGBoost, TensorFlow pour le ML
- SHAP pour l'explainability
