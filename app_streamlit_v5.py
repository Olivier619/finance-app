"""
Application Streamlit V5 : Version Complète avec Toutes les Améliorations
Interface à onglets : Analyse, Watchlist, Alertes, Portfolio, News, Backtesting
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

# Imports des modules créés
from config import *
from utils import *
from database import Database
from watchlist import Watchlist
from alerts import AlertSystem
from paper_trading import PaperTrading
from performance_metrics import PerformanceMetrics
from news_fetcher import NewsFetcher
from economic_calendar import EconomicCalendar
from fundamental_data import FundamentalData
from sentiment_analysis import SentimentAnalysis
from technical_indicators import calculate_all_advanced_indicators, get_stochastic_signal, get_ichimoku_signal
from pattern_detection import detect_all_patterns, get_recent_patterns
from support_resistance import identify_support_resistance, analyze_price_action
from volume_profile import analyze_volume_profile
from backtesting import Backtesting
from ml_models import MLModels
from model_explainability import ModelExplainability
from trading_strategies import get_all_strategies, compare_strategies, MACrossover, RSIStrategy, BollingerBandsStrategy, MACDStrategy, CoupledStrategy
from strategy_backtester import StrategyBacktester

# Fonctions IA (inline pour éviter conflits d'import)
def preparer_donnees_ia_avance(df):
    """Prépare les données pour l'IA"""
    data = df.copy()
    macd = ta.macd(data['close'])
    if macd is not None:
        data = pd.concat([data, macd], axis=1)
        data['MACD'] = data[macd.columns[0]]
    data['RSI'] = ta.rsi(data['close'], length=14)
    bb = ta.bbands(data['close'], length=20, std=2)
    if bb is not None:
        data['BB_Pct'] = (data['close'] - bb[bb.columns[0]]) / (bb[bb.columns[2]] - bb[bb.columns[0]])
    data['ROC'] = ta.roc(data['close'], length=10)
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    return data

def calculer_toutes_probabilites(df_full):
    """Calcule les probabilités de hausse pour différents horizons"""
    features = ['RSI', 'MACD', 'BB_Pct', 'ROC', 'ATR']
    if not all(col in df_full.columns for col in features): 
        return {}
    last_row = df_full.iloc[[-1]][features]
    if last_row.isna().any().any(): 
        return {} 

    horizons = {'Demain (1j)': 1, 'Semaine (5j)': 5, 'Mois (20j)': 20, 'Année (252j)': 252}
    resultats = {}
    
    for nom, shift in horizons.items():
        df_train = df_full.copy()
        df_train['Target'] = (df_train['close'].shift(-shift) > df_train['close']).astype(int)
        df_train = df_train.dropna(subset=features + ['Target'])
        
        if len(df_train) > 200: 
            X = df_train[features]
            y = df_train['Target']
            if len(y.unique()) < 2:
                prob = 99.0 if y.iloc[0] == 1 else 1.0
            else:
                try:
                    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
                    model.fit(X, y)
                    prob = model.predict_proba(last_row)[0][1] * 100
                except:
                    prob = 50.0
            resultats[nom] = prob
        else:
            resultats[nom] = 50.0
    return resultats

# Configuration de la page
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# CSS personnalisé
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 300px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation des objets
@st.cache_resource
def init_objects():
    return {
        'db': Database(),
        'watchlist': Watchlist(),
        'alerts': AlertSystem(),
        'paper_trading': PaperTrading(),
        'performance': PerformanceMetrics(),
        'news': NewsFetcher(),
        'calendar': EconomicCalendar(),
        'fundamentals': FundamentalData(),
        'sentiment': SentimentAnalysis(),
        'backtesting': Backtesting(),
        'ml_models': MLModels(),
        'explainability': ModelExplainability()
    }

objects = init_objects()

# Sidebar
with st.sidebar:
    st.header("🔎 Configuration")
    
    # Sélection de catégorie
    categorie = st.selectbox("Catégorie", list(CATEGORIES.keys()))
    
    # Sélection du symbole
    if categorie == "Actions":
        symbol_input = st.text_input("Ticker", "NVDA").upper()
    else:
        choix = st.selectbox("Sélection", CATEGORIES[categorie])
        symbol_input = SECTEURS_ETF.get(choix, choix)
    
    st.caption(f"Symbole : **{symbol_input}**")
    asset_name = get_asset_name(symbol_input)
    if asset_name and asset_name != symbol_input:
        st.write(f"🏢 **{asset_name}**")
    
    # Bouton d'analyse
    analyser = st.button("⚡ Analyser", type="primary", use_container_width=True)

# Titre principal
st.title(f"{APP_ICON} {APP_TITLE}")

# Onglets principaux
# Onglets principaux
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Analyse", 
    "⭐ Watchlist", 
    "🔔 Alertes", 
    "💼 Portfolio", 
    "📰 News & Sentiment",
    "🧪 AI Backtesting",
    "📈 Stratégies",
    "💎 Synthèse Globale"
])

# ===== ONGLET 1: ANALYSE =====
with tab1:
    if analyser:
        with st.spinner(f"Analyse de {symbol_input}..."):
            # Nom de l'actif
            asset_name = get_asset_name(symbol_input)
            st.header(f"📊 {asset_name if asset_name else symbol_input} ({symbol_input})")
            
            # Récupérer les données
            df = cached_fetch_data(symbol_input, period='10y')
            
            if not df.empty and len(df) > 50:
                # Calculer les indicateurs avancés
                df_enhanced, fib_levels = calculate_all_advanced_indicators(df)
                
                # Prédictions IA
                df_ia = preparer_donnees_ia_avance(df)
                probs = calculer_toutes_probabilites(df_ia)
                
                # Section 1: Prédictions IA
                st.subheader("🔮 Prédictions de Tendance")
                c1, c2, c3, c4 = st.columns(4)
                
                if probs:
                    keys = list(probs.keys())
                    for i, col in enumerate([c1, c2, c3, c4]):
                        if i < len(keys):
                            val = probs[keys[i]]
                            col.metric(keys[i], f"{val:.1f}%", f"{val-50:.1f}%")
                            col.progress(int(val))
                            if val >= 60:
                                col.success("HAUSSIER")
                            elif val <= 40:
                                col.error("BAISSIER")
                            else:
                                col.warning("NEUTRE")
                
                st.divider()
                
                # Section 2: Indicateurs Techniques Avancés
                st.subheader("📈 Indicateurs Techniques Avancés")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Stochastique**")
                    if 'stoch_k' in df_enhanced.columns:
                        stoch_k = df_enhanced['stoch_k'].iloc[-1]
                        stoch_d = df_enhanced['stoch_d'].iloc[-1]
                        st.metric("%K", f"{stoch_k:.2f}")
                        st.metric("%D", f"{stoch_d:.2f}")
                        signal = get_stochastic_signal(stoch_k, stoch_d)
                        if signal == 'BUY':
                            st.success("Signal: ACHAT")
                        elif signal == 'SELL':
                            st.error("Signal: VENTE")
                        else:
                            st.info("Signal: NEUTRE")
                
                with col2:
                    st.write("**VWAP**")
                    if 'vwap' in df_enhanced.columns:
                        vwap = df_enhanced['vwap'].iloc[-1]
                        current_price = df_enhanced['close'].iloc[-1]
                        st.metric("VWAP", format_currency(vwap))
                        diff = ((current_price - vwap) / vwap) * 100
                        st.metric("Prix vs VWAP", f"{diff:+.2f}%")
                
                with col3:
                    st.write("**Ichimoku**")
                    if all(col in df_enhanced.columns for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
                        signal = get_ichimoku_signal(
                            df_enhanced['close'].iloc[-1],
                            df_enhanced['tenkan_sen'].iloc[-1],
                            df_enhanced['kijun_sen'].iloc[-1],
                            df_enhanced['senkou_span_a'].iloc[-1],
                            df_enhanced['senkou_span_b'].iloc[-1]
                        )
                        if signal == 'BULLISH':
                            st.success("Tendance: HAUSSIÈRE")
                        elif signal == 'BEARISH':
                            st.error("Tendance: BAISSIÈRE")
                        else:
                            st.info("Tendance: NEUTRE")
                
                st.divider()
                
                # Section 3: Patterns de Chandeliers
                st.subheader("🕯️ Patterns de Chandeliers Récents")
                patterns = get_recent_patterns(df_enhanced, lookback_days=30)
                
                if patterns:
                    pattern_df = pd.DataFrame(patterns)
                    pattern_df['timestamp'] = pd.to_datetime(pattern_df['timestamp'])
                    pattern_df = pattern_df.sort_values('timestamp', ascending=False)
                    
                    for _, pattern in pattern_df.head(5).iterrows():
                        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                        col1.write(pattern['timestamp'].strftime('%Y-%m-%d'))
                        col2.write(f"**{pattern['pattern']}**")
                        
                        if pattern['type'] == 'Bullish':
                            col3.success(pattern['type'])
                        elif pattern['type'] == 'Bearish':
                            col3.error(pattern['type'])
                            col3.info(pattern['type'])
                            
                    # Légende explication
                    with st.expander("📚 Comprendre les Patterns de Chandeliers"):
                        st.markdown("""
                        **Patterns Haussiers (Bullish) 🟢**
                        - **Hammer (Marteau)** : Petit corps en haut, longue mèche basse. Indique que les acheteurs reprennent le contrôle après une baisse.
                        - **Engulfing Bullish** : Une grosse bougie verte avale totalement la petite bougie rouge précédente. Signe fort de retournement.
                        - **Morning Star** : Une bougie rouge, un doji/toupie, puis une bougie verte. Le soleil se lève sur la tendance !
                        
                        **Patterns Baissiers (Bearish) 🔴**
                        - **Shooting Star (Étoile filante)** : Petit corps en bas, longue mèche haute. Les vendeurs rejettent la hausse.
                        - **Engulfing Bearish** : Une grosse bougie rouge avale la verte précédente. Chute probable.
                        - **Evening Star** : Une bougie verte, un doji, puis une bougie rouge. Le soleil se couche, la nuit tombe.
                        
                        **Patterns d'Indécision ⚪**
                        - **Doji** : Le prix d'ouverture est égal au prix de fermeture (forme de croix). Le marché hésite.
                        - **Spinning Top (Toupie)** : Petit corps centré. Les deux camps s'affrontent sans vainqueur.
                        """)
                        
                        col4.write(f"{pattern['confidence']:.0%}")
                else:
                    st.info("Aucun pattern détecté récemment")
                
                st.divider()
                
                # Section 4: Support/Résistance
                st.subheader("📍 Support & Résistance")
                
                with st.expander("ℹ️ Qu'est-ce que Support et Résistance ?", expanded=False):
                    st.markdown("""
                    **Support** : Niveau de prix où la demande est forte, empêchant le prix de baisser davantage.
                    - Le prix "rebondit" souvent sur un support
                    - Si le support est cassé, il peut devenir une résistance
                    
                    **Résistance** : Niveau de prix où l'offre est forte, empêchant le prix de monter davantage.
                    - Le prix "plafonne" souvent à une résistance
                    - Si la résistance est cassée, elle peut devenir un support
                    
                    **Force** : Nombre de fois où le prix a touché ce niveau (plus c'est élevé, plus le niveau est important)
                    """)
                
                levels = identify_support_resistance(df_enhanced)
                analysis = analyze_price_action(df_enhanced, levels)
                
                current_price = analysis['current_price']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔴 Résistances** (au-dessus du prix)")
                    resistances = [r for r in levels['resistance'] if r['level'] > current_price]
                    
                    if resistances:
                        for r in resistances[:3]:
                            distance = ((r['level'] - current_price) / current_price) * 100
                            formatted_level = format_currency(r['level'])
                            st.write(f"{formatted_level} - Force: {r['strength']} - (+{distance:.1f}%)")
                    else:
                        st.info("Aucune résistance proche détectée")
                
                with col2:
                    st.write("**🟢 Supports** (en-dessous du prix)")
                    supports = [s for s in levels['support'] if s['level'] < current_price]
                    
                    if supports:
                        for s in supports[:3]:
                            distance = ((current_price - s['level']) / current_price) * 100
                            formatted_level = format_currency(s['level'])
                            st.write(f"{formatted_level} - Force: {s['strength']} - (-{distance:.1f}%)")
                    else:
                        st.info("Aucun support proche détecté")
                
                # Afficher le prix actuel et la tendance
                trend_color = "🟢" if analysis['trend'] == 'BULLISH' else "🔴" if analysis['trend'] == 'BEARISH' else "⚪"
                formatted_price = format_currency(current_price)
                st.info(f"{trend_color} **Prix actuel: {formatted_price}** - Tendance: {analysis['trend']}")
                
                # Explication de la tendance
                if analysis['trend'] == 'BULLISH':
                    st.success("📈 Le prix est au-dessus de la résistance la plus proche → Tendance haussière")
                elif analysis['trend'] == 'BEARISH':
                    st.error("📉 Le prix est en-dessous du support le plus proche → Tendance baissière")
                else:
                    st.warning("➡️ Le prix est entre support et résistance → Tendance neutre")
                
                st.divider()
                
                # Section 5: Graphique Principal
                st.subheader("📊 Graphique Technique")
                
                fig = go.Figure()
                
                # Chandeliers
                fig.add_trace(go.Candlestick(
                    x=df_enhanced['timestamp'],
                    open=df_enhanced['open'],
                    high=df_enhanced['high'],
                    low=df_enhanced['low'],
                    close=df_enhanced['close'],
                    name='Prix'
                ))
                
                # VWAP
                if 'vwap' in df_enhanced.columns:
                    fig.add_trace(go.Scatter(
                        x=df_enhanced['timestamp'],
                        y=df_enhanced['vwap'],
                        line=dict(color='orange', width=1),
                        name='VWAP'
                    ))
                
                # Support/Résistance
                current_price = df_enhanced['close'].iloc[-1]
                for r in levels['resistance'][:3]:
                    if abs(r['level'] - current_price) / current_price < 0.1:  # Dans 10%
                        fig.add_hline(y=r['level'], line_dash="dash", line_color="red", 
                                     annotation_text=f"R: ${r['level']:.2f}")
                
                for s in levels['support'][:3]:
                    if abs(s['level'] - current_price) / current_price < 0.1:
                        fig.add_hline(y=s['level'], line_dash="dash", line_color="green",
                                     annotation_text=f"S: ${s['level']:.2f}")
                
                # Configuration
                last_date = df_enhanced['timestamp'].iloc[-1]
                one_year_ago = last_date - timedelta(days=365)
                
                fig.update_layout(
                    title=f"{symbol_input} - Analyse Technique",
                    template="plotly_dark",
                    height=600,
                    xaxis=dict(range=[one_year_ago, last_date], rangeslider=dict(visible=True)),
                    yaxis_title="Prix ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Données non disponibles ou insuffisantes")

# ===== ONGLET 2: WATCHLIST =====
with tab2:
    st.subheader("⭐ Ma Watchlist")
    
    # Ajouter un symbole
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_symbol = st.text_input("Ajouter un symbole", key="watchlist_symbol").upper()
    with col2:
        category = st.selectbox("Catégorie", ["Actions", "Crypto", "Forex", "Commodities"], key="watchlist_cat")
    with col3:
        if st.button("➕ Ajouter", use_container_width=True):
            if new_symbol:
                if objects['watchlist'].add_symbol(new_symbol, category):
                    st.success(f"{new_symbol} ajouté !")
                    st.rerun()
                else:
                    st.warning(f"{new_symbol} déjà dans la watchlist")
    
    # Afficher la watchlist
    watchlist_data = objects['watchlist'].get_current_prices()
    
    if watchlist_data:
        df_watchlist = pd.DataFrame(watchlist_data)
        
        # Formater l'affichage
        for idx, row in df_watchlist.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            
            col1.write(f"**{row['symbol']}**")
            asset_name_wl = get_asset_name(row['symbol'])
            if asset_name_wl and asset_name_wl != row['symbol']:
                col1.caption(asset_name_wl)
            col2.write(row['category'])
            
            if row['current_price'] != 'N/A' and row['current_price'] != 'Error':
                col3.metric("Prix", format_currency(row['current_price']))
                
                change_pct = row['change_pct']
                if change_pct > 0:
                    col4.success(f"+{change_pct:.2f}%")
                elif change_pct < 0:
                    col4.error(f"{change_pct:.2f}%")
                else:
                    col4.info("0.00%")
            else:
                col3.write(row['current_price'])
            
            if col5.button("🗑️", key=f"del_{row['symbol']}"):
                objects['watchlist'].remove_symbol(row['symbol'])
                st.rerun()
    else:
        st.info("Votre watchlist est vide. Ajoutez des symboles ci-dessus.")

# ===== ONGLET 3: ALERTES =====
with tab3:
    st.subheader("🔔 Alertes de Prix")
    
    # Créer une alerte
    st.write("**Créer une nouvelle alerte**")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        alert_symbol = st.text_input("Symbole", key="alert_symbol").upper()
    with col2:
        alert_type = st.selectbox("Type", ["ABOVE", "BELOW"], key="alert_type")
    with col3:
        threshold = st.number_input("Seuil ($)", min_value=0.01, value=100.0, key="alert_threshold")
    with col4:
        if st.button("➕ Créer", use_container_width=True):
            if alert_symbol:
                objects['alerts'].create_alert(alert_symbol, alert_type, threshold)
                st.success("Alerte créée !")
                st.rerun()
    
    st.divider()
    
    # Alertes actives
    st.write("**Alertes Actives**")
    active_alerts = objects['alerts'].get_active_alerts()
    
    if active_alerts:
        for alert in active_alerts:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
            
            col1.write(f"**{alert['symbol']}**")
            asset_name_al = get_asset_name(alert['symbol'])
            if asset_name_al and asset_name_al != alert['symbol']:
                col1.caption(asset_name_al)
            col2.write(alert['alert_type'])
            col3.write(f"Seuil: {format_currency(alert['threshold_price'])}")
            col4.write(f"Créée: {alert['created_date'][:10]}")
            
            if col5.button("🗑️", key=f"del_alert_{alert['id']}"):
                objects['alerts'].delete_alert(alert['id'])
                st.rerun()
    else:
        st.info("Aucune alerte active")
    
    st.divider()
    
    # Historique
    st.write("**Historique des Alertes Déclenchées**")
    history = objects['alerts'].get_alert_history(limit=10)
    
    if history:
        for h in history:
            st.write(f"✅ {h['symbol']} - {h['alert_type']} {format_currency(h['threshold'])} - {h['triggered'][:10]}")
    else:
        st.info("Aucune alerte déclenchée")

# ===== ONGLET 4: PORTFOLIO =====
with tab4:
    st.subheader("💼 Paper Trading Portfolio")
    
    # Résumé du portfolio
    summary = objects['paper_trading'].get_portfolio_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cash", format_currency(summary['cash']))
    col2.metric("Valeur Investie", format_currency(summary['total_invested']))
    col3.metric("Valeur Actuelle", format_currency(summary['total_current_value']))
    
    pnl_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
    col4.metric("P&L Total", format_currency(summary['total_pnl']), 
                f"{summary['total_pnl_pct']:.2f}%", delta_color=pnl_color)
    
    # Dividendes cumulés
    st.write(f"💰 **Total Dividendes Perçus : {format_currency(summary['total_dividends'])}**")
    
    # Bouton de rafraîchissement des dividendes
    if st.button("🔄 Vérifier les Dividendes", help="Vérifie si de nouveaux dividendes ont été payés"):
        with st.spinner("Vérification des dividendes..."):
            count = objects['paper_trading'].process_dividends()
            if count > 0:
                st.success(f"{count} nouveaux dividendes crédités !")
                st.rerun()
            else:
                st.info("Aucun nouveau dividende à créditer.")
    
    st.divider()
    
    # Trading
    st.write("**Passer un Ordre**")
    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
    
    with col1:
        trade_symbol = st.text_input("Symbole", key="trade_symbol").upper()
    with col2:
        trade_type = st.selectbox("Type", ["BUY", "SELL"], key="trade_type")
    with col3:
        quantity = st.number_input("Quantité", min_value=1, value=1, key="trade_qty")
    with col4:
        if st.button("📊 Exécuter", use_container_width=True):
            if trade_symbol:
                if trade_type == "BUY":
                    result = objects['paper_trading'].buy(trade_symbol, quantity)
                else:
                    result = objects['paper_trading'].sell(trade_symbol, quantity)
                
                if result['success']:
                    st.success(f"Ordre exécuté ! Nouveau solde: {format_currency(result['new_balance'])}")
                    st.rerun()
                else:
                    st.error(result['error'])
    
    st.divider()
    
    # Positions
    st.write("**Positions Actuelles**")
    if summary['positions']:
        for pos in summary['positions']:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            col1.write(f"**{pos['symbol']}**")
            asset_name_pt = get_asset_name(pos['symbol'])
            if asset_name_pt and asset_name_pt != pos['symbol']:
                col1.caption(asset_name_pt)
            col2.write(f"{pos['quantity']:.0f} actions")
            col3.write(f"Avg: {format_currency(pos['avg_price'])}")
            col4.write(f"Now: {format_currency(pos['current_price'])}")
            
            if pos['pnl'] >= 0:
                col5.success(f"+{pos['pnl_pct']:.2f}%")
            else:
                col5.error(f"{pos['pnl_pct']:.2f}%")
    else:
        st.info("Aucune position ouverte")
    
    st.divider()
    
    # Historique des dividendes
    with st.expander("📝 Historique des Dividendes"):
        div_history = objects['paper_trading'].get_dividend_history()
        if div_history:
            df_divs = pd.DataFrame(div_history)
            st.dataframe(df_divs[['symbol', 'ex_date', 'amount_per_share', 'quantity', 'total_amount']], 
                        use_container_width=True, hide_index=True)
        else:
            st.info("Aucun dividende perçu pour le moment.")
            
    st.divider()
    
    # Métriques de performance
    st.write("**Métriques de Performance**")
    
    # Vérifier s'il y a des transactions
    transactions = objects['paper_trading'].get_transaction_history(limit=1)
    
    if transactions:
        metrics = objects['performance'].get_portfolio_metrics()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("ROI", f"{metrics['roi']:.2f}%", 
                   help="Return on Investment - Rendement total du portfolio")
        col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                   help="Rendement ajusté au risque (>1 = bon, >2 = excellent)")
        col3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%",
                   help="Perte maximale depuis un pic")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", metrics['total_trades'])
        col2.metric("Win Rate", f"{metrics['win_rate']:.2f}%",
                   help="Pourcentage de trades gagnants")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}",
                   help="Ratio gains/pertes (>1 = profitable)")
    else:
        st.info("💡 Effectuez des trades pour voir les métriques de performance")

    st.divider()
    
    # Sauvegarde et Restauration
    st.write("📂 **Sauvegarde & Restauration**")
    col_exp, col_imp = st.columns(2)
    
    with col_exp:
        # Export
        data_json = json.dumps(objects['paper_trading'].export_portfolio(), indent=2)
        st.download_button(
            label="💾 Exporter le Portfolio (JSON)",
            data=data_json,
            file_name=f"portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
        
    with col_imp:
        # Import
        uploaded_file = st.file_uploader("Restaurer un Portfolio", type="json")
        if uploaded_file is not None:
            if st.button("📂 Importer ce Portfolio", type="secondary", use_container_width=True):
                try:
                    import_data = json.load(uploaded_file)
                    if objects['paper_trading'].import_portfolio(import_data):
                        st.success("✅ Portfolio restauré avec succès !")
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors de l'importation.")
                except Exception as e:
                    st.error(f"❌ Fichier invalide : {e}")

# ===== ONGLET 5: NEWS & SENTIMENT =====
with tab5:
    st.subheader("📰 News & Analyse de Sentiment")
    
    # Sélection du symbole
    news_symbol = st.text_input("Symbole pour les news", symbol_input).upper()
    
    if st.button("🔍 Rechercher News"):
        with st.spinner("Récupération des news..."):
            # News
            articles = objects['news'].get_news(news_symbol, days=7, max_articles=10)
            
            if articles:
                # Nom de l'actif
                asset_name_news = get_asset_name(news_symbol)
                st.write(f"📰 **{asset_name_news if asset_name_news else news_symbol}** ({news_symbol})")
                
                # Analyse de sentiment
                analyzed_articles = objects['sentiment'].analyze_news_batch(articles)
                sentiment_agg = objects['sentiment'].calculate_aggregate_sentiment(analyzed_articles)
                
                # Afficher le sentiment global
                st.write("**Sentiment Global**")
                col1, col2, col3 = st.columns(3)
                
                score = sentiment_agg['sentiment_score']
                col1.metric("Score de Sentiment", f"{score:.1f}/100")
                col2.metric("Label", sentiment_agg['sentiment_label'])
                col3.metric("Articles Analysés", sentiment_agg['total_articles'])
                
                # Jauge de sentiment
                st.progress(int(score))
                
                st.write(objects['sentiment'].interpret_sentiment(sentiment_agg))
                
                st.divider()
                
                # Afficher les articles
                st.write("**Articles Récents**")
                for article in analyzed_articles[:5]:
                    with st.expander(f"{article['title'][:80]}..."):
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Date:** {article['published_at'][:10]}")
                        st.write(f"**Sentiment:** {article['sentiment']['sentiment']} ({article['sentiment']['polarity']:.2f})")
                        st.write(article['description'])
                        st.write(f"[Lire l'article]({article['url']})")
            else:
                st.info("Aucune news trouvée")
    
    st.divider()
    
    # Calendrier économique
    st.write("**Calendrier Économique**")
    upcoming = objects['calendar'].get_upcoming_events(news_symbol, days_ahead=30)
    
    if upcoming['earnings']:
        st.write(f"📊 **Prochain Earnings:** {upcoming['earnings']['next_earnings_date']}")
    
    if upcoming['dividends']:
        st.write(f"💰 **Prochain Dividende:** {upcoming['dividends']['ex_dividend_date']} (${upcoming['dividends']['dividend_rate']:.2f})")
    
    if upcoming['economic_events']:
        st.write("**Événements Économiques à Venir:**")
        for event in upcoming['economic_events'][:5]:
            st.write(f"- {event['date']}: {event['event']} ({event['importance']})")

# ===== ONGLET 6: AI BACKTESTING =====
with tab6:
    st.subheader("🧪 Backtesting des Modèles IA")
    
    # Explications
    with st.expander("ℹ️ Qu'est-ce que le Backtesting ?", expanded=False):
        st.markdown("""
        Le **backtesting** permet de tester les performances d'un modèle de prédiction sur des données historiques.
        
        **Comment ça fonctionne ?**
        1. On divise l'historique en fenêtres d'entraînement et de test
        2. Le modèle apprend sur les données passées (fenêtre d'entraînement)
        3. Il fait des prédictions sur les données futures (fenêtre de test)
        4. On compare les prédictions avec la réalité
        
        **Métriques expliquées :**
        - **Accuracy** : % de prédictions correctes (hausse/baisse)
        - **Precision** : Quand le modèle prédit une hausse, à quelle fréquence a-t-il raison ?
        - **Recall** : Parmi toutes les hausses réelles, combien le modèle en a-t-il détectées ?
        - **F1-Score** : Moyenne harmonique de Precision et Recall (équilibre)
        
        **Matrice de Confusion :**
        - **True Positives (TP)** : Hausse prédite ✅ et hausse réelle ✅
        - **True Negatives (TN)** : Baisse prédite ✅ et baisse réelle ✅
        - **False Positives (FP)** : Hausse prédite ❌ mais baisse réelle
        - **False Negatives (FN)** : Baisse prédite ❌ mais hausse réelle
        
        **Interprétation :**
        - Accuracy > 55% : Le modèle bat le hasard (50%)
        - Accuracy > 60% : Très bon modèle
        - Accuracy > 70% : Excellent modèle (rare !)
        """)
    
    backtest_symbol = st.text_input("Symbole pour backtesting", symbol_input).upper()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 Le backtesting nécessite au moins 500 jours de données historiques (environ 2 ans)")
    with col2:
        run_backtest = st.button("🚀 Lancer Backtesting", type="primary", use_container_width=True)
    
    if run_backtest:
        with st.spinner("Backtesting en cours... Cela peut prendre 30-60 secondes"):
            # Récupérer les données
            df = cached_fetch_data(backtest_symbol, period='5y')
            
            if not df.empty and len(df) > 500:
                # Préparer les données
                df_ia = preparer_donnees_ia_avance(df)
                
                features = ['RSI', 'MACD', 'BB_Pct', 'ROC', 'ATR']
                
                # Entraîner un modèle Random Forest
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Backtesting
                results = objects['backtesting'].walk_forward_analysis(
                    df_ia, model, features, 
                    train_window=252, test_window=20, horizon=1
                )
                
                if results:
                    # Afficher les résultats
                    asset_name_bt = get_asset_name(backtest_symbol)
                    st.success(f"✅ Backtesting terminé pour **{asset_name_bt if asset_name_bt else backtest_symbol}** ({backtest_symbol}) ! {results['total_predictions']} prédictions testées")
                    
                    st.divider()
                    
                    # Métriques principales avec explications
                    st.write("**📊 Métriques de Performance**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    accuracy = results['accuracy']
                    col1.metric("Accuracy", f"{accuracy:.2%}", 
                               help="Pourcentage de prédictions correctes. >55% = bat le hasard, >60% = très bon")
                    
                    precision = results['precision']
                    col2.metric("Precision", f"{precision:.2%}",
                               help="Quand le modèle prédit une hausse, à quelle fréquence a-t-il raison ?")
                    
                    recall = results['recall']
                    col3.metric("Recall", f"{recall:.2%}",
                               help="Parmi toutes les hausses réelles, combien le modèle en a détecté ?")
                    
                    f1 = results['f1_score']
                    col4.metric("F1-Score", f"{f1:.2%}",
                               help="Équilibre entre Precision et Recall (moyenne harmonique)")
                    
                    # Interprétation
                    st.divider()
                    st.write("**💡 Interprétation**")
                    
                    if accuracy >= 0.70:
                        st.success("🌟 **Excellent modèle !** L'accuracy est supérieure à 70%, ce qui est rare et très performant.")
                    elif accuracy >= 0.60:
                        st.success("✅ **Très bon modèle !** L'accuracy est supérieure à 60%, le modèle est fiable.")
                    elif accuracy >= 0.55:
                        st.info("👍 **Bon modèle.** L'accuracy bat le hasard (50%), mais il y a une marge d'amélioration.")
                    else:
                        st.warning("⚠️ **Modèle peu fiable.** L'accuracy est proche du hasard. Essayez d'autres indicateurs ou périodes.")
                    
                    # Matrice de confusion avec explications
                    st.divider()
                    st.write("**📈 Matrice de Confusion**")
                    
                    cm = results['confusion_matrix']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Prédictions Positives (Hausse)**")
                        tp = cm[1][1]
                        fp = cm[0][1]
                        st.metric("✅ True Positives", tp, 
                                 help="Hausse prédite ET hausse réelle (CORRECT)")
                        st.metric("❌ False Positives", fp,
                                 help="Hausse prédite MAIS baisse réelle (ERREUR)")
                    
                    with col2:
                        st.write("**Prédictions Négatives (Baisse)**")
                        tn = cm[0][0]
                        fn = cm[1][0]
                        st.metric("✅ True Negatives", tn,
                                 help="Baisse prédite ET baisse réelle (CORRECT)")
                        st.metric("❌ False Negatives", fn,
                                 help="Baisse prédite MAIS hausse réelle (ERREUR)")
                    
                    # Statistiques supplémentaires
                    total_correct = tp + tn
                    total_incorrect = fp + fn
                    total = total_correct + total_incorrect
                    
                    st.info(f"**Résumé** : {total_correct} prédictions correctes sur {total} ({(total_correct/total)*100:.1f}%)")
                    
                    # Rapport complet
                    st.divider()
                    with st.expander("📄 Rapport Détaillé", expanded=False):
                        report = objects['backtesting'].generate_backtest_report(results)
                        st.text(report)
                else:
                    st.error("❌ Erreur lors du backtesting. Vérifiez que le symbole a suffisamment de données.")
            else:
                st.error("❌ Données insuffisantes pour le backtesting. Minimum requis : 500 jours (~2 ans d'historique)")
                st.info(f"Données disponibles : {len(df)} jours")

# ===== ONGLET 7: STRATÉGIES =====
with tab7:
    st.subheader("📈 Backtesting de Stratégies Trading")
    
    st.info("Comparez différentes stratégies techniques classiques sur des données historiques.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        strat_symbol = st.text_input("Symbole", symbol_input, key="strat_symbol").upper()
    with col2:
        capital = st.number_input("Capital Initial ($)", value=100000, step=1000)
    with col3:
        period_years = st.selectbox("Période", ["1y", "2y", "5y", "10y"], index=2)
    
    st.divider()
    
    col_config, col_run = st.columns([3, 1])
    
    with col_config:
        st.write("**Sélection des Stratégies**")
        strategies_to_test = []
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_ma = st.checkbox("MA Crossover", value=True)
            st.caption("📈 **Moyennes Mobiles**\nAchète quand la tendance court terme dépasse la tendance long terme (le prix accélère).")
            if use_ma:
                ma_short = st.number_input("MA Court", 10, 100, 50, help="Période courte (réactive)")
                ma_long = st.number_input("MA Long", 50, 365, 200, help="Période longue (tendance de fond)")
                strategies_to_test.append(MACrossover(ma_short, ma_long))
        
        with c2:
            use_rsi = st.checkbox("RSI Strategy", value=True)
            st.caption("🔄 **Surachat / Survente**\nAchète quand le prix a trop baissé (survendu) et commence à remonter.")
            if use_rsi:
                rsi_period = st.number_input("RSI Période", 5, 30, 14)
                rsi_low = st.number_input("Survente", 10, 40, 30, help="Niveau d'achat (<30)")
                rsi_high = st.number_input("Surachat", 60, 90, 70, help="Niveau de vente (>70)")
                strategies_to_test.append(RSIStrategy(rsi_period, rsi_low, rsi_high))
                
        with c3:
            use_bb = st.checkbox("Bollinger Bands", value=True)
            st.caption("💥 **Volatilité**\nAchète ou vend quand le prix sort de ses limites habituelles (bandes).")
            if use_bb:
                bb_period = st.number_input("BB Période", 10, 50, 20)
                bb_std = st.number_input("BB Std Dev", 1.0, 3.0, 2.0, help="Écart-type (largeur des bandes)")
                strategies_to_test.append(BollingerBandsStrategy(bb_period, bb_std))
                
        with c4:
            use_macd = st.checkbox("MACD", value=True)
            st.caption("🌊 **Momentum**\nCapture les changements de dynamique entre deux moyennes mobiles.")
            if use_macd:
                macd_fast = st.number_input("Fast", 5, 20, 12)
                macd_slow = st.number_input("Slow", 20, 50, 26)
                macd_sig = st.number_input("Signal", 5, 15, 9)
                strategies_to_test.append(MACDStrategy(macd_fast, macd_slow, macd_sig))
    
    with col_run:
        st.write("") # Spacer
        st.write("") # Spacer
        run_strat = st.button("🚀 Lancer Comparaison", type="primary", use_container_width=True)
    
    if run_strat and strategies_to_test:
        with st.spinner(f"Backtesting de {len(strategies_to_test)} stratégies sur {strat_symbol}..."):
            # Nom de l'actif
            asset_name_strat = get_asset_name(strat_symbol)
            st.write(f"🔬 **{asset_name_strat if asset_name_strat else strat_symbol}** ({strat_symbol})")
            
            # Récupérer données
            df_strat = cached_fetch_data(strat_symbol, period=period_years)
            
            if not df_strat.empty:
                # Exécuter les stratégies
                results = []
                backtester = StrategyBacktester(initial_capital=capital)
                
                for strategy in strategies_to_test:
                    res = strategy.backtest(df_strat, initial_capital=capital)
                    if res:
                        # Calculer métriques avancées
                        metrics = backtester.calculate_advanced_metrics(res['equity_curve'], res['trades'])
                        res['metrics'] = metrics
                        results.append(res)
                
                if results:
                    st.divider()
                    
                    # 1. Tableau comparatif
                    st.subheader("🏆 Classement des Stratégies")
                    
                    comparison_data = []
                    for res in results:
                        m = res['metrics']
                        comparison_data.append({
                            "Stratégie": res['strategy'],
                            "Rendement Total": f"{res['total_return']:.2f}%",
                            "Profit Net": f"${res['final_capital'] - res['initial_capital']:,.2f}",
                            "Sharpe": f"{m.get('sharpe_ratio', 0):.2f}",
                            "Max Drawdown": f"{m.get('max_drawdown_pct', 0):.1f}%",
                            "Trades": res['total_trades'],
                            "Win Rate": f"{res['win_rate']:.1f}%"
                        })
                    
                    df_comp = pd.DataFrame(comparison_data)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    
                    # 2. Graphique comparatif
                    st.subheader("📈 Comparaison des Courbes d'Équité")
                    fig_equity = backtester.create_equity_chart(results)
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    # 3. Détails par stratégie
                    st.subheader("📝 Détails par Stratégie")
                    
                    tabs_strat = st.tabs([r['strategy'] for r in results])
                    
                    for idx, tab in enumerate(tabs_strat):
                        res = results[idx]
                        m = res['metrics']
                        
                        with tab:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Rendement", f"{res['total_return']:.2f}%", 
                                     f"${res['final_capital'] - res['initial_capital']:,.2f}")
                            c2.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
                            c3.metric("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.2f}%", delta_color="inverse")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                chart_dd = backtester.create_drawdown_chart(res['equity_curve'])
                                st.plotly_chart(chart_dd, use_container_width=True, key=f"dd_{idx}")
                            
                            with col2:
                                st.write("**Stats Trading**")
                                st.write(f"- Win Rate: {res['win_rate']:.1f}%")
                                st.write(f"- Profit Factor: {res['profit_factor']:.2f}")
                                st.write(f"- Durée Moyenne: {m.get('avg_trade_duration', 0):.1f} jours")
                                st.write(f"- Max Gains Consécutifs: {m.get('max_consecutive_wins', 0)}")
                                st.write(f"- Max Pertes Consécutives: {m.get('max_consecutive_losses', 0)}")
                                
                            with st.expander("Voir le rapport textuel"):
                                report = backtester.generate_report(res, m)
                                st.text(report)

            else:
                st.error("Données indisponibles pour ce symbole")
    elif run_strat:
        st.warning("Veuillez sélectionner au moins une stratégie")

# ===== ONGLET 8: SYNTHÈSE GLOBALE =====
with tab8:
    st.subheader("💎 Synthèse Globale des Stratégies")
    
    st.info("Combinez plusieurs stratégies pour obtenir un signal de 'Consensus' plus robuste.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        synth_symbol = st.text_input("Symbole Consensus", symbol_input, key="synth_symbol").upper()
    with col2:
        capital_synth = st.number_input("Capital ($)", value=100000, step=1000, key="synth_cap")
    with col3:
        period_synth = st.selectbox("Période", ["1y", "2y", "5y", "10y"], index=2, key="synth_per")

    st.divider()

    col_conf, col_act = st.columns([3, 1])

    with col_conf:
        st.write("**Sélectionnez les stratégies à combiner**")
        
        c1, c2, c3, c4 = st.columns(4)
        strategies_selected = []
        
        with c1:
            if st.checkbox("MA Crossover", value=True, key="synth_ma"):
                strategies_selected.append(MACrossover(50, 200))
        with c2:
            if st.checkbox("RSI (30/70)", value=True, key="synth_rsi"):
                strategies_selected.append(RSIStrategy(14, 30, 70))
        with c3:
            if st.checkbox("Bollinger Bands", value=True, key="synth_bb"):
                strategies_selected.append(BollingerBandsStrategy(20, 2))
        with c4:
            if st.checkbox("MACD", value=True, key="synth_macd"):
                strategies_selected.append(MACDStrategy(12, 26, 9))

    with col_act:
        st.write("")
        st.write("")
        gen_synth = st.button("✨ Générer Consensus", type="primary", use_container_width=True)

    if gen_synth:
        if not strategies_selected:
            st.warning("Veuillez sélectionner au moins une stratégie.")
        else:
            with st.spinner("Calcul du Consensus et Backtesting..."):
                df_synth = cached_fetch_data(synth_symbol, period=period_synth)
                
                if not df_synth.empty and len(df_synth) > 50:
                    # Créer la stratégie couplée
                    coupled = CoupledStrategy(strategies_selected)
                    
                    # Backtest de la stratégie couplée
                    res_coupled = coupled.backtest(df_synth, initial_capital=capital_synth)
                    
                    # Backtest des stratégies individuelles pour comparaison
                    individual_results = []
                    for s in strategies_selected:
                        res = s.backtest(df_synth, initial_capital=capital_synth)
                        if res:
                            individual_results.append(res)
                    
                    if res_coupled:
                        # --- AFFICHAGE DU SIGNAL ACTUEL ---
                        st.divider()
                        asset_name_synth = get_asset_name(synth_symbol)
                        st.subheader(f"📢 Signal Actuel : {asset_name_synth if asset_name_synth else synth_symbol} ({synth_symbol})")
                        
                        # Recalculer les signaux sur la dernière ligne pour l'état actuel
                        df_signals = coupled.generate_signals(df_synth)
                        current_signal_val = df_signals['signal'].iloc[-1]
                        last_date = df_signals['timestamp'].iloc[-1]
                        
                        c1, c2, c3 = st.columns([1, 4, 1])
                        with c2:
                             if current_signal_val == 1:
                                 st.success(f"## 🟢 ACHAT FORT (Date: {last_date.date()})")
                                 st.write("### Le consensus des stratégies indique une opportunité d'achat.")
                             elif current_signal_val == -1:
                                 st.error(f"## 🔴 VENTE FORTE (Date: {last_date.date()})")
                                 st.write("### Le consensus des stratégies indique une opportunité de vente.")
                             else:
                                 st.info(f"## ⚪ NEUTRE (Date: {last_date.date()})")
                                 st.write("### Les stratégies sont partagées ou neutres. Attente conseillée.")
                        
                        st.caption("ℹ️ **Note sur la Tendance** : Les stratégies de tendance (MA, MACD) sont désormais configurées en mode 'Continu'. Elles indiquent ACHAT tant que la tendance est haussière et VENTE tant que la tendance est baissière, réduisant ainsi les périodes 'Neutres'.")

                        st.divider()
                        
                        # --- RÉSULTATS BACKTEST ---
                        st.subheader("📊 Performance Comparée")
                        
                        col_chart, col_table = st.columns([3, 2])
                        
                        all_results = individual_results + [res_coupled]
                        
                        with col_chart:
                            # Graphique rendement bar chart
                            perf_data = []
                            for r in all_results:
                                is_coupled = r['strategy'].startswith("Coupled")
                                perf_data.append({
                                    "Stratégie": r['strategy'],
                                    "Rendement %": r['total_return'],
                                    "Couleur": "#FF4B4B" if is_coupled else "#31333F"
                                })
                            
                            df_perf = pd.DataFrame(perf_data)
                            
                            fig_perf = go.Figure(data=[
                                go.Bar(x=df_perf['Stratégie'], y=df_perf['Rendement %'], marker_color=df_perf['Couleur'])
                            ])
                            fig_perf.update_layout(title="Rendement Total par Stratégie", template="plotly_dark")
                            st.plotly_chart(fig_perf, use_container_width=True)
                        
                        with col_table:
                            # Tableau détaillé
                            comparison_data = []
                            for res in all_results:
                                 comparison_data.append({
                                    "Stratégie": res['strategy'],
                                    "Rendement": f"{res['total_return']:.2f}%",
                                    "Win Rate": f"{res['win_rate']:.1f}%",
                                    "Profit Fact.": f"{res['profit_factor']:.2f}",
                                    "Drawdown": f"{res['max_drawdown']:.2f}%"
                                })
                            
                            st.write("**Détails Chiffrés**")
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                        
                        # Equity Curve du Coupled
                        st.subheader("📈 Courbe de Capital (Consensus)")
                        
                        df_equity = res_coupled['equity_curve']
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=df_equity['date'], 
                            y=df_equity['equity'], 
                            mode='lines', 
                            name='Consensus', 
                            line=dict(color='#FF4B4B', width=2),
                            fill='tozeroy'
                        ))
                        fig_eq.update_layout(
                            title=f"Évolution avec Stratégie Couplée ({res_coupled['total_return']:.2f}%)", 
                            template="plotly_dark", 
                            yaxis_title="Capital ($)"
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

                else:
                    st.error("Données insuffisantes pour l'analyse.")


# Footer
st.divider()
st.caption(f"Market Predictor Pro V5 - {len(objects)} modules actifs | Données fournies par yfinance")
