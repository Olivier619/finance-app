"""
Application Streamlit V4 : UI Optimisée (Sidebar fine) + Zoom Auto + Commodities/Forex
"""
import os
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

# --- CSS PERSONNALISÉ (Sidebar Fine) ---
st.set_page_config(page_title="Market Predictor Pro", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 250px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Configuration & Données ---
# Dictionnaire enrichi avec Commodities & Forex
SECTEURS_ETF = {
    # Actions / Secteurs
    "Technologie": "XLK", "Santé": "XLV", "Finance": "XLF",
    "Immobilier": "VNQ", "Automobile": "CARZ", "Énergie": "XLE",
    # Matières Premières (Futures)
    "Or (Gold)": "GC=F", "Argent (Silver)": "SI=F", 
    "Pétrole (WTI)": "CL=F", "Gaz Naturel": "NG=F",
    "Blé (Wheat)": "ZW=F", "Maïs (Corn)": "ZC=F",
    # Crypto
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
    # Forex (Monnaies)
    "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", 
    "GBP/USD": "GBPUSD=X", "USD/CHF": "CHF=X"
}

def fetch_market_data(symbol):
    """Récupère 10 ans d'historique pour l'entraînement."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="10y") 
        df.reset_index(inplace=True)
        # Nettoyage colonnes
        df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", 
                              "Low": "low", "Close": "close", "Volume": "volume"})
        
        # Gestion Timezone (suppression pour éviter conflits)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
        return df
    except Exception:
        return pd.DataFrame()

# --- Moteur IA (Identique V3 pour stabilité) ---
def preparer_donnees_ia_avance(df):
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
    features = ['RSI', 'MACD', 'BB_Pct', 'ROC', 'ATR']
    if not all(col in df_full.columns for col in features): return {}
    last_row = df_full.iloc[[-1]][features]
    if last_row.isna().any().any(): return {} 

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

# --- Interface ---
with st.sidebar:
    st.header("🔎 Paramètres")
    
    # Menu Catégories
    categorie = st.selectbox("Catégorie", ["Actions", "Secteurs/Indices", "Matières Premières", "Forex (Devises)", "Cryptos"])
    
    symbol_input = "AAPL" # Valeur par défaut
    
    if categorie == "Actions":
        symbol_input = st.text_input("Ticker (ex: TSLA, NVDA)", "NVDA").upper()
    
    elif categorie == "Secteurs/Indices":
        choix = st.selectbox("Indice", ["Technologie", "Finance", "Immobilier", "Énergie", "Automobile"])
        symbol_input = SECTEURS_ETF[choix]
        
    elif categorie == "Matières Premières":
        choix = st.selectbox("Commodity", ["Or (Gold)", "Argent (Silver)", "Pétrole (WTI)", "Gaz Naturel", "Blé (Wheat)", "Maïs (Corn)"])
        symbol_input = SECTEURS_ETF[choix]
        
    elif categorie == "Forex (Devises)":
        choix = st.selectbox("Paire", ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF"])
        symbol_input = SECTEURS_ETF[choix]

    elif categorie == "Cryptos":
        choix = st.selectbox("Crypto", ["Bitcoin", "Ethereum", "Solana"])
        symbol_input = SECTEURS_ETF[choix]
    
    lancer = st.button("⚡ Analyser", type="primary")
    st.caption(f"Symbole : {symbol_input}")

# --- Zone Principale ---
st.title(f"Analyse : {symbol_input}")

if lancer:
    with st.spinner(f"Analyse de {symbol_input}..."):
        df = fetch_market_data(symbol_input)
        
        if not df.empty and len(df) > 50:
            df_ia = preparer_donnees_ia_avance(df)
            probs = calculer_toutes_probabilites(df_ia)
            
            # --- Jauges IA ---
            st.subheader("🔮 Prédictions de Tendance")
            c1, c2, c3, c4 = st.columns(4)
            keys = list(probs.keys())
            
            def show_gauge(col, label, val):
                col.metric(label, f"{val:.1f}%", f"{val-50:.1f}%")
                col.progress(int(val))
                if val >= 60: col.success("HAUSSIER")
                elif val <= 40: col.error("BAISSIER")
                else: col.warning("NEUTRE")

            if len(keys) >= 4:
                show_gauge(c1, keys[0], probs[keys[0]])
                show_gauge(c2, keys[1], probs[keys[1]])
                show_gauge(c3, keys[2], probs[keys[2]])
                show_gauge(c4, keys[3], probs[keys[3]])

            st.divider()

            # --- Graphique Amélioré (Zoom Auto) ---
            st.subheader("📊 Graphique Technique")
            
            fig = go.Figure()
            # Chandeliers
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                       low=df['low'], close=df['close'], name='Prix'))
            
            # Bollinger (Visuel)
            if 'BB_Pct' in df_ia.columns:
                bb = ta.bbands(df_ia['close'], length=20, std=2)
                if bb is not None:
                    fig.add_trace(go.Scatter(x=df_ia['timestamp'], y=bb[bb.columns[0]], 
                                           line=dict(color='rgba(0,0,255,0.1)'), name='BB Bas'))
                    fig.add_trace(go.Scatter(x=df_ia['timestamp'], y=bb[bb.columns[2]], 
                                           line=dict(color='rgba(0,0,255,0.1)'), name='BB Haut', fill='tonexty'))

            # Configuration Zoom par défaut (Dernière année)
            last_date = df['timestamp'].iloc[-1]
            one_year_ago = last_date - timedelta(days=365)
            
            fig.update_layout(
                title=f"{symbol_input} - Zoom 1 An (par défaut)",
                template="plotly_dark",
                height=600,
                xaxis=dict(
                    range=[one_year_ago, last_date], # Force le zoom initial
                    rangeslider=dict(visible=True),   # Garde le slider en bas pour voir avant
                    type="date"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Données non disponibles ou insuffisantes (vérifiez le symbole).")
