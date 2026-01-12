"""
Module de modèles ML multiples
Random Forest, XGBoost, LSTM
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# XGBoost (optionnel, installé séparément)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# TensorFlow/Keras pour LSTM (optionnel)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
        """
        Entraîne un modèle Random Forest
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            n_estimators: Nombre d'arbres (défaut: 100)
            max_depth: Profondeur maximale (défaut: None)
            random_state: Seed aléatoire (défaut: 42)
        
        Returns:
            Modèle entraîné
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Entraîne un modèle XGBoost
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            n_estimators: Nombre d'arbres (défaut: 100)
            max_depth: Profondeur maximale (défaut: 6)
            learning_rate: Taux d'apprentissage (défaut: 0.1)
            random_state: Seed aléatoire (défaut: 42)
        
        Returns:
            Modèle entraîné ou None si XGBoost non disponible
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost n'est pas installé. Installez-le avec: pip install xgboost")
            return None
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        return model
    
    def train_lstm(self, X_train, y_train, sequence_length=10, epochs=50, batch_size=32):
        """
        Entraîne un modèle LSTM
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            sequence_length: Longueur de la séquence (défaut: 10)
            epochs: Nombre d'époques (défaut: 50)
            batch_size: Taille du batch (défaut: 32)
        
        Returns:
            Modèle entraîné ou None si TensorFlow non disponible
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow n'est pas installé. Installez-le avec: pip install tensorflow")
            return None
        
        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers['lstm'] = scaler
        
        # Créer des séquences
        X_seq, y_seq = self._create_sequences(X_scaled, y_train.values, sequence_length)
        
        if len(X_seq) == 0:
            print("Pas assez de données pour créer des séquences")
            return None
        
        # Construire le modèle LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Entraîner
        model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
        
        self.models['lstm'] = model
        
        return model
    
    def _create_sequences(self, X, y, sequence_length):
        """Crée des séquences pour LSTM"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def predict(self, model_name, X):
        """
        Fait une prédiction avec un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('random_forest', 'xgboost', 'lstm')
            X: Features pour la prédiction
        
        Returns:
            Prédictions
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non entraîné")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            # Normaliser et créer des séquences pour LSTM
            scaler = self.scalers.get('lstm')
            if scaler is None:
                raise ValueError("Scaler LSTM non trouvé")
            
            X_scaled = scaler.transform(X)
            # Pour la prédiction, on utilise les dernières valeurs comme séquence
            # Simplification : on prend juste la dernière ligne
            predictions = model.predict(X_scaled)
            return (predictions > 0.5).astype(int).flatten()
        else:
            return model.predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Retourne les probabilités de prédiction
        
        Args:
            model_name: Nom du modèle
            X: Features pour la prédiction
        
        Returns:
            Probabilités
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non entraîné")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            scaler = self.scalers.get('lstm')
            if scaler is None:
                raise ValueError("Scaler LSTM non trouvé")
            
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        else:
            return model.predict_proba(X)
    
    def compare_models(self, X_test, y_test):
        """
        Compare les performances de tous les modèles entraînés
        
        Args:
            X_test: Features de test
            y_test: Target de test
        
        Returns:
            Dict avec les performances de chaque modèle
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                y_pred = self.predict(model_name, X_test)
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
            except Exception as e:
                print(f"Erreur lors de l'évaluation de {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def get_best_model(self, X_test, y_test, metric='accuracy'):
        """
        Retourne le meilleur modèle basé sur une métrique
        
        Args:
            X_test: Features de test
            y_test: Target de test
            metric: Métrique à utiliser (défaut: 'accuracy')
        
        Returns:
            Tuple (nom_modèle, score)
        """
        results = self.compare_models(X_test, y_test)
        
        best_model = None
        best_score = 0
        
        for model_name, metrics in results.items():
            if metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model, best_score
    
    def save_model(self, model_name, filepath):
        """Sauvegarde un modèle"""
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            model.save(filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
    
    def load_model(self, model_name, filepath):
        """Charge un modèle"""
        if model_name == 'lstm':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow non disponible")
            self.models[model_name] = keras.models.load_model(filepath)
        else:
            with open(filepath, 'rb') as f:
                self.models[model_name] = pickle.load(f)
