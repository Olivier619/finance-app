"""
Module d'explainability avec SHAP
Affiche l'importance des features et explique les prédictions
"""

import pandas as pd
import numpy as np

# SHAP (optionnel)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelExplainability:
    def __init__(self):
        self.explainers = {}
    
    def create_explainer(self, model, X_train, model_type='tree'):
        """
        Crée un explainer SHAP pour un modèle
        
        Args:
            model: Modèle entraîné
            X_train: Données d'entraînement
            model_type: Type de modèle ('tree', 'linear', 'deep')
        
        Returns:
            Explainer SHAP
        """
        if not SHAP_AVAILABLE:
            print("SHAP n'est pas installé. Installez-le avec: pip install shap")
            return None
        
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_train)
        elif model_type == 'deep':
            explainer = shap.DeepExplainer(model, X_train)
        else:
            explainer = shap.Explainer(model, X_train)
        
        self.explainers[model_type] = explainer
        
        return explainer
    
    def calculate_shap_values(self, explainer, X):
        """
        Calcule les SHAP values pour des données
        
        Args:
            explainer: Explainer SHAP
            X: Données à expliquer
        
        Returns:
            SHAP values
        """
        if not SHAP_AVAILABLE:
            return None
        
        shap_values = explainer.shap_values(X)
        
        return shap_values
    
    def get_feature_importance(self, shap_values, feature_names):
        """
        Calcule l'importance des features basée sur SHAP
        
        Args:
            shap_values: SHAP values
            feature_names: Noms des features
        
        Returns:
            DataFrame avec importance des features
        """
        if shap_values is None:
            return None
        
        # Si shap_values est une liste (classification binaire), prendre le premier élément
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe positive
        
        # Calculer l'importance moyenne absolue
        importance = np.abs(shap_values).mean(axis=0)
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        return df_importance
    
    def explain_prediction(self, shap_values, feature_names, feature_values, class_idx=1):
        """
        Explique une prédiction spécifique
        
        Args:
            shap_values: SHAP values
            feature_names: Noms des features
            feature_values: Valeurs des features pour cette prédiction
            class_idx: Index de la classe (défaut: 1 pour classe positive)
        
        Returns:
            Dict avec explication
        """
        if shap_values is None:
            return None
        
        # Si shap_values est une liste, prendre la classe spécifiée
        if isinstance(shap_values, list):
            shap_vals = shap_values[class_idx][0]  # Première prédiction
        else:
            shap_vals = shap_values[0]
        
        # Créer un DataFrame avec features et leurs contributions
        df_explanation = pd.DataFrame({
            'feature': feature_names,
            'value': feature_values,
            'shap_value': shap_vals
        })
        
        df_explanation = df_explanation.sort_values('shap_value', key=abs, ascending=False)
        
        return df_explanation
    
    def generate_explanation_text(self, explanation_df, top_n=5):
        """
        Génère une explication en langage naturel
        
        Args:
            explanation_df: DataFrame retourné par explain_prediction
            top_n: Nombre de features à expliquer (défaut: 5)
        
        Returns:
            str: Explication textuelle
        """
        if explanation_df is None or explanation_df.empty:
            return "Aucune explication disponible."
        
        explanation = []
        explanation.append("🔍 EXPLICATION DE LA PRÉDICTION\n")
        explanation.append(f"Top {top_n} facteurs influençant la prédiction:\n")
        
        for idx, row in explanation_df.head(top_n).iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row['shap_value']
            
            direction = "augmente" if shap_val > 0 else "diminue"
            impact = "fortement" if abs(shap_val) > 0.1 else "légèrement"
            
            explanation.append(f"  • {feature} = {value:.2f}")
            explanation.append(f"    → {impact} {direction} la probabilité de hausse (impact: {shap_val:+.3f})\n")
        
        return "\n".join(explanation)
    
    def get_feature_importance_simple(self, model, feature_names):
        """
        Récupère l'importance des features sans SHAP (pour Random Forest/XGBoost)
        
        Args:
            model: Modèle entraîné (Random Forest ou XGBoost)
            feature_names: Noms des features
        
        Returns:
            DataFrame avec importance des features
        """
        try:
            # Pour Random Forest et XGBoost
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                df_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                })
                
                df_importance = df_importance.sort_values('importance', ascending=False)
                
                return df_importance
            else:
                return None
        except:
            return None
    
    def visualize_feature_importance(self, importance_df):
        """
        Crée une visualisation simple de l'importance des features
        
        Args:
            importance_df: DataFrame avec colonnes 'feature' et 'importance'
        
        Returns:
            str: Visualisation ASCII
        """
        if importance_df is None or importance_df.empty:
            return "Aucune donnée d'importance disponible."
        
        max_importance = importance_df['importance'].max()
        
        viz = []
        viz.append("📊 IMPORTANCE DES FEATURES\n")
        
        for idx, row in importance_df.head(10).iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Barre de progression ASCII
            bar_length = int((importance / max_importance) * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            viz.append(f"{feature:15s} {bar} {importance:.4f}")
        
        return "\n".join(viz)
