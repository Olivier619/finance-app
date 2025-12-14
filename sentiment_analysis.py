"""
Module d'analyse de sentiment
Analyse des titres de news avec NLP basique
"""

from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta


class SentimentAnalysis:
    def __init__(self):
        pass
    
    def analyze_text(self, text):
        """
        Analyse le sentiment d'un texte
        
        Args:
            text: Texte à analyser
        
        Returns:
            Dict avec polarity et subjectivity
        """
        try:
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 (négatif) à +1 (positif)
                'subjectivity': blob.sentiment.subjectivity  # 0 (objectif) à 1 (subjectif)
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_article(self, article):
        """
        Analyse le sentiment d'un article de news
        
        Args:
            article: Dict avec 'title' et 'description'
        
        Returns:
            Dict avec sentiment
        """
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Analyser le titre (poids plus important)
        title_sentiment = self.analyze_text(title)
        
        # Analyser la description
        desc_sentiment = self.analyze_text(description)
        
        # Moyenne pondérée (titre = 70%, description = 30%)
        combined_polarity = (title_sentiment['polarity'] * 0.7) + (desc_sentiment['polarity'] * 0.3)
        combined_subjectivity = (title_sentiment['subjectivity'] * 0.7) + (desc_sentiment['subjectivity'] * 0.3)
        
        # Classifier le sentiment
        if combined_polarity > 0.1:
            sentiment_label = 'POSITIVE'
        elif combined_polarity < -0.1:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'polarity': combined_polarity,
            'subjectivity': combined_subjectivity,
            'sentiment': sentiment_label,
            'title_polarity': title_sentiment['polarity'],
            'desc_polarity': desc_sentiment['polarity']
        }
    
    def analyze_news_batch(self, articles):
        """
        Analyse le sentiment d'un batch d'articles
        
        Args:
            articles: List d'articles
        
        Returns:
            List d'articles avec sentiment ajouté
        """
        analyzed_articles = []
        
        for article in articles:
            sentiment = self.analyze_article(article)
            
            article_with_sentiment = article.copy()
            article_with_sentiment['sentiment'] = sentiment
            
            analyzed_articles.append(article_with_sentiment)
        
        return analyzed_articles
    
    def calculate_aggregate_sentiment(self, articles, days=7):
        """
        Calcule le sentiment agrégé sur une période
        
        Args:
            articles: List d'articles avec sentiment
            days: Nombre de jours à considérer (défaut: 7)
        
        Returns:
            Dict avec sentiment agrégé
        """
        if not articles:
            return {
                'avg_polarity': 0.0,
                'sentiment_score': 50.0,
                'sentiment_label': 'NEUTRAL',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }
        
        # Filtrer les articles récents
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_articles = []
        
        for article in articles:
            try:
                pub_date = datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00'))
                if pub_date >= cutoff_date:
                    recent_articles.append(article)
            except:
                continue
        
        if not recent_articles:
            recent_articles = articles  # Utiliser tous les articles si aucun récent
        
        # Calculer les statistiques
        polarities = [a.get('sentiment', {}).get('polarity', 0) for a in recent_articles]
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        
        # Convertir en score 0-100
        sentiment_score = ((avg_polarity + 1) / 2) * 100
        
        # Compter les sentiments
        positive_count = sum(1 for p in polarities if p > 0.1)
        negative_count = sum(1 for p in polarities if p < -0.1)
        neutral_count = len(polarities) - positive_count - negative_count
        
        # Label global
        if avg_polarity > 0.2:
            sentiment_label = 'VERY POSITIVE'
        elif avg_polarity > 0.05:
            sentiment_label = 'POSITIVE'
        elif avg_polarity < -0.2:
            sentiment_label = 'VERY NEGATIVE'
        elif avg_polarity < -0.05:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'avg_polarity': avg_polarity,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(recent_articles),
            'days_analyzed': days
        }
    
    def get_sentiment_trend(self, articles, window_days=7):
        """
        Calcule la tendance du sentiment au fil du temps
        
        Args:
            articles: List d'articles avec sentiment
            window_days: Fenêtre glissante en jours (défaut: 7)
        
        Returns:
            DataFrame avec sentiment par période
        """
        if not articles:
            return pd.DataFrame()
        
        # Convertir en DataFrame
        df_articles = []
        
        for article in articles:
            try:
                pub_date = datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00'))
                polarity = article.get('sentiment', {}).get('polarity', 0)
                
                df_articles.append({
                    'date': pub_date.date(),
                    'polarity': polarity
                })
            except:
                continue
        
        if not df_articles:
            return pd.DataFrame()
        
        df = pd.DataFrame(df_articles)
        df = df.sort_values('date')
        
        # Calculer la moyenne mobile
        df['sentiment_ma'] = df['polarity'].rolling(window=window_days, min_periods=1).mean()
        
        return df
    
    def interpret_sentiment(self, sentiment_data):
        """
        Interprète les données de sentiment en recommandation
        
        Args:
            sentiment_data: Dict retourné par calculate_aggregate_sentiment
        
        Returns:
            str: Interprétation textuelle
        """
        score = sentiment_data['sentiment_score']
        label = sentiment_data['sentiment_label']
        total = sentiment_data['total_articles']
        positive = sentiment_data['positive_count']
        negative = sentiment_data['negative_count']
        
        interpretation = f"Sentiment global: {label} (Score: {score:.1f}/100)\n\n"
        interpretation += f"Basé sur {total} articles:\n"
        interpretation += f"  • {positive} articles positifs\n"
        interpretation += f"  • {negative} articles négatifs\n\n"
        
        if score >= 70:
            interpretation += "🟢 Le sentiment est très positif. Les news récentes sont favorables."
        elif score >= 55:
            interpretation += "🟡 Le sentiment est légèrement positif. Tendance plutôt favorable."
        elif score >= 45:
            interpretation += "⚪ Le sentiment est neutre. Pas de tendance claire."
        elif score >= 30:
            interpretation += "🟠 Le sentiment est légèrement négatif. Prudence recommandée."
        else:
            interpretation += "🔴 Le sentiment est très négatif. Les news récentes sont défavorables."
        
        return interpretation
