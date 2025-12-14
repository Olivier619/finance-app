"""
Module de récupération de news en temps réel
Intégration avec NewsAPI et Finnhub
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class NewsFetcher:
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
    
    def fetch_from_newsapi(self, query, from_date=None, language='en', page_size=20):
        """
        Récupère des news depuis NewsAPI
        
        Args:
            query: Mot-clé de recherche (ex: "AAPL", "Tesla")
            from_date: Date de début (défaut: 7 jours)
            language: Langue (défaut: 'en')
            page_size: Nombre d'articles (défaut: 20)
        
        Returns:
            List d'articles
        """
        if not self.newsapi_key:
            return []
        
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date,
            'language': language,
            'pageSize': page_size,
            'sortBy': 'publishedAt',
            'apiKey': self.newsapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                
                return [
                    {
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'image_url': article.get('urlToImage', '')
                    }
                    for article in articles
                ]
            else:
                return []
        
        except Exception as e:
            print(f"Erreur NewsAPI: {e}")
            return []
    
    def fetch_from_finnhub(self, symbol, from_date=None):
        """
        Récupère des news depuis Finnhub
        
        Args:
            symbol: Symbole de l'actif
            from_date: Date de début (défaut: 7 jours)
        
        Returns:
            List d'articles
        """
        if not self.finnhub_key:
            return []
        
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = 'https://finnhub.io/api/v1/company-news'
        params = {
            'symbol': symbol.upper(),
            'from': from_date,
            'to': to_date,
            'token': self.finnhub_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()
            
            return [
                {
                    'source': article.get('source', 'Finnhub'),
                    'title': article.get('headline', ''),
                    'description': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'image_url': article.get('image', '')
                }
                for article in articles
            ]
        
        except Exception as e:
            print(f"Erreur Finnhub: {e}")
            return []
    
    def get_news(self, symbol, days=7, max_articles=20):
        """
        Récupère des news depuis toutes les sources disponibles
        
        Args:
            symbol: Symbole de l'actif
            days: Nombre de jours à récupérer (défaut: 7)
            max_articles: Nombre maximum d'articles (défaut: 20)
        
        Returns:
            List d'articles triés par date
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        all_news = []
        
        # Finnhub (spécifique au symbole)
        finnhub_news = self.fetch_from_finnhub(symbol, from_date)
        all_news.extend(finnhub_news)
        
        # NewsAPI (recherche générale)
        newsapi_news = self.fetch_from_newsapi(symbol, from_date, page_size=max_articles)
        all_news.extend(newsapi_news)
        
        # Supprimer les doublons basés sur l'URL
        seen_urls = set()
        unique_news = []
        
        for article in all_news:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_news.append(article)
        
        # Trier par date (plus récent en premier)
        unique_news.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        return unique_news[:max_articles]
    
    def filter_by_keywords(self, articles, keywords):
        """
        Filtre les articles par mots-clés
        
        Args:
            articles: List d'articles
            keywords: List de mots-clés
        
        Returns:
            List d'articles filtrés
        """
        if not keywords:
            return articles
        
        filtered = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            # Vérifier si au moins un mot-clé est présent
            if any(keyword.lower() in title or keyword.lower() in description for keyword in keywords):
                filtered.append(article)
        
        return filtered
    
    def score_relevance(self, article, symbol):
        """
        Score la pertinence d'un article pour un symbole
        
        Args:
            article: Dict de l'article
            symbol: Symbole de l'actif
        
        Returns:
            float: Score de pertinence (0-1)
        """
        score = 0.0
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        symbol_lower = symbol.lower()
        
        # Présence du symbole dans le titre
        if symbol_lower in title:
            score += 0.5
        
        # Présence du symbole dans la description
        if symbol_lower in description:
            score += 0.3
        
        # Bonus pour les sources fiables
        reliable_sources = ['reuters', 'bloomberg', 'wsj', 'financial times', 'cnbc']
        source = article.get('source', '').lower()
        
        if any(rs in source for rs in reliable_sources):
            score += 0.2
        
        return min(score, 1.0)
