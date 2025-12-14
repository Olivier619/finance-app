"""
Module de gestion du cache intelligent
Cache multi-niveaux avec persistance
"""

import pickle
import hashlib
import time
from pathlib import Path
from functools import wraps
import pandas as pd


class CacheManager:
    """Gestionnaire de cache multi-niveaux"""
    
    def __init__(self, cache_dir='data/cache', default_ttl=300):
        """
        Initialise le gestionnaire de cache
        
        Args:
            cache_dir: Répertoire pour le cache persistant
            default_ttl: Durée de vie par défaut en secondes (5 min)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = {}  # Cache en mémoire
        self.default_ttl = default_ttl
    
    def _generate_key(self, func_name, args, kwargs):
        """Génère une clé unique pour la fonction et ses arguments"""
        # Créer une représentation string des arguments
        key_parts = [func_name]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        
        key_string = "|".join(key_parts)
        
        # Hash pour avoir une clé courte
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key, check_disk=True):
        """
        Récupère une valeur du cache
        
        Args:
            key: Clé du cache
            check_disk: Si True, vérifie aussi le cache disque
        
        Returns:
            Tuple (value, found) - (None, False) si non trouvé
        """
        # Vérifier le cache mémoire
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry['expires_at']:
                return entry['value'], True
            else:
                # Expiré, supprimer
                del self.memory_cache[key]
        
        # Vérifier le cache disque
        if check_disk:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if time.time() < entry['expires_at']:
                        # Charger dans le cache mémoire
                        self.memory_cache[key] = entry
                        return entry['value'], True
                    else:
                        # Expiré, supprimer
                        cache_file.unlink()
                except:
                    pass
        
        return None, False
    
    def set(self, key, value, ttl=None, persist=False):
        """
        Stocke une valeur dans le cache
        
        Args:
            key: Clé du cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes (None = default_ttl)
            persist: Si True, sauvegarde aussi sur disque
        """
        if ttl is None:
            ttl = self.default_ttl
        
        entry = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
        
        # Stocker en mémoire
        self.memory_cache[key] = entry
        
        # Stocker sur disque si demandé
        if persist:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
            except:
                pass
    
    def clear_memory(self):
        """Vide le cache mémoire"""
        self.memory_cache.clear()
    
    def clear_disk(self):
        """Vide le cache disque"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
    
    def clear_all(self):
        """Vide tous les caches"""
        self.clear_memory()
        self.clear_disk()
    
    def get_stats(self):
        """Retourne des statistiques sur le cache"""
        memory_size = len(self.memory_cache)
        disk_size = len(list(self.cache_dir.glob("*.pkl")))
        
        # Calculer la taille totale sur disque
        total_disk_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        total_disk_mb = total_disk_bytes / (1024 * 1024)
        
        return {
            'memory_entries': memory_size,
            'disk_entries': disk_size,
            'disk_size_mb': total_disk_mb
        }


# Instance globale
_cache_manager = CacheManager()


def cached(ttl=300, persist=False):
    """
    Décorateur pour mettre en cache les résultats d'une fonction
    
    Args:
        ttl: Durée de vie du cache en secondes
        persist: Si True, persiste sur disque
    
    Example:
        @cached(ttl=600, persist=True)
        def expensive_function(arg1, arg2):
            # Calculs coûteux
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Générer la clé
            key = _cache_manager._generate_key(func.__name__, args, kwargs)
            
            # Vérifier le cache
            value, found = _cache_manager.get(key, check_disk=persist)
            
            if found:
                return value
            
            # Calculer la valeur
            result = func(*args, **kwargs)
            
            # Stocker dans le cache
            _cache_manager.set(key, result, ttl=ttl, persist=persist)
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires
def clear_cache():
    """Vide tous les caches"""
    _cache_manager.clear_all()


def get_cache_stats():
    """Retourne les statistiques du cache"""
    return _cache_manager.get_stats()
