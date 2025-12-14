"""
Module de calculs parallèles
Optimisation des performances avec multiprocessing
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
from functools import partial


def _process_symbol_helper(args):
    """Helper pour process_symbol"""
    symbol, data_fetcher, indicator_calculator = args
    try:
        df = data_fetcher(symbol)
        if df is not None and not df.empty:
            df_with_indicators = indicator_calculator(df)
            return (symbol, df_with_indicators)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    return (symbol, None)

def _backtest_strategy_helper(args):
    """Helper pour backtest_strategies"""
    name, strategy, df = args
    try:
        result = strategy.backtest(df.copy())
        return result
    except Exception as e:
        print(f"Error backtesting {name}: {e}")
        return None

class ParallelComputing:
    """Gestionnaire de calculs parallèles"""
    
    def __init__(self, max_workers=None):
        """
        Initialise le gestionnaire
        
        Args:
            max_workers: Nombre max de workers (None = auto)
        """
        self.max_workers = max_workers
    
    def process_batch(self, func, items, use_threads=False):
        """
        Traite un batch d'items en parallèle
        
        Args:
            func: Fonction à appliquer à chaque item
            items: Liste d'items à traiter
            use_threads: Si True, utilise ThreadPoolExecutor au lieu de ProcessPoolExecutor
        
        Returns:
            Liste des résultats
        """
        if not items:
            return []
        
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        results = []
        with Executor(max_workers=self.max_workers) as executor:
            # Soumettre les tâches
            futures = [executor.submit(func, item) for item in items]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in parallel execution: {e}")
                    results.append(None)
        
        return results
    
    def calculate_indicators_parallel(self, symbols, data_fetcher, indicator_calculator):
        """
        Calcule les indicateurs pour plusieurs symboles en parallèle
        
        Args:
            symbols: Liste de symboles
            data_fetcher: Fonction pour récupérer les données (symbol) -> df
            indicator_calculator: Fonction pour calculer les indicateurs (df) -> df
        
        Returns:
            Dict {symbol: df_with_indicators}
        """
        # Préparer les arguments pour chaque tâche
        tasks = [(symbol, data_fetcher, indicator_calculator) for symbol in symbols]
        
        # Sur Windows, ProcessPoolExecutor a souvent du mal avec des fonctions complexes
        # On utilise ThreadPoolExecutor ici car c'est souvent I/O bound (fetch data)
        results = self.process_batch(_process_symbol_helper, tasks, use_threads=True)
        
        return {symbol: df for symbol, df in results if df is not None}
    
    def backtest_strategies_parallel(self, strategies, df):
        """
        Teste plusieurs stratégies en parallèle
        
        Args:
            strategies: Dict de stratégies {name: strategy}
            df: DataFrame avec données
        
        Returns:
            Liste de résultats
        """
        # Préparer les arguments : on doit passer df à chaque fois
        tasks = [(name, strategy, df) for name, strategy in strategies.items()]
        
        # Backtesting est CPU bound, mais pour compatibilité max on peut utiliser threads
        # ou processes si les objets sont picklables.
        # Attention: passer un gros DF à chaque process peut être lent.
        # Essayer ProcessPoolExecutor (use_threads=False) mais avec fonction globale
        results = self.process_batch(_backtest_strategy_helper, tasks, use_threads=False)
        
        return [r for r in results if r is not None]


# Instance globale
_parallel_computer = ParallelComputing()


def parallel_map(func, items, use_threads=False, max_workers=None):
    """
    Map parallèle simple
    
    Args:
        func: Fonction à appliquer
        items: Liste d'items
        use_threads: Utiliser threads au lieu de processes
        max_workers: Nombre de workers
    
    Returns:
        Liste des résultats
    """
    computer = ParallelComputing(max_workers=max_workers)
    return computer.process_batch(func, items, use_threads=use_threads)
