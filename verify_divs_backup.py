
import json
import os
import sys
# Mock streamlit
class MockStreamlit:
    def cache_data(self, **kwargs): return lambda f: f
    def cache_resource(self, **kwargs): return lambda f: f
sys.modules['streamlit'] = MockStreamlit()

from paper_trading import PaperTrading
from database import Database

def test_full_system():
    # 1. Test Backup/Restore
    pt = PaperTrading()
    print("--- Test Backup ---")
    data = pt.export_portfolio()
    print(f"Export réussi. Cash actuel: {data['cash']}")
    
    with open('test_backup.json', 'w') as f:
        json.dump(data, f)
    
    print("--- Test Reset and Restore ---")
    pt.reset_portfolio(500000.0)
    print(f"Reset fait. Nouveau cash: {pt.get_portfolio_summary()['cash']}")
    
    with open('test_backup.json', 'r') as f:
        import_data = json.load(f)
    pt.import_portfolio(import_data)
    print(f"Restore fait. Cash restauré: {pt.get_portfolio_summary()['cash']}")
    
    # 2. Test Dividends
    # Simuler une transaction passée pour AAPL pour forcer un calcul de dividende
    # AAPL ex-date était 2024-11-08 (par exemple)
    print("\n--- Test Dividendes (Simulé) ---")
    db = Database()
    # On ajoute manuellement une transaction passée pour tester le script process_dividends
    db.add_transaction("AAPL", "BUY", 100, 150.0, 0, "Fake historic trade for div test")
    
    print("Traitement des dividendes...")
    count = pt.process_dividends()
    print(f"Dividendes traités: {count}")
    
    summary = pt.get_portfolio_summary()
    print(f"Total dividendes en DB: {summary['total_dividends']}")
    
    # Clean up
    if os.path.exists('test_backup.json'):
        os.remove('test_backup.json')

if __name__ == "__main__":
    test_full_system()
