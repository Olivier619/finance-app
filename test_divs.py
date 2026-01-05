
import yfinance as yf
import pandas as pd

def test_dividends(symbol):
    ticker = yf.Ticker(symbol)
    print(f"--- Dividendes pour {symbol} ---")
    
    # Historique des dividendes
    divs = ticker.dividends
    if not divs.empty:
        print("Historique récent:")
        print(divs.tail(5))
    else:
        print("Aucun dividende trouvé via ticker.dividends")
    
    # Info générale
    info = ticker.info
    print(f"\nEx-Dividend Date (info): {info.get('exDividendDate')}")
    print(f"Dividend Rate: {info.get('dividendRate')}")
    print(f"Last Dividend Value: {info.get('lastDividendValue')}")

if __name__ == "__main__":
    test_dividends("AAPL")
    test_dividends("MSFT")
    test_dividends("T") # AT&T (High dividend)
