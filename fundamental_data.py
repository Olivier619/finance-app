"""
Module de données fondamentales
Ratios financiers, valorisation, rentabilité, croissance
"""

import yfinance as yf


class FundamentalData:
    def __init__(self):
        pass
    
    def get_valuation_ratios(self, symbol):
        """
        Récupère les ratios de valorisation
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les ratios de valorisation
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'peg_ratio': info.get('pegRatio', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'ev_to_revenue': info.get('enterpriseToRevenue', None),
                'ev_to_ebitda': info.get('enterpriseToEbitda', None),
                'market_cap': info.get('marketCap', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération des ratios de valorisation pour {symbol}: {e}")
            return None
    
    def get_profitability_ratios(self, symbol):
        """
        Récupère les ratios de rentabilité
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les ratios de rentabilité
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'gross_margin': info.get('grossMargins', None),
                'ebitda_margin': info.get('ebitdaMargins', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération des ratios de rentabilité pour {symbol}: {e}")
            return None
    
    def get_dividend_data(self, symbol):
        """
        Récupère les données de dividendes
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les données de dividendes
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'dividend_yield': info.get('dividendYield', None),
                'dividend_rate': info.get('dividendRate', None),
                'payout_ratio': info.get('payoutRatio', None),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', None),
                'trailing_annual_dividend_rate': info.get('trailingAnnualDividendRate', None),
                'trailing_annual_dividend_yield': info.get('trailingAnnualDividendYield', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération des données de dividendes pour {symbol}: {e}")
            return None
    
    def get_growth_metrics(self, symbol):
        """
        Récupère les métriques de croissance
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les métriques de croissance
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),
                'revenue_per_share': info.get('revenuePerShare', None),
                'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération des métriques de croissance pour {symbol}: {e}")
            return None
    
    def get_financial_health(self, symbol):
        """
        Récupère les indicateurs de santé financière
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les indicateurs de santé financière
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'total_debt': info.get('totalDebt', None),
                'total_cash': info.get('totalCash', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'operating_cash_flow': info.get('operatingCashflow', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération de la santé financière pour {symbol}: {e}")
            return None
    
    def get_key_statistics(self, symbol):
        """
        Récupère les statistiques clés
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec les statistiques clés
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'beta': info.get('beta', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', None),
                '52_week_low': info.get('fiftyTwoWeekLow', None),
                '50_day_average': info.get('fiftyDayAverage', None),
                '200_day_average': info.get('twoHundredDayAverage', None),
                'shares_outstanding': info.get('sharesOutstanding', None),
                'float_shares': info.get('floatShares', None),
                'shares_short': info.get('sharesShort', None),
                'short_ratio': info.get('shortRatio', None),
                'short_percent_of_float': info.get('shortPercentOfFloat', None)
            }
        
        except Exception as e:
            print(f"Erreur lors de la récupération des statistiques clés pour {symbol}: {e}")
            return None
    
    def get_all_fundamentals(self, symbol):
        """
        Récupère toutes les données fondamentales
        
        Args:
            symbol: Symbole de l'actif
        
        Returns:
            Dict avec toutes les données fondamentales
        """
        return {
            'valuation': self.get_valuation_ratios(symbol),
            'profitability': self.get_profitability_ratios(symbol),
            'dividends': self.get_dividend_data(symbol),
            'growth': self.get_growth_metrics(symbol),
            'financial_health': self.get_financial_health(symbol),
            'key_statistics': self.get_key_statistics(symbol)
        }
    
    def format_fundamentals_table(self, fundamentals):
        """
        Formate les données fondamentales en tableau lisible
        
        Args:
            fundamentals: Dict retourné par get_all_fundamentals
        
        Returns:
            str: Tableau formaté
        """
        output = []
        
        # Valorisation
        if fundamentals.get('valuation'):
            val = fundamentals['valuation']
            output.append("📊 VALORISATION")
            output.append(f"  P/E Ratio: {val.get('pe_ratio', 'N/A')}")
            output.append(f"  Forward P/E: {val.get('forward_pe', 'N/A')}")
            output.append(f"  P/B Ratio: {val.get('pb_ratio', 'N/A')}")
            output.append(f"  P/S Ratio: {val.get('ps_ratio', 'N/A')}")
            output.append(f"  PEG Ratio: {val.get('peg_ratio', 'N/A')}")
            output.append("")
        
        # Rentabilité
        if fundamentals.get('profitability'):
            prof = fundamentals['profitability']
            output.append("💰 RENTABILITÉ")
            output.append(f"  ROE: {prof.get('roe', 'N/A')}")
            output.append(f"  ROA: {prof.get('roa', 'N/A')}")
            output.append(f"  Profit Margin: {prof.get('profit_margin', 'N/A')}")
            output.append(f"  Operating Margin: {prof.get('operating_margin', 'N/A')}")
            output.append("")
        
        # Dividendes
        if fundamentals.get('dividends'):
            div = fundamentals['dividends']
            output.append("💵 DIVIDENDES")
            output.append(f"  Dividend Yield: {div.get('dividend_yield', 'N/A')}")
            output.append(f"  Dividend Rate: ${div.get('dividend_rate', 'N/A')}")
            output.append(f"  Payout Ratio: {div.get('payout_ratio', 'N/A')}")
            output.append("")
        
        # Croissance
        if fundamentals.get('growth'):
            growth = fundamentals['growth']
            output.append("📈 CROISSANCE")
            output.append(f"  Revenue Growth: {growth.get('revenue_growth', 'N/A')}")
            output.append(f"  Earnings Growth: {growth.get('earnings_growth', 'N/A')}")
            output.append("")
        
        return "\n".join(output)
