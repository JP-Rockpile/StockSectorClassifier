import yfinance as yf
import pandas as pd
import json

with open('nasdaq_tickers.json', 'r') as f:
    nasdaq_tickers = json.load(f)

with open('nyse_tickers.json', 'r') as f:
    nyse_tickers = json.load(f)

symbols = nasdaq_tickers + nyse_tickers

data = []

for symbol in symbols:
    print(symbol)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        data.append({
            'symbol': symbol,
            'current_price': info.get('currentPrice'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'eps': info.get('trailingEps'),
            'dividend_yield': info.get('dividendYield'),
            'market_cap': info.get('marketCap'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'roa': info.get('returnOnAssets'),
            'roe': info.get('returnOnEquity'),
            'beta': info.get('beta'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'asset_turnover': info.get('assetTurnover'),
            'debt_to_equity': info.get('debtToEquity'),
            'interest_coverage': info.get('interestCoverage'),
            'operating_margin': info.get('operatingMargins'),
            'net_profit_margin': info.get('netProfitMargins'),
            'free_cash_flow': info.get('freeCashflow'),
            'sector': info.get('sector')
        })
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('stock_data.csv', index=False)

print("Data saved to stock_data.csv")

with open('stock_data.csv', 'r') as file:
    for i in range(100):
        line = file.readline()
        print(line.strip())

print("Data saved to stock_data.csv and first 100 lines printed.")
