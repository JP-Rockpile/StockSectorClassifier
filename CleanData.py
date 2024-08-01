import pandas as pd

df = pd.read_csv('stock_data.csv')

print("Total number of rows: 1", len(df))

# List of columns to drop
columns_to_drop = [
    'symbol',
    '52_week_high',
    '52_week_low',
    'ps_ratio',
    'revenue_growth',
    'earnings_growth',
    'roa',
    'beta',
    'quick_ratio',
    'interest_coverage',
    'free_cash_flow',
    'net_profit_margin',
    'asset_turnover'
]

# Drop the selected columns
df.drop(columns=columns_to_drop, inplace=True)
print("Total number of rows 2: ", len(df))

# # Remove rows with any missing data
# df.dropna(inplace=True)
#
# print("Total number of rows 3: ", len(df))

df.to_csv('cleaned_stock_data.csv', index=False)

with open('cleaned_stock_data.csv', 'r') as file:
    for i in range(100):
        line = file.readline()
        print(line.strip())

print("Data saved to cleaned_stock_data.csv and first 100 lines printed.")
