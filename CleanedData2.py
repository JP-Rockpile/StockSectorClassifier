import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_stock_data.csv')

# Drop rows that are fully blank
df.dropna(how='all', inplace=True)

# Fill missing values with sector-based mean
def fill_missing_with_sector_mean(df):
    # Replace inf and -inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the mean of the column grouped by sector
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df.groupby('sector')[column].transform(lambda x: x.fillna(x.mean()))
    return df

# Fill missing values
df = fill_missing_with_sector_mean(df)

# Transform sector names to 'Other' IF NOT 'Energy', 'Utilities', or 'Technology'
sectors_to_keep = ['Energy', 'Utilities', 'Technology']
df['sector'] = df['sector'].apply(lambda x: x if x in sectors_to_keep else 'Other')

# Total number of rows
print("Total number of rows:", len(df))

df.to_csv('cleaned_stock_data2.csv', index=False)

with open('cleaned_stock_data2.csv', 'r') as file:
    for i in range(100):
        line = file.readline()
        print(line.strip())

print("Data saved to cleaned_stock_data2.csv and first 100 lines printed.")
