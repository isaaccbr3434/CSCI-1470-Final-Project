import pandas as pd
import yfinance as yf
from datetime import datetime

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]
ticker_symbols = sp500_table['Symbol'].tolist()

start_date = '2023-11-01'
end_date = '2024-03-01'

all_stock_data = []

for symbol in ticker_symbols:
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data['Symbol'] = symbol
        all_stock_data.append(stock_data)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

all_stock_data = pd.concat(all_stock_data)

#Cleaning up the data by removing stocks with insufficient data
missing_data_counts = all_stock_data.isnull().sum(axis=0)
missing_data_threshold = 0.1 * len(all_stock_data)
stocks_with_insufficient_data = missing_data_counts[missing_data_counts > missing_data_threshold].index.tolist()
all_stock_data = all_stock_data[~all_stock_data['Symbol'].isin(stocks_with_insufficient_data)]

# Removing the rows with None values
cleaned_stock_data = all_stock_data.dropna()
cleaned_stock_data.to_csv('cleaned_sp500_stock_data.csv')
print("Data fetching and cleaning completed.")
