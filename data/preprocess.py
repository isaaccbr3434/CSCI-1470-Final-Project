import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

def preprocess():
    #Only needs to be ran once! 
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    # print(sp500_table)
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
    # print(type(all_stock_data))

    #Cleaning up the data by removing stocks with insufficient data
    missing_data_counts = all_stock_data.isnull().sum(axis=0)
    missing_data_threshold = 0.1 * len(all_stock_data)
    stocks_with_insufficient_data = missing_data_counts[missing_data_counts > missing_data_threshold].index.tolist()
    all_stock_data = all_stock_data[~all_stock_data['Symbol'].isin(stocks_with_insufficient_data)]

    # Removing the rows with None values
    cleaned_stock_data = all_stock_data.dropna()
    cleaned_stock_data.to_csv('cleaned_sp500_stock_data.csv')


def prepare_training_data():
    """
    499 stocks, 82 days, 6 features
    """
    cleaned_stock_data = pd.read_csv('cleaned_sp500_stock_data.csv')

    #create a dictionary that is Symbol (str) -> Data across time (data frame)
    stock_data_dict = dict()
    for symbol, data in cleaned_stock_data.groupby('Symbol'):
        stock_data_dict[symbol] = data.drop(columns=['Symbol', 'Date'])

    
    # From the dictionary create a (499, 81, 7) (#_stocks, seq_length, num_feature) X_train and a (499, 81) (#_stocks, binary) label

    X_train = []
    Y_train = []

    for symbol, data in stock_data_dict.items():
         #Slicing off last date for labeling purpose
        feature = np.array(data.values)[:-1]
        ## Binary Task, where 1 if greater, 0 if not
        target = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)[:-1] 

        # print(symbol)
        # print(data)
        # print(len(feature))
        # print(len(target))

        X_train.append(feature)
        Y_train.append(target)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

x, y = prepare_training_data()