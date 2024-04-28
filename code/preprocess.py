import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def one_time_preprocess():
    #Only needs to be ran once! 

    #------------------------- Independent Variables ---------------------------
    start_date = '2023-11-01'
    end_date = '2024-03-01'
    #---------------------------------------------------------------------------

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    # print(sp500_table)
    ticker_symbols = sp500_table['Symbol'].tolist()

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


def prepare_training_data(window_size):
    cleaned_stock_data = pd.read_csv('cleaned_sp500_stock_data.csv')
    # print(cleaned_stock_data)

    #create a dictionary that is Symbol (str) -> Data across time (data frame)
    stock_data_dict = dict()
    for symbol, data in cleaned_stock_data.groupby('Symbol'):
        stock_data_dict[symbol] = data.drop(columns=['Symbol', 'Date'])
    
    # Populate training & testing features + labels 
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for symbol, data in stock_data_dict.items():

        #-------------Create all of the features + labels necessary-------------
        feature = np.array(data.values)[:-1] #This is (days, features)
        # Binary Task, where 1 if greater the next day 0 if not
        # Last date not included since can't make decision
        target = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)[:-1] #Bn
        assert(feature.shape[0] > window_size)
        assert(len(feature) == len(target))


        #------------------Split into train/test feature/label------------------
        train_length = math.ceil(len(feature) * 0.8)

        train_features = feature[:train_length]
        train_labels = target[:train_length]
        test_features = feature[train_length:]
        test_labels = target[train_length:]

        # -------------Need to scale the data to be betwene 0 and 1-------------
        scaler = MinMaxScaler()
        # Fit the scaler on the training data and transform the training data
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        #----------create a series of windows for training and testing----------
        num_windows_train = len(train_features) + 1 - window_size
        for i in range(num_windows_train):
            X_train.append(train_features[i:i+window_size])
            Y_train.append(train_labels[i+window_size-1])

        num_windows_test = len(test_features) + 1 - window_size
        for i in range(num_windows_test):
            X_test.append(test_features[i:i+window_size])
            Y_test.append(test_labels[i+window_size-1])


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test
    #X_train = (num_stock * num_windows_train, window_size, num_features)
    #For each window of num_features there should be 1 