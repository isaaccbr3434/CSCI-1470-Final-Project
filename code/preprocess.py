import pandas as pd
import yfinance as yf
import tensorflow as tf
from datetime import datetime
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

#---------------------------------- CONSTANTS ----------------------------------
start_date = '2002-05-01'
end_date = '2020-02-01'

omx30_tickers = ['HM-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SWED-A.ST', 'VOLV-B.ST', 
    'TELIA.ST', 'SHB-A.ST', 'ATCO-A.ST', 'SKA-B.ST', 'INVE-B.ST', 
    'ELUX-B.ST', 'ASSA-B.ST', 'SCA-B.ST', 'INDU-C.ST', 'AZN.ST', 
    'ATCO-B.ST', 'GETI-B.ST', 'KINV-B.ST', 'HEXA-B.ST', 'ALIV-SDB.ST', 
    'SKF-B.ST', 'LATO-B.ST', 'SAND.ST', 'INVE-A.ST', 'MTG-B.ST', 
    'FING-B.ST']

def one_time_preprocess():
    #Only needs to be ran once! 

    print(len(omx30_tickers), " stocks included in adjusted portfolio. Read comments for details ")
    
    #'LUMI-SDB.ST', 'SWMA.ST', are currently unpopulated excel cells
    #'ESSITY-B.ST' not included due to lack of data for the dates of interest! Footnote 1, Pg 5
    # Kevin note - ESSITY-A starts from 2017, which was dropped later by Connie/Isaac. Removing here
    # 'ALFA.ST' dropped for now to avoid inhomogenous data

    #-------------------------In case tickers wrong---------------------------

    # tickers = [
    # 'SEB-A.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'INVE-B.ST', 'ABB.ST', 
    # 'SWED-A.ST', 'NDA-SE.ST', 'VOLV-B.ST', 'SKF-B.ST', 'ERIC-B.ST', 
    # 'ALFA.ST', 'EVO.ST', 'BOL.ST', 'SHB-A.ST', 'SCA-B.ST', 
    # 'SAND.ST', 'KINV-B.ST', 'NIBE-B.ST', 'AZN.ST', 'ELUX-B.ST', 
    # 'HM-B.ST', 'ASSA-B.ST', 'ALIV-SDB.ST', 'TEL2-B.ST', 'GETI-B.ST', 
    # 'TELIA.ST', 'HEXA-B.ST', 'SBB-B.ST', 'SINCH.ST']

    # set1 = set(omx30_tickers)
    # set2 = set(tickers)
    # print("Shared: ", list(set1 & set2), "Total count: ", len(set1 & set2))
    # print("Unique to 1: ", list(set1 - set2))
    # print("Unique to 2: ", list(set2 - set1))

    # sys.exit()


    #-----------------------Pull in data for each 28 ticker---------------------
    stock_data = {}
    for ticker in omx30_tickers:
        try:
            df = yf.download(tickers=ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No data found for ticker: {ticker}. Skipping...")
                continue
            stock_data[ticker] = df['Close'].pct_change()
        except Exception as e:
            print(f"Error retrieving data for ticker {ticker}: {str(e)}")
            continue

    omx30_df = pd.DataFrame(stock_data)

    median_return = omx30_df.median(axis=1)

    omx30_df['Median_Return'] = median_return
    
    for ticker in omx30_tickers:
        if ticker not in omx30_df.columns:
            print(f"No data found for ticker: {ticker}. Skipping...")
            continue
        omx30_df[f"{ticker}_Target"] = (omx30_df[ticker] > median_return).astype(int)

    omx30_df.to_csv('omx_30_data.csv')


def prepare_data1():
    
    cleaned_omx = pd.read_csv('omx_30_data.csv')

    #Pg. 6 Constants
    sequence_length = 240
    rolling_window = 30

    train_length = 750
    validation_length = 270
    test_length = 270

    # Initialize lists to store data
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    cleaned_omx = cleaned_omx.drop([0, 1, 2]) #Drop first 3 days to make it 4500 rows

    # Kevin Edits: Currently overcounts training data. There should be (4500-1290)/30 = 107 training blocks
    # (0, 3960) is used for training with 270 sized windows (4500-540)
    # (750, 4230) is used for validation
    # (1020, 4500) used for testing, with the last 30 technically not included for each.

    for ticker in omx30_tickers:
        if ticker not in cleaned_omx.columns:
            print(f"Ticker {ticker} not found in dataset. Skipping...")
            continue

        ticker_data = cleaned_omx[ticker]
        targets = cleaned_omx[f"{ticker}_Target"]


        # -----------------------Training processing----------------------------
        train_last_index = len(cleaned_omx) - 270 - 270 - 240
        train_first_index = 0

        i = train_first_index

        while i <= train_last_index:
            # Extract sequences of length 240
            sequence = ticker_data[i:i+sequence_length]
            target = targets.iloc[i+sequence_length]

            x_train.append(sequence.values)
            y_train.append(target)

            i += rolling_window

        # print(len(x_train))
        # sys.exit()

        # -----------------------Validation processing--------------------------
        val_last_index = len(cleaned_omx) - 270 - 240
        val_first_index = 750

        i = val_first_index
        while i <= val_last_index:
            # Extract sequences of length 240
            sequence = ticker_data[i:i+sequence_length]
            target = targets.iloc[i+sequence_length]

            x_val.append(sequence.values)
            y_val.append(target)

            i += rolling_window

        # print(len(x_val))
        # print(len(y_val))
        # sys.exit()

        # -----------------------Validation processing--------------------------
        test_last_index = len(cleaned_omx) - 270
        test_first_index = 750 + 270

        i = test_first_index
        while i <= val_last_index:
            # Extract sequences of length 240
            sequence = ticker_data[i:i+sequence_length]
            target = targets.iloc[i+sequence_length]

            x_test.append(sequence.values)
            y_test.append(target)

            i+=rolling_window

        # print(len(x_test))
        # print(len(y_test))
        # sys.exit()

    # Convert lists to arrays
    x_train = np.array(x_train).reshape(-1, sequence_length, 1)
    y_train = np.array(y_train)
    x_val = np.array(x_val).reshape(-1, sequence_length, 1)
    y_val = np.array(y_val)
    x_test = np.array(x_test).reshape(-1, sequence_length, 1)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

def prepare_data():
    cleaned_omx = pd.read_csv('omx_30_data.csv')
    cleaned_omx = cleaned_omx.drop([0, 1, 2]) #Drop first 3 days to make it 4500 rows
    # print(cleaned_omx)

    #Pg. 6 Constants
    SEQUENCE_LENGTH = 240
    ROLLING_WINDOW = 30
    TRAIN_LENGTH = 750
    VALIDATION_LENGTH = 270
    TEST_LENGTH = 270 #NEVER USED

    # Initialize lists to store data
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    # For each of the (26) stocks
    for ticker in omx30_tickers:
        if ticker not in cleaned_omx.columns:
            print(f"Ticker {ticker} not found in dataset. Skipping...")
            continue

        # Select the data for the current ticker
        ticker_data = cleaned_omx[ticker]
        targets = cleaned_omx[f"{ticker}_Target"]

        for i in range(0, 3210 + 1, ROLLING_WINDOW):
            # -----------------------Training processing------------------------
            train_end = i + TRAIN_LENGTH
            train_data = ticker_data[i : train_end]
            train_targets = targets[i : train_end + 1].values

            for j in range(0, len(train_data) - SEQUENCE_LENGTH + 1, ROLLING_WINDOW):
                x_train.append(train_data.iloc[j:j+240].values)
                y_train.append(train_targets[j+240])

            # ----------------------Validation processing-----------------------
            val_start = train_end
            val_end = val_start + VALIDATION_LENGTH
            val_data = ticker_data[val_start : val_end]
            val_targets = targets[val_start : val_end + 1].values

            for j in range(0, len(val_data) - SEQUENCE_LENGTH + 1, ROLLING_WINDOW):
                x_val.append(val_data.iloc[j:j+240].values)
                y_val.append(val_targets[j+240])

            # -----------------------Testing processing-------------------------
            test_start = val_end
            test_end = test_start + SEQUENCE_LENGTH
            
            x_test.append(ticker_data[test_start : test_end].values)
            y_test.append(targets[test_end])


    # Convert lists to arrays
    x_train = np.array(x_train).reshape(-1, SEQUENCE_LENGTH, 1)
    y_train = np.array(y_train)
    x_val = np.array(x_val).reshape(-1, SEQUENCE_LENGTH, 1)
    y_val = np.array(y_val)
    x_test = np.array(x_test).reshape(-1, SEQUENCE_LENGTH, 1)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def prepare_data2():
    SEQUENCE_LENGTH = 240
    ROLLING_WINDOW = 30
    TRAIN_LENGTH = 750
    VALIDATION_LENGTH = 270
    TEST_LENGTH = 270 #NEVER USED

    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(np.genfromtxt('omx_30_data.csv', skip_header = 4, delimiter = ',', usecols = [i for i in range(1, 27)]))

    y_data = np.genfromtxt('omx_30_data.csv', skip_header = 4, delimiter = ',', usecols = [i for i in range(28, 54)])
    # with open('test.txt', 'w') as f:
    #     print(data, file = f)
    x_train = x_data[0:TRAIN_LENGTH]
    y_train = y_data[0:TRAIN_LENGTH]
    val_start = TRAIN_LENGTH
    x_val = x_data[val_start:val_start+VALIDATION_LENGTH]
    y_val = y_data[val_start:val_start+VALIDATION_LENGTH] 
    test_start = val_start + VALIDATION_LENGTH
    x_test = x_data[test_start:test_start + SEQUENCE_LENGTH]
    y_test = y_data[test_start:test_start + SEQUENCE_LENGTH]
    return x_train, y_train, x_val, y_val, x_test, y_test
# one_time_preprocess()
# x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()
# prepare_data2()



# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_val shape:", x_val.shape)
# print("y_val shape:", y_val.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)


# print("x_train len:", len(x_train))
# print("y_train len:", len(y_train))
# print("x_val len:", len(x_val))
# print("y_val len:", len(y_val))
# print("x_test len:", len(x_test))
# print("y_test len:", len(y_test))

# with open('test.txt', 'w') as f:
#     print(x_test, file = f)
#     print(y_test, file = f)