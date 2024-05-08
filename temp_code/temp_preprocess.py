import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt

omx30_tickers = ['HM-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SWED-A.ST', 'VOLV-B.ST', 
    'TELIA.ST', 'SHB-A.ST', 'ATCO-A.ST', 'SKA-B.ST', 'INVE-B.ST', 
    'ELUX-B.ST', 'ASSA-B.ST', 'SCA-B.ST', 'INDU-C.ST', 'AZN.ST', 
    'ATCO-B.ST', 'GETI-B.ST', 'KINV-B.ST', 'HEXA-B.ST', 'ALIV-SDB.ST', 
    'SKF-B.ST', 'LATO-B.ST', 'SAND.ST', 'INVE-A.ST', 'MTG-B.ST', 
    'FING-B.ST']

omx30_file_name = 'omx_30_data.csv'

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


def one_time_preprocess():
    #Only needs to be ran once!

    #------------------------- Independent Variables ---------------------------
    start_date = '2002-05-01'
    end_date = '2020-02-01'
    #---------------------------------------------------------------------------
    global omx30_tickers
    print(len(omx30_tickers), " stocks included in adjusted portfolio. Read comments for details ")

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


    #This stock had no data
    #    omx30_df.drop(columns=['ESSITY-A.ST'], inplace=True)
    #    omx30_tickers.remove('ESSITY-A.ST')


    for ticker in omx30_tickers:
        if ticker not in omx30_df.columns:
            print(f"No data found for ticker: {ticker}. Skipping...")
            continue
        omx30_df[f"{ticker}_Target"] = (omx30_df[ticker] > median_return).astype(int)

    omx30_df.to_csv(omx30_file_name)
    print(omx30_df.head())


def prepare_data():
   #------------------------- Independent Variables ---------------------------
   start_date = '2002-05-01'
   end_date = '2020-02-01'
   #---------------------------------------------------------------------------

   cleaned_omx = pd.read_csv(omx30_file_name)
   cleaned_omx = cleaned_omx.drop([0, 1, 2]) #Drop first 3 days to make it 4500 rows

   sequence_length = 240
   rolling_window = 30

   train_length = 750
   validation_length = 270
   test_length = 270

   # Initialize lists to store data
   x_train, y_train = [], []
   x_val, y_val = [], []
   x_test, y_test = [], []

   for ticker in omx30_tickers:
       if ticker not in cleaned_omx.columns:
           print(f"Ticker {ticker} not found in dataset. Skipping...")
           continue

       ticker_data = cleaned_omx[ticker]
       targets = cleaned_omx[f"{ticker}_Target"]

       i_train, i_val, i_test = 0, 0, 0 
       i_indexer = 0
        # Initialize counters

       i = 0

       counter = 0

       while i + 1290 < len(ticker_data):

            i_indexer = counter * 30
            
            if i_train + sequence_length < train_length:
                sequence = ticker_data[i_indexer:i_indexer + sequence_length]
                target = targets.iloc[i_indexer + sequence_length]
                x_train.append(sequence.values)
                y_train.append(target)
                i_train += 1

            elif i_val + sequence_length < validation_length:
                sequence = ticker_data[i_indexer:i_indexer + sequence_length]
                target = targets.iloc[i_indexer + sequence_length]
                x_val.append(sequence.values)
                y_val.append(target)
                i_train += 1
                i_val += 1

            elif i_test + sequence_length <= sequence_length:
                sequence = ticker_data[i_indexer:i_indexer + sequence_length]
                target = targets.iloc[i_indexer + sequence_length]
                x_test.append(sequence.values)
                y_test.append(target)
                i_test += 1

            else:
                i_train, i_val, i_test = 0, 0, 0
                i += rolling_window 
                counter +=1 # Initialize counters
                    
   # Convert lists to arrays
   x_train = np.array(x_train).reshape(-1, sequence_length, 1)
   y_train = np.array(y_train)
   x_val = np.array(x_val).reshape(-1, sequence_length, 1)
   y_val = np.array(y_val)
   x_test = np.array(x_test).reshape(-1, sequence_length, 1)
   y_test = np.array(y_test)


   return x_train, y_train, x_val, y_val, x_test, y_test

# one_time_preprocess()
# x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()


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


# num_samples = 5
# for i in range(num_samples):
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_train[i], label='Sequence')
#     plt.axhline(y=y_train[i], color='r', linestyle='--', label='Target')
#     plt.title(f"Sample Sequence {i+1} (Target: {y_train[i]})")
#     plt.xlabel("Days")
#     plt.ylabel("Rate of Return")
#     plt.legend()
#     plt.show()

