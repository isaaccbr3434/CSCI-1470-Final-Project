import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

#---------------------------------- CONSTANTS ----------------------------------
start_date = '2002-05-01'
end_date = '2020-02-01'

def one_time_preprocess():
    #Only needs to be ran once! 

    omx30_tickers = ['HM-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SWED-A.ST', 'VOLV-B.ST', 
    'TELIA.ST', 'SHB-A.ST', 'ATCO-A.ST', 'SKA-B.ST', 'INVE-B.ST', 
    'ELUX-B.ST', 'ASSA-B.ST', 'SCA-B.ST', 'INDU-C.ST', 
    'AZN.ST', 'ATCO-B.ST', 'GETI-B.ST', 'KINV-B.ST', 'HEXA-B.ST', 
    'ALIV-SDB.ST', 'SKF-B.ST', 'LATO-B.ST', 'SAND.ST', 'INVE-A.ST', 
    'MTG-B.ST', 'FING-B.ST']

    print(len(omx30_tickers), " stocks included in adjusted portfolio. Read comments for details ")
    
    #'LUMI-SDB.ST', 'SWMA.ST', are currently unpopulated excel cells
    #'ESSITY-B.ST' not included due to lack of data for the dates of interest! Footnote 1, Pg 5
    # Kevin note - ESSITY-A starts from 2017, which was dropped later by Connie/Isaac. Removing here
    # 'ALFA.ST' dropped for now to avoid inhomogenous data

    #-------------------------In case tickers wrong---------------------------

    # tickers = [
    # 'SEB-A.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'INVE-B.ST', 'ESSITY-B.ST', 
    # 'ABB.ST', 'SWED-A.ST', 'NDA-SE.ST', 'VOLV-B.ST', 'SKF-B.ST', 
    # 'ERIC-B.ST', 'ALFA.ST', 'EVO.ST', 'BOL.ST', 'SHB-A.ST', 
    # 'SCA-B.ST', 'SAND.ST', 'KINV-B.ST', 'NIBE-B.ST', 'AZN.ST', 
    # 'ELUX-B.ST', 'HM-B.ST', 'ASSA-B.ST', 'ALIV-SDB.ST', 'TEL2-B.ST', 
    # 'GETI-B.ST', 'TELIA.ST', 'HEXA-B.ST', 'SBB-B.ST', 'SINCH.ST']

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


def prepare_data():
    
    cleaned_omx = pd.read_csv('omx_30_data.csv')
    print(cleaned_omx)

    X0 = []
    Y0 = []
    
    X1 = []
    Y1 = []

    X2 = []
    Y2 = []

    # Pg 6 constants
    train_length = 750
    validation_length = 270
    test_length = 270
    rolling_window = 30

    cleaned_omx = cleaned_omx.drop([0, 1, 2]) #Drop first 3 days to make it 4500 rows
    # Kevin Edits: Currently overcounts training data. There should be (4500-1290)/30 = 107 training blocks
    # (0, 3960) is used for training with 270 sized windows (4500-540)
    # (750, 4230) is used for validation
    # (1020, 4500) used for testing, with the last 30 technically not included for each.

    train_last_index = len(cleaned_omx) - 270 - 270 - 240
    for i in range(0, train_last_index, 750):
        block = cleaned_omx.iloc[i:i+train_length]
        for j in range(0, len(block), rolling_window):
            X0.append(block.iloc[j:j+240])
            X1.append(block.iloc[j+240]) #The median measurement is for the day after

    

    for dataset, length in [(train_sequences, 750), (validation_sequences, 270), (test_sequences, 270)]:
        for i in range(0, len(cleaned_omx), length):
            block = cleaned_omx.iloc[i:i+length]
            for j in range(0, len(block), rolling_window):
                sequence = block.iloc[j:j+240]
                dataset.append(sequence)
    
    print(len(train_sequences))
    print(len(train_sequences[150]))

    return train_sequences, validation_sequences, test_sequences

# one_time_preprocess()
prepare_data()