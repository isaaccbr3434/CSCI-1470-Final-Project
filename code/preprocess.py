import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def one_time_preprocess():
    #Only needs to be ran once! 

    #------------------------- Independent Variables ---------------------------
    start_date = '2002-05-01'
    end_date = '2020-02-01'
    #---------------------------------------------------------------------------
    omx30_tickers = ['HM-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SWED-A.ST', 'VOLV-B.ST', 'TELIA.ST', 'SHB-A.ST', 'ATCO-A.ST', 'SKA-B.ST',
                    'INVE-B.ST', 'ELUX-B.ST', 'ASSA-B.ST', 'SCA-B.ST', 'LUMI-SDB.ST', 'ALFA.ST', 'INDU-C.ST', 'AZN.ST', 'ATCO-B.ST',
                    'ESSITY-B.ST', 'GETI-B.ST', 'KINV-B.ST', 'HEXA-B.ST', 'SWMA.ST', 'ALIV-SDB.ST', 'SKF-B.ST', 'LATO-B.ST', 'SAND.ST',
                    'INVE-A.ST', 'MTG-B.ST', 'FING-B.ST', 'ESSITY-A.ST']
    
    stock_data = {}
    for ticker in omx30_tickers:
        # Download stock data for the current ticker
        df = yf.download(tickers=ticker, start=start_date, end=end_date)
        # Calculate daily returns for the current ticker
        stock_data[ticker] = df['Close'].pct_change()

    omx30_df = pd.DataFrame(stock_data)


    median_return = omx30_df.median(axis=1)

    omx30_df['Median_Return'] = median_return
    
    #This stock had no data
    omx30_df.drop(columns=['ESSITY-A.ST'], inplace=True)
    omx30_tickers.remove('ESSITY-A.ST')

    for ticker in omx30_tickers:
        omx30_df[f"{ticker}_Target"] = (omx30_df[ticker] > median_return).astype(int)

    omx30_df.to_csv('omx_30_data.csv')
    print(omx30_df.head())

def prepare_data():
    cleaned_omx = pd.read_csv('omx_30_data.csv')

    train_length = 750
    validation_length = 270
    test_length = 270

    train_sequences = []
    validation_sequences = []
    test_sequences = []

    for dataset, length in [(train_sequences, train_length), (validation_sequences, validation_length), (test_sequences, test_length)]:
        for i in range(0, len(cleaned_omx), length):
            block = cleaned_omx.iloc[i:i+length]
            for j in range(0, len(block), 30):
                sequence = block.iloc[j:j+240]
                dataset.append(sequence)

    return train_sequences, validation_sequences, test_sequences


#one_time_preprocess()
prepare_data()