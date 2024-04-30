# CSCI-1470-Final-Project
Final project for Deep Learning which is a binary classification model trained to decide when to buy and sell stocks to maximize profit

Development process:
-Preprocess S&P 500 stock data to get all of the [Date,Open,High,Low,Close,Adj Close,Volume] for each sticker from 2023-11-01 through 2024-03-01
    - Note that for the 82 days, we trimmed off the last day because we don't know if we should hold/sell that day. For a given index on Y_train, let's say 5, if it is 1 that means that on day 5 the stock should be held, and if it's 0 it should be sold.
-Create a LSTM model
4/30/24
-Gonna use a single stock, stll all 6 features, and see where that leads us.