# CSCI-1470-Final-Project
Final project for Deep Learning which is a binary classification model trained to decide when to buy and sell stocks to maximize profit

Introduction:
Our project aims to leverage deep learning techniques for stock market prediction, using an ensemble of LSTMs based on Carmina Fjellstrom’s paper on “Long Short-Term Memory Neural Network for Financial Time Series” Our project independently and concurrently trains 11 LSTM models, each initialized with different weight settings, to create an ensemble of predictions for every stock’s daily movement within the Swedish OMX Stockholm 30 index. We then applied the predictions from our deep learning model to a trading bot that aimed to beat the index’s performance through day trading on the binary classification task model. Additionally, we implemented our strategy to our trading bot to see how well our model worked. With our strategy, our portfolio outperformed the OMX 30 with the buy and hold strategy from 2008-2009.

User Instructions:
- ONLY USE temp_code FOLDER AND temp_models!!!! Ignore everything else
- User only needs to run temp_predictions (modify comments in main) OR temp_trading_bot, and nothing else.


File Descriptions
- temp_preprocess is used to create the OMX30 csv get all X0, X1, Y0, Y1, and testing data
- temp_model_train trains 11 LSTM models and store them to temp_models
    - Takes 5~7hrs to train, so use the temp_models folder to load the models
- temp_predictions is used to make stock predictions for any given day on any stock
- temp_trading_bot implements a trading program using our model outputs. 


Development process:
-Preprocess S&P 500 stock data to get all of the [Date,Open,High,Low,Close,Adj Close,Volume] for each sticker from 2023-11-01 through 2024-03-01
    - Note that for the 82 days, we trimmed off the last day because we don't know if we should hold/sell that day. For a given index on Y_train, let's say 5, if it is 1 that means that on day 5 the stock should be held, and if it's 0 it should be sold.
-Create a LSTM model
4/30/24
-Gonna use a single stock, stll all 6 features, and see where that leads us.
-Model predicted only sell for 30 day windows

-Decided to pursue a new preprocessing that yields much more windows (many repeating) in temp_preprocess
-Training the 11 LSTM models took an entire night with more data (1.4 million vs. 50K) in temp_model_train
x_train shape: (1418820, 240, 1)
y_train shape: (1418820,)
x_val shape: (83460, 240, 1)
y_val shape: (83460,)
x_test shape: (2782, 240, 1)
y_test shape: (2782,)

-Takes ~20 seconds to load the models, and < 3s to get a daily stock_prediction for any given day in temp_predictions

-accuracy data for x_test is in temp_predictions, with distribution of data showing accuracy is not due to pure true negatives and false positives

-All temp_files are linked - omx_30 stock list and the file directory are shared

-Well developed trading bot by Connie

