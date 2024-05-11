import numpy as np
import time
import tensorflow as tf
from scipy import stats
from temp_preprocess import prepare_data, omx30_tickers, omx30_file_name
from temp_model_train import model_folder
from tensorflow.keras.models import load_model
import os
import pandas as pd

# ------------------------ Loading saved models ----------------------------
# Not a global because 
model_time = time.time()
models = []
weight_initializers = ["random_normal", "random_uniform", "truncated_normal", "zeros", 
                        "ones", "glorot_normal", "glorot_uniform", "identity",
                        "orthogonal", "constant", "variance_scaling"]
print("Loading models...")
os.makedirs(model_folder, exist_ok=True)
for index, weight in enumerate(weight_initializers):
    model = load_model(f'{model_folder}model_{index}_{weight}')
    print(f'Model {weight} loaded from {model_folder}model_{index}_{weight}\n')
    models.append(model)
end_model_time = time.time()
elapsed_model = end_model_time - model_time
print(f"Time took to load models: {elapsed_model} seconds\n")


def get_stock_predictions(ticker_symbol, target_date):
    cleaned_omx = pd.read_csv(omx30_file_name)
    cleaned_omx['Date'] = pd.to_datetime(cleaned_omx['Date'])

    # Set the 'Date' column as the index
    cleaned_omx.set_index('Date', inplace=True)

    #ticker_data = cleaned_omx[ticker_symbol]
    target_date_index = cleaned_omx.index.get_loc(pd.to_datetime(target_date))
    sequence_length = 240
    i_test = 0
    i_indexer = target_date_index-269

    x_test = []
    while i_test + sequence_length <= sequence_length:
        # cleaned_omx.iloc[start_index:target_date_index + 1][ticker_symbol]
        sequence = cleaned_omx.iloc[i_indexer:i_indexer + sequence_length][ticker_symbol]
        x_test.append(sequence.values)
        i_test += 1

    x_window = np.array(x_test).reshape(-1, sequence_length, 1)
    # print(x_window.shape) #(1, 240, 1)

    predictions = [model.predict(x_window, batch_size=64) for model in models]
    predictions = np.array(predictions) #Originally shaped (11, 1, 1)
    binary_predictions = (predictions.reshape(11,)>0.5).astype(int)
    print(binary_predictions) #The 11 model's binary predictions 

    prediction = stats.mode(binary_predictions)[0]
    verbal_prediction = "BUY" if prediction == 1 else "SELL"

    print(f"Prediction of {ticker_symbol} on {target_date}: {verbal_prediction}\n")

    return prediction

def loop_stocks(target_date):
    #Time Measurement
    daily_prediction_time = time.time()
    
    all_predictions = []
    cleaned_omx = pd.read_csv(omx30_file_name)

    #For every stock, get the prediction of the day in binary
    for ticker in omx30_tickers:
       if ticker not in cleaned_omx.columns:
           print(f"No data found for ticker: {ticker}. Skipping...")
           continue
       stock_prediction = get_stock_predictions(ticker, target_date)
       all_predictions.append(stock_prediction)

    print(all_predictions)
    map_predictions = dict()

    #Convert all binary into verbal decision and print decision
    verbal_predictions = ["Buy/Hold" if mode == 1 else "Sell/Abstain" for mode in all_predictions]

    print(f"LSTM Model predictions for {target_date}:")
    for b_prediction, v_prediction, stock in zip(all_predictions, verbal_predictions, omx30_tickers):
        map_predictions[stock] = b_prediction
        print(f"Portfolio should {v_prediction} {stock}")

    print("Total buys:", all_predictions.count(1))
    print("Total sells:", all_predictions.count(0))

    #End time measurement and return MAP OF DECISIONS
    end_daily_prediction_time = time.time()
    print(f"Daily index stock prediction time: {end_daily_prediction_time - daily_prediction_time} seconds\n")

    return map_predictions


def accuracy_x_test():
    #Preprocessing necssary for X_test and Y_test
    print("Preprocessing Data...")
    preprocessing_time = time.time()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()
    end_preprocessing_time = time.time()
    print(f"X_test's shape: {X_test.shape}") #(2782, 240, 1)
    y_ups = sum(y == 1 for y in Y_test) 
    print(f"Percentage of stock ups: {round(y_ups / Y_test.shape[0]*100,2)}%")
    print(f"Preprocessing Time: {end_preprocessing_time - preprocessing_time} seconds\n")


    #Get all predictions for every model
    predictions = [model.predict(X_test, batch_size=64) for model in models]
    predictions = np.array(predictions)
    print(f"predictions shape: {predictions.shape}") #(11, 2782, 1)

    #Make them binary predictions and then get the model mode's for all 2782
    binary_predictions = (predictions > 0.5).astype(int)
    mode_result = stats.mode(binary_predictions, axis=0)
    mode_predictions = mode_result.mode.flatten()
    print(f"{len(mode_predictions)} moded predictions\n") #2782 preidctions

    #Understand false/true positives/negatives
    x_ups = sum(x == 1 for x in mode_predictions) 
    print(f"Percentage of stock ups predicted: {round(x_ups / len(mode_predictions)*100,2)}%")
    y_ups = sum(y == 1 for y in Y_test) 
    print(f"Percentage of stock ups actual: {round(y_ups / Y_test.shape[0]*100,2)}%\n")

    true_positives = sum(x == y == 1 for x, y in zip(mode_predictions, Y_test))
    false_positives = sum(x ==1 and y == 0 for x, y in zip(mode_predictions, Y_test))
    true_negatives = sum(x == y == 0 for x, y in zip(mode_predictions, Y_test))
    false_negatives = sum(x ==0 and y == 1 for x, y in zip(mode_predictions, Y_test))

    print(f"Number of true positives: {true_positives}, {round(true_positives/len(mode_predictions)*100,4)}%")
    print(f"Number of false positives: {false_positives}, {round(false_positives/len(mode_predictions)*100,4)}%")
    print(f"Number of true negatives: {true_negatives}, {round(true_negatives/len(mode_predictions)*100,4)}%")
    print(f"Number of false negatives: {false_negatives}, {round(false_negatives/len(mode_predictions)*100,4)}%\n")

    # Replaced bce with accuracy since all are whole numbers, and measure % accuracy.
    accuracy = tf.keras.metrics.BinaryAccuracy()
    accuracy.update_state(Y_test, mode_predictions)
    accuracy = accuracy.result().numpy() #32byte float
    print(f"Accuracy of LSTM(X_test) to Y_test (2808 tests): {round(accuracy*100, 4)}%")
    return accuracy


#This takes around 40 minutes to get 2 ~(192, 26, 11) prediction & label numpys
def plot_accuracies():
    all_predictions = []
    all_performances = []
    for year in range(2004, 2020):
        for month in range(1, 13):
            date = ''
            if (month < 10):
                date = f"{year}-0{month}-01"
            else:
                date = f"{year}-{month}-01"
            print(date)

            start_time = time.time()

            stock_predictions = []
            stock_performances = []

            for ticker_symbol in omx30_tickers:
                print(ticker_symbol)
                cleaned_omx = pd.read_csv(omx30_file_name)

                #--------------------Get window of training data ---------------
                cleaned_omx['Date'] = pd.to_datetime(cleaned_omx['Date'])

                # Set the 'Date' column as the index
                cleaned_omx.set_index('Date', inplace=True)

                #TRY CATCH cause trading days don't always start on the 1st
                try:
                    date_index = cleaned_omx.index.get_loc(pd.to_datetime(date))
                except KeyError:
                    print(f"{month}-01 is not a trading day. trying 02")
                    date = date[:-1] + str(int(date[-1]) + 1)
                    try:
                        date_index = cleaned_omx.index.get_loc(date)
                    except KeyError:
                        print(f"{month}-02 is not a trading day either. trying 03")
                        date = date[:-1] + str(int(date[-1]) + 1)
                        try:
                            date_index = cleaned_omx.index.get_loc(date)
                        except KeyError:
                            print(f"{month}-03 is not a trading day either. trying 04 (only May 2009)")
                            date = date[:-1] + str(int(date[-1]) + 1)
                            date_index = cleaned_omx.index.get_loc(date)

                sequence_length = 240
                i_test = 0
                i_indexer = date_index-239

                x_test = []
                while i_test + sequence_length <= sequence_length:
                    sequence = cleaned_omx.iloc[i_indexer:i_indexer + sequence_length][ticker_symbol]
                    # print(sequence.values)
                    x_test.append(sequence.values)
                    i_test += 1
                x_window = np.array(x_test).reshape(1, sequence_length, 1)
                # print(x_window.shape) #(1, 240, 1)

                #---------------Get prediction off of training data ------------
                predictions = [model.predict(x_window, batch_size=256) for model in models]
                predictions = np.array(predictions) #Originally shaped (11, 1, 1)
                binary_predictions = (predictions > 0.5).astype(int)
                binary_predictions_model = binary_predictions.reshape(11,)
                print(binary_predictions_model)
                # print(f"Mode prediction for stock {ticker_symbol}: {mode_predictions}")
                stock_predictions.append(binary_predictions_model)

                #-----------------------Get the actual performance------------------
                targets = cleaned_omx[f"{ticker_symbol}_Target"]
                target = targets.iloc[date_index]
                print(target)
                stock_performances.append(target)

            end_time = time.time()
            print(f"Took {end_time - start_time} seconds to get prediction over 30 models for {month}, {year} of 192 data points\n")
        all_performances.append(stock_performances)
        all_predictions.append(stock_predictions)

    all_predictions = np.array(all_predictions)
    all_performances = np.array(all_performances)
    
    np.save("my_all_predictions.npy", all_predictions)
    np.save("my_all_performances.npy", all_performances)

    print(f"All predictions shape: {all_predictions.shape}")
    print(f"All performances shape: {all_performances.shape}")

            

def main(args):
    # main_time = time.time()
    # accuracy_x_test()

    # # -------------- Portfolio action based on model predictions ---------------
    # target_date = '2015-01-29'
    # map_decision = loop_stocks(target_date)
    # print(map_decision)

    plot_accuracies()
    # end_main_time = time.time()
    # print(f"Total Time of program: {end_main_time - main_time} seconds\n")
    return None

def parse_args():
    #saved up to model 6 glorot uniform
    None 

if __name__ == '__main__':
    main(parse_args())
