from preprocess import one_time_preprocess
from model import LSTMModel


def parse_args():
    None

def main(args):
    #---------------------------- Hyper Parameter ------------------------------
    window_size = 60 #MUST BE LESS THAN 
    #---------------------------------------------------------------------------
    #Run once
    # one_time_preprocess()
    # # Prepare training and testing data
    X_train, y_train, X_test, y_test = prepare_training_data(window_size)

    # print(X_train.shape)

    # # Instantiate LSTMModel
    model = LSTMModel(window_size)
    # # Compile the model
    model.compile_model(learning_rate = 0.01)
    # # Train the model
    history = model.train_model(X_train, y_train, X_test, y_test, epochs=10)
    # # Evaluate the model
    loss = model.evaluate_model(X_test, y_test)

    # # Print loss or any other evaluation metrics
    print("Test Loss:", loss)


if __name__ == '__main__':
    main(parse_args())
