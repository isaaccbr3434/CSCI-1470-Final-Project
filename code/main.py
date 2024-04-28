from preprocess import prepare_training_data
from model import LSTMModel

def parse_args():
    None

def main(args):
    #---------------------------- Hyper Parameter ------------------------------
    window_size = 8 #MUST BE LESS THAN seq_len
    #---------------------------------------------------------------------------

    X0, Y0, X1, Y1 = prepare_training_data(window_size)

    # Prepare training and testing data
    X_train, y_train, X_test, y_test = prepare_training_data(window_size)

    # Instantiate LSTMModel
    model = LSTMModel()

    # Compile the model
    model.compile_model()

    # Train the model
    history = model.train_model(X_train, y_train, X_test, y_test)

    # Evaluate the model
    loss = model.evaluate_model(X_test, y_test)

    # Print loss or any other evaluation metrics
    print("Test Loss:", loss)


if __name__ == '__main__':
    main(parse_args())
