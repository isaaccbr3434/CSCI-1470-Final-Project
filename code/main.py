from preprocess import prepare_training_data
from model import LSTMModel

def parse_args():
    None

def main(args):
    #---------------------------- Hyper Parameter ------------------------------
    window_size = 8 #MUST BE LESS THAN seq_len
    #---------------------------------------------------------------------------

    X0, Y0, X1, Y1 = prepare_training_data(window_size)

    model = LSTMModel()
    model.compile_model()
    model.train_model()

    # Pull the data
    # Run into model
    

if __name__ == '__main__':
    main(parse_args())
