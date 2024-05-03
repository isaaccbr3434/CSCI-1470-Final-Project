import numpy as np
import tensorflow as tf
from scipy import stats
from preprocess import prepare_data
from tensorflow.keras.models import load_model

class MyLSTM(tf.keras.Model):
    def __init__(self, hidden_size=3, weight_initializer="glorot_uniform"):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_dense = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=3, #Should be able to not have to rely on hard code
                kernel_initializer=weight_initializer,
                dropout=0.06,
                recurrent_dropout=0.14
            ),
            tf.keras.layers.Dense(
                units=1,
                activation='sigmoid',
                use_bias=True,
                bias_initializer="truncated_normal"
            )
        ])
        
    def call(self, inputs):
        output = self.lstm_dense(inputs)
        return output
    
def main(args):
    # Prepare training and testing data.
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()
    
    # Initialize hyperparameters from paper
    learning_rate = 0.0075
    epochs = 10
    batch_size = 6800

    # Instantiate models.
    weight_initializers = ["random_normal", "random_uniform", "truncated_normal", "zeros", 
                           "ones", "glorot_normal", "glorot_uniform", "identity",
                           "orthogonal", "constant", "variance_scaling"] # FIX CONSTANT.
    
    models = []
    for index, weight in enumerate(weight_initializers):
        model = MyLSTM(weight)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        model.fit(
            X_train, 
            Y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val, Y_val), 
            verbose=1
        )

       # -------------------Save the 11 models with initializer----------------
        model_path = f'models/model_{index}_{weight}'
        model.save(model_path, save_format='tf')
        print(f'Model saved to {model_path}')

        models.append(model)
    
    # # ------------------------ Loading saved models ----------------------------
    # for index, weight in enumerate(weight_initializers):
    #     model = load_model(f'models/model_{index}_{weight}')
    #     print(f'\nModel {weight} loaded from models/model_{index}_{weight}\n')
    #     models.append(model)

    # predictions = [(model.predict(X_test, batch_size=batch_size) > 0.5).astype("int32") for model in models] #(11, 2808, 1)
    # predictions = np.array(predictions)
    # mode_result = stats.mode(predictions, axis=0)
    # mode_predictions = mode_result.mode.flatten() #(2808, )

    #Get predictions from all 11 models across 2808 tests in X_tests
    predictions = np.concatenate([model.predict(X_test, batch_size=batch_size) for model in models], axis=1) #(2808, 11)
    mode_predictions = stats.mode(predictions, axis=1).mode.flatten() #(2808, )
    
    #Replaced bce with accuracy since all are whole numbers, and measure % accuracy
    accuracy = tf.keras.metrics.BinaryAccuracy()
    accuracy.update_state(Y_test, mode_predictions)
    print("Accuracy out of 2808 tests: ", accuracy.result().numpy()) #Always 0.50035614

def parse_args():
    None 

if __name__ == '__main__':
    main(parse_args())