import numpy as np
import tensorflow as tf
from scipy import stats
from preprocess import prepare_data

class LSTM(tf.keras.Model):
    def __init__(self, hidden_size=3, weight_initializer="glorot_uniform"):
        super().__init__()
        self.hidden_size = 3
        self.LSTM = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.hidden_size,
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
        output = self.LSTM(inputs)
        return output
    
def main(args):
    # Prepare training and testing data.
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()
    
    # Initialize hyperparameters.
    learning_rate = 0.0075
    epochs = 10
    batch_size = 6800 # RECALCULATE.

    # Instantiate models.
    weight_initializers = ["random_normal", "random_uniform", "truncated_normal", "zeros", 
                           "ones", "glorot_normal", "glorot_uniform", "identity",
                           "orthogonal", "constant", "variance_scaling"] # FIX CONSTANT.
    models = []

    for weight in weight_initializers:
        model = LSTM(weight)
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
        models.append(model)
        
    predictions = [(model.predict(X_test, batch_size=batch_size) > 0.5).astype("int32") for model in models]
    predictions = np.array(predictions)
    mode_result = stats.mode(predictions, axis=0)  
    mode_predictions = mode_result.mode.flatten()
    # predictions = (stats.mode(predictions)).flatten()
    # predictions = tf.convert_to_tensor(predictions)
    
    bce = tf.keras.losses.BinaryCrossentropy()
    test_loss = bce(Y_test, predictions)
    print("Test Loss: ", test_loss)
    
def parse_args():
    None

if __name__ == '__main__':
    main(parse_args())
    