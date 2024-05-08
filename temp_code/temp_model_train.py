import numpy as np
import time
import tensorflow as tf
from scipy import stats
from temp_preprocess import prepare_data
from tensorflow.keras.models import load_model
import os

#################### DON'T RUN FILE IF temp_models EXIST!!!#####################

model_folder = 'temp_models/'

class MyLSTM(tf.keras.Model):
    def __init__(self, weight_initializer, hidden_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        if weight_initializer == "constant":
            weight_initializer = tf.keras.initializers.Constant(0.5)

        print("Model weight: ", weight_initializer)

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
    return None
    start_time = time.time()

    # Prepare training and testing data.
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", Y_val.shape)
    print("Y_val shape:", Y_val.shape)
    
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
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), 
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
        os.makedirs(model_folder, exist_ok=True)

        model_path = f'{model_folder}model_{index}_{weight}'
        model.save(model_path, save_format='tf')

        print(f'Model saved to {model_path}')

        #----------------------- Print train time ------------------------------
        model_time = time.time()
        elapsed_time = start_time - model_time
        minutes = int((elapsed_time_seconds % 3600) // 60)
        print(f"Time took to train model {weight}: {minutes} minutes")
        models.append(model)


    # --------------------Time took to train all models ------------------------
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)

    print(f"Elapsed time to train 11 models: {hours} hours, {minutes} minutes, {seconds} seconds")

if __name__ == '__main__':
    None
    # main(parse_args())