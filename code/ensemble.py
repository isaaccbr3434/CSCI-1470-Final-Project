import numpy as np
import tensorflow as tf
from scipy import stats
from preprocess import prepare_data, prepare_data2
from tensorflow.keras.models import load_model
import keras
from keras import layers

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

class MyGRU(tf.keras.Model):
    def __init__(self, hidden_size=3, weight_initializer="glorot_uniform"):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_dense = tf.keras.Sequential([
            tf.keras.layers.GRU(
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

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, kernel_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Conv1D(filters=1, kernel_size = kernel_size, activation = 'relu')
        self.pos_emb = layers.Embedding(input_dim=200, output_dim=embed_dim)

    def call(self, x):
        x = tf.expand_dims(x, axis = 1)
        positions = tf.range(0, len(x))
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class MyTransformer(tf.keras.Model):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self):
        super().__init__()

        self.model_layers = tf.keras.Sequential([
            TokenAndPositionEmbedding(32, 32),
            TransformerBlock(32, 2, 32),
            tf.keras.layers.Dense(units = 10, activation = 'relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])


    def call(self, inputs):
        print(inputs.shape)
        output = self.model_layers(inputs)
        return output

    
def main(args):
    # Prepare training and testing data.
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

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
        model = MyLSTM()
        model = MyGRU()
        # model = MyTransformer()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        model.fit(
            X_train, 
            Y_train, 
            epochs=epochs, 
            batch_size=32, 
            validation_data=(X_val, Y_val), 
            verbose=1
        )

       # -------------------Save the 11 models with initializer----------------
        model_path = f'models/model_{index}_{weight}'
        model.save(model_path, save_format='tf')
        print(f'Model saved to {model_path}')

        models.append(model)
    
    # # ------------------------ Loading saved models ----------------------------
    for index, weight in enumerate(weight_initializers):
        model = load_model(f'models/model_{index}_{weight}')
        print(f'\nModel {weight} loaded from models/model_{index}_{weight}\n')
        models.append(model)

    predictions = [(model.predict(X_test, batch_size=batch_size) > 0.5).astype("int32") for model in models] #(11, 2808, 1)
    predictions = np.array(predictions)
    mode_result = stats.mode(predictions, axis=0)
    mode_predictions = mode_result.mode.flatten() #(2808, )

    #Get predictions from all 11 models across 2808 tests in X_tests
    # predictions = np.concatenate([model.predict(X_test, batch_size=batch_size) for model in models], axis=1) #(2808, 11)
    # mode_predictions = stats.mode(predictions, axis=1).mode.flatten() #(2808, )
    
    #Replaced bce with accuracy since all are whole numbers, and measure % accuracy
    accuracy = tf.keras.metrics.BinaryAccuracy()
    accuracy.update_state(Y_test, mode_predictions)
    print("Accuracy out of 2808 tests: ", accuracy.result().numpy()) #Always 0.50035614

def parse_args():
    None 

if __name__ == '__main__':
    main(parse_args())