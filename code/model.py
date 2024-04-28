import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential

class LSTMModel(tf.keras.Model):
    def __init__(self, embed_size=64):
        super().__init__()
        self.embed_size = embed_size
        self.lstm_layer = tf.keras.layers.LSTM(embed_size, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense()

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = tf.keras.layers.Flatten()(x) 
        output = self.dense_layer(x)
        return output

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        model = Sequential([self])
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test):
        model = Sequential([self])
        model.compile(optimizer='adam', loss='mean_squared_error')
        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss