import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, GRU
from tensorflow.python.keras.layers.recurrent import LSTM, GRU
from tensorflow.python.keras.models import Sequential

class LSTMModel(tf.keras.Model):
    def __init__(self, window_size, units=50):
        super().__init__()
        self.window_size = window_size
        self.units = units
        self.lstm_layer = LSTM(units=self.units, return_sequences=True, input_shape = (window_size, 1), dropout = 0.06, recurrent_dropout = 0.14)
        self.lstm_layer2 = LSTM(units = self.units, return_sequences=False, dropout = 0.06, recurrent_dropout = 0.14)
        self.dense_1 = Dense(units=100, activation = 'relu', dropout = 0.06, recurrent_dropout = 0.14)
        self.dense_final = Dense(units=1, activation = 'sigmoid') #Needs sigmoid for binary

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = self.lstm_layer2(inputs)
        x = tf.keras.layers.Flatten()(x) 
        x = self.dense_1(x)
        output = self.dense_final(x)
        return output

    def compile_model(self, optimizer='adam', learning_rate = 0.0075, loss='binary_crossentropy', metrics=['accuracy']):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=optimizer, loss=loss, metrics = metrics)

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        history = self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test):
        loss = self.evaluate(X_test, y_test, verbose=0)
        return loss
    
class GRUModel(tf.keras.Model):
    def __init__(self, window_size, units=50):
        super().__init__()
        self.window_size = window_size
        self.units = units
        self.lstm_layer = LSTM(units=self.units, return_sequences=True, input_shape = (window_size, 1), dropout = 0.06, recurrent_dropout = 0.14)
        self.lstm_layer2 = LSTM(units = self.units, return_sequences=False, dropout = 0.06, recurrent_dropout = 0.14)
        self.dense_1 = Dense(units=100, activation = 'relu', dropout = 0.06, recurrent_dropout = 0.14)
        self.dense_final = Dense(units=1, activation = 'sigmoid') #Needs sigmoid for binary

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = self.lstm_layer2(inputs)
        x = tf.keras.layers.Flatten()(x) 
        x = self.dense_1(x)
        output = self.dense_final(x)
        return output

    def compile_model(self, optimizer='adam', learning_rate = 0.0075, loss='binary_crossentropy', metrics=['accuracy']):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=optimizer, loss=loss, metrics = metrics)

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        history = self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test):
        loss = self.evaluate(X_test, y_test, verbose=0)
        return loss