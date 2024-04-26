import model as model
from data import preprocess
import logging 
class TradingStrategy:
    def __init__(self):
        super().__init__()
    
    def generate_signals(self, predictions):
        pass


class Backtester:
    def __init__(self):
        super().__init__()

    def backtest_strategy(self, strategy, test_data):
        pass


class TradingBot:
    def __init__(self, data_file):
        self.model = model.LSTMModel()
        self.strategy = TradingStrategy()
        self.backtester = Backtester()

#This logger will be used to log errors or other important information
    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run(self):
            try:
                data = self.data_handler.load_data()
                processed_data = self.data_handler.preprocess_data(data)
                train_data, test_data = self.data_handler.split_data(processed_data)
                X_train, y_train = self.data_handler.create_sequences(train_data)
                X_test, y_test = self.data_handler.create_sequences(test_data)

                self.model.build_model()
                self.model.train_model(X_train, y_train, X_test, y_test)
                self.model.evaluate_model(X_test, y_test)

                predictions = self.model.predict(X_test)
                signals = self.strategy.generate_signals(predictions)
                self.backtester.backtest_strategy(signals, test_data)

                self.logger.info("Trading bot run completed successfully.")

            except Exception as e:
                self.logger.error(f"Error occurred: {str(e)}")