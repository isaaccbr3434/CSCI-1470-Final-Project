import model as model
from data import preprocess
import logging 
import pandas as pd
from lumibot.brokers import Alpaca
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

API_KEY = "PKWB28G9RORVCGQPLZGT"
API_SECRET = "ziIaOkXTkmAebTlyKbHIxaph7hGhmvTCLJe4JP4t"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":API_KEY,
    "API_SECRET":API_SECRET,
    "PAPER":True
}

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
                data = pd.read_csv(('cleaned_sp500_stock_data.csv'))
                processed_data = None
                train_data, test_data = None
                X_train, y_train = None
                X_test, y_test = None

                #self.model.build_model()
                self.model.train_model(X_train, y_train, X_test, y_test)
                self.model.evaluate_model(X_test, y_test)

                #Need to implement the functions below
                #predictions = self.model.predict(X_test)
                #signals = self.strategy.generate_signals(predictions)
                #self.backtester.backtest_strategy(signals, test_data)

                self.logger.info("Trading bot run completed successfully.")

            except Exception as e:
                self.logger.error(f"Error occurred: {str(e)}")