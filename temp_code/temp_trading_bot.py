from datetime import datetime, timezone

# Make the datetime object timezone-aware

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.entities import Asset
from temp_predictions import loop_stocks


#Credentials to the Alpaca Broker
API_KEY = "PKWB28G9RORVCGQPLZGT"
API_SECRET = "ziIaOkXTkmAebTlyKbHIxaph7hGhmvTCLJe4JP4t"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":API_KEY,
    "API_SECRET":API_SECRET,
    "PAPER":True
}


#Buy and Hold Strategy (OMXS30)
class BuyHoldOMXStrategy(Strategy):
    def initialize(self, symbol:str="OMX"):
        self.symbol = symbol
        self.sleeptime = "24H"

    def on_trading_iteration(self):            
        if self.first_iteration:
            omxs30_price = self.get_last_price(self.symbol)
            quantity = self.portfolio_value // omxs30_price
            order = self.create_order(self.symbol, quantity, "buy")
            self.submit_order(order)

#Log file location
# datastring = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# stat_file = f"trading_logs/my_strategy_{datastring}.csv"


class LSTM_Strategy(Strategy):
    #Overloading lifecycle methods

    def initialize(self):
        self.sleeptime = "24H"
        date = self.get_datetime()
        date = date.date()

        print(date)
        predictions_dict = loop_stocks(date)
        self.portfolio_value
        self.predictions_dict = predictions_dict


    def on_trading_iteration(self): 
        #Prediction Strategy Goes here:
        #First day, we buy if it'll go up  
        prediction_threshold = 0.5  
        if self.first_iteration:
            for symbol,prediction in self.predictions_dict.items():
                if (prediction > prediction_threshold):
                    order = self.create_order(symbol, 10, "buy")
                    self.submit_order(order)
        #
        else:
            if self.portfolio_value >= 1000000:
                self.sell_all()
            for symbol,prediction in self.predictions_dict.items():
                if (prediction > prediction_threshold):
                    order = self.create_order(symbol, 10, "buy")
                    self.submit_order(order)
                else:
                    order = self.create_order(symbol, 10, "sell")
                    self.submit_order(order)


# Pick the dates that you want to start and end your backtest
# and the allocated budget
backtesting_start = datetime(2012, 1, 2)
backtesting_end = datetime(2013, 1, 2)

# benchmark_asset = Asset(symbol="OMXS30", asset_type="stock")


# Run the backtest
LSTM_Strategy.backtest(
    YahooDataBacktesting,
    backtesting_start,
    backtesting_end,
    #stats_file=stat_file,
)

#Backtesting BuyHoldOMX30Strategy
BuyHoldOMXStrategy.backtest(
    YahooDataBacktesting,
    backtesting_start,
    backtesting_end,
)

