class sgx_market_data:
    def __init__(self, slice):

        #self.timestamp = slice['t'].tz_localize('Asia/Singapore')
        self.timestamp = slice['t']
        self.action = slice['mc']
        self.isb = slice['isb']

        #print(slice)
        # 这里是单次成交的成交价格和成交量？
        self.last_price = slice['dp0']
        self.last_quantity = slice['dq0']

        self.bid1price = slice['bp00']
        self.bid1quantity = slice['bq00']

        self.bid2price = slice['bp01']
        self.bid2quantity = slice['bq01']

        self.bid3price = slice['bp02']
        self.bid3quantity = slice['bq02']

        self.bid4price = slice['bp03']
        self.bid4quantity = slice['bq03']

        self.bid5price = slice['bp04']
        self.bid5quantity = slice['bq04']

        self.bid6price = slice['bp05']
        self.bid6quantity = slice['bq05']

        self.bid7price = slice['bp06']
        self.bid7quantity = slice['bq06']

        self.bid8price = slice['bp07']
        self.bid8quantity = slice['bq07']

        self.bid9price = slice['bp08']
        self.bid9quantity = slice['bq08']

        self.bid10price = slice['bp09']
        self.bid10quantity = slice['bq09']

        self.ask1price = slice['ap00']
        self.ask1quantity = slice['aq00']

        self.ask2price = slice['ap01']
        self.ask2quantity = slice['aq01']

        self.ask3price = slice['ap02']
        self.ask3quantity = slice['aq02']

        self.ask4price = slice['ap03']
        self.ask4quantity = slice['aq03']

        self.ask5price = slice['ap04']
        self.ask5quantity = slice['aq04']

        self.ask6price = slice['ap05']
        self.ask6quantity = slice['aq05']

        self.ask7price = slice['ap06']
        self.ask7quantity = slice['aq06']

        self.ask8price = slice['ap07']
        self.ask8quantity = slice['aq07']

        self.ask9price = slice['ap08']
        self.ask9quantity = slice['aq08']

        self.ask10price = slice['ap09']
        self.ask10quantity = slice['aq09']
        
        self.bid_prices = []
        self.bid_quantities = []
        self.sum_bid_quantity = 0
        self.ask_prices = []
        self.ask_quantities = []
        self.sum_ask_quantity = 0
        self.sum_quantity_in_levels = 0
        self.bid_vwap_3 = 0
        self.ask_vwap_3 = 0
        self.market_spread = 0
        self.order_quantity_imbalance = 0
        self.last_price_x_quantity = 0

    def is_valid(self):
        return self.bid1price > 0 and self.ask1price > 0 and self.bid1quantity > 0 and self.ask1quantity > 0
    
    def calculate(self):
        self.bid_prices = [self.bid1price,
                           self.bid2price,
                           self.bid3price,
                           self.bid4price,
                           self.bid5price,
                           self.bid6price,
                           self.bid7price,
                           self.bid8price,
                           self.bid9price]

        self.bid_quantities = [self.bid1quantity,
                               self.bid2quantity,
                               self.bid3quantity,
                               self.bid4quantity,
                               self.bid5quantity,
                               self.bid6quantity,
                               self.bid7quantity,
                               self.bid8quantity,
                               self.bid9quantity]

        self.sum_bid_quantity = sum(self.bid_quantities)

        self.ask_prices = [self.ask1price,
                           self.ask2price,
                           self.ask3price,
                           self.ask4price,
                           self.ask5price,
                           self.ask6price,
                           self.ask7price,
                           self.ask8price,
                           self.ask9price]

        self.ask_quantities = [self.ask1quantity,
                               self.ask2quantity,
                               self.ask3quantity,
                               self.ask4quantity,
                               self.ask5quantity,
                               self.ask6quantity,
                               self.ask7quantity,
                               self.ask8quantity,
                               self.ask9quantity]

        self.sum_ask_quantity = sum(self.ask_quantities)

        self.sum_quantity_in_levels = self.sum_bid_quantity + self.sum_ask_quantity

        self.bid_vwap_3 = (self.bid1price * self.bid1quantity +
                           self.bid2price * self.bid2quantity +
                           self.bid3price * self.bid3quantity) / (sum(self.bid_quantities[0:3]))

        self.ask_vwap_3 = (self.ask1price * self.ask1quantity +
                           self.ask2price * self.ask2quantity +
                           self.ask3price * self.ask3quantity) / (sum(self.ask_quantities[0:3]))

        self.market_spread = self.ask_vwap_3 - self.bid_vwap_3
        self.order_quantity_imbalance = ((self.sum_bid_quantity - self.sum_ask_quantity) /
                                         (self.sum_bid_quantity + self.sum_ask_quantity))

        self.last_price_x_quantity = self.last_price * self.last_quantity

    def to_string(self):
        print("timestamp: {}, action: {}, isb: {}, last_price: {}, last_quantity: {}， "
              "bid1price: {}, bid1quantity: {}, ask1price: {}, ask1quantity: {}".format(
            self.timestamp, self.action, self.isb, self.last_price, self.last_quantity,
            self.bid1price, self.bid1quantity, self.ask1price, self.ask1quantity)
        )
