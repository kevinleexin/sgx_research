import collections
import pandas as pd
import datetime
from sgx_market_data import sgx_market_data
import matplotlib.pyplot as plt
import numpy as np

import bayes_garch_poisson_spread_to_predict_price




class PriceLevel_ActiveOrderVolume:

    def __init__(self):
        self.active_buyorder_quantity = 0
        self.active_sellorder_quantity = 0

    def set_buyorder(self, quantity):
        self.active_buyorder_quantity += quantity

    def set_sellorder(self, quantity):
        self.active_sellorder_quantity += quantity

    def to_string(self) -> str:
        return str("buy_order_quantity: {}, sell_order_quantity: {}".format(
          self.active_buyorder_quantity,
          self.active_sellorder_quantity
        ))


class OrderFlowIndicator:
    def __init__(self, timestamp, delta, normal_delta, price_chg):
        self.timestamp = timestamp
        self.delta = delta
        self.norm_delta = normal_delta
        self.price_change = price_chg

    def to_string(self, last_price):
        print("timestamp: {}, last_price: {} delta: {}, norm_delta: {}, price_chg: {}".format(
            self.timestamp,
            last_price,
            self.delta,
            self.norm_delta,
            self.price_change
        ))


class OrderFlowTable:
    """
        //= == == == == == == == == == == == == == == == == == == == == == == == == == == == = //
    // 计算limitorderbook上，主动交易量的供需不平衡性
    // 供给失衡：根据拍卖理论，bar内斜对角价位进行对比，某个价位主动卖单数量远大于高一档价位的主动买单数量，表示空头力量强势，比值通常设为3: 1
    // 需求失衡：根据拍卖理论，bar内斜对角价位进行对比，某个价位主动买单数量远大于低一档价位的主动卖单数量，表示多头力量强势，比值通常设为3: 1
    // 目的是判断当某个方向上供需不平衡时，可以在流动性缺失处，考虑提供流动性，获取价格的revert盈利。
    // 但需要做好风控，从其他信号处设置下单的止损价格。
    // 交易原则是cut big_loss, cut big_loss.
    //= == == == == == == == == == == == == == == == == == == == == == == == == == == == = //
    """

    def __init__(self, tick_width, time_span):
        self.tick_width = tick_width

        # cached real-time order_flow_table
        self.table = list()
        # active_buy and active_sell data-structure
        self.active_table_interval = dict()
        self.last_price_for_graph = list()
        self.ts_for_graph = list()
        self.last_price = list()

        self.start_ts = None
        self.prev_ts = None

        self.market_spread = list()

        self.time_span = time_span

    def on_tick(self, data: sgx_market_data):
        # record every event which could be affected the order flow or order book
        # make these event to construct order flow table to generate tick-class signal for trading
        assert (data, sgx_market_data)
        if self.start_ts is None:
            self.start_ts = data.timestamp
            self.active_trade_event(data)
            self.prev_ts = data.timestamp
        else:
            time_diff = data.timestamp - self.start_ts
            if time_diff.seconds >= self.time_span:
                self.start_ts = data.timestamp
                if self.last_price:
                    self.clear_and_save_data(data.timestamp, self.last_price[-1], data.ask1price - data.bid1price)
                else:
                    self.clear_and_save_data(data.timestamp, None, data.ask1price - data.bid1price)
                self.prev_ts = data.timestamp
            else:
                self.active_trade_event(data)

    def active_trade_event(self, data):
        if data.action == 67 or data.action == 69:
            self.last_price.append(data.last_price)
            if data.isb is True and data.last_price in self.active_table_interval.keys():
                self.active_table_interval[data.last_price].set_buyorder(abs(data.last_quantity))
            elif data.isb is True and data.last_price not in self.active_table_interval.keys():
                price_level = PriceLevel_ActiveOrderVolume()
                price_level.set_buyorder(abs(data.last_quantity))
                self.active_table_interval[data.last_price] = price_level

            elif data.isb is False and data.last_price in self.active_table_interval.keys():
                self.active_table_interval[data.last_price].set_sellorder(data.last_quantity)
            elif data.isb is False and data.last_price not in self.active_table_interval.keys():
                price_level = PriceLevel_ActiveOrderVolume()
                price_level.set_sellorder(data.last_quantity)
                self.active_table_interval[data.last_price] = price_level
            else:
                pass

    def clear_and_save_data(self, timestamp, last_price, spread):

        indicator = self.calc_table_in_realtime(timestamp)
        print("---------------------delta------------------------")
        indicator.to_string(last_price)
        print("---------------------level------------------------")
        for key, value in self.active_table_interval.items():
            print("price: {} ".format(
                key
            ) + value.to_string())
        print("--------------------------------------------------")
        self.table.append(indicator)
        self.last_price_for_graph.append(last_price)
        self.ts_for_graph.append(timestamp)
        self.market_spread.append(spread)
        self.active_table_interval.clear()

    def calc_table_in_realtime(self, timestamp) -> OrderFlowIndicator:
        delta, normal_delta, price_change = self.calc_delta()

        return OrderFlowIndicator(timestamp, delta, normal_delta, price_change)

    def calc_delta(self) -> (int, float):

        interval_delta = 0
        normal_interval_delta = 0
        total_trade_quantity = 0
        price_change = 0
        if self.active_table_interval:
            for key, value in self.active_table_interval.items():
                if isinstance(value, PriceLevel_ActiveOrderVolume):
                    interval_delta += value.active_buyorder_quantity + value.active_sellorder_quantity
                    total_trade_quantity += value.active_buyorder_quantity + abs(value.active_sellorder_quantity)
                else:
                    pass
            price_change = (max(self.active_table_interval.keys()) - min(self.active_table_interval.keys())) / self.tick_width
        if total_trade_quantity:
            normal_interval_delta = interval_delta / total_trade_quantity

        return interval_delta, normal_interval_delta, price_change

    def plot(self):
        global price_changes_window
        deltas = list()
        price_change = list()
        for indicator in self.table:
            deltas.append(indicator.delta)
            price_change.append(indicator.price_change)
        log_returns = np.diff(np.log(self.last_price_for_graph[4:]))

        print(len(self.ts_for_graph[5:]), len(log_returns), len(deltas[5:]))
        # 创建图形
        fig, ax1 = plt.subplots()

        # 绘制第一个数据（使用左侧y轴）
        ax1.plot(self.ts_for_graph[5:], log_returns, 'g-', label='LogReturn')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('LogReturn (Dimension 1)', color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        # 创建第二个y轴，绘制第二个数据（使用右侧y轴）
        ax2 = ax1.twinx()
        ax2.plot(self.ts_for_graph[5:], deltas[5:], 'b-', label='NormDelta')
        ax2.set_ylabel('NormDelta (Dimension 2)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # 显示图形
        plt.title('Delta-Price')
        plt.show()

        correlation_matrix = np.corrcoef(log_returns, deltas[5:])
        print(correlation_matrix)

        # 初始化参数
        P_up = 0.5
        P_down = 0.5

        mu_price_up, sigma_price_up = 0.001, 0.02  # 上涨时的价格变化分布
        mu_price_down, sigma_price_down = -0.001, 0.02  # 下跌时的价格变化分布
        lambda_volume_up = 0.25  # 初始成交量均值
        lambda_volume_down = 0.25  # 初始成交量均值

        beta = 0.5  # 波动率的敏感系数
        window_size = 10  # 滚动窗口大小

        price_changes = []
        delta_changes = []

        for price_change, delta_change in zip(price_change[4:], deltas[4:]):
            bayes_garch_poisson_spread_to_predict_price.update_price_changes(price_change, price_changes, delta_change, delta_changes, window_size)
            mu_price_up, mu_price_sigma = bayes_garch_poisson_spread_to_predict_price. update_distribution_parameters(price_changes, window_size)
            P_up, P_down, price_changes_window = bayes_garch_poisson_spread_to_predict_price.process_new_data(price_change, delta_change, P_up,
                                                                                                              P_down, price_changes)

            # 根据后验概率预测趋势
            if P_up > 0.5:
                print("    预测：上涨\n")
            else:
                print("    预测：下跌\n")


if __name__ == "__main__":

    time_span = 300
    order_flow = OrderFlowTable(5, time_span)
    # market_data = pd.read_parquet('./data/20230904/20230904-SG28228467-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230905/20230905-SG28231055-P1-NKZ23-8323274.lvls.parquet')
    market_data = pd.read_parquet('./data/20230906/20230906-SG28232495-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230907/20230907-SG28233935-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230908/20230908-SG28235375-P1-NKZ23-8323274.lvls.parquet')
    # print(market_data.head())
    for index, row in market_data.iterrows():
        order_flow.on_tick(sgx_market_data(row))

    order_flow.plot()
