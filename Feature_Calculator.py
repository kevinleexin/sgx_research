from sgx_market_data import sgx_market_data
from SGX_order_action import Action
from collections import deque
import pandas as pd
import numpy as np
from Trading_Session import Nikkei_225_Index_Futures_Trading_Session
from typing import Optional
import vpin_calculator
import jump_detection

import seaborn as sns
import matplotlib.pyplot as plt


class ImmediateImpactStruct:
    def __init__(self, ts, isb=False, level_diff=0, price=0.0, quantity=0, bid_liquidity_diff=0, ask_liquidity_diff=0,
                 action=0,
                 spread=0, b_market_depth=0, a_market_depth=0, imbalance=0.0, wimbalance=0.0,
                 short_ai_ratio=0.0, long_ai_ratio=0.0,
                 vwap=0.0, vol=0.0, add_delete_ratio=0.0,
                 book_pressure=0.0, vpin=0.0):
        self.ts = ts
        self.isb = isb
        self.price_level_diff = level_diff  # y
        self.executed_price = price
        self.executed_quantity = quantity
        self.bid_side_liquidity_diff = bid_liquidity_diff
        self.ask_side_liquidity_diff = ask_liquidity_diff

        self.action = action
        # liquidity factor
        self.market_spread = spread  # vwap_ask1-3 - vwap_bid1-3
        self.bid_market_depth = b_market_depth  # sum(bid1-10.quantity)
        self.ask_market_depth = a_market_depth  # sum(ask1-10.quantity)
        self.Imbalance = imbalance  # (sum_bid_quantity - sum_ask_quantity) / (sum_bid_quantity + sum_ask_quantity)
        self.weighted_Imbalance = wimbalance
        self.short_Amihub_Illiquidity_Ratio = short_ai_ratio  # (price_return_t) / sum_last_quantity_t
        self.long_Amihub_Illiquidity_Ratio = long_ai_ratio
        self.vwap = vwap

        # volatility
        self.vol = vol

        # call_delete_ration
        self.real_time_add_delete_ratio = add_delete_ratio

        # book_pressure
        self.book_pressure = book_pressure

        # vpin
        self.vpin = vpin

    def to_string(self):
        print(f"ts: {self.ts},"
              f"isb: {self.isb}, "
              f"price_level_diff: {self.price_level_diff},"
              f"executed_price: {self.executed_price}, "
              f"executed_quantity: {self.executed_quantity}, "
              f"bid_side_liquidity_diff: {self.bid_side_liquidity_diff}, "
              f"ask_side_liquidity_diff: {self.ask_side_liquidity_diff},"
              f"action: {self.action},"
              f"market_spread: {self.market_spread},"
              f"bid_market_depth: {self.bid_market_depth},"
              f"ask_market_depth: {self.ask_market_depth},"
              f"order_quantity_imbalance: {self.Imbalance},"
              f"weighted_imbalance: {self.weighted_Imbalance},"
              f"short_amihub_illiquidity_ratio: {self.short_Amihub_Illiquidity_Ratio},"
              f"long_amihub_illiquidity_ratio: {self.long_Amihub_Illiquidity_Ratio},"
              f"vwap: {self.vwap},"
              f"volatility: {self.vol},"
              f"real_time_add_delete_ratio: {self.real_time_add_delete_ratio},"
              f"book_pressure: {self.book_pressure},"
              f"vpin: {self.vpin}")

    def to_dict(self) -> dict:
        return {'ts': self.ts,
                'isb': self.isb,
                'price_level_diff': self.price_level_diff,
                'executed_price': self.executed_price,
                'executed_quantity': self.executed_quantity,
                'bid_side_liquidity_diff': self.bid_side_liquidity_diff,
                'ask_side_liquidity_diff': self.ask_side_liquidity_diff,
                'action': self.action,
                'market_spread': self.market_spread,
                'bid_market_depth': self.bid_market_depth,
                'ask_market_depth': self.ask_market_depth,
                'order_quantity_imbalance': self.Imbalance,
                'weighted_imbalance': self.weighted_Imbalance,
                'short_amihub_illiquidity_ratio': self.short_Amihub_Illiquidity_Ratio,
                'long_amihub_illiquidity_ratio': self.long_Amihub_Illiquidity_Ratio,
                'vwap': self.vwap,
                'volatility': self.vol,
                'real_time_add_delete_ratio': self.real_time_add_delete_ratio,
                'book_pressure': self.book_pressure,
                'vpin': self.vpin}


class ImmediateImpact:
    """
    统计瞬时冲击
    """

    def __init__(self, tick_width, initial_trading_date: str, short_term_lookback: int, long_term_lookback: int):
        self.cur_ts = None
        self.prev_ts = None
        self.tick_width = tick_width
        self.short_term_lookback = short_term_lookback
        self.long_term_lookback = long_term_lookback


        # event statistics
        # add_order, delete_order, order_execution
        # 计算报撤单率
        self.add_order_event = list()
        self.delete_order_event = list()

        # 维护计算实时波动率和流动性feature
        self.order_executed_price = list()
        self.order_executed_quantity = list()
        self.order_executed_direction = list()
        self.order_executed_timestamp = list()
        self.accumulated_executed_quantity = 0
        # 缓存过去2000条消息
        self.order_book_cache = deque(maxlen=2000)

        # 价格冲击指标
        # 创建feature字典->pd.DataFrame, 用于最后模型计算
        self.t_session_feature_map = pd.DataFrame(columns=['ts', 'isb', 'price_level_diff',
                                                           'executed_price', 'executed_quantity',
                                                           'bid_side_liquidity_diff', 'ask_side_liquidity_diff',
                                                           'action', 'market_spread', 'bid_market_depth',
                                                           'ask_market_depth', 'order_quantity_imbalance',
                                                           'weighted_imbalance'
                                                           'short_amihub_illiquidity_ratio',
                                                           'long_amihub_illiquidity_ratio',
                                                           'vwap', 'volatility', 'real_time_add_delete_ratio',
                                                           'book_pressure', 'vpin'])

        self.t1_session_feature_map = pd.DataFrame(columns=['ts', 'isb', 'price_level_diff',
                                                            'executed_price', 'executed_quantity',
                                                            'bid_side_liquidity_diff', 'ask_side_liquidity_diff',
                                                            'action', 'market_spread', 'bid_market_depth',
                                                            'ask_market_depth', 'order_quantity_imbalance',
                                                            'weighted_imbalance',
                                                            'short_amihub_illiquidity_ratio',
                                                            'long_amihub_illiquidity_ratio',
                                                            'vwap', 'volatility', 'real_time_add_delete_ratio',
                                                            'book_pressure', 'vpin'])


        # trading session manager
        self.trading_session_mgr = Nikkei_225_Index_Futures_Trading_Session(initial_trading_date)

        # real-time calc data cache
        self.sum_last_price_last_quantity = 0
        self.sum_last_quantity = 0
        self.real_time_vwap = 0.0
        self.short_term_ai_ratio = 0.0  # 对应于short_term_lookback
        self.long_term_ai_ratio = 0.0  # long_term_lookback

        self.real_time_add_cancel_ratio = 0.0
        self.real_time_realized_volatility = 0.0

        self.vpin = vpin_calculator.VpinCalculator(200)
        self.current_vpin = 0.0

        config = jump_detection.JumpConfig(
            window_size=100,
            threshold_multiplier=4.0,
            min_jump_size=0.0001,
            decay_factor=0.94
        )

        self.jump_detector = jump_detection.JumpDetector(config)

    def on_tick(self, data: sgx_market_data):
        if self.market_data_filter(data) is False:
            return

        self.order_book_cache.append(data)
        if self.prev_ts is None:
            self.prev_ts = data.timestamp

        else:
            self.event_statistic(data)
            self.prev_ts = data.timestamp

    def calc_price_level_impact_after_order_executed(self, ts: pd.Timestamp):
        """
        calc 价格瞬时冲击
        :return:
        """
        self.real_time_vwap = self.real_time_vwap_calc()
        self.short_term_ai_ratio = self.real_time_ai_ratio_calc(self.short_term_lookback)
        self.long_term_ai_ratio = self.real_time_ai_ratio_calc(self.long_term_lookback)
        self.real_time_realized_volatility = self.real_time_volatility_calc()
        bid_side_liquidity_diff = self.order_book_cache[-1].sum_bid_quantity - self.order_book_cache[
            -2].sum_bid_quantity
        ask_side_liquidity_diff = self.order_book_cache[-1].sum_ask_quantity - self.order_book_cache[
            -2].sum_ask_quantity

        if self.order_book_cache[-1].isb:
            level_diff = (self.order_book_cache[-2].bid1price - self.order_book_cache[
                -1].bid1price) / self.tick_width

        else:
            level_diff = (self.order_book_cache[-1].ask1price - self.order_book_cache[
                -2].ask1price) / self.tick_width

        cur_executed_result = ImmediateImpactStruct(
            ts,
            self.order_book_cache[-1].isb,
            level_diff,
            self.order_executed_price[-1],
            self.order_executed_quantity[-1],
            bid_side_liquidity_diff,
            ask_side_liquidity_diff,
            self.order_book_cache[-1].action,
            self.order_book_cache[-1].market_spread,
            self.order_book_cache[-1].sum_bid_quantity,
            self.order_book_cache[-1].sum_ask_quantity,
            self.order_book_cache[-1].order_quantity_imbalance,
            self.order_book_cache[-1].weighted_imbalance,
            self.short_term_ai_ratio,
            self.long_term_ai_ratio,
            self.real_time_vwap,
            self.real_time_realized_volatility,
            self.real_time_add_cancel_ratio,
            self.order_book_cache[-1].book_pressure,
            self.current_vpin
        )

        cur_executed_result.to_string()
        if self.trading_session_mgr.is_t_session_opening_range(ts):
            self.t_session_feature_map.loc[len(self.t_session_feature_map)] = cur_executed_result.to_dict()
        if self.trading_session_mgr.is_t1_session_opening_range(ts):
            self.t1_session_feature_map.loc[len(self.t1_session_feature_map)] = cur_executed_result.to_dict()

    def norm_state_stat(self, ts: pd.Timestamp):
        bid_side_liquidity_diff = self.order_book_cache[-1].sum_bid_quantity - self.order_book_cache[
            -2].sum_bid_quantity
        ask_side_liquidity_diff = self.order_book_cache[-1].sum_ask_quantity - self.order_book_cache[
            -2].sum_ask_quantity

        cur_result = ImmediateImpactStruct(
            ts,
            self.order_book_cache[-1].isb,
            0,
            self.order_executed_price[-1],
            self.order_executed_quantity[-1],
            bid_side_liquidity_diff,
            ask_side_liquidity_diff,
            self.order_book_cache[-1].action,
            self.order_book_cache[-1].market_spread,
            self.order_book_cache[-1].sum_bid_quantity,
            self.order_book_cache[-1].sum_ask_quantity,
            self.order_book_cache[-1].order_quantity_imbalance,
            self.order_book_cache[-1].weighted_imbalance,
            self.short_term_ai_ratio,
            self.long_term_ai_ratio,
            self.real_time_vwap,
            self.real_time_realized_volatility,
            self.real_time_add_cancel_ratio,
            self.order_book_cache[-1].book_pressure,
            self.current_vpin
        )

        # cur_result.to_string()
        cur_result.to_string()
        if self.trading_session_mgr.is_t_session_opening_range(ts):
            self.t_session_feature_map.loc[len(self.t_session_feature_map)] = cur_result.to_dict()
        if self.trading_session_mgr.is_t1_session_opening_range(ts):
            self.t1_session_feature_map.loc[len(self.t1_session_feature_map)] = cur_result.to_dict()

    def event_statistic(self, data):
        cur_vpin = self.vpin.on_tick
        if cur_vpin:
            self.current_vpin = cur_vpin
        else:
            last_vpin = self.vpin.get_last_vpin()
            if last_vpin:
                self.current_vpin = last_vpin
            else:
                self.current_vpin = 0.0

        if data.action == Action.SecondOrderExecuted or data.action == Action.FirstOrderExecuted:
            self.order_executed_timestamp.append(data.timestamp)
            self.order_executed_price.append(data.last_price)
            self.order_executed_quantity.append(data.last_quantity)
            self.accumulated_executed_quantity += data.last_quantity
            if data.isb:
                # minus represent sell
                self.order_executed_direction.append(-1)
            else:
                # positive represent buy
                self.order_executed_direction.append(1)
            self.calc_price_level_impact_after_order_executed(data.timestamp)
            jump_result = self.jump_detector.detect_jump(data.last_price, data.timestamp)
            print(jump_result['is_jump'])
            return

        elif data.action == Action.AddOrder:
            add_quantity = data.sum_quantity_in_levels - self.order_book_cache[-2].sum_quantity_in_levels
            self.add_order_event.append(add_quantity)
            self.real_time_add_cancel_ratio = self.real_time_add_delete_ratio_calc(self.long_term_lookback)
            self.norm_state_stat(data.timestamp)
            return

        elif data.action == Action.OrderDelete:
            delete_quantity = self.order_book_cache[-2].sum_quantity_in_levels - data.sum_quantity_in_levels
            self.delete_order_event.append(delete_quantity)
            self.real_time_add_cancel_ratio = self.real_time_add_delete_ratio_calc(self.long_term_lookback)
            self.norm_state_stat(data.timestamp)
            return

    def market_data_filter(self, data):
        # filter non-trading session
        if data.bid1price >= data.ask1price:
            # print(f"bid1price: {data.bid1price}, ask1price: {data.ask1price}")
            return False
        # filter non-opening session
        elif self.trading_session_mgr.is_t_session_pre_open_range(data.timestamp):
            # print(f"is_t_session_pre_open_range: {data.timestamp}")
            return False
        elif self.trading_session_mgr.is_t_session_pre_closing_range(data.timestamp):
            # print(f"is_t_session_pre_closing_range: {data.timestamp}")
            return False
        elif self.trading_session_mgr.is_t1_session_pre_open_range(data.timestamp):
            # print(f"is_t1_session_pre_open_range: {data.timestamp}")
            return False

        else:
            # print (f"opening: {data.timestamp}")
            return True

    def real_time_vwap_calc(self) -> float:
        # 实时维持vwap的计算
        self.sum_last_price_last_quantity += self.order_book_cache[-1].last_price_x_quantity
        self.sum_last_quantity += self.order_book_cache[-1].last_quantity
        return self.sum_last_price_last_quantity / self.sum_last_quantity

    def real_time_ai_ratio_calc(self, lookback) -> float:
        # 实时维持短期和长期的ai_ratio的计算
        if len(self.order_executed_price) >= lookback:
            ret = ((self.order_executed_price[-1] - self.order_executed_price[-lookback]) /
                   self.order_executed_price[-lookback])
            sum_quantity = sum(self.order_executed_price[-lookback:])
            return ret / sum_quantity
        else:
            return 0.0

    def real_time_volatility_calc(self, lookback=1000) -> float:
        # 实时维护已实现的波动率计算
        log_returns = np.diff(np.log(self.order_executed_price[-lookback:]))
        realized_volatility = np.sqrt(np.sum(log_returns ** 2)) * 100
        # print(f"Realized_Volatility: {realized_volatility}")
        return realized_volatility

    def real_time_add_delete_ratio_calc(self, lookback):
        # print(f"add_order_event: {self.add_order_event}, delete_order_event: {self.delete_order_event}, loopback: {lookback}")
        if len(self.add_order_event) >= lookback and len(self.delete_order_event) >= lookback:
            return 0.0 if sum(self.add_order_event[-lookback:]) == 0 else sum(
                self.delete_order_event[-lookback:]) / sum(self.add_order_event[-lookback:])
        else:
            return 0.0


class PermanentImpact:
    """
    统计延迟冲击
    """

    def __init__(self):
        pass


if __name__ == "__main__":

    # 实际跑全量feature计算时，使用processpool来并行多天的feature的同时计算

    immediate_impact = ImmediateImpact(5, '2023-09-06 ', 10, 50)
    # market_data = pd.read_parquet('./data/20230904/20230904-SG28228467-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230905/20230905-SG28231055-P1-NKZ23-8323274.lvls.parquet')
    market_data = pd.read_parquet('./data/20230906/20230906-SG28232495-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230907/20230907-SG28233935-P1-NKZ23-8323274.lvls.parquet')
    # market_data = pd.read_parquet('./data/20230908/20230908-SG28235375-P1-NKZ23-8323274.lvls.parquet')
    # print(market_data.head())

    market_data['t'] = pd.to_datetime(market_data['t']).dt.tz_localize('UTC').dt.tz_convert('Asia/Singapore')

    for index, row in market_data.iterrows():
        md = sgx_market_data(row)
        if md.is_valid():
            md.calculate()
            immediate_impact.on_tick(md)

    immediate_impact.t_session_feature_map.to_csv('./data/20230906/t_session_feature_map.csv')
    immediate_impact.t1_session_feature_map.to_csv('./data/20230906/t1_session_feature_map.csv')

    # bid_level_diff_array = list()
    # ask_level_diff_array = list()
    #
    # for bid in immediate_impact.bid_side_impact:
    #     if bid.price_level_diff != 0:
    #         bid_level_diff_array.append(bid.price_level_diff)
    #
    # for ask in immediate_impact.ask_side_impact:
    #     if ask.price_level_diff != 0:
    #         ask_level_diff_array.append(ask.price_level_diff)
    #
    # # 使用Seaborn绘制分布图
    # sns.histplot(ask_level_diff_array, kde=True, bins=30, color='blue')
    #
    # # 添加标题和标签
    # plt.title('Distribution ask_level_diff_impact')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.show()
