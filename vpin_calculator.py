import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict
import scipy
import pandas as pd
from sgx_market_data import sgx_market_data


@dataclass
class Trade:
    timestamp: pd.Timestamp
    price: float
    volume: float
    direction: Optional[int] = None  # 1 buy, -1 sell


@dataclass
class Quote:
    bid: float
    ask: float
    timestamp: pd.Timestamp


class VPINBucket:
    """
    One vpin bucket
    """

    def __init__(self, target_volume: float):
        self.target_volume = target_volume
        self.current_volume = 0
        self.buy_volume = 0
        self.sell_volume = 0
        self.start_time: Optional[pd.Timestamp] = None
        self.end_time: Optional[pd.Timestamp] = None

    def is_complete(self) -> bool:
        return self.current_volume >= self.target_volume

    @property
    def imbalance(self) -> float:
        if self.current_volume == 0:
            return 0
        return abs(self.buy_volume - self.sell_volume) / self.current_volume

    def duration(self) -> Optional[pd.Timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def transfer_market_data_to_trade(data: sgx_market_data) -> Optional[Trade]:
    direct = data.get_direction()
    if direct:
        newTrade = Trade(data.timestamp, data.last_price, data.last_quantity, direct)
        return newTrade
    else:
        return None


def transfer_market_data_to_quote(data: sgx_market_data) -> Quote:
    newQuote = Quote(data.ask1price, data.bid1price, data.timestamp)
    return newQuote


class VpinCalculator:
    def __init__(self, bucket_volume: float, num_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets

        self.current_bucket = VPINBucket(bucket_volume)
        self.completed_buckets: Deque[VPINBucket] = deque(maxlen=num_buckets)

        self.current_quote: Optional[Quote] = None

        self.metrics: Dict[str, deque] = {
            'vpin_values': deque(maxlen=1000),
            'bucket_durations': deque(maxlen=1000),
            'trade_counts': deque(maxlen=1000)
        }

    def process_trade(self, trade: Trade) -> Optional[float]:
        if self.current_bucket.start_time is None:
            self.current_bucket.start_time = trade.timestamp

        buy_volume = abs(trade.volume) if trade.direction == 1 else 0
        sell_volume = abs(trade.volume) if trade.direction == -1 else 0

        self.current_bucket.buy_volume += buy_volume
        self.current_bucket.sell_volume += sell_volume
        self.current_bucket.current_volume += trade.volume

        if self.current_bucket.is_complete():
            self.current_bucket.end_time = trade.timestamp
            if self.current_bucket.duration() is not None:
                self.metrics['bucket_durations'].append(self.current_bucket.duration())

            self.completed_buckets.append(self.current_bucket)

            vpin = self.calculate_vpin()
            self.metrics['vpin_values'].append(vpin)

            # Start new bucket with remaining volume
            overflow = self.current_bucket.current_volume - self.bucket_volume
            self.current_bucket = VPINBucket(self.bucket_volume)

            if overflow > 0:
                # Process overflow volume
                overflow_trade = Trade(
                    timestamp=trade.timestamp,
                    price=trade.price,
                    volume=overflow,
                    direction=trade.direction
                )
                self.process_trade(overflow_trade)
            return vpin

        return None

    def calculate_vpin(self) -> float:
        """Calculate VPIN from completed buckets"""
        if not self.completed_buckets:
            return 0.0

        return np.mean([bucket.imbalance for bucket in self.completed_buckets])

    def get_toxicity_level(self) -> str:
        """Get current toxicity level based on VPIN"""
        vpin = self.calculate_vpin()
        if vpin < 0.2:
            return "LOW"
        elif vpin < 0.4:
            return "NORMAL"
        elif vpin < 0.6:
            return "ELEVATED"
        else:
            return "HIGH"

    def get_last_vpin(self) -> Optional[float]:
        if len(self.metrics['vpin_value']) > 0:
            return self.metrics['vpin_value'][-1]
        else:
            return None

    def on_tick(self, data: sgx_market_data) -> Optional[float]:
        if data.is_trade():
            incoming_trade = transfer_market_data_to_trade(data)
            if incoming_trade:
                current_vpin = self.process_trade(incoming_trade)
                return current_vpin
            else:
                return None
        elif data.is_quote():
            incoming_quote = transfer_market_data_to_quote(data)
            if incoming_quote:
                self.current_quote = incoming_quote
                return None
            else:
                return None
        else:
            return None
