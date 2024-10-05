import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict
import time
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


class StreamingVPIN:
    def __init__(self,
                 bucket_volume: float,
                 num_buckets: int = 50,
                 sample_size: int = 1000,
                 estimation_window: float = 300):  # 5 minutes in seconds
        """
        Initialize Streaming VPIN calculator

        Parameters:
        - bucket_volume: Target volume for each bucket
        - num_buckets: Number of buckets to maintain for VPIN calculation
        - sample_size: Number of recent trades to keep for BVC calculation
        - estimation_window: Time window (seconds) for volatility estimation
        """
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets
        self.sample_size = sample_size
        self.estimation_window = estimation_window

        # State management
        self.current_bucket = VPINBucket(bucket_volume)
        self.completed_buckets: Deque[VPINBucket] = deque(maxlen=num_buckets)

        # Price history for BVC
        self.price_history: Deque[float] = deque(maxlen=sample_size)
        self.volume_history: Deque[float] = deque(maxlen=sample_size)
        self.timestamp_history: Deque[pd.Timestamp] = deque(maxlen=sample_size)

        # Volatility estimation
        self.returns_history: Deque[float] = deque(maxlen=sample_size)
        self.last_volatility_update = time.time()
        self.current_volatility: Optional[float] = None

        # Quote tracking
        self.current_quote: Optional[Quote] = None

        # Metrics tracking
        self.metrics: Dict[str, deque] = {
            'vpin_values': deque(maxlen=1000),
            'bucket_durations': deque(maxlen=1000),
            'trade_counts': deque(maxlen=1000)
        }

    def update_volatility(self, timestamp: float) -> None:
        """Update volatility estimate if enough time has passed"""
        if not self.returns_history:
            return

        if timestamp - self.last_volatility_update >= self.estimation_window:
            # Filter returns within estimation window
            recent_returns = [r for t, r in zip(self.timestamp_history, self.returns_history)
                              if timestamp - t <= self.estimation_window]

            if recent_returns:
                self.current_volatility = np.std(recent_returns)
                self.last_volatility_update = timestamp

    def classify_trade_bulk_volume(self, trade: Trade) -> float:
        """
        Classify trade using Bulk Volume Classification (BVC)
        Returns probability of buy (between 0 and 1)
        """
        if not self.price_history:
            self.price_history.append(trade.price)
            return 0.5

        price_return = np.log(trade.price / self.price_history[-1])
        self.returns_history.append(price_return)
        self.price_history.append(trade.price)
        self.volume_history.append(trade.volume)
        self.timestamp_history.append(trade.timestamp)

        # Update volatility estimate
        self.update_volatility(trade.timestamp)

        if not self.current_volatility or self.current_volatility == 0:
            return 0.5

        # Calculate buy probability using normal CDF
        z_score = price_return / self.current_volatility
        buy_prob = 0.5 + 0.5 * scipy.stats.norm.cdf(z_score)

        return buy_prob

    def process_trade(self, trade: Trade) -> Optional[float]:
        """
        Process incoming trade and return updated VPIN if bucket is completed
        """
        if self.current_bucket.start_time is None:
            self.current_bucket.start_time = trade.timestamp

        # Classify trade
        if trade.direction is None:
            buy_prob = self.classify_trade_bulk_volume(trade)
            buy_volume = trade.volume * buy_prob
            sell_volume = trade.volume * (1 - buy_prob)
        else:
            # Use provided trade direction if available
            buy_volume = trade.volume if trade.direction == 1 else 0
            sell_volume = trade.volume if trade.direction == -1 else 0

        # Update current bucket
        self.current_bucket.buy_volume += buy_volume
        self.current_bucket.sell_volume += sell_volume
        self.current_bucket.current_volume += trade.volume

        # Check if bucket is complete
        if self.current_bucket.is_complete():
            self.current_bucket.end_time = trade.timestamp

            # Store metrics
            if self.current_bucket.duration() is not None:
                self.metrics['bucket_durations'].append(self.current_bucket.duration())

            # Add to completed buckets
            self.completed_buckets.append(self.current_bucket)

            # Calculate VPIN
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

    def get_metrics(self) -> Dict[str, float]:
        """Get current VPIN metrics"""
        metrics = {
            'vpin': self.calculate_vpin(),
            'current_bucket_volume': self.current_bucket.current_volume,
            'current_bucket_imbalance': self.current_bucket.imbalance
        }

        if self.completed_buckets:
            metrics.update({
                'avg_bucket_duration': np.mean(self.metrics['bucket_durations']),
                'vpin_stddev': np.std(self.metrics['vpin_values']),
                'vpin_max': max(self.metrics['vpin_values'])
            })

        return metrics

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

    def get_suggested_spread_multiplier(self) -> float:
        """Get suggested spread multiplier based on current VPIN"""
        vpin = self.calculate_vpin()
        base_multiplier = 1.0

        if not self.metrics['vpin_values']:
            return base_multiplier

        vpin_std = np.std(self.metrics['vpin_values'])
        vpin_mean = np.mean(self.metrics['vpin_values'])

        # Calculate z-score of current VPIN
        z_score = (vpin - vpin_mean) / vpin_std if vpin_std > 0 else 0

        # Exponential increase in spread as VPIN deviates from mean
        return base_multiplier * np.exp(max(0, z_score - 1))


def transfer_market_data_to_trade(data: sgx_market_data) -> Optional[Trade]:
    direct = data.get_direction()
    if direct:
        newTrade = Trade(data.last_price, data.last_quantity, data.timestamp, direct )
        return newTrade
    else:
        return None


def transfer_market_data_to_quote(data: sgx_market_data) -> Quote:
    newQuote = Quote(data.ask1price, data.bid1price, data.timestamp)
    return newQuote


class OnlineVPIN:
    def __init__(self,
                 base_spread: float,
                 vpin_calculator: StreamingVPIN):

        self.base_spread = base_spread
        self.vpin_calculator = vpin_calculator
        self.current_position = 0
        self.last_quote: Optional[Quote] = None

    def process_market_update(self, data: sgx_market_data) -> Dict:
        """
        Process market updates and return updated market making parameters
        """
        # Update VPIN if trade received
        vpin_update = None
        if data.is_trade():
            vpin_update = self.vpin_calculator.process_trade(transfer_market_data_to_trade(data))

        # Update quote if received
        if data.is_quote():
            self.last_quote = data

        # Get current market making parameters
        spread_multiplier = self.vpin_calculator.get_suggested_spread_multiplier()
        toxicity_level = self.vpin_calculator.get_toxicity_level()

        # Calculate suggested parameters
        suggested_params = {
            'spread': self.base_spread * spread_multiplier,
            'size_multiplier': 1.0 / spread_multiplier,  # Reduce size when spread increases
            'toxicity_level': toxicity_level,
            'vpin': self.vpin_calculator.calculate_vpin()
        }

        if vpin_update is not None:
            suggested_params['vpin_update'] = vpin_update

        return suggested_params
