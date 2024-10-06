import numpy as np
from typing import Tuple, List, Dict
import pandas as pd
from dataclasses import dataclass

@dataclass
class JumpConfig:
    """Configuration parameters for jump detection"""
    window_size: int = 100  # Rolling window size for volatility estimation
    threshold_multiplier: float = 4.0  # Number of standard deviations for jump detection
    min_jump_size: float = 0.0001  # Minimum absolute price change to consider
    decay_factor: float = 0.94  # Exponential decay factor for volatility estimation

class JumpDetector:
    def __init__(self, config: JumpConfig):
        self.config = config
        self.volatility = None
        self.last_price = None
        self.returns_buffer = []
    
    def estimate_volatility(self, returns: np.ndarray) -> float:
        """
        Estimate volatility using exponentially weighted moving standard deviation
        """
        weights = np.power(
            self.config.decay_factor,
            np.arange(len(returns))[::-1]
        )
        weights /= weights.sum()
        return np.sqrt(np.sum(weights * returns * returns))
    
    def detect_jump(self, price: float, timestamp: pd.Timestamp) -> Dict:
        """
        Detect if a price movement constitutes a jump
        
        Args:
            price: Current price
            timestamp: Current timestamp
        
        Returns:
            Dictionary containing jump detection results
        """
        if self.last_price is None:
            self.last_price = price
            return {"is_jump": False, "jump_size": 0.0, "threshold": 0.0}
        
        # Calculate return
        returns = np.log(price / self.last_price)
        self.returns_buffer.append(returns)
        
        # Maintain rolling window
        if len(self.returns_buffer) > self.config.window_size:
            self.returns_buffer.pop(0)
        
        # Update volatility estimate
        self.volatility = self.estimate_volatility(np.array(self.returns_buffer))
        
        # Calculate threshold
        threshold = self.volatility * self.config.threshold_multiplier
        
        # Detect jump
        is_jump = (abs(returns) > threshold and 
                  abs(returns) > self.config.min_jump_size)
        
        result = {
            "is_jump": is_jump,
            "jump_size": returns,
            "threshold": threshold,
            "volatility": self.volatility,
            "timestamp": timestamp,
            "price": price
        }
        
        self.last_price = price
        return result


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = JumpConfig(
        window_size=100,
        threshold_multiplier=4.0,
        min_jump_size=0.0001,
        decay_factor=0.94
    )
