# VPIN (Volume-Synchronized Probability of Informed Trading)

## 1. Theoretical Foundation

VPIN is designed to measure order flow toxicity - the probability that informed traders (those with superior information) are adversely selecting liquidity providers. Key concepts:

### Core Principles
- Time-based sampling can be misleading due to varying trading intensity
- Volume bucketing provides more stable measurement periods
- Order flow imbalances indicate informed trading
- Sequential probability analysis reveals toxic flow patterns

### Why VPIN Matters
- High VPIN indicates increased risk of adverse selection
- Market makers should widen spreads during high VPIN periods
- Can predict significant price movements
- Helps in risk management and position sizing

## 2. Calculation Components

### A. Volume Bucketing
- Trade data is grouped into buckets of equal volume rather than time
- Each bucket represents a standardized trading intensity
- Typical bucket size: 1% of daily volume

### B. Trade Classification
Three main methods:
1. **Tick Rule**: 
   - Compare trade price to previous trade
   - Price up = Buy, Price down = Sell
   
2. **Quote Rule**:
   - Compare trade price to prevailing quotes
   - Above midpoint = Buy, Below midpoint = Sell
   
3. **Bulk Volume Classification (BVC)**:
   - Probabilistic approach using price changes
   - More robust for high-frequency data

### C. Order Imbalance
- Calculate Buy/Sell volume imbalance within each bucket
- Normalized by bucket size
- Running average over multiple buckets

## 3. VPIN Formula

VPIN = Average(|VS - VB|) / V

Where:
- VS = Sell volume in bucket
- VB = Buy volume in bucket
- V = Total bucket volume
- Average taken over n recent buckets

## 4. Interpretation

VPIN Values:
- 0.0-0.2: Low toxicity
- 0.2-0.4: Normal market conditions
- 0.4-0.6: Elevated toxicity
- 0.6+: High toxicity, potential price movement

