import pandas as pd

# market_data = pd.read_parquet('./data/20230904/20230904-SG28228467-P1-NKZ23-8323274.lvls.parquet')
# market_data = pd.read_parquet('./data/20230905/20230905-SG28231055-P1-NKZ23-8323274.lvls.parquet')
# market_data = pd.read_parquet('./data/20230906/20230906-SG28232495-P1-NKZ23-8323274.lvls.parquet')
# market_data = pd.read_parquet('./data/20230907/20230907-SG28233935-P1-NKZ23-8323274.lvls.parquet')
market_data = pd.read_parquet('./data/20230908/20230908-SG28235375-P1-NKZ23-8323274.lvls.parquet')


statistical_result = market_data[market_data['mc'] == 69].groupby(['t'])['mc'].value_counts().sort_values()
print(statistical_result)
