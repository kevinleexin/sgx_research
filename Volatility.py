import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from arch import arch_model


def bayes_statistical_inference_for_vol(df, window_size = 15):
    predicted_volatility = []

    # 遍历每个时间点，计算 15 分钟滑动窗口波动率，并使用贝叶斯推断预测下一周期的波动率
    for i in range(window_size, len(df)):
        # 获取当前窗口内的对数收益率
        window_data = df['log_return'].iloc[i - window_size:i].dropna()

        # 构建贝叶斯模型
        with pm.Model() as model:
            # 假设波动率服从半正态分布，均值为0，方差是我们要推断的参数
            volatility = pm.HalfNormal('volatility', sigma=0.1)

            # 似然：观测值为对数收益率，服从均值为0、方差为波动率的正态分布
            likelihood = pm.Normal('likelihood', mu=0, sigma=volatility, observed=window_data)

            # 采样：生成后验分布
            trace = pm.sample(1000, cores=1, return_inferencedata=False, progressbar=False)

            # 计算后验分布的均值，作为预测的波动率
            predicted_vol = trace['volatility'].mean()
            predicted_volatility.append(predicted_vol)

    # 将预测的波动率添加到原 DataFrame 中
    if len(predicted_volatility) < len(df.loc[window_size:]):
        predicted_volatility = np.append(predicted_volatility,
                                         [np.nan] * (len(df.loc[window_size:]) - len(predicted_volatility)))

    df.loc[window_size:, 'predicted_volatility'] = predicted_volatility

    # 打印结果
    print(df[['close', 'log_return', 'predicted_volatility']])

    # 实时数据流更新GARCH模型的流程
    window_size = 100  # 选择一个合适的窗口大小，比如100个周期
    price_changes = []  # 初始化一个空列表，存储价格变化

def update_price_changes(new_price_change, price_changes, window_size):
    price_changes.append(new_price_change)
    if len(price_changes) > window_size:
        price_changes.pop(0)  # 维持滚动窗口大小
    return price_changes

def predict_volatility(price_changes):
    if len(price_changes) < window_size:
        return None  # 如果数据不足，返回空值

    # 创建并拟合GARCH模型
    # scale_factor = 1e4
    garch_model = arch_model(price_changes, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')

    # 预测下一个周期的波动率
    forecast = garch_fit.forecast(horizon=1)
    predicted_volatility = np.sqrt(forecast.variance.iloc[-1, 0])

    return predicted_volatility


if __name__ == "__main__":
    bar_df = pd.read_csv("./data/20230904/NKZ23-1min.csv")
    bar_df['log_return'] = (np.log(bar_df['close'] / bar_df['close'].shift(1))) * 1e4
    bar_df = bar_df.dropna(subset=['log_return'])
    window_size = 5

    # bayes_statistical_inference_for_vol(bar_df, window_size)
    price_changes = list()

    for logret in bar_df['log_return']:
        price_changes = update_price_changes(logret, price_changes, window_size)

        # 实时预测波动率
        predicted_volatility = predict_volatility(price_changes)
        if predicted_volatility is not None:
            print(f"Predicted volatility for the next period: {predicted_volatility}")

