import numpy as np
from scipy.stats import norm, poisson
from arch import arch_model

# 初始化参数
P_up = 0.5
P_down = 0.5

mu_price_up, sigma_price_up = 0.1, 0.02  # 上涨时的价格变化分布
mu_price_down, sigma_price_down = -0.1, 0.02  # 下跌时的价格变化分布
lambda_volume_up = 1000  # 初始成交量均值
lambda_volume_down = 500  # 初始成交量均值

beta = 0.5  # 波动率的敏感系数
window_size = 10  # 滚动窗口大小

# 保存最近窗口的价格变化数据
price_changes_window = []

# 检查数据是否包含NaN或inf值
def clean_data(data):
    return [x for x in data if np.isfinite(x)]  # 保留有限值


def update_price_changes(new_price_change, price_changes, new_delta_change, delta_changes, window_size=10):
    # 将新的价格变化加入序列
    price_changes.append(new_price_change)
    delta_changes.append(new_delta_change)
    # 如果价格变化数据超过窗口大小，则删除最早的数据
    if len(price_changes) > window_size:
        price_changes.pop(0)
        delta_changes.pop(0)

    return price_changes, delta_changes



def update_distribution_parameters(price_changes, window_size):
    # 使用滚动窗口动态计算均值和标准差
    mu_price_up = np.mean(price_changes[-window_size:])  # 动态均值
    sigma_price_up = np.std(price_changes[-window_size:])  # 动态标准差
    return mu_price_up, sigma_price_up


def compute_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = 100. - 100. / (1. + rs)

    for delta in deltas[window:]:
        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta
        up = (up * (window - 1) + up_val) / window
        down = (down * (window - 1) + down_val) / window
        rs = up / down
        rsi = 100. - 100. / (1. + rs)
    return rsi

# 使用 GARCH 模型实时计算波动率（基于窗口价格数据）
def calculate_volatility_garch(price_changes_window):
    clean_window = clean_data(price_changes_window)
    if len(clean_window) < window_size:
        # 如果数据不足，返回一个默认波动率值
        return 0.02
    else:
        # 将价格变化的窗口数据缩放
        scale_factor = 1e4
        scaled_window = np.array(clean_window) * scale_factor  # 放大数据
        am = arch_model(scaled_window, vol='Garch', p=1, q=1)
        res = am.fit(disp="off")
        return np.sqrt(res.conditional_volatility[-1]) / scale_factor # 只返回最新的波动率

# 动态更新泊松均值 (引入市场波动率)
def update_lambda_with_volatility(base_lambda, volatility, beta):
    return base_lambda * (1 + beta * volatility)

# 实时贝叶斯更新函数（考虑盘口价差影响）
# 贝叶斯更新函数，加入检查
def update_bayesian(price_change, volume_change, P_up, P_down, lambda_up, lambda_down):
    # 计算价格变化的似然
    likelihood_price_up = norm.pdf(price_change, mu_price_up, sigma_price_up)
    likelihood_price_down = norm.pdf(price_change, mu_price_down, sigma_price_down)

    # 计算成交量变化的似然
    likelihood_volume_up = poisson.pmf(volume_change, lambda_up)
    likelihood_volume_down = poisson.pmf(volume_change, lambda_down)

    # 联合似然函数
    likelihood_up = likelihood_price_up * likelihood_volume_up
    likelihood_down = likelihood_price_down * likelihood_volume_down

    print(f"Likelihood up: {likelihood_up}, Likelihood down: {likelihood_down}")

    # 计算分母，防止为 0 或 NaN
    denominator = likelihood_up * P_up + likelihood_down * P_down
    if denominator == 0 or not np.isfinite(denominator):
        # 如果分母为 0 或 NaN，返回默认值
        return 0.5, 0.5  # 默认认为上涨和下跌的概率各为 50%

    # 贝叶斯更新公式
    posterior_up = (likelihood_up * P_up) / denominator
    posterior_down = 1 - posterior_up

    return posterior_up, posterior_down

# 实时处理数据流函数
def process_new_data(price_change, volume_change, P_up, P_down, price_changes_window):
    # 更新价格变化的滚动窗口
    price_changes_window.append(price_change)
    if len(price_changes_window) > window_size:
        price_changes_window.pop(0)

    # 计算当前的波动率
    volatility = calculate_volatility_garch(price_changes_window)

    # 动态调整泊松分布的均值
    lambda_up = update_lambda_with_volatility(lambda_volume_up, volatility, beta)
    lambda_down = update_lambda_with_volatility(lambda_volume_down, volatility, beta)

    # 贝叶斯更新
    P_up, P_down = update_bayesian(price_change, volume_change, P_up, P_down, lambda_up, lambda_down)

    # 输出实时更新后的信息
    print(f"实时数据处理:")
    print(f"    价格变化: {price_change:.4f}, 成交量变化: {volume_change}")
    print(f"    动态波动率: {volatility:.4f}")
    print(f"    上涨概率: {P_up:.4f}, 动态上涨泊松均值: {lambda_up:.2f}")
    print(f"    下跌概率: {P_down:.4f}, 动态下跌泊松均值: {lambda_down:.2f}")

    # 返回更新后的后验概率和窗口数据
    return P_up, P_down, price_changes_window


if __name__ == "__main__":

    # 模拟实时数据流
    price_changes_stream = np.array([0.02, 0.05, -0.03, 0.12, -0.1, 0.04, 0.02, -0.02])  # 模拟价格变化
    volume_changes_stream = np.array([800, 1200, 400, 1500, 300, 1100, 900, 700])  # 模拟成交量变化
    spread_changes_stream = np.array([0.02, 0.01, 0.015, 0.025, 0.02, 0.01, 0.012, 0.018])  # 模拟盘口价差

    # 处理每一笔实时数据
    for price_change, volume_change, spread_change in zip(price_changes_stream, volume_changes_stream, spread_changes_stream):
        P_up, P_down, price_changes_window = process_new_data(price_change, volume_change, spread_change, P_up, P_down, price_changes_window)

        # 根据后验概率预测趋势
        if P_up > 0.5:
            print("    预测：上涨\n")
        else:
            print("    预测：下跌\n")