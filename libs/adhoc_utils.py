import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

def days_to_reach_confidence(sr_target, sr_benchmark, skewness, kurtosis, confidence):
    """
    Usage:
    #   Example usage
    sr_target = 2.0  # Target Sharpe ratio
    sr_benchmark = 0.0  # We are comparing against 0
    skewness = 0.0  # Assuming normal distribution of returns
    kurtosis = 3.0  # Normal distribution kurtosis
    confidence = 0.95  # 95% confidence

    days_needed = days_to_reach_confidence(sr_target, sr_benchmark, skewness, kurtosis, confidence)
    print(f"Number of days needed: {days_needed}")
    """
    # Set the z-value for the desired confidence level (e.g., 95% confidence)
    z_value = norm.ppf(confidence)

    # Initialize N (number of days)
    N = 2

    while True:
        sr_std = np.sqrt(
            (1 + 0.5 * sr_target ** 2 - skewness * sr_target + (kurtosis - 3) / 4 * sr_target ** 2) / (N - 1))
        psr = norm.cdf((sr_target - sr_benchmark) / sr_std)

        # Check if we have reached the confidence level
        if psr >= confidence:
            return N

        N += 1

def rolling_max_drawdown(returns, window_size):
    rolling_drawdowns = []

    # Iterate through the returns in a rolling window
    for i in range(len(returns) - window_size + 1):
        window = returns[i:i + window_size]
        cumulative_returns = np.cumsum(window)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        rolling_drawdowns.append(np.max(drawdown))

    return np.array(rolling_drawdowns)

def simulate_portfolio_returns(annual_return, annual_vol, n_days, df=5):
    # Convert annual return and volatility to daily return and volatility
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Generate random returns using Student's t-distribution with df degrees of freedom
    random_returns = daily_return + daily_vol * t.rvs(df, size=n_days)

    return random_returns
def simulate_portfolios_with_rolling_drawdowns(annual_return, annual_vol, n_days, n_samples, window_size, df=5):
    """
    # Example usage
    annual_return = 0.1  # 10% annual return
    annual_vol = 0.2  # 20% annual volatility
    n_days = 252  # Simulate 1 year of daily returns
    n_samples = 1000  # Number of portfolios to simulate
    window_size = 30  # Rolling window of 30 days for max drawdown

    # Simulate portfolios and get rolling drawdowns
    sharpe_ratios, rolling_drawdowns = simulate_portfolios_with_rolling_drawdowns(annual_return, annual_vol, n_days,
                                                                              n_samples, window_size)
    # Plot histogram of 30-day rolling max drawdowns
    plot_rolling_drawdowns_histogram(rolling_drawdowns)

    # Calculate the probability of a rolling drawdown greater than x
    x = 0.1  # Example drawdown threshold (10%)
    drawdown_prob = probability_of_rolling_drawdown_exceeding(x, rolling_drawdowns)
    print(f"Probability of seeing a drawdown greater than {x * 100}% in 30 days: {drawdown_prob * 100:.2f}%")
    """
    all_sharpe_ratios = []
    all_rolling_drawdowns = []

    for _ in range(n_samples):
        returns = simulate_portfolio_returns(annual_return, annual_vol, n_days, df)

        sr = sharpe_ratio(returns)
        max_drawdowns = rolling_max_drawdown(returns, window_size)

        all_sharpe_ratios.append(sr)
        all_rolling_drawdowns.extend(max_drawdowns)  # Collect all rolling drawdowns

    return np.array(all_sharpe_ratios), np.array(all_rolling_drawdowns)

def plot_rolling_drawdowns_histogram(max_drawdowns):
    plt.figure(figsize=(12, 6))
    plt.hist(max_drawdowns, bins=30, color='green', alpha=0.7)
    plt.title('Histogram of 30-Day Maximum Drawdowns')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Frequency')
    plt.show()


def probability_of_rolling_drawdown_exceeding(x, rolling_drawdowns):
    exceedance_prob = np.mean(rolling_drawdowns >= x)
    return exceedance_prob

def sharpe_ratio(returns, risk_free_rate=0):
    return (np.mean(returns) - risk_free_rate)*252 / (np.std(returns))*np.sqrt(252)

def max_drawdown(returns):
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return np.max(drawdown)

def simulate_portfolios(annual_return, annual_vol, n_days, n_samples, df=5):
    all_sharpe_ratios = []
    all_max_drawdowns = []

    for _ in range(n_samples):
        returns = simulate_portfolio_returns(annual_return, annual_vol, n_days, df)

        sr = sharpe_ratio(returns)
        md = max_drawdown(returns)

        all_sharpe_ratios.append(sr)
        all_max_drawdowns.append(md)

    return np.array(all_sharpe_ratios), np.array(all_max_drawdowns)

def plot_histograms(sharpe_ratios, max_drawdowns):
    # Plot Sharpe Ratios
    plt.figure(figsize=(12, 6))
    plt.hist(sharpe_ratios, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.show()

    # Plot Max Drawdowns
    plt.figure(figsize=(12, 6))
    plt.hist(max_drawdowns, bins=30, color='red', alpha=0.7)
    plt.title('Histogram of Maximum Drawdowns')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Frequency')
    plt.show()

def probability_of_drawdown_exceeding(x, max_drawdowns):
    """
    # Example usage
    annual_return = 0.02  # 10% annual return
    annual_vol = 0.03  # 20% annual volatility
    n_days = 252  # Simulate 1 year of daily returns
    n_samples = 1000  # Number of portfolios to simulate

    # Simulate portfolios
    sharpe_ratios, max_drawdowns = simulate_portfolios(annual_return, annual_vol, n_days, n_samples)

    # Plot histograms
    plot_histograms(sharpe_ratios, max_drawdowns)

    # Calculate the probability of a drawdown greater than x
    x = 0.1  # Example drawdown threshold
    drawdown_prob = probability_of_drawdown_exceeding(x, max_drawdowns)
    print(f"Probability of seeing a drawdown greater than {x}: {drawdown_prob * 100:.2f}%")
    :param x:
    :param max_drawdowns:
    :return:
    """
    exceedance_prob = np.mean(max_drawdowns >= x)
    return exceedance_prob