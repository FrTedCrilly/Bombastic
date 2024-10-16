import pandas as pd
import numpy as np
import copy
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from libs.c_speed import c_utils
import matplotlib.pyplot as plt

def keepSig(sig, keeplong, keepshort):
    """
    :param sig: the base entry signal
    :param keeplong: the condition for which to keep the signal long
    :param keepshort: the condition for which to keep the signal short
    :return: full signal.
    """
    newSig = sig.copy()

    for i in range(len(sig) - 1):  # Correct iteration
        j = i + 1
        original = sig.iloc[j]  # Use iloc for positional indexing
        olds = newSig.iloc[j - 1]

        # Corrected logical conditions
        if olds == 1 and keeplong.iloc[j] and original >= 0:
            newSig.iloc[j] = olds
        elif olds == -1 and keepshort.iloc[j] and original <= 0:
            newSig.iloc[j] = olds
        else:
            newSig.iloc[j] = original

    return(newSig)


def getFreqData(macroData):
    """
    Takes in a time series, checks that it is a DataFrame with an index,
    then determines the data frequency (daily, weekly, monthly, quarterly, or yearly).

    :param macroData: pandas DataFrame
    :return: frequency of the data as a string
    """
    if not isinstance(macroData, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if macroData.index is None or not isinstance(macroData.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Use pandas built-in frequency inference
    inferred_freq = pd.infer_freq(macroData.index)

    if inferred_freq is None:
        # Fallback to custom frequency determination if infer_freq fails
        delta = macroData.index.to_series().diff().dropna().mode()[0]

        if delta.days == 1:
            return "D"
        elif delta.days >= 7 and delta.days < 30:
            return "W"
        elif delta.days >= 30 and delta.days < 90:
            return "M"
        elif delta.days >= 90 and delta.days < 365:
            return "Q"
        elif delta.days >= 365:
            return "Y"
        else:
            return "unknown"
    else:
        # Map the inferred frequency to custom frequency labels
        freq_map = {
            'D': 'D',
            'W': 'W',
            'M': 'M',
            'Q': 'Q',
            'A': 'Y',
            'H': 'hour',
            'T': 'minute',
            'S': 'second',
            'B': 'D',  # Business daily
            'BM': 'M',  # Business month-end
            'BQ': 'Q',  # Business quarter-end
            'BA': 'Y',  # Business year-end
            'ME': 'M',  # Month-end
            'MS': 'M',  # Month-start
            'QS': 'Q',  # Quarter-start
            'AS': 'Y'  # Year-start
        }
        return freq_map.get(inferred_freq, 'unknown')


def convertFreq(freq):
    """
    Converts a given frequency label to a conversion factor that normalizes the period to a daily equivalent.

    Parameters:
    freq (str): The frequency label ('D', 'W', 'M', 'Q', 'Y', 'H', 'minute', 'second', etc.)

    Returns:
    float: The conversion factor to normalize the period to a daily equivalent.
    """

    # Define the conversion factors based on the provided map
    freq_conversion = {
        'D': 1,  # Daily, no conversion needed
        'W': 5,  # Weekly, assume 5 trading days in a week
        'M': 21,  # Monthly, assume 21 trading days in a month
        'Q': 63,  # Quarterly, 21 trading days per month * 3 months
        'Y': 260,  # Annual, assume 260 trading days in a year (equivalent to 'Y')
        'hour': 1 / 24,  # Hourly, 24 hours in a day
        'minute': 1 / 1440,  # Minutely, 1440 minutes in a day
        'second': 1 / 86400,  # Secondly, 86400 seconds in a day
        'B': 1,  # Business daily, same as daily
        'ME': 21,  # Business month-end, assume 21 trading days in a month
        'QE': 63,  # Business quarter-end, 21 trading days per month * 3 months
        'YE': 260,  # Business year-end, assume 260 trading days in a year
        'ME': 21,  # Month-end, assume 21 trading days in a month
        'QS': 63,  # Quarter-start, 21 trading days per month * 3 months
        'YS': 260  # Year-start, assume 260 trading days in a year
    }

    # Return the conversion factor, or 1 if the frequency is unknown
    return freq_conversion.get(freq, 1)

def check_and_clean_data(df):
    """Check for missing values, remove outliers, and perform additional data cleaning."""
    # Ensure data is in return format (e.g., daily returns are typically between -1 and 1)
    if df.apply(lambda x: (x < -1).any() or (x > 1).any()).any():
        print("Warning: Some values are outside the typical range for returns. Please verify your data.")

    # Convert extreme values to NaN (values outside 6 standard deviations)
    mean = df.mean()
    std = df.std()
    is_outlier = (df < (mean - 20 * std)) | (df > (mean + 20 * std))
    if df.loc[is_outlier.values].values.shape[0] > 0:
        print("Warning: Some values are outside the typical, as below")
        print(df.loc[is_outlier.values])
        df.loc[is_outlier.values] = np.nan

    # Report on detected outliers
    if is_outlier.any().any():
        print("Outliers detected and set to NaN. Outliers are values beyond 20 standard deviations from the mean.")

    # Drop rows with NaN values (from outliers and missing values)
    original_row_count = len(df)
    df = df.dropna()
    cleaned_row_count = len(df)
    if original_row_count > cleaned_row_count:
        print(f"Rows with missing values or outliers removed: {original_row_count - cleaned_row_count}")

    # Additional checks could include:
    # - Verifying the frequency and continuity of the datetime index (for time series data).
    # - Ensuring no duplicate dates/indexes exist.
    # - Checking for sudden spikes in volatility that aren't tagged as outliers by the standard deviation criterion.

    return df
def annualize_factor(frequency):
    """Return the annualization factor based on the data frequency."""
    factors = {
        'daily': 260,
        "D": 260,
        "d": 260,
        "W":52,
        "Weekly": 52,
        'weekly': 52,
        'monthly': 12,
        'M': 12,
        'ME': 12,
        'Monthly' : 12
    }
    return factors.get(frequency.lower(), 260)  # Default to daily if not found

def sharpe_ratio(df, risk_free_rate=0,frequency='daily', diff_rets=False):
    """Calculate the Sharpe Ratio of a DataFrame of returns.
       Assumes that you input arithmetic returns for the stats
    """
    if diff_rets:
        df = df.diff().dropna()
    df = check_and_clean_data(df)
    mean_returns = df.mean()*annualize_factor(frequency)
    std_returns = df.std()*(annualize_factor(frequency) ** 0.5)
    sharpe_ratio = (mean_returns - risk_free_rate) / std_returns
    return sharpe_ratio

def sortino_ratio(df, risk_free_rate=0,frequency='daily', diff_rets=False):
    """Calculate the Sortino Ratio of a DataFrame of returns."""
    if diff_rets:
        df = df.diff().dropna()
    df = check_and_clean_data(df)
    mean_returns = df.mean()*annualize_factor(frequency)
    downside_returns = df[df < risk_free_rate].copy()
    downside_std = downside_returns.std()*(annualize_factor(frequency) ** 0.5)
    sortino_ratio = (mean_returns - risk_free_rate) / downside_std
    return sortino_ratio

def skewness(df,diff_rets=False):
    """Calculate skewness of a DataFrame of returns."""
    if diff_rets:
        df = df.diff().dropna()
    df = check_and_clean_data(df)
    return df.apply(skew)

def Getkurtosis(df,diff_rets=False):
    """Calculate kurtosis of a DataFrame of returns."""
    if diff_rets:
        df = df.diff().dropna()
    df = check_and_clean_data(df)
    return df.apply(kurtosis)


def maxDD(data, window=None, arithmetic = True):
    """
    Calculate the rolling maximum drawdown for each series in a DataFrame.
    If input is a Series, convert it to a DataFrame first.

    Parameters:
    data (pd.Series or pd.DataFrame): A pandas Series or DataFrame representing cumulative returns.
    window (int or None): The rolling window size (in days) over which to calculate the maximum drawdown.
                          If None, use an expanding window.

    Returns:
    pd.DataFrame: A pandas DataFrame of the rolling maximum drawdown.
    """
    # If the input is a Series, convert it to a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if window is None:
        # Use an expanding window (expanding max)
        rolling_max = data.expanding(min_periods=1).max()
    else:
        # Use a fixed rolling window
        rolling_max = data.rolling(window, min_periods=1).max()

    # Calculate the drawdown for each column
    drawdown = (data - rolling_max) / (100 if arithmetic else rolling_max) # assuming 100 start point.

    # Calculate the rolling max drawdown (minimum drawdown) for each column
    rolling_max_drawdown = drawdown.min()

    return rolling_max_drawdown

def hp_filter(timeseries, lamb):
    """
    Apply the Hodrick-Prescott (HP) filter to decompose a timeseries into its trend component.

    Parameters:
    - timeseries: The time series data as a numpy array.
    - lamb: The smoothing parameter (lambda) for the HP filter.

    Returns:
    - The trend component of the time series.
    """

    # Define the loss function for the HP filter
    def hp_filter_loss(tau, y, lamb):
        return np.sum((y - tau) ** 2) + lamb * np.sum(np.diff(tau, 2) ** 2)

    # Initial guess for the trend is the original data
    initial_tau = np.copy(timeseries)

    # Minimize the HP filter loss function to find the trend
    result = minimize(hp_filter_loss, initial_tau, args=(timeseries, lamb), method='L-BFGS-B')

    # Extract the trend from the optimization result
    trend = result.x

    return trend


def apply_zscorePY(var, Zwin, Zexpand, rmMean=True, cheat=True):
    if not isinstance(var, pd.DataFrame):
        raise AttributeError("var does not exist or is not a DataFrame. You need to provide a valid DataFrame.")

    if Zexpand:
        if rmMean:
            var = var.apply(lambda x: (x - x.expanding().mean()) / x.expanding().std())
        else:
            var = var.apply(lambda x: (x) / x.expanding().std())

    else:
        if rmMean:
            var = var.apply(lambda x: (x - x.rolling(window=Zwin, min_periods=1).mean()) / x.rolling(window=Zwin, min_periods=1).std())
        else:
            var = var.apply(lambda x: (x ) / x.rolling(window=Zwin, min_periods=1).std())

    if cheat:
        var = var.apply(lambda x: x.ffill().bfill())

    return var


def apply_quantilePY(var, Qwin, Qexpand):
    if not isinstance(var, pd.DataFrame):
        raise AttributeError("var does not exist or is not a DataFrame. You need to provide a valid DataFrame.")

    if Qexpand:
        var = var.apply(lambda x: x.expanding().apply(lambda y: pd.Series(y).rank(pct=True).iloc[-1]))
    else:
        var = var.apply(
            lambda x: x.rolling(window=Qwin, min_periods=1).apply(lambda y: pd.Series(y).rank(pct=True).iloc[-1]))

    return var

def sigmoid_expand(series):
    transformed = []
    for i in range(len(series)):
        if i == 0:
            transformed.append(0.5)  # Sigmoid(0) = 0.5 as a neutral starting point
            continue
        window = series[:i+1]
        mean = window.mean()
        std = window.std()
        if std == 0:
            std = 1  # Prevent division by zero
        standardized_value = (series[i] - mean) / std
        sigmoid_value = 1 / (1 + np.exp(-standardized_value))
        transformed.append(sigmoid_value)
    return pd.Series(transformed, index=series.index)

def sigmoidSig(value):
    return np.sin(np.pi * (value - 0.5))

def continousSigPY(value):
    if (value >= 0.45) and (value <= 0.55):
        return 0.0
    else:
        return np.floor(value * 20) / 20


def apply_sandPY(df, sand, signal_column = None):
    if signal_column == None:
        adjusted_signal = copy.deepcopy(df)
    else:
        adjusted_signal = copy.deepcopy(df[signal_column])

    previous_value = adjusted_signal.iloc[0]

    for i in range(1, len(adjusted_signal)):
        current_value = adjusted_signal.iloc[i]
        if abs(current_value - previous_value) >= sand:
            previous_value = current_value
        else:
            adjusted_signal.iloc[i] = previous_value

    return adjusted_signal

def sig_sandPY(quantiles, pEnter, pExit):
    signals = np.zeros(len(quantiles))
    long_enter = 1 - pEnter
    long_exit = 1 - pExit
    short_enter = pEnter
    short_exit = pExit
    position = 0  # 1 for long, -1 for short, 0 for no position

    for i in range(len(quantiles)):
        quantile = quantiles[i]

        if position == 0:
            if quantile > long_enter:
                position = 1
                signals[i] = 1
            elif quantile < short_enter:
                position = -1
                signals[i] = -1
        elif position == 1:
            if quantile < long_exit:
                position = 0
                signals[i] = 0
            else:
                signals[i] = 1
        elif position == -1:
            if quantile > short_exit:
                position = 0
                signals[i] = 0
            else:
                signals[i] = -1

    return signals

def expanding_corr(df, start_expanding_from=200):
    """
    Calculate expanding correlation from a specified observation and backfill the value
    at the starting observation backwards for the initial values.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    var_col (str): Column name for the var series.
    close_col (str): Column name for the close series.
    start_expanding_from (int): Observation from which to start the expanding window.

    Returns:
    pd.Series: Series with expanding correlation and backfilled initial values.
    """

    # Calculate the expanding correlation starting from the specified observation
    expanding_corr = pd.Series(index=df.index, dtype=float)
    expanding_corr[start_expanding_from:] = df[start_expanding_from:].expanding().corr(
        df[var_col][start_expanding_from:])

    # Backfill the initial values with the value at the starting observation
    expanding_corr[:start_expanding_from] = expanding_corr[start_expanding_from]

    return expanding_corr

def apply_kalman_filter(timeseries, transition_matrices, observation_matrices,
                        transition_covariance, observation_covariance,
                        initial_state_mean, initial_state_covariance):
    """
    Applies the Kalman Filter to a given time series.

    Parameters:
    - timeseries: The observed time series data.
    - transition_matrices: Specifies the transition model of the Kalman filter.
    - observation_matrices: Specifies the observation model of the Kalman filter.
    - transition_covariance: Specifies the covariance of the transition model.
    - observation_covariance: Specifies the covariance of the observation model.
    - initial_state_mean: The mean of the initial state distribution.
    - initial_state_covariance: The covariance of the initial state distribution.

    Returns:
    - Filtered time series as per the specified Kalman Filter parameters.
    """
    """    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      transition_covariance=transition_covariance,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance)

    # Use the Kalman Filter to estimate states
    filtered_state_means, filtered_state_covariances = kf.filter(timeseries)"""

    return 1

def apply_zscore(df, Zwin = 250, Zexpand = True, rmMean = True, cheat = True):
    zs = pd.DataFrame(c_utils.apply_zscore(df.to_numpy(), Zwin=Zwin, Zexpand=Zexpand, rmMean=rmMean, cheat=cheat), index=df.dropna().index, columns=df.columns)
    return zs
def apply_quantile(df, Qwin = 250, Qexpand = True):
    qs = pd.DataFrame(c_utils.apply_quantile(df.to_numpy(), Qwin, Qexpand), index=df.dropna().index, columns=df.columns)
    return qs
def apply_sand(df, sand = 0.05):
    ss = pd.DataFrame(c_utils.apply_sand(df.to_numpy(), sand = sand), index=df.dropna().index, columns=df.columns)
    return ss
def sigSand(df, pEnter, pExit):
    sig_s = pd.DataFrame(c_utils.sigSand(df.to_numpy(), pEnter = pEnter, pExit = pExit), index=df.dropna().index, columns=df.columns)
    return sig_s

def continousSig(df):
    sig_s = pd.DataFrame(c_utils.continousSig(df.to_numpy()), index=df.dropna().index, columns=df.columns)
    return sig_s