from datetime import timedelta
from libs.dates_io import tslagged


def detrend_price_series(total_ret, lookback=365):
    avg_lookback = total_ret.rolling(window=lookback, min_periods=1).mean()
    macd = total_ret - avg_lookback
    return macd


def generate_signals(seas, lookfwd, pentry, pexit, initialYr=4):
    """
    Generate trading signals based on seasonal patterns and MACD values.
    :param seas: Seasonal pattern time series
    :param lookfwd: Number of days to look forward in the forecast
    :param pentry: Entry threshold for scaled value
    :param pexit: Exit threshold for scaled value
    :param initialYr: Number of initial years to skip before generating signals
    :param updateFreq: Frequency in days to update the forecast.
    :return: A time series of signals
    """
    updateFreq = 1 # set like this given code bug meaning signal wont change unless udatefreq is 1
    # Ensure  seas are pandas Series with DateTimeIndex
    if not isinstance(seas.index, pd.DatetimeIndex):
        seas.index = pd.to_datetime(seas.index)
    # Initialize variables

    start_date = seas.index[0] + pd.DateOffset(years=initialYr)
    signals = pd.Series(index=seas.index, data=0)
    last_update = updateFreq

    # Convert indices to integers for processing
    seas_dates = seas.index
    init_date = seas[start_date:].index[0]
    seas_start_idx = seas.index.get_loc(init_date)

    for i in range(seas_start_idx, len(seas_dates)):
        # Update the forecast only every updateFreq days
        if (i - last_update) >= updateFreq:
            forecast_start_idx = i - 364
            forecast = seas.iloc[forecast_start_idx:i+1 ]
            last_update = i

        # Take the first lookfwd rows of the forecast
        truncated_forecast = forecast.head(lookfwd)

        # Today's value
        curr_value = truncated_forecast.iloc[1]

        # Quantile calculation
        min_forecast = truncated_forecast.min()
        max_forecast = truncated_forecast.max()
        range_forecast = max_forecast - min_forecast
        scaled_value = (curr_value - min_forecast) / range_forecast if range_forecast != 0 else 0.5
        print(scaled_value)
        # Determine signal
        signal = signals.iloc[i - 1]

        if scaled_value >= pentry:
            signal = -1
        elif scaled_value <= (1 - pentry):
            signal = 1
        elif signal == -1 and scaled_value < pexit:
            signal = 0
        elif signal == 1 and scaled_value > (1 - pexit):
            signal = 0

        signals.iloc[i] = signal

    return tslagged(signals, k = 0, setDaily = True).dropna()

def obsfreq(ts, freq='Y'):
    """Calculate the frequency of observations per specified period starting from the first full year,
       excluding the last year if it is not a full year, and including only full months."""

    # Ensure the index is a DateTimeIndex
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index)

    # Check the frequency and handle accordingly
    if freq == 'Y':
        # Ensure there is at least one full year of data
        start_date = ts.index.min()
        end_date = ts.index.max()
        if (end_date - start_date).days < 365:
            raise ValueError("The time series must contain at least one full year of data")

        # Adjust the start date to the beginning of the next full year if necessary
        if start_date.month != 1 or start_date.day != 1:
            first_full_year_start = pd.Timestamp(year=start_date.year + 1, month=1, day=1)
        else:
            first_full_year_start = pd.Timestamp(year=start_date.year, month=1, day=1)

        # Adjust the end date to the end of the previous full year if necessary
        last_full_year_end = pd.Timestamp(year=end_date.year, month=12, day=31)
        if end_date.month != 12 or end_date.day != 31:
            last_full_year_end = pd.Timestamp(year=end_date.year - 1, month=12, day=31)

        # Ensure the series has at least one full year after adjustments
        if (last_full_year_end - first_full_year_start).days < 365:
            raise ValueError(
                "The time series must contain at least one full year of data starting from the first full year")

        # Filter the time series to include only the full years
        ts_full_period = ts[first_full_year_start:last_full_year_end]

        # Calculate and return the average observations per year
        return ts_full_period.resample('YE').count().mean()

    elif freq == 'M':
        # Ensure there is at least one full month of data
        start_date = ts.index.min()
        end_date = ts.index.max()
        if (end_date - start_date).days < 30:
            raise ValueError("The time series must contain at least one full month of data")

        # Adjust the start date to the beginning of the next full month if necessary
        if start_date.day != 1:
            first_full_month_start = pd.Timestamp(year=start_date.year, month=start_date.month, day=1) + pd.DateOffset(
                months=1)
        else:
            first_full_month_start = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)

        # Adjust the end date to the end of the previous full month if necessary
        last_full_month_end = (pd.Timestamp(year=end_date.year, month=end_date.month, day=1) - pd.DateOffset(
            days=1)).normalize() + pd.offsets.MonthEnd(0)
        if end_date != last_full_month_end:
            last_full_month_end = (pd.Timestamp(year=end_date.year, month=end_date.month, day=1) - pd.DateOffset(
                days=1)).normalize()

        # Ensure the series has at least one full month after adjustments
        if (last_full_month_end - first_full_month_start).days < 30:
            raise ValueError(
                "The time series must contain at least one full month of data starting from the first full month")

        # Filter the time series to include only the full months
        ts_full_period = ts[first_full_month_start:last_full_month_end]

        # Calculate and return the average observations per month
        return ts_full_period.resample('ME').count().mean()

    else:
        raise ValueError("Unsupported frequency")

def rmNA(arr):
    """Remove NA values from the array."""
    return arr[~np.isnan(arr)]

def calculate_seasonal_pattern(macd, years, dts_extended, Usemedian = True):

    # Find the index of the first date in dts_extended that is >= dt_end
    iend = next(i for i, date in enumerate(dts_extended.index) if date >= macd.index[-1])

    def get_sample_dates(signal_date, years):
        sample_dates = []
        for y in range(years + 1):
            # Create a sample date with the same day and month but from previous years
            try:
                sample_date = signal_date.replace(year=signal_date.year - y)
            except ValueError:
                # Handle leap year case for February 29
                sample_date = signal_date.replace(year=signal_date.year - y, day=28)
            # If the sample_date falls on a weekend, adjust to the next business day
            if sample_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                sample_date = sample_date - pd.offsets.BDay()
            sample_dates.append(sample_date)
        return sample_dates

    seasonal_pattern = []
    for i in range(iend):
        signal_date = dts_extended.index[i]
        sample_dates = get_sample_dates(signal_date, years)
        macdx = rmNA(macd.loc[macd.index.intersection(sample_dates)].values)
        if Usemedian:
            mu = np.median(macdx)
        else:
            mu = np.mean(macdx)
        std = np.std(macdx)
        seasonal_pattern.append([mu, std])

    seasonal_pattern_df = pd.DataFrame(seasonal_pattern, columns=['mu', 'std'], index=dts_extended.index[:iend])
    return seasonal_pattern_df.fillna(0)


def GetSeas(total_ret, lookfwd=260, pentry=0.95, pexit=0.80, initialYr=4, cyclYears=1, detrend="Y", diagnositc = False):
    """
    Generate trading signals based on seasonal patterns and detrended price series.

    :param total_ret: Time series of total returns
    :param lookfwd: Number of days to look forward in the forecast
    :param pentry: Entry threshold for scaled value
    :param pexit: Exit threshold for scaled value
    :param initialYr: Number of initial years to skip before generating signals
    :param cyclYears: Number of years to cycle for the seasonal pattern
    :param detrend: Whether to detrend the price series
    :param updateFreq: Frequency in days to update the forecast
    :return: A time series of trading signals
    """

    # Convert total_ret to log returns
    total_ret = np.log(total_ret)

    # Create an extended date index from one day after the first date to 365 days after the last date
    dts_index = pd.date_range(start=total_ret.index[1] + timedelta(days=1),
                              end=total_ret.index[-1] + timedelta(days=365), freq='D')

    # Initialize an extended DataFrame with zeros
    dts_extended = pd.DataFrame(np.zeros(len(dts_index)), index=dts_index)

    # Calculate the number of observations per year and per month
    nr = len(total_ret)
    dyear = round(obsfreq(total_ret, 'Y'), 0)
    years = int(nr / dyear)
    months = int(obsfreq(total_ret, 'M'))

    # Step 1: Detrend the price series using a 1-year average lookback
    macd = detrend_price_series(total_ret, lookback=int(dyear))

    # Step 2: Calculate the seasonal component (seas) for each day in the time series
    seas = calculate_seasonal_pattern(macd, years, dts_extended, Usemedian=False)

    # Apply centered moving average to smooth the seasonal pattern
    window_size = 4  # This window size considers the previous day, current day, and next day
    seas['smoothed_mu'] = seas['mu'].rolling(window=8, center=True).mean().ffill()

    # Step 3: Generate forecast and signals
    signals = generate_signals(seas['smoothed_mu'], lookfwd=lookfwd, pentry=pentry, pexit=pexit, initialYr=initialYr)
    if diagnositc:
        return signals, seas
    else:
        return signals

if __name__ == "__main__":
    # Example usage
    if False:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from libs.Backtester.Backtester import getAssetBT
        from libs.seasonal_utils import *
        systemname, start_date, end_date = "Equity", "2000-01-01", "2021-01-01"
        portdf, ohlc_data = getAssetBT(systemname, "SP")
        total_ret = ohlc_data['SP']['Total_Return']
        seas_sigs = GetSeas(total_ret)
        seas_sigs.plot()
        plt.show()



