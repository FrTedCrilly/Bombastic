import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, BMonthEnd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, MO
from datetime import datetime
import bisect

# Define UK holidays (simplified version)
class UKHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        Holiday('Boxing Day', month=12, day=26, observance=nearest_workday),
        # Add other UK holidays here
    ]


# Create CustomBusinessDay objects for each country
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
uk_bd = CustomBusinessDay(calendar=UKHolidayCalendar())


import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import holidays

def tslagged(df, k, country='US', setDaily=False, fwdLooking=False):
    """
    Shifts a time series DataFrame by a set number of business days based on the country's calendar.

    :param df: DataFrame with DateTimeIndex.
    :param k: Number of business days to shift. Positive for future, negative for lag.
    it shifts by k days and will create the extra days in the future as well (at the tail of the df)
    :param country: "US" or "UK" for the business day calendar.
    :param setDaily: If True, converts non-daily data to daily frequency, aligning to business days.
    :param fwdLooking: will only return future dates if asked for it.
    """
    # Define the holiday calendars
    if country == 'US':
        holidays_list = holidays.US()
    elif country == 'UK':
        holidays_list = holidays.UK()
    else:
        raise ValueError("Country not supported. Please use 'US' or 'UK'.")

    # Create the custom business day offset
    bd = CustomBusinessDay(calendar=holidays_list)

    if setDaily:
        # Resample to daily frequency using the business day offset, filling missing values with NaN
        df_daily = df.resample('B').asfreq()
    else:
        df_daily = df.copy()

    # Remove duplicate indices
    df_daily = df_daily[~df_daily.index.duplicated(keep='first')]

    # Shift the index by the specified number of business days
    if fwdLooking:
        shifted_dates = df_daily.index + (k * bd)
        return df_daily.set_index(shifted_dates)
    else:
        shifted_dates = df.index + (k * bd)
        return df_daily.set_index(shifted_dates).reindex(df.index)

# Example usage
# Assuming `data` is a DataFrame with a DateTimeIndex
# shifted_df = tslagged(data, k=5, country='US', setDaily=True, fwdLooking=True)

def align_time_series(varIn, varToAlignTo, carry_over=True):
    """
    Aligns varIn DataFrame to the dates of varToAlignTo DataFrame, Series, or DatetimeIndex.
    Parameters:
    - varIn: The DataFrame to be aligned.
    - varToAlignTo: The DataFrame, Series, or DatetimeIndex whose index dates varIn will be aligned to.
    - carry_over: If True, carries over the last known value from varIn when no direct match exists.
                  If False, missing values will be NaN.
    Returns:
    - A new DataFrame with varIn's data aligned to the index of varToAlignTo.
    """
    # Extract the index from varToAlignTo, depending on its type
    if isinstance(varToAlignTo, (pd.DataFrame, pd.Series)):
        align_index = varToAlignTo.index
    elif isinstance(varToAlignTo, pd.DatetimeIndex):
        align_index = varToAlignTo
    else:
        raise ValueError("varToAlignTo must be a DataFrame, Series, or DatetimeIndex")

    # Check if align_index is a DatetimeIndex
    if not isinstance(align_index, pd.DatetimeIndex):
        raise TypeError("The index of varToAlignTo is not a DatetimeIndex")

    # Ensure varIn's 'Date' column is in datetime format and set it as the index if it's not already
    if not isinstance(varIn.index, pd.DatetimeIndex):
        if 'Date' in varIn.columns:
            varIn['Date'] = pd.to_datetime(varIn['Date'])
            varIn.set_index('Date', inplace=True)
        else:
            raise ValueError("varIn must have a 'Date' column or be indexed by datetime")

    # Reindex varIn to match the index of varToAlignTo, carrying over values if specified
    aligned_df = varIn.reindex(align_index, method='ffill' if carry_over else None)

    return aligned_df


# Main function to get business month ends
def get_BizME(start_date, end_date = datetime.now(), country='US'):
    # Select the appropriate holiday calendar
    if country == 'US':
        holiday_calendar = USFederalHolidayCalendar()
    elif country == 'UK':
        holiday_calendar = UKHolidayCalendar()
    else:
        raise ValueError("Unsupported country. Please choose 'US' or 'UK'.")

    # Create a CustomBusinessDay object considering the selected holiday calendar
    business_day = CustomBusinessDay(calendar=holiday_calendar)

    # Generate a range of business days
    business_days = pd.date_range(start=start_date, end=end_date, freq=business_day)

    # Create a series with the month for each business day
    months = business_days.month

    month_changes = months.diff() !=0

    # return the biz days of the day before the month changes.
    return business_days.shift(-1)[month_changes]

# Example usage
# Assuming `your_timeseries_df` is your DataFrame with a DateTimeIndex

def find_index(dates_or_df, input_date, x = 0, date_column='date'):
    # Convert input_date to datetime object if it is a string
    if isinstance(input_date, pd.Timestamp):
        input_date = input_date.to_pydatetime()
    elif isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
    else:
        raise ValueError("input_date must be a string or pandas.Timestamp")

    # Determine if input is list or DataFrame
    if isinstance(dates_or_df, list):
        dates = pd.to_datetime(dates_or_df).sort_values().tolist()
    elif isinstance(dates_or_df, pd.DataFrame):
        if date_column in dates_or_df.columns:
            dates = pd.to_datetime(dates_or_df[date_column]).sort_values().tolist()
        elif isinstance(dates_or_df.index, pd.DatetimeIndex):
            dates = pd.to_datetime(dates_or_df.index).sort_values().tolist()
        else:
            raise ValueError("DataFrame must have a date index or a date column.")
    else:
        raise ValueError("Input must be a list or a pandas DataFrame.")

    # Create a pandas Series for date indexing
    date_index = pd.Series(dates).sort_values().reset_index(drop=True)

    # Look for the exact match
    if input_date in date_index.values:
        nearest_index = date_index[date_index == input_date].index[0]
    else:
        # Convert dates to timestamps for comparison
        timestamps = [date.timestamp() for date in dates]
        input_timestamp = input_date.timestamp()

        # Use binary search to find the nearest date
        pos = bisect.bisect_left(timestamps, input_timestamp)

        # Determine the nearest index
        if pos == 0:
            nearest_index = 0
        elif pos == len(dates):
            nearest_index = len(dates) - 1
        else:
            before = timestamps[pos - 1]
            after = timestamps[pos]
            nearest_index = pos if (after - input_timestamp) < (input_timestamp - before) else pos - 1

    # Calculate the index x places after the nearest date
    target_index = nearest_index + x

    # Retrieve the date at target_index
    if 0 <= target_index < len(dates):
        return dates[target_index].strftime('%Y-%m-%d')
    else:
        return None  # Handle the case where the index is out of range


def rangeDates(start, end, interval='1M', include_start_date = False):
    """
    Generates dates from start_date to finish_date by adding the specified interval.

    Parameters:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        finish_date (str): The finish date in 'YYYY-MM-DD' format.
        interval (str): The interval to add ('1M', '3M', '6M', '1Y').

    Returns:
        List of dates from start_date to finish_date.
    """

    # Ensure start_date and finish_date are pandas Timestamps
    start_date = pd.to_datetime(start)
    finish_date = pd.to_datetime(end)

    # Check if the interval is valid
    if interval not in ['1M', '3M', '6M', '1Y']:
        raise ValueError("Interval must be one of '1M', '3M', '6M', '1Y'")

    # Determine the offset based on the interval
    if interval == '1M':
        offset = pd.DateOffset(months=1)
    elif interval == '3M':
        offset = pd.DateOffset(months=3)
    elif interval == '6M':
        offset = pd.DateOffset(months=6)
    elif interval == '1Y':
        offset = pd.DateOffset(years=1)

    # Generate dates
    dates = []
    current_date = start_date

    if not include_start_date:
        current_date += offset

    while current_date <= finish_date:
        dates.append(current_date)
        current_date += offset

    return dates
