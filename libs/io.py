# Getting raw files and raw data
import copy
import pandas as pd
import numpy as np
import yfinance as yf
import os
from pandas.tseries.offsets import BDay
from libs.trade_utils import calculate_atr
from libs.dates_io import align_time_series
from libs.utils_setup import modDF


def getFolioFile(SystemName, sysversion = None):
    """
    :param SystemName:
    :return: the csv file of the system needed. All params which are the attributes of the system
    """
    # TODO: obviously its all hardcoded atm
    baseDir = r'C:\Users\edgil\Documents\git_working\Research\config\asset_config'
    if sysversion == None:
        sysversion = SystemName
    folioFile = pd.read_csv(baseDir + '\\'  + SystemName + '\\' + sysversion + ".csv")
    return(folioFile)


def getAssetFutures(asset_name, asset_class= None):
    """
    ideally call to a database with the OHLC closes alongside the stitched future and or the swap pv01 TR series.
    multiple if statements to handle to different conventions, stitching, carry, roll, expiry etc,
    :param asset_name:
    :return:
    """
    outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
    front_df = pd.read_csv(outDir + "SP_front.csv", index_col=0, parse_dates=True)
    back_df = pd.read_csv(outDir + "SP_back.csv", index_col=0, parse_dates=True)
    return((front_df, back_df))

def getQuickData(ticker_symbol, start_date, end_date):
    # Define the ticker symbol for the S&P 500 ETF as an example
    ticker_symbol = '^GSPC'  # For the S&P 500 index itself
    # or 'SPY' for the S&P 500 ETF, which tracks the performance of the S&P 500

    # Download historical data for the S&P 500
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

    # Display the first few rows of the data
    return(data)

def getAssetBT(SystemName,asset = None):
    """
    :param SystemName:
    :param asset:
    :return:
    """
    portdf = getFolioFile(SystemName)
    if asset != None:
        try:
            # Make it iterate over a list of assets and bring back only those.
            # weights of subset will be proportional.
            portdf = portdf.explode('Asset').query('Asset in @asset').drop_duplicates()
            if portdf.empty:
                print("Asset doesn't exist in portfolio:", asset)
        except Exception as e:
            # This block executes if an exception occurs in the try block
            print(f"Error with asset '{asset}' in system '{SystemName}': {e}")
    # Initialize an empty dictionary to store the OHLC data for each asset
    ohlc_data_dict = {}

    # Iterate over each row in the portfolio DataFrame to fetch OHLC data
    for index, row in portdf.iterrows():
        # Check if yFinanceTicker is available for the asset
        if pd.notna(row['Asset']):
            asset = row['Asset']
            fx_quote = row['quotecurrency']
            fx_risk = row['riskcurr']
            instrument_type = row['class']
            if instrument_type == "Future":
                front_df, back_df = getAssetFutures(asset)
                # TODO: obviously its all hardcoded atm
                # Store the OHLC data in the dictionary with the asset as the key
                backAdj_Prices = back_adjust_futures(front_df, back_df)
                backAdj_Prices['quotecurrency'] = getFXClose( fx_quote + "USD", backAdj_Prices['Close'])
                backAdj_Prices['riskcurr'] = getFXClose(fx_risk + "USD", backAdj_Prices['Close'])
                ohlc_data_dict[asset] = backAdj_Prices
            elif instrument_type == "FX":
                baseDollar = row['baseDollar']
                termDollar = row['termDollar']
                backAdj_Prices= getFXData(asset)
                backAdj_Prices['baseDollar'] = getFXClose(baseDollar, backAdj_Prices['Close'])
                backAdj_Prices['termDollar'] = getFXClose(termDollar, backAdj_Prices['Close'])
                ohlc_data_dict[asset] = backAdj_Prices

    return(portdf, ohlc_data_dict)

def getFXClose(ticker, var = None):
    """
    get the ohlc and add expiry contract date, add labellign for OHLC and also add the ATR (14)
    :param data:
    :return:
    """
    outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
    fx_rate = pd.read_csv(outDir + ticker +".csv", index_col=0, parse_dates=True)
    fx_rate.index = pd.to_datetime(fx_rate.index,  format="%d/%m/%Y")
    if isinstance(var, (pd.DataFrame, pd.Series)):
        return align_time_series(fx_rate.dropna(), var.index)
    else:
        return (fx_rate.dropna())


def get_spot_rate(asset):
    # Placeholder: Implement logic to fetch the spot rate
    return 1.0

def get_implied_yield(asset):
    # Placeholder: Implement logic to fetch the implied yield
    return 0.05

def FXCarry(asset, spot,fwd, usePoints = False, fwd_date = "Settle_DT", spot_date = "Reference_Date",  daily = True):
    if spot is None or fwd is None:
        # Add columns for rates based on the asset
        spot_rate = get_spot_rate(asset)
        implied_yield = get_implied_yield(asset)

        # Calculate forward rate
        fwd_rate = modDF(spot_rate, implied_yield, "Add")

        # Calculate implied yield based on the current forward and spot rates
        implied_yield = modDF(modDF(fwd_rate, spot_rate, "sub"), fwd_rate, "div")

        # Convert string dates to datetime
        fwd_date = pd.to_datetime(fwd_date, errors='coerce')
        spot_date = pd.to_datetime(spot_date, errors='coerce')

        # Calculate the number of days between fwd_date and spot_date
        delta_days = (fwd_date - spot_date).dt.days

        # Handle cases where dates are NA
        delta_days.fillna(30, inplace=True)  # Default to 30 days if dates are NA

        # Calculate daily carry
        daily_carry = implied_yield / delta_days
        daily_carry.columns = ['daily_carry']
        implied_yield.columns = ['implied_yield']
    else:
        if isinstance(fwd, (pd.Series)):
            fwd = pd.DataFrame(fwd)
        if fwd.columns.str.contains("1M").any():
            implied_yield = modDF(modDF(spot,fwd, "Sub"), fwd, "div")
            # find days in month from there to 1 m out?
            days_in_month = pd.DataFrame((spot.index+ pd.tseries.offsets.DateOffset(months = 1) - spot.index).days, index = spot.index)
            daily_carry = modDF(implied_yield, days_in_month,"div") # rough carry, will need to look at the days between spot vale and settle date.

    return daily_carry , implied_yield

def WriteSig(folder_name, asset_name, new_signal_data):
    """
    Writes or appends aggregated signal data to a CSV file without overwriting existing data.

    :param folder_name: Directory to save or append the CSV.
    :param asset_name: Name of the asset, used for the CSV file name.
    :param new_signal_data: DataFrame with new signals as columns and dates as either index or a 'Date' column.
    """
    file_path = os.path.join(folder_name, f"{asset_name}_signals.csv")

    # Check if 'Date' is a column and set it as index if necessary
    if not isinstance(new_signal_data.index, pd.DatetimeIndex):
        print("Error: The index is not a datetime index. Attempting to convert it.")
        try:
            new_signal_data.index = pd.to_datetime(new_signal_data.index)
        except Exception as e:
            print(f"Error: The index cannot be converted to datetime. {e}")
            return

    # Check if the index is sorted in chronological order
    if not new_signal_data.index.is_monotonic_increasing:
        print("Warning: The index is not sorted in chronological order. Sorting the index.")
        new_signal_data.sort_index(inplace=True)

    # Check if the file exists and load existing data
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        combined_data = merge_df_with_checks(existing_data, new_signal_data)
    else:
        combined_data = new_signal_data

    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)
    # todo: ave the name as it needs to be recognised for the params
    # Save combined data to CSV, ensuring no index duplication
    combined_data = combined_data.loc[~combined_data.index.duplicated(keep='first')]
    combined_data.to_csv(file_path)


def merge_df_with_checks(existing_data, new_data):
    """
    Merges two DataFrames with checks for duplicate column names and aligns them on the union of their date indices.

    If duplicate column names have different values, renames the column in new_data.
    If new_data has more dates, reindexes existing_data to align with new_data's date index.

    :param existing_data: The existing DataFrame with a DatetimeIndex.
    :param new_data: The new DataFrame to merge with existing_data, also with a DatetimeIndex.
    :return: Merged DataFrame with unique column names and aligned date indices.
    """

    # Ensure that the index is a DatetimeIndex
    if not isinstance(existing_data.index, pd.DatetimeIndex) or not isinstance(new_data.index, pd.DatetimeIndex):
        raise ValueError("Both DataFrames must have a DatetimeIndex.")

    # Get the union of both date indices
    combined_index = existing_data.index.union(new_data.index)

    # Reindex both DataFrames to the combined index
    existing_data = existing_data.reindex(combined_index)
    new_data = new_data.reindex(combined_index)

    for col in new_data.columns:
        if col in existing_data.columns:
            if not new_data[col].equals(existing_data[col]):
                new_col_name = f"{col}_1"
                new_data.rename(columns={col: new_col_name}, inplace=True)
                print(f"Column names are the same but values are different for '{col}'. Renamed to '{new_col_name}'.")
            else:
                # cols are equal, then just drop that col
                new_data.drop(columns=[col], inplace=True)

    # Concatenate the two DataFrames along the columns
    combined_data = pd.concat([existing_data, new_data], axis=1)

    return combined_data


def seriesMerge(dfs, pos="union", how="NA"):
    if not isinstance(dfs, list) or len(dfs) < 2:
        raise ValueError("Input must be a list of at least two DataFrames.")

    if pos == "union":
        result = pd.concat(dfs, axis=1)
        if how == "NA":
            result = result.fillna(method='ffill').fillna(method='bfill')
        return result
    else:
        raise NotImplementedError("Only 'union' position is implemented")


def back_adjust_futures(front_df, back_df, roll_day = 5):
    """
    Cleans and back adjusts the front futures prices using the differences in closing prices
    between the front and back futures on specified roll days before the expiration of the front contract.

    Parameters:
    - front_df: DataFrame of the front futures with columns ['Date', 'LastTradeDate', 'Open', 'High', 'Low', 'Close'], indexed by 'Date'.
    - back_df: DataFrame of the back futures with the same columns, indexed by 'Date'.
    - roll_day: Integer specifying the number of days before the final trade date to calculate the adjustment.

    Returns:
    - DataFrame with adjusted OHLC prices for the front futures, with NaN values handled.
    """

    # Ensure dates are in datetime format and set 'Date' as the index if not already
    for df in [front_df, back_df]:
        df['LastTradeDate'] = pd.to_datetime(df['LastTradeDate'], format='%Y%m%d')
         # Forward fill NaN values in OHLC data to ensure continuity
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')

    # Calculate days until expiration for the front contract
    front_df['DaysToExpiration'] = (front_df['LastTradeDate'] - front_df.index).dt.days

    # Initialize adjustment factors with zero
    adjustment_factors = {'Open': 0, 'High': 0, 'Low': 0, 'Close': 0}
    front_df['Is_Roll_Day'] = 0

    # Get unique last trade dates to handle different futures contracts
    unique_last_trade_dates = front_df['LastTradeDate'].unique()

    for last_trade_date in unique_last_trade_dates:
        # Calculate the target roll date as a business day
        target_roll_date = last_trade_date - BDay(roll_day)

        # Find the index closest to the target roll date
        # Calculate the absolute difference between the target_roll_date and each date in the index
        # Then find the index of the minimum difference
        nearest_date_index = np.abs(front_df.index - target_roll_date).argmin()
        # Mark the identified date as a roll date
        front_df.iloc[nearest_date_index, front_df.columns.get_loc('Is_Roll_Day')] = 1

    # Iterate over each day in the front futures dataframe
    for date, row in front_df.iterrows():
        if row['Is_Roll_Day'] == 1:
            # Calculate the difference in closing prices between the front and back contracts for the roll day
            if date in back_df.index:  # Ensure back contract has data for this date
                for price_type in ['Open', 'High', 'Low', 'Close']:
                    difference = front_df.at[date, price_type] - back_df.at[date, price_type]
                    adjustment_factors[price_type] += difference

        # Apply the cumulative adjustment factors to the historical data up to this date
        for price_type in ['Open', 'High', 'Low', 'Close']:
            adjusted_col_name = f'Adjusted_{price_type}'
            front_df.at[date, adjusted_col_name] = front_df.at[date, price_type] - adjustment_factors[price_type]
    # Calculate daily returns
    front_df['Daily_Returns'] = (front_df['Adjusted_Close'] - front_df['Adjusted_Close'].shift(1)) / front_df[
        'Close'].shift(1)

    # Calculate cumulative product of daily returns + 1 (to represent total return)
    front_df['Total_Return'] = (front_df['Daily_Returns'] + 1).cumprod()
    front_df['ATR'] = calculate_atr(front_df[['Close', 'Low', 'High']], window = 15)
    return front_df

def getFXData(asset):
    """
    Cleans and back adjusts the front futures prices using the differences in closing prices
    between the front and back futures on specified roll days before the expiration of the front contract.

    Parameters:
    - front_df: DataFrame of the front futures with columns ['Date', 'LastTradeDate', 'Open', 'High', 'Low', 'Close'], indexed by 'Date'.
    - back_df: DataFrame of the back futures with the same columns, indexed by 'Date'.
    - roll_day: Integer specifying the number of days before the final trade date to calculate the adjustment.

    Returns:
    - DataFrame with adjusted OHLC prices for the front futures, with NaN values handled.
    """
    outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
    fx_spot = pd.read_csv(outDir +"FX_spot_outright.csv", index_col=0, parse_dates=True)
    fxSpot = copy.deepcopy(fx_spot[asset].dropna())
    fx_fwd = pd.read_csv(outDir +"FX_forward_outright.csv", index_col=0, parse_dates=True) # a check to see if you are getting points or fwd rate?
    fxFwd = copy.deepcopy(fx_fwd.dropna()[asset+ "1MF"])
    fxFwd = align_time_series(fxFwd, fxSpot.index)
    daily_carry, implied_yield = FXCarry(asset, fxSpot,fxFwd)
    total_return, spot_return = FXTotalRet(fxSpot, daily_carry)
    OHLC = pd.concat([fxSpot,fxFwd,implied_yield,daily_carry,total_return, spot_return], axis = 1)
    OHLC.columns =        ['Close', 'CloseFwd', 'implied_yield', "daily_carry", "Total_Return", "Spot_Total_Return"]
    stdev = fx_spot[asset].diff().dropna().std()
    OHLC['Low'] = fxSpot - 2*stdev # Need to get OHLC data.
    OHLC['High'] =fxSpot + 2*stdev
    OHLC['ATR'] = calculate_atr(OHLC[['Close', 'Low', 'High']], window = 15)
    return OHLC

def FXTotalRet(spot, daily_carry = None, depo_rate = None, term_depo = None, base_depo = None):
    """
    Get FX total return based on crude carry application.
    :param spot:
    :param fwd:
    :return:
    """
    accuraldays = pd.DataFrame(spot.index.diff().days , index = spot.index)
    if daily_carry is not None:
        # Interest per trading day.
        carryAccured = modDF(accuraldays, daily_carry, "mult")
        lagcarryAccured = carryAccured.shift(1).fillna(0)
        spotRet = (spot - spot.shift(1)) / spot.shift(1)
        dailyRetIncCarry = modDF(spotRet, lagcarryAccured, "add")
    if depo_rate is not None:
        Y =  (spot.shift(1)/spot)
        X1 = 1 + (accuraldays)*term_depo/36500
        X2 = 1 + (accuraldays)*base_depo/36500
        X3 = X1/X2
        dailyRetIncCarry = 1 - (Y*X3)
    totRet = (dailyRetIncCarry + 1).cumprod()
    spotTR = (spotRet + 1).cumprod()
    return totRet, spotTR




def identify_and_mark_roll_dates(df, roll_day):
    """
    Identifies the nearest valid roll dates in the DataFrame and marks them.

    Parameters:
    - df: DataFrame with futures data, indexed by date.
    - roll_day: Integer specifying the number of business days before the final trade date to calculate the adjustment.

    Modifies the DataFrame in-place by adding a 'Is_Roll_Day' column marked with 1 on identified roll dates.
    """
    # Initialize the 'Is_Roll_Day' column with zeros
    df['Is_Roll_Day'] = 0

    # Get unique last trade dates to handle different futures contracts
    unique_last_trade_dates = df['LastTradeDate'].unique()

    for last_trade_date in unique_last_trade_dates:
        # Calculate the target roll date as a business day
        target_roll_date = pd.to_datetime(last_trade_date) - BDay(roll_day)

        # Find the nearest date in the DataFrame's index to this target roll date
        nearest_date = df.index.get_loc(target_roll_date, method='nearest')

        # Mark the identified date as a roll date
        df.iloc[nearest_date, df.columns.get_loc('Is_Roll_Day')] = 1