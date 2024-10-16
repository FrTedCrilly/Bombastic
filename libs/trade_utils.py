import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_atr(ohlc, window=14):
    high = ohlc['High']
    low = ohlc['Low']
    close = ohlc['Close']

    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Calculate Average True Range (ATR) using a rolling window
    return tr.rolling(window=window).mean()


def apply_atr_trailing_stop(ohlc_data, signal_column, atr_lookback=14, atr_multiple=2):
    """
    Applies an ATR-based trailing stop strategy and creates a new signal column.

    Parameters:
    - ohlc_data: DataFrame containing OHLC data.
    - signal_column: The name of the column in ohlc_data containing the raw trading signals.
    - atr_lookback: The lookback period for ATR calculation.
    - atr_multiple: The multiple of ATR to set the stop level.

    Returns:
    - DataFrame with a new signal column named "<original_signal_column>_Trailing_<atr_multiple>ATRStop",
      indicating the modified signals with trailing stops applied.
    """
    atr = calculate_atr(ohlc_data, window=atr_lookback)
    new_signal_column = f"{signal_column}_Trailing_{atr_multiple}ATRStop"
    ohlc_data[new_signal_column] = ohlc_data[signal_column]  # Copy original signals
    ohlc_data['stop_level'] = np.nan  # Initialize stop level column

    for i in range(1, len(ohlc_data)):
        if ohlc_data[signal_column].iloc[i] != 0 and ohlc_data[signal_column].iloc[i-1] == 0:
            # Set stop level at trade initiation
            stop_level = ohlc_data['Close'].iloc[i] - atr_multiple * atr.iloc[i] if ohlc_data[signal_column].iloc[i] > 0 else ohlc_data['Close'].iloc[i] + atr_multiple * atr.iloc[i]
            ohlc_data['stop_level'].iloc[i] = stop_level
        else:
            # Carry forward the stop level for active trades
            ohlc_data['stop_level'].iloc[i] = ohlc_data['stop_level'].iloc[i-1]

        # Adjust the signal based on the stop level
        if (ohlc_data[signal_column].iloc[i] > 0 and ohlc_data['Close'].iloc[i] < ohlc_data['stop_level'].iloc[i]) or \
           (ohlc_data[signal_column].iloc[i] < 0 and ohlc_data['Close'].iloc[i] > ohlc_data['stop_level'].iloc[i]):
            ohlc_data[new_signal_column].iloc[i] = 0  # Stop out

    # Clean up by removing the temporary 'stop_level' column
    ohlc_data.drop(columns=['stop_level'], inplace=True)

    return ohlc_data

