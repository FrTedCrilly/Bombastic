import pandas as pd
import numpy as np
from libs.trade_utils import calculate_atr
from libs.signal_utils import keepSig
from statsmodels.tsa.filters.hp_filter import hpfilter

class TrendSystem:
    def __init__(self, ohlc_data):
        """
        Initializes the TrendSystem with OHLC data.

        :param ohlc_data: A Pandas DataFrame containing OHLC data with a DateTimeIndex.
        """
        self.ohlc_data = ohlc_data

    def moving_average(self, window):
        """
        Calculates the simple moving average for a given window size.
        """
        return self.ohlc_data['Close'].rolling(window=window).mean()

    def exponential_moving_average(self, window):
        """
        Calculates the exponential moving average for a given window size.
        """
        return self.ohlc_data['Close'].ewm(span=window, adjust=False).mean()

    def moving_average_crossover(self, short_window, long_window):
        """
        Generates signals based on moving average crossovers.
        """
        signals = pd.DataFrame(index=self.ohlc_data.index)
        signals['signal'] = 0
        signals['short_mavg'] = self.moving_average(short_window)
        signals['long_mavg'] = self.moving_average(long_window)
        # Corrected to use .iloc for positional indexing
        signals['sigPure'].iloc[short_window:] = np.where(
            signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, np.where(
            signals['short_mavg'].iloc[short_window:] < signals['long_mavg'].iloc[short_window:], -1.0, 0) )
        return signals[['short_mavg', 'long_mavg', 'sigPure']]

    def ema_crossover(self, short_window, long_window):
        """
        Generates signals based on exponential moving average crossovers.
        """
        short_window = int(short_window)
        long_window = int(long_window)
        signals = pd.DataFrame(index=self.ohlc_data.index)
        signals['sigPure'] = 0
        signals['short_ema'] = self.exponential_moving_average(short_window)
        signals['long_ema'] = self.exponential_moving_average(long_window)
        # Corrected to use .iloc for positional indexing
        signals['sigPure'].iloc[short_window:] = np.where(
            signals['short_ema'].iloc[short_window:] > signals['long_ema'].iloc[short_window:], 1.0, np.where(
            signals['short_ema'].iloc[short_window:] < signals['long_ema'].iloc[short_window:], -1.0, 0) )
        return signals[['short_ema', 'long_ema', 'sigPure']]

    def breakout_signal(self, days):
        """
        Identifies breakout signals where the price moves above the high or below the low over a specified number of days.
        Depending on the breakout
        """
        signals = pd.DataFrame(index=self.ohlc_data.index)
        signals['high'] = self.ohlc_data['High'].rolling(window=days).max().shift(1)
        signals['low'] = self.ohlc_data['Low'].rolling(window=days).min().shift(1)
        signals['sigPure'] = 0.0
        # Here, direct comparison without need for positional slicing, so no change needed
        signals['sigPure'] = np.where(self.ohlc_data['Close'] > signals['high'], 1,
                                     np.where(self.ohlc_data['Close'] < signals['low'], -1, 0))
        signals['keep_long'] = np.where(self.ohlc_data['Close'] > signals['low'], 1, 0)
        signals['keep_short'] = np.where(self.ohlc_data['Close'] < signals['high'], -1, 0)
        signals['sigPure'] = keepSig(signals["sigPure"], signals['keep_long'], signals['keep_short'])
        return signals[['high', 'low', 'sigPure']]

    def calculate_adx(self, window=14):
        high = self.ohlc_data['High']
        low = self.ohlc_data['Low']
        close = self.ohlc_data['Close']

        # Use the ATR function to calculate ATR
        atr = calculate_atr(self.ohlc_data, window)

        # Calculate directional movements
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        # Smooth +DM and -DM
        smooth_plus_dm = plus_dm.rolling(window=window).sum()
        smooth_minus_dm = minus_dm.rolling(window=window).sum()

        # Calculate +DI and -DI
        plus_di = 100 * (smooth_plus_dm / atr)
        minus_di = 100 * (smooth_minus_dm / atr)

        # Calculate the DX and then the ADX
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=window).mean()

        return adx, plus_di, minus_di

    def calculate_rsi(self, window=14):
        delta = self.ohlc_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def apply_hp_filter(self, lamb=1600):
        cycle, trend = hpfilter(self.ohlc_data['Close'], lamb=lamb)
        return trend

    def get_HP_ADX_signals(self, window=14, lamb=1600):
        ## Initialize signals column
        signals = pd.DataFrame(index=self.ohlc_data.index)
        signals['trend_hp'] = self.apply_hp_filter(lamb=lamb)
        adx, plus_di, minus_di = self.calculate_adx(window)

        signals['sigPure'] = 0.0

        # Buy signal: +DI > -DI and current price > HP trend
        buy_signals = (plus_di > minus_di) & (self.ohlc_data['Close'] > signals['trend_hp'])
        signals.loc[buy_signals, 'sigPure'] = 1

        # Sell signal: spot < HP average and -DI > +DI
        sell_signals = (self.ohlc_data['Close'] < signals['trend_hp']) & (minus_di > plus_di)
        signals.loc[sell_signals, 'sigPure'] = -1

        return signals

    def bbands(self, window=20, num_std_dev=2, reversal=True):
        # need to use previous days bands, otherwise you might miss the breakthrough?
        middle_band = self.ohlc_data['Close'].rolling(window=window).mean().shift(1)
        std_dev = self.ohlc_data['Close'].rolling(window=window).std().shift(1)

        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        upper_band_exit = middle_band + ((1.75 * std_dev) * num_std_dev)
        lower_band_exit = middle_band - ((1.75 * std_dev) * num_std_dev)
        # Initialize signals column
        signals = pd.DataFrame(index=self.ohlc_data.index)
        signals['sigPure_base'] = 0.0

        if reversal:
            # Reversal strategy: Sell when above upper band, buy when below lower band
            signals['sigPure_base'] =  np.where(self.ohlc_data['Close'] > upper_band, -1,
                                     np.where(self.ohlc_data['Close'] < lower_band, 1, 0)) # Sell
            signals['keep_short'] = np.where(self.ohlc_data['Close'] < upper_band_exit, -1, 0)
            signals['keep_long']  = np.where(self.ohlc_data['Close'] > lower_band_exit, 1, 0)
            signals['sigPure'] = keepSig(signals['sigPure_base'], signals['keep_long'], signals['keep_short'])
        else:
            # Trend-following strategy: Buy when above upper band, sell when below lower band
            signals['sigPure_base'] =  np.where(self.ohlc_data['Close'] > upper_band, 1,
                                     np.where(self.ohlc_data['Close'] < lower_band, -1, 0)) # Sell
            signals['keep_long'] = np.where(self.ohlc_data['Close'] > (upper_band - 0.75*std_dev) , 1, 0)
            signals['keep_short']  = np.where(self.ohlc_data['Close'] < (upper_band + 0.75*std_dev), -1, 0)
            signals['sigPure'] = keepSig(signals['sigPure_base'], signals['keep_long'], signals['keep_short'])
        signals['upper_band'] = upper_band
        signals['middle_band'] = middle_band
        signals['lower_band'] = lower_band
        signals['upper_band_exit'] = upper_band_exit
        signals['lower_band_exit'] = lower_band_exit
        return signals


    def ad_line(self, volname = "Volume"):
        """
            Calculate the Accumulation/Distribution Line (A/D Line) for a given DataFrame.

            Parameters:
            - df: DataFrame with columns ['High', 'Low', 'Close', 'Volume']

            Returns:
            - Series containing the A/D Line.
            """
        if(volname in self.ohlc_data.columns):
            mfv = ((self.ohlc_data['Close'] - self.ohlc_data['Low']) - (
                        self.ohlc_data['High'] - self.ohlc_data['Close'])) / (
                              self.ohlc_data['High'] - self.ohlc_data['Low'])
            mfv = mfv.fillna(0)  # Handle division by zero
            mfv *= self.ohlc_data['Volume']  # Multiply by volume
            ad_line = mfv.cumsum()  # Cumulative sum to get the A/D Line

            return ad_line
        else:
            no_volData = self.ohlc_data['Close']*0
            no_volData.columns = ['NoVolDataProvided']
            return(no_volData)





