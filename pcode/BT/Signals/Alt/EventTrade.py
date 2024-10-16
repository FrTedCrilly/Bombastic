import pandas as pd
import numpy as np
from scipy.special import expit  # For sigmoid function
from libs.signal_utils import apply_zscore , apply_quantile
from libs.dates_io import align_time_series

class EventTrade:
    def __init__(self, eventName,  events, price_series, lookback_period=250, tradeSign = "+ve", probEnter=0.1, daysToHold=5,
                 nperiods = 1, log_returns = False,
                 doZ = True, expand = True, sigmoid=False):
        """
        Initialize the class with event dates, price series, and parameters.

        :param events: List of event dates (e.g., ['2022-02-02', ...])
        :param price_series: DataFrame or Series with prices indexed by date.
        :param lookback_period: Number of periods for quantile lookback.
        :param tradeSign: +1 for trading in direction of move, -1 for opposite.
        :param probEnter: Percentile threshold to enter trades (e.g., 0.1 for 10%).
        :param daysToHold: Number of periods to hold the trade signal.
        :param sigmoid: If True, use a sigmoid function for signal transitions.
        """
        self.varName = eventName
        self.events = pd.to_datetime(events)
        self.price_series = price_series
        self.lookback_period = lookback_period
        self.tradeSign = tradeSign
        self.probEnter = probEnter
        self.daysToHold = daysToHold
        self.doZ = doZ
        self.expand = expand
        self.sigmoid = sigmoid
        self.eventRets = self.getEventRet(nperiods, log_returns)


    def getEventRet(self,  n: int = 1, log_returns: bool = False) -> pd.Series:
        """
        Calculate the n-period return of a given price series.

        Parameters:
            price_series (pd.Series): A series of prices.
            n (int): Number of periods for the return calculation. Default is 1 for 1-period return.
            log_returns (bool): If True, calculate log returns. Otherwise, calculate percentage returns.

        Returns:
            pd.Series: A series containing the n-period returns, with NA values filled with 0.
        """
        if log_returns:
            # Use for FX
            # Logarithmic returns: ln(current price / price n periods ago)
            return_series = np.log(self.price_series / self.price_series.shift(n))
        else:
            # Regular percentage returns
            return_series = self.price_series.pct_change(periods=n)

        # Fill any NaN values with 0
        self.eventRets = return_series[self.events].dropna()

        return return_series


    def generate_signals(self):
        """
        Generate trading signals based on the event returns and quantiles.
        """

        if len(self.eventRets) < self.lookback_period:
            return None  # Not enough data for quantile calculation

        if self.doZ:
            # if Zexpand is false, we walk forward until we get to the rolling window size,
            zs = apply_zscore(self.eventRets, self.lookback_period, self.expand, rmMean=True, cheat=False)
            # Define the thresholds
            upper_threshold = 50
            lower_threshold = -50
            # Identify extreme values
            extreme_mask = (zs > upper_threshold) | (zs < lower_threshold)
            # Replace extreme values with NaN
            zs[extreme_mask.values] = np.nan
            zs = zs.dropna()
        else:
            zs = self.eventRets.dropna()

        prob = apply_quantile(zs, self.lookback_period, self.expand)

        # Next handle the signs.
        if self.tradeSign == "-ve":
            prob = 1 - prob
        # now expand the event series to match all trading days.

        prob_full_sample = align_time_series(prob, self.price_series)
        signals = self.get_sig(prob_full_sample.fillna(0), self.probEnter, self.daysToHold)

        return signals

    def sigmoid_decay(self, length):
        """
        Create a sigmoid decay pattern for signal values.
        """
        x = np.linspace(-6, 6, length)
        return expit(x)  # Sigmoid function

    def get_sig(self, data: pd.DataFrame, probEnter: float, hold_days: int) -> pd.DataFrame:
        """
        Generates a trading signal based on the probability threshold and holds the signal for X days.
        If a new significant event (signal) occurs, it overrides the current signal and restarts the holding period.

        Parameters:
            data (pd.DataFrame): DataFrame containing values between 0 and 1.
            probEnter (float): Threshold for signal generation.
            hold_days (int): Number of days to hold the signal after it is generated.

        Returns:
            pd.DataFrame: DataFrame with the generated signals (1, -1, or 0).
        """
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)  # Initialize a DataFrame of 0s

        for col in data.columns:
            current_signal = 0
            hold_counter = 0

            for i in range(len(data)):
                value = data.loc[i, col]  # Use .loc[] to access the value safely

                # Generate a new signal
                new_signal = 0
                if value > (1 - probEnter):
                    new_signal = 1  # Buy signal
                elif value < probEnter:
                    new_signal = -1  # Sell signal

                # Check if there is a new event (significant signal) during the hold period
                if new_signal != 0:
                    current_signal = new_signal  # Override the current signal
                    hold_counter = hold_days  # Restart the hold period

                # Continue holding the signal if the hold counter is still active
                if hold_counter > 0:
                    signals.loc[i, col] = current_signal  # Use .loc[] to safely assign the signal
                    hold_counter -= 1
                else:
                    signals.loc[i, col] = 0  # No signal if hold period has expired
        sigName = self.nameSig()

        if len(signals.columns) == 1:
            signals.columns = [sigName]

        return signals

    def nameSig(self):

        signal_name = self.var_name

        if self.doZdoZ:
            if self.expand:
                signal_name += "_recZ"
            else:
                signal_name += "_Z"
            if self.lookback_period:
                signal_name += str(self.lookback_period)
        else:
            signal_name += "_noZ"

        # we always quantile
        signal_name += f"Q{self.lookback_period}_"
        if not self.expand:
            signal_name += f"movQ{self.lookback_period}_"

        signal_name += f"P{int(self.probEnter * 100)}_PerHold{int(self.daysToHold)}"

        return signal_name



#############################################################
#Test


import pandas as pd
import numpy as np

# Simulate a price series with drift and volatility
np.random.seed(42)

# Parameters for simulation
n_periods = 100  # Number of periods
drift = 0.0002  # Drift component (e.g., 0.02% per period)
volatility = 0.01  # Volatility component (e.g., 1% volatility)
initial_price = 100  # Initial price

# Generate random returns with drift and volatility
returns = drift + np.random.normal(0, volatility, n_periods)

# Simulate the price series from returns
price_series = pd.Series(initial_price * (1 + returns).cumprod(),
                         index=pd.date_range(start='2022-01-01', periods=n_periods))

# Simulate event dates occurring randomly within the time series
event_dates = pd.date_range(start='2022-01-15', periods=5, freq='20D')  # 5 events, one every 20 days

# Print out the simulated price series and event dates
price_series.head(), event_dates


# Number of jumps
n_jumps = 5
jump_size = 0.03  # 3% jump

# Randomly select indices in the price series for the jumps
random_jump_indices = np.random.choice(price_series.index, size=n_jumps, replace=False)

# Apply a 3% jump at the selected indices
price_series[random_jump_indices] *= (1 + jump_size)

# Display the price series with the jumps applied
price_series[random_jump_indices], random_jump_indices

trade_system = EventTrade(
    eventName="SimulatedEvent",
    events=event_dates,
    price_series=price_series, lookback_period=3,
    probEnter=0.2,
    daysToHold=5
)