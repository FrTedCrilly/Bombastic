import pandas as pd
import numpy as np
from libs.signal_utils import keepSig, apply_zscore , apply_quantile, sigmoid_expand, sigmoidSig, continousSig, apply_sand, sigSand
from libs.signal_utils import getFreqData, convertFreq
from scipy.stats import zscore
from libs.dates_io import align_time_series, tslagged
from libs.io import WriteSig
import os
import copy
import itertools

# how to add in US only or EA only results etc.
class MacroSignal:
    def __init__(self, varData, OHLC,tradeSign,start_date, end_date, regionForLag, assetName,  uselag = "None", lagK = 0, sig_name = None, freq = None, outDir = None):
        """
        Initializes the TrendSystem with OHLC data.

        :param ohlc_data: A Pandas DataFrame containing OHLC data with a DateTimeIndex.
        """
        self.macroData,  self.sig_name = self.getMacroData(varData, sig_name)
        self.ohlc_data = copy.deepcopy(pd.DataFrame(OHLC))
        self.tradeSign = tradeSign
        self.start_date = start_date
        self.end_date = end_date
        self.regionForLag = regionForLag
        self.assetName = assetName
        self.freq = convertFreq(getFreqData(self.macroData))
        self.lag = self.getLag(uselag, regionForLag, lagK)
        self.lagCor = self.lag
        self.outDir = outDir

        if self.macroData.shape[0] != self.ohlc_data.shape[0]:
            print("Macro data and price data mismatched, please ensure macro data corresponds to the asset data")
        elif  self.macroData.shape[0] == 1:
            self.useTS = True
        else:
            # this is for cross sectional ranking of the macro data and a price of different assets etc.
            print("Macro data and price data looks to be used for cross sectional")
            self.useTS = False


    def getMacroData(self, varData, sig_name):
        """
        if the data is already there thats fine, else source it from db
        :param macroData:
        :return:
        """

        if isinstance(varData, pd.Series) or isinstance(varData, pd.DataFrame):
            # If macroData is a time series, return it as is
            x =  pd.DataFrame(copy.deepcopy(varData))
        elif isinstance(varData, str):
            if os.path.isfile(varData):
                # If macroData is a file path, read the CSV file
                x = pd.read_csv(varData)
            else:
                # If macroData is a string but not a file path, make a dummy database call
                # This is a placeholder for the actual database call
                # You can replace this with your actual database query code
                return "dummy_database_call"
        else:
            raise ValueError("Unsupported data type for macroData")
        if sig_name is not None:
            if isinstance(sig_name, str):
                x.columns = [sig_name]
            else:
                print("Sig Name given but not a string")
        else:
            if len(x.columns) == 1:
                sig_name = str(x.columns[0])
            else:
                sig_name = x.columns

        return x, sig_name

    def getLag(self, uselag, regionForLag, lagK):
        """
        All implies 1 day for EA and US but 2 for asia.
        lagK allows you to add on a arbiraty lag if needed.
        if you want to lag the US by 1 and EU/APAC by 1 more, then use lagK = 1 and lag region EA/APAC
        :return:
        """
        recognisedLags = [None, "None","US","EA", "EA/APAC", "Europe", "All"]

        if uselag not in recognisedLags:
            print("Lag not recognised, default to 1, Please choose lag from ", recognisedLags)
            lag = 1

        if uselag == "None" or uselag == None:
            lag = 0
        if uselag == "All":
            if regionForLag == "US" or regionForLag == "EA" or regionForLag =="Europe":
               lag = 1 + lagK
            else:
                lag = 2 + lagK

        if uselag == "EA" or uselag == "EA/APAC":
            if regionForLag == "US":
                lag = 0 + lagK
            else:
                lag = 1 + lagK
        if uselag == "APAC":
            if regionForLag == "US" or regionForLag == "Europe":
                lag = 0 + lagK
            else:
                lag = 1 + lagK

        return lag


    def corrFilter(self, corChg, corEntry, corExit, corWindow, corSign, sig, expand = False, cheat=False):
        """
        This function calculates the correlation between the 'var' timeseries and
        the 'ohlc_data['close']' series based on a linear difference (chg). It computes
        the rolling correlation using a specified lookback window. If cheat is true,
        it backfills the data to the start of the timeseries. The align_time_series
        function aligns the close price with the var series. It returns the correlation timeseries.

        Once the correlation is calculated, it adjusts the signal (sig) based on the correlation:
        - If corSign is positive, set values in sig to 0 on days when the correlation is negative or below corEntry.
        - If corSign is negative, set values in sig to 0 on days when the correlation is positive or above corEntry.
        - If corSign is "unk" or "Unknown" or "Unk", multiply the signal by 1 when the correlation is above corEntry,
          and multiply the signal by -1 when the correlation is below -corEntry.
        - For corSign positive, remove signal only if correlation falls below corExit after rising above corEntry.
        - For corSign negative, remove signal only if correlation rises above -corExit after falling below corEntry.

        Parameters:
        chg (float): Linear difference scalar.
        corEntry (float): Entry correlation threshold.
        corExit (float): Exit correlation threshold.
        window (int): Lookback window for rolling correlation.
        corSign (str): Correlation signal direction ('+ve', '-ve', 'unk', 'Unknown', 'Unk').
        sig (pd.Series): Signal series to be adjusted.
        cheat (bool): If True, backfills the data to the start of the timeseries.

        Returns:
        pd.Series: Adjusted signal series.
        """

        # Align the time series
        # TODO: change from infinite carry over?
        aligned_close = align_time_series(self.ohlc_data['Close'], self.var.ffill())
        # Calculate the linear difference for the close prices
        close_diff = aligned_close.diff(corChg)
        var_diff = self.var.ffill().diff(corChg)

        # Calculate the rolling correlation
        if not expand:
            rolling_corr = close_diff.rolling(window=corWindow).corr(var_diff)
        else:
            rolling_corr = close_diff.expanding(window=corWindow).corr(var_diff)

        # Apply the cheat mode if required
        if cheat:
            rolling_corr = rolling_corr.reindex_like(self.var).fillna(method='bfill')
        else:
            rolling_corr = rolling_corr.reindex_like(self.var).dropna()

        corSig = self.getCorSig(rolling_corr, corEntry, corExit)

        self.rollingCorr = rolling_corr
        self.corSignal = corSig

        # Adjust the signal based on the correlation
        adjusted_sig = copy.deepcopy(sig)
        alignedCor = align_time_series(corSig, adjusted_sig, carry_over=False).fillna(0)
        if corSign == "+ve":
            adjusted_sig[alignedCor <= 0] = 0
        elif corSign == "-ve":
            adjusted_sig[alignedCor >= 0] = 0
        else:
            # assumes the var has a baseline positive signal set up
            adjusted_sig = adjusted_sig*alignedCor

        return tslagged(adjusted_sig, self.lagCor)

    def getCorSig(self, rolling_corr, corEntry, corExit):
        """
        Generates entry and exit signals based on rolling correlation thresholds.

        Parameters:
        rolling_corr (pd.Series): The rolling correlation data.
        corEntry (float): The entry threshold for generating signals.
        corExit (float): The exit threshold for generating signals.

        Returns:
        pd.Series: A series of signals where 1 indicates a positive entry, -1 indicates a negative entry,
                   and 0 indicates no signal or an exit.
        """
        corSig = pd.Series(0, index=rolling_corr.index)

        # pos only
        entry_triggered = (rolling_corr > corEntry) & (rolling_corr > 0)
        exit_triggered = (rolling_corr < corExit) & (rolling_corr > 0)
        # neg side
        entry_triggeredN = (rolling_corr < -corEntry) & (rolling_corr < 0)
        exit_triggeredN = (rolling_corr > -corExit) & (rolling_corr < 0)
        holding_signal = False
        holding_signalN = False
        for i in range(len(rolling_corr)):
            if entry_triggered.iloc[i].values[0] and not holding_signal:
                holding_signal = True
                corSig.iloc[i] = 1
            elif holding_signal and exit_triggered.iloc[i].values[0]:
                corSig.iloc[i] = 0
                holding_signal = False
            elif entry_triggeredN.iloc[i].values[0] and not holding_signalN:
                holding_signalN = True
                corSig.iloc[i] = -1
            elif holding_signalN and exit_triggeredN.iloc[i].values[0]:
                 corSig.iloc[i] = 0
                 holding_signalN = False
            else:
                if i == 0:
                    corSig.iloc[i] = 0
                else:
                    corSig.iloc[i] =  corSig.iloc[i-1]
        return(corSig)

    def dataTran(self, time_series = None, chg=None, short=None, long=None, MA=None, Zscore=False, rmMean=True, pctchg=False,
                 useLog=False, cheat=False, setVar = True):
        """
        Given a time series input, perform transformations based on parameters.

        :param time_series: The input time series as a pandas DataFrame or Series
        :param chg: Number for differencing
        :param short: Lookback period for MACD short-term
        :param long: Lookback period for MACD long-term
        :param MA: Lookback period for moving average
        :param Zscore: Boolean, if True apply z-score
        :param rmMean: Boolean, if False get z-score without subtracting the mean
        :param pctchg: Boolean, if True compute percentage change instead of linear difference
        :param useLog: Boolean, if True take logs first before transformations
        :param cheat: Boolean, if True backfill the first value
        :return: Transformed time series
        """
        if time_series == None:
            time_series = self.macroData

        if not isinstance(time_series, (pd.Series, pd.DataFrame)):
            raise ValueError("Input must be a pandas Series or DataFrame")

        if useLog:
            time_series = np.log(time_series)

        if chg is not None and short is not None and long is not None:
            raise ValueError("Either chg or (short and long for MACD) must be inputted separately")

        if MA is not None:
            if MA > len(time_series):
                raise ValueError("MA value cannot be greater than the length of the time series")
            time_series = time_series.rolling(window=MA).mean()

        if chg is not None:
            if chg > len(time_series):
                raise ValueError("chg value cannot be greater than the length of the time series")
            if pctchg:
                dts = time_series.pct_change(periods=chg)
            else:
                dts = time_series.diff(periods=chg)

        if short is not None and long is not None:
            if short > len(time_series) or long > len(time_series):
                raise ValueError("Lookback periods for MACD cannot be greater than the length of the time series")
            ema_short = time_series.ewm(span=short, adjust=False).mean()
            ema_long = time_series.ewm(span=long, adjust=False).mean()
            dts = ema_short - ema_long

        if Zscore:
            if rmMean:
                zts = zscore(dts, nan_policy='omit')
            else:
                mean = dts.mean()
                std = dts.std()
                zts = (dts - mean) / std if std != 0 else dts
        else:
            zts = dts

        if cheat:
            self.var = copy.deepcopy(zts.fillna(method='bfill'))
        else:
            self.var = copy.deepcopy(zts)

        del time_series
        del zts
        del dts

        if MA != None:
            self.var_name = f"MA{MA}_{self.sig_name}"
        else:
            self.var_name = self.sig_name

        if chg != None:
            self.var_name = f"C{chg}_{self.var_name}"
        elif short != None and long != None:
            self.var_name = f"MACD_{short}_{long}_{self.var_name}"

        return self.var.dropna()

    def getSignal(self,pEnter, pExit, doZ, doQ, Zwin, Qwin, Zexpand,
                  corChg, corEntry, corExit, corWindow, corSign,
                  Qexpand = True, cheat = False, subtractMean = True, longOnly = False, shortOnly = False, diagnostic=False, altVar = None):
        """
        DataTran must be run before we can run the getSignal.
        This will generate a signal from a transformed macrovariable.

        """
        # check if self.var exists and is a dataframe
        if isinstance(altVar, pd.DataFrame) or isinstance(altVar, pd.Series):
            self.var = copy.deepcopy(altVar)
        else:
            if not hasattr(self, 'var') and not isinstance(self.var, pd.DataFrame):
                raise AttributeError("self.var does not exist or is not a DataFrame. You need to run DataTran first.")

        # lag the data as needed.
        # TODO: figure out the alignment error in the data.
        sigData = tslagged(self.var, self.lag)

        if (pEnter == 0.5) & (pExit == 0.5):
            doContinous = True
        else:
            doContinous = False

        if doZ:
            # if Zexpand is false, we walk forward until we get to the rolling window size
            zs = apply_zscore(sigData.dropna(), Zwin, Zexpand, rmMean=subtractMean, cheat=cheat)
            # Define the thresholds
            upper_threshold = 50
            lower_threshold = -50
            # Identify extreme values
            extreme_mask = (zs > upper_threshold) | (zs < lower_threshold)
            # Replace extreme values with NaN
            zs[extreme_mask.values] = np.nan
            zs = zs.dropna()
        else:
            zs = sigData.dropna()

        # Log the dates and values
        #log_entries = zs[extreme_mask][['Date', 'Z_Score']].apply(
        #    lambda row: f"{row['Date']}: Z-score of {row:.2f} exceeds threshold\n", axis=1)

        if doQ:
            #:TODO the prob function and also the score need to either use NaN until we have enough data to make a correct value. otherwise early sample will be a bit stupid.
            prob = apply_quantile(zs, Qwin, Qexpand)
        else:
            # use sigmoid
            prob = sigmoid_expand(zs)

        # Next handle the signs.
        if self.tradeSign == "-ve":
            prob = 1 - prob

        if doContinous:
            baseSig = prob.apply(sigmoidSig)
            sig = apply_sand(baseSig,0.05)
            # apply used for a dataframe will not take each element of the df but only each column. to go over the elements you need to convert to a pd.series or use applymap
            condition = (sig >= 0.45) & (sig <= 0.55)
            jj = np.where(condition, 0.0, np.floor(sig * 20) / 20)
            sig = pd.DataFrame(jj, index = sig.index, columns=sig.columns)
        else:
            # binary non linear signal structure.
            sig = sigSand(prob, pEnter, pExit)

        # now check the correlation filter.
        if corChg > 0:
            sig = self.corrFilter(corChg = corChg, corEntry =  corEntry, corExit = corExit,corWindow = corWindow,corSign =  corSign, sig = sig)

        sigName = self.nameSig(pEnter, pExit, doZ, doQ, Zwin, Qwin, Zexpand, Qexpand,
                corChg, corEntry, corExit, corWindow, corSign)

        if len(sig.columns) == 1:
            sig.columns = [sigName]

        if longOnly:
            sig = sig[ sig > 0]
        if shortOnly:
            sig = sig[ sig < 0]

        if diagnostic:
            # Write log entries to a log file
            log_file = sigName
            #with open(log_file, 'w') as file:
            #    file.writelines(log_entries)
            self.sendDiagnostic(self, zs, prob, sig)

        self.FinalSigName = sigName
        self.FinalSig = sig

    def sendDiagnostic(self, zs, prob, sig):
        """
        Send it to the outDir folder, and save each input as a separate CSV, with the name of the file as z_score for zs,
        quantile for prob and self.sigName for sig.
        :param zs: DataFrame for z-scores
        :param prob: DataFrame for probabilities
        :param sig: DataFrame for signal
        :return: None
        """
        if self.outDir is None:
            print("Error: Output directory is not specified.")
            return

        # Check if the outDir exists, if not create it
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

        # Save each DataFrame to a CSV file
        zs_path = f"{self.outDir}/z_score.csv"
        prob_path = f"{self.outDir}/quantile.csv"
        sig_path = f"{self.outDir}/{self.sigName}.csv" if self.sigName else f"{self.outDir}/signal.csv"

        zs.to_csv(zs_path)
        prob.to_csv(prob_path)
        sig.to_csv(sig_path)
        if self.corSignal is not None:
            corPath = f"{self.outDir}/Correl_Signal.csv"
            self.corSignal.to_csv(corPath)
        if self.rollingCorr is not None:
            rollingPath = f"{self.outDir}/ActualCorrel.csv"
            self.rollingCorr.to_csv(rollingPath)

    def analyze_signal(self,signal = None):
        """
        Perform various analyses on FinalSig and ohlc_data['Close'].
        Returns a dictionary with analysis results.
        """
        results = {}

        # Ensure FinalSig is a Series and has the same index as ohlc_data
        if not signal:
            # Ensure FinalSig is a Series and has the same index as ohlc_data
            if not isinstance(self.FinalSig, (pd.Series, pd.DataFrame)):
                raise ValueError("FinalSig must be a pandas Series")
            if isinstance(self.FinalSig, pd.DataFrame):
                finalSig = self.FinalSig.iloc[:, 0]  # Use the first column if it's a DataFrame
            else:
                finalSig = self.FinalSig
        else:
            finalSig = signal

        finalSig = align_time_series(finalSig, self.ohlc_data['Close'])

        # Calculate the ratio of long signals to short signals
        long_signals = (finalSig > 0).sum()
        short_signals = (finalSig < 0).sum()
        ratio_long_short = long_signals / short_signals if short_signals != 0 else np.inf
        results['ratio_long_short'] = ratio_long_short

        # Calculate the sparsity of the signal
        non_zero_signals = (finalSig != 0).sum()
        total_days = len(finalSig)
        sparsity = non_zero_signals / total_days
        results['sparsity'] = sparsity

        # Calculate the turnover of the signal
        positive_turnover = (finalSig > 0).astype(int).diff().abs().sum()
        negative_turnover = (finalSig < 0).astype(int).diff().abs().sum()
        total_turnover = positive_turnover + negative_turnover
        results['turnover'] = total_turnover

        # Calculate the x = FinalSig - 1_month_rolling_average of FinalSig
        rolling_avg = finalSig.rolling(window=30, min_periods=1).mean()
        x = finalSig - rolling_avg
        avg_x = x.mean()
        results['avg_x'] = avg_x

        return results

    def AnalyseFwdRet(self, signal = None, totalRet = None):
        """
        Calculate the average 2-week ahead move in the close price based on FinalSig conditions.
        Returns a dictionary with the results.
        """
        results = {}
        if not isinstance(signal, (pd.Series, pd.DataFrame)):
            # Ensure FinalSig is a Series and has the same index as ohlc_data
            if not isinstance(self.FinalSig, (pd.Series, pd.DataFrame)):
                raise ValueError("FinalSig must be a pandas Series")
            if isinstance(self.FinalSig, pd.DataFrame):
                finalSig = self.FinalSig.iloc[:, 0]  # Use the first column if it's a DataFrame
            else:
                finalSig = self.FinalSig
        else:
            finalSig = signal

        if totalRet is not None:
            close_prices = totalRet
        else:
            # Calculate the 2-week ahead move in the close price
            close_prices = self.ohlc_data['Close']

        forward_returns = pd.DataFrame(close_prices.shift(-14) / close_prices - 1).dropna()
        alignedSig = align_time_series(finalSig.dropna(), forward_returns)
        # Average 2-week ahead move when FinalSig > 0
        avg_move_sig_positive = forward_returns[alignedSig.iloc[:,0] > 0].dropna().mean()
        results['avg_2_week_move_sig_positive'] = round(avg_move_sig_positive, 3)

        # Average 2-week ahead move when FinalSig < 0
        avg_move_sig_negative = forward_returns[alignedSig.iloc[:,0] < 0].mean()
        results['avg_2_week_move_sig_negative'] = round(avg_move_sig_negative, 3)

        # Average 2-week ahead move when FinalSig > 0.75
        avg_move_sig_high_positive = forward_returns[alignedSig.iloc[:,0] > 0.75].mean()
        results['avg_2_week_move_sig_high_positive'] = round(avg_move_sig_high_positive, 3)

        # Average 2-week ahead move when FinalSig < -0.75
        avg_move_sig_high_negative = forward_returns[alignedSig.iloc[:,0] < -0.75].mean()
        results['avg_2_week_move_sig_high_negative'] = round(avg_move_sig_high_negative, 3)

        return results
    def nameSig(self, pEnter, pExit, doZ, doQ, Zwin, Qwin, Zexpand, Qexpand,
                corChg, corEntry, corExit, corWindow, corSign, cheat=False, subtractMean=True, longOnly=False, shortOnly=False):

        signal_name = self.var_name

        if doZ:
            if Zexpand:
                signal_name += "_recZ"
            else:
                signal_name += "_Z"
            if Zwin:
                signal_name += str(Zwin)
        else:
            signal_name += "_noZ"

        if doQ:
            signal_name += f"Q{Qwin}_"
        if not Qexpand:
            signal_name += f"movQ{Qexpand}_"

        signal_name += f"P{int(pEnter * 100)}_P{int(pExit * 100)}"

        if corWindow > 0:
            signal_name += f"_Cor_win{corWindow}_C{corChg}_{int(corEntry*100)}_{int(corExit*100)}"

        return signal_name

    # Check re the send signal module, reuse the code?
    # then set up the parameter scan part.
    def macroParamScan(self, setCorSign = "+ve", runCorr = False):
        """
        This function will set yp various params for a macro indicator to be set up, changes and macd short, med and long term.
        create the signal and save it to the dir. also change the frequency depending on the freq.
        this one will require some deeper thought, its about doing it well so you can see the indicator benefits and also the get the correct performance stats asap.
        :return:
        """
        # set the param space.
        changes = [ 10, 30, 60, 90, 120, 250]
        changes = [max(1, int(round(i / self.freq, 0))) for i in changes]
        short_window = [5,   10,  20, 30,  120, 220]
        long_window =  [20,  60,  90, 120, 250, 600]
        short_window = [max(1, int(round(i / self.freq, 0))) for i in short_window]
        long_window = [max(1, int(round(i / self.freq, 0))) for i in long_window]
        pEnter = [0.05, 0.20,  0.50]
        pExit  = [0.10, 0.25,  0.50]
        doZ = [True, False]
        Zwin =  [max(12, int(round(i / self.freq, 0))) for i in [750, 1250]]
        Zexpand = [False]
        # Ensure the lengths of short_window and long_window are the same
        assert len(short_window) == len(long_window)
        assert len(pEnter) == len(pExit)

        p_combinations = list(zip(pEnter, pExit))
        if runCorr:
            corChg = [max(1, int(round(i / self.freq, 0))) for i in [10, 30, 60]]
            corWin = [max(12, int(round(i / self.freq, 0))) for i in [260, 750, 750]]
            corEntry = [0.10, 0.20]
            corExit = [0.05, 0.10]
            corParams = list(zip(corChg, corWin, corEntry, corExit ))
        else:
            corParams = [(0,0,0,0)]

        for chg in changes:
            for pE, z, zwin, zexpand, corP in itertools.product(p_combinations, doZ, Zwin, Zexpand,corParams):
                varTran = self.dataTran(chg = chg)
                self.getSignal(pEnter =  pE[0], pExit = pE[1], doZ = z, doQ = True, Zwin = zwin, Zexpand = zexpand,Qwin = zwin,
                                corChg = corP[0], corEntry = corP[1], corExit = corP[2], corWindow =  corP[3], corSign = setCorSign, altVar = varTran)
                WriteSig(self.outDir,  self.assetName , self.FinalSig)

        # When chg == 0, iterate over short_window and long_window pairs
        chg = 0
        for macd, pE, z, zwin, zexpand in itertools.product(zip(short_window, long_window), p_combinations, doZ, Zwin, Zexpand):
            varTran = self.dataTran(short = macd[0], long =  macd[1])
            self.getSignal(pEnter= pE[0], pExit= pE[1], doZ=z, doQ=True, Zwin=zwin, Zexpand = zexpand, Qwin=zwin,
                           corChg=corP[0], corEntry=corP[1], corExit=corP[2], corWindow=corP[3], corSign=setCorSign,
                           altVar=varTran)
            WriteSig(self.outDir, self.assetName, self.FinalSig)







