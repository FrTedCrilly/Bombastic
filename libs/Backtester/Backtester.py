import numpy as np
import pandas as pd
from libs.io import getAssetBT
import time
import os
import re
from libs.utils_setup import get_last_business_day, create_unique_date_folder, create_log_folder
from libs.dates_io import align_time_series
from libs.utils_setup import modDF
from libs.signal_utils import sortino_ratio, sharpe_ratio,skewness, Getkurtosis
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Backtest:
    def __init__(self, systemname, sig_folder, strats, start_date= "1992-01-01", end_date = None, config_df=None, scaling = 1, vol_target = 0.1, testN="StratTest", folderN = "TestOutput",outputFolder = "",
                 notional = 100000000, refpl = None, costFactor = 1, diagnostic = False, anger = 1/15,  MonthlyResize = False):
        """
        Initializes the BTrun backtester for futures contracts.
        Parameters:
        - asset_df: DataFrame containing OHLC data, futures expiry dates, and ATR for an asset.
        - signals: Series/DataFrame containing trading signals for the asset.
        - config_df: DataFrame with futures point sizing, commission costs.
        - notional: USD notional value to base position sizes on.
        - scaling: Scaling factor for adjusting position sizes.
        - vol_target: Target volatility for position sizing.
        """
        self.systemname = systemname
        self.sig_folder = sig_folder
        self.start_date = start_date
        self.end_date = end_date
        self.notional = notional
        self.strats = self.parse_signals(strats)
        self.config_df = config_df
        self.notional = notional
        self.scaling = scaling
        self.vol_target = vol_target
        self.testN = testN
        self.folderN = folderN
        self.costFactor = costFactor
        self.diagnostic = diagnostic
        self.outp = outputFolder
        self.refpl = refpl
        self.anger = anger
        self.resize = MonthlyResize

    def parse_signals(self, strats):
        """
        Parses the input 'strats' and returns results based on the input type:
        - For a list, returns a dictionary with a single key 'SingleStrat' and values as each strategy with a weight of 1.
        - For a file path, reads the CSV:
            - If one column, treat it as a list of strategies, each with a weight of 1.
            - If multiple columns, assumes alternating columns of strategy names and weights, returns a nested dictionary,
              processing rows until an empty cell is encountered in either strategy name or weight column.
        - For a dictionary, handles it directly.
        At the end, performs a final check to ensure that weights sum to 1 in each nested dictionary, where applicable.
        :param strats: list, file path string, or dictionary
        :return: dictionary or nested dictionary based on input
        """
        result_dict = {}

        if isinstance(strats, list):
            result_dict = {strat: {strat: 1} for strat in strats}
        elif isinstance(strats, str) and os.path.exists(strats):
            df = pd.read_csv(strats)
            if df.shape[1] == 1:
                result_dict = {'SingleStrat': {row[0]: 1 for index, row in df.dropna().iterrows()}}
            else:
                for i in range(0, df.shape[1], 2):
                    folio_name = df.columns[i]
                    weight_col = df.columns[i + 1]
                    result_dict[folio_name] = {}
                    for j in range(len(df)):
                        if pd.isna(df.at[j, folio_name]) or pd.isna(df.at[j, weight_col]):
                            break
                        result_dict[folio_name][df.at[j, folio_name]] = df.at[j, weight_col]
        elif isinstance(strats, dict):
            result_dict = strats

        # Final check for weight sums in each nested dict within result_dict
        for folio, strategies in result_dict.items():
            if isinstance(strategies, dict) and all(isinstance(weight, (int, float)) for weight in strategies.values()):
                total_weight = sum(strategies.values())
                if folio != "SingleStrat" and not round(total_weight, 5) == 1:
                    raise ValueError(f"Weights in {folio} do not sum to 1")

        return result_dict

    def runStats(self, pnl_res, outp):
        """

        :param pnl_res:
        :param outp:
        :return:
        """


    def getSW(self, asset = None):
        """
        :param systemname:
        :return:
        """
        self.portdf, self.ohlc_data = getAssetBT(self.systemname, asset)
        # get asset weights, assumed in pro-rata if not selecting full folio
        self.portdf['rel_asset_weight'] = self.portdf['weight'] / self.portdf['weight'].sum()
        if self.end_date == None:
            self.end_date = get_last_business_day()
        if asset == None:
            assets = self.portdf['Asset']
        if not isinstance(asset, list):
            assets = [asset]
        # Create the folder to save to.
        self.outp = create_unique_date_folder(self.sig_folder, self.folderN)
        # make log file folder to save extra bits
        self.logfile = create_log_folder(self.outp)
        # send the data deets from OHLC to a csv in the foler for data checking and repeatability.
        # Now you need to get the signals from whenever they are stored... you need a sig_folder
        return self.logfile


    def RunPnL(self, assets):
        """
        :param assets:
        :return:
        """
        start_time = time.time()  # Start time
        assetCount = self.getAssetCount(self.ohlc_data)
        instrument_type = self.portdf['class']
        folio_results = dict()
        # How to handle a list of sigs, or a dict of signals?
        # asset level first
        tot_pnl = 0
        for folio in self.strats.keys():
            agg_sig = 0
            agg_wgt = 0
            strat_dict = dict()
            strat_stats = dict()
            for asset in assets:
                for strat_name in self.strats[folio]:
                    weight = self.strats[folio][strat_name]
                    sig = self.getSigs(self.sig_folder, asset, strat_name)
                    agg_sig += sig[asset]*weight # add up signal contribution where needed.
                    agg_wgt += weight
                    # THE TUPLE IS SIG AND WEIGHT?
                    # HOW TO ACCESS THE WEIGHT?
                    # I dont think dict and tuple value will work, how else will you be able to look up the stuff?
                    # assuming this will go through the cols?
                if round(agg_wgt,5) == 1:
                    pnl = getPnLFutures(asset, self.portdf, self.ohlc_data[asset], agg_sig, self.start_date,
                                        self.end_date, self.notional,
                                        instrument_type, assetCount,
                                        self.costFactor, self.diagnostic)
                    ret = pnl/self.notional
                    tot_pnl += pnl
                    strat_stats[folio + '_' + asset] = self.plStats(ret, agg_sig)
                    strat_dict[folio + '_' + asset] = pnl
                else:
                    print("Weights not adding up.")
                folio_results[folio] = strat_dict
        j = folio_results
        return(j)
        #self.runStats(j, self.outp)
        end_time = time.time()

    def plStats(self,rets, agg_sig):
        """
        Given a timeseries of cumulative arithmetic returns, I want you to

        :return:
        """
        sortino = sortino_ratio(rets,diff_rets=True)
        sr = sharpe_ratio(rets, diff_rets=True)
        skew = skewness(rets, diff_rets=True)
        kurt = Getkurtosis(rets, diff_rets = True)
        valAdd, refAdd, retCorrelation = self.getU(rets, self.refpl)
        trade_analysis = self.calculate_trading_metrics(rets, agg_sig)
        signal_analysis = self.SigImpact(rets, agg_sig)
        return {"SR": sr, "Sortio": sortino, "skew": skew , "kurt": kurt, "valAdd": valAdd, "refAdd": refAdd , "retCorr": retCorrelation, "TAnalysis" : trade_analysis}
        #eventMoves = self.calculate_event_impact(rets)


    def PlotPdf(self, data_dict, filename):
            """
            Plots time series and performance statistics for each key in the nested data dictionary
            and saves the plots to a PDF file.
            Parameters:
            filename (str): The name of the output PDF file.
            """
            with PdfPages(filename) as pdf:
                for outer_key, inner_dict in data_dict.items():
                    for key, value in inner_dict.items():
                        # Create a new figure
                        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                        # Plot the time series
                        axs[0].plot(value[0], label='Cumulative Returns')
                        axs[0].set_title(f'Cumulative Returns for {outer_key} - {key}')
                        axs[0].legend()

                        # Prepare the stats table
                        stats = value[1]
                        stats_names = ['SR', 'Sortino', 'skew', 'kurt', 'valAdd', 'refAdd', 'retCorr']
                        stats_values = [
                            stats[stat].values[0] if isinstance(stats[stat], (pd.Series, pd.DataFrame)) else stats[stat]
                            for stat in stats_names]

                        # Create a table for the stats
                        cell_text = [[f'{val:.4f}' if isinstance(val, float) else str(val)] for val in stats_values]
                        table = axs[1].table(cellText=cell_text,
                                             rowLabels=stats_names,
                                             colLabels=['Value'],
                                             loc='center', cellLoc = 'center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(8)
                        table.scale(1, 1)
                        for key, cell in table.get_celld().items():
                            cell.set_edgecolor('black')
                            cell.set_linewidth(0.1)
                        axs[1].axis('off')
                        axs[1].set_title(f'Stats for {outer_key} - {key}')

                        # Add TAnalysis details if present
                        if 'TAnalysis' in stats:
                            tanalysis = stats['TAnalysis']
                            ta_names = list(tanalysis.keys())
                            ta_values = [
                                tanalysis[ta].values[0] if isinstance(tanalysis[ta], (pd.Series, pd.DataFrame)) else
                                tanalysis[ta] for ta in ta_names]
                            ta_text = [[f'{val:.4f}' if isinstance(val, float) else str(val)] for val in ta_values]
                            ta_table = axs[1].table(cellText=ta_text,
                                                    rowLabels=ta_names,
                                                    colLabels=['Value'],
                                                    loc='bottom',cellLoc = 'center')
                            ta_table.auto_set_font_size(False)
                            ta_table.set_fontsize(8)
                            ta_table.scale(0.8, 1)

                        # Save the figure to the PDF
                        pdf.savefig(fig)
                        plt.close(fig)

    def calculate_trading_metrics(self, cumulative_rets, signals):
        """
        Calculates the returns from long and short positions separately, the proportion of long to short signals,
        and the hit rate of trades based on a given signal series.

        Parameters:
            cumulative_rets (pd.Series): Time series of cumulative returns.
            signals (pd.Series): Time series of signals ranging from -1 (short) to 1 (long).

        Returns:
            dict: A dictionary containing long returns, short returns, proportion of long signals, and hit rate.
        """
        # Ensure index alignment and forward fill signals
        signals = signals.reindex(cumulative_rets.index).ffill()

        # Calculate daily returns from cumulative returns
        daily_rets = cumulative_rets.diff().fillna(0)
        long_sigs = signals > 0
        short_sigs   = signals < 0
        # Separate returns based on the signal
        long_rets = daily_rets[long_sigs.values]  # Returns when signal is positive
        short_rets = daily_rets[short_sigs.values]  # Negative returns when signal is negative

        # Calculate the proportion of long signals to total non-zero signals
        long_signals = np.sum(signals > 0)
        short_signals = np.sum(signals < 0)
        total_signals = long_signals + short_signals
        # not actually a hit rate fwiw
        proportion_long = (long_signals.iloc[0]) / total_signals.iloc[0] if total_signals.iloc[0] != 0 else np.nan

        # Calculate hit rate: profitable trades versus total trades
        profitable_long_trades = (long_rets > 0).sum()
        profitable_short_trades = (short_rets > 0).sum()
        total_trades = (signals != 0).sum()
        hit_rate = (profitable_long_trades.iloc[0] + profitable_short_trades.iloc[0]) / total_trades.iloc[0] if total_trades.iloc[0] != 0 else np.nan

        return {
            'Long Returns': long_rets.sum(),
            'Short Returns': short_rets.sum(),
            'Proportion Long to Short': proportion_long,
            'Hit Rate': hit_rate
        }
    def SigImpact(self, returns, signal):
        """
        Quantile the signal look returs from 0-0.25 0.25-0.5,0.5-0.75 and 0.75 to 1 (and check megative too).

        :return: stats of sig strength and pnl outcome, is it as an ideal condidate would expect?
        """
        """
        Calculate the average returns based on different conditions on the signal.

        Parameters:
        signal (pd.Series): A pandas Series representing the signal values.
        returns (pd.Series): A pandas Series representing the corresponding returns.

        Returns:
        dict: A dictionary with the average returns under different conditions.
        """
        # Convert Series to DataFrame if necessary
        if isinstance(signal, pd.Series):
            signal = signal.to_frame(name='signal')
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name='returns')

        # Ensure the DataFrames have the same index
        if not signal.index.equals(returns.index):
            raise ValueError("Signal and returns must have the same index.")

        # Dictionary to store the results
        averages = {
            'avg_rets_sigLess_-0.75': returns[signal['signal'] < 0.75]['returns'].mean(),
            'avg_rets_sigGrtr_0.75': returns[signal['signal'] > 0.75]['returns'].mean(),
            'avg_rets_sigGrtr_0': returns[signal['signal'] > 0]['returns'].mean(),
            'avg_rets_sigLess_0': returns[signal['signal'] < 0]['returns'].mean()
        }

        return averages


    def calcEventRets(self, event_dates, cumulative_rets):
        pre_event_rets = []
        post_event_rets = []

        for date in event_dates:
            # Calculate 5-day pre-event returns
            pre_start = date - pd.Timedelta(days=5)
            if pre_start in cumulative_rets.index:
                pre_event_ret = cumulative_rets[date] / cumulative_rets[pre_start] - 1
                pre_event_rets.append(pre_event_ret)

            # Calculate 5-day post-event returns
            post_end = date + pd.Timedelta(days=5)
            if post_end in cumulative_rets.index:
                post_event_ret = cumulative_rets[post_end] / cumulative_rets[date] - 1
                post_event_rets.append(post_event_ret)

        return np.array(pre_event_rets), np.array(post_event_rets)

    def calculate_event_impact(self,cumulative_rets, fed_csv='fed_events.csv', us_econ_csv='US_econ_events.csv'):
        """
        Calculates the returns 5 days before and after events listed in two CSV files and computes
        the max, min, and median returns for those events.

        Parameters:
            cumulative_rets (pd.Series): Time series of cumulative returns.
            fed_csv (str): Path to the CSV file containing Federal event dates.
            us_econ_csv (str): Path to the CSV file containing US economic event dates.

        Returns:
            dict: A dictionary containing statistics for pre and post event returns for both event types.
        """
        folderN = r"C:\Users\edgil\Documents\OUTDIR\EquitySystem\EquitySystem_2024-03-16"
        # Read and process event dates
        fed_events = pd.read_csv(folderN + '/' + fed_csv, parse_dates=[0], squeeze=True)
        us_econ_events = pd.read_csv(folderN + '/' + us_econ_csv, parse_dates=[0], squeeze=True)

        # Helper function to calculate returns around events
        # Calculate returns around Fed and US economic events
        fed_pre_rets, fed_post_rets = self.calcEventRets(fed_events)
        us_econ_pre_rets, us_econ_post_rets = self.calcEventRets(us_econ_events)

        # Prepare the output with statistics
        output = {
            'Fed Events': {
                'Pre-Event Max': np.max(fed_pre_rets),
                'Pre-Event Min': np.min(fed_pre_rets),
                'Pre-Event Median': np.median(fed_pre_rets),
                'Post-Event Max': np.max(fed_post_rets),
                'Post-Event Min': np.min(fed_post_rets),
                'Post-Event Median': np.median(fed_post_rets)
            },
            'US Econ Events': {
                'Pre-Event Max': np.max(us_econ_pre_rets),
                'Pre-Event Min': np.min(us_econ_pre_rets),
                'Pre-Event Median': np.median(us_econ_pre_rets),
                'Post-Event Max': np.max(us_econ_post_rets),
                'Post-Event Min': np.min(us_econ_post_rets),
                'Post-Event Median': np.median(us_econ_post_rets)
            }
        }

        return output

    def getU(self, rets, refPnL):
        """
        Performs operations on two time series, scales them by the full sample standard deviation,
        runs a regression where the dependent variable is a vector of ones, adjusts beta coefficients,
        and calculates the full sample correlation between the two normalized return series.

        Parameters:
            rets (pd.Series): Time series of returns.
            refPnL (pd.Series): Time series of reference profit and loss.

        Returns:
            dict: Dictionary containing the beta coefficients and the correlation between the series.
        """
        # Normalize the returns using the full sample standard deviation
        if refPnL is None:
            return (0, 0, 0)
        refPnL = align_time_series(refPnL, rets, carry_over=False)
        std_rets = rets.std()
        std_ref = refPnL.std()

        # Scale both series by the std of rets
        scaled_rets = rets * (std_ref / std_rets)  # scaling rets to refPnL std for comparison
        scaled_refPnL = refPnL

        # Dependent variable: vector of ones
        Y = np.ones(len(scaled_rets))

        # Independent variables: no constant added
        X = pd.DataFrame({'Scaled_Rets': scaled_rets, 'Scaled_RefPnL': scaled_refPnL})

        # Model: OLS regression without a constant
        model = sm.OLS(Y, X)
        results = model.fit()

        # Extract and adjust the beta coefficients
        betas = results.params
        if betas['Scaled_Rets'] < 0:
            betas['Scaled_Rets'] = 0
            betas['Scaled_RefPnL'] = 1

        if betas['Scaled_RefPnL'] < 0:
            betas['Scaled_RefPnL'] = 0
            # Adjust the rets beta to sum to 1 after setting refPnL beta to zero
            betas['Scaled_Rets'] = 1
        # Check if the sum of betas exceeds 1 and scale them if necessary
        beta_sum = betas.sum()
        if beta_sum > 1:
            # After scaling, check if refPnL beta is negative and set it to zero if it i
            betas = betas / beta_sum

        # Calculate full sample correlation between the two normalized return series
        correlation = scaled_rets.corr(scaled_refPnL)

        return betas['Scaled_Rets'], betas['Scaled_RefPnL'], correlation


    def getAssetCount(self, ohlc_data):
        """
        Create a DataFrame that counts how many assets have non-NaN 'Close' data starting from their
        first available non-NaN close price.

        Parameters:
        - ohlc_data (dict): Dictionary with asset names as keys and DataFrames as values. Each DataFrame
                            should have a 'Close' column and be indexed by 'Date'.

        Returns:
        - DataFrame with a single column indicating the count of assets available for trading per day.
        """
        # Initialize an empty DataFrame to store the availability of each asset
        asset_availability = pd.DataFrame()

        # Iterate over each asset's DataFrame in the dictionary
        for asset, data in ohlc_data.items():
            # Drop rows where 'Close' is NaN
            valid_data = data.dropna(subset=['Close'])

            # Find the first date where 'Close' is not NaN
            if not valid_data.empty:
                start_date = valid_data.index.min()
                # Create a series from the start date till the end of the asset's data marked as 1
                valid_dates = pd.Series(1, index=pd.date_range(start=start_date, end=data.index.max()))
                asset_availability[asset] = valid_dates

        # Fill NaNs with 0 for days where the asset is not available
        asset_availability.fillna(0, inplace=True)

        # Sum the availability across all assets to get the count of available assets per day
        total_assets_available = asset_availability.sum(axis=1).to_frame(name='Available Assets Count')

        return total_assets_available

    def getSigs(self, directory, asset_names, columns):
        """
        Load specified columns from CSV files for given assets in a specified directory.

        Parameters:
        - directory (str): Path to the directory containing the CSV files.
        - asset_names (str or list): Name(s) of the asset(s) whose data is to be loaded.
        - columns (dict of aggnames and strats): dict of columns to extract from the asset's CSV file.

        Returns:
        - Dictionary with asset names as keys and DataFrames as values containing the specified columns.
        - If a file or columns do not exist for an asset, appropriate messages are printed.
        """

        if isinstance(asset_names, str):
            asset_names = [asset_names]  # Convert to list if only one asset name is provided
        if isinstance(columns, str):
            columns = [columns]  # Convert to list if only one asset name is provided
        results = {}
        for asset_name in asset_names:
            file_path = os.path.join(directory, f"{asset_name}_signals.csv")
            if not os.path.exists(file_path):
                print(f"No file found for asset: {asset_name}")
                continue  # Skip to the next asset
            try:
                data = pd.read_csv(file_path)
                # col = get the names of the strats from the dict, the name is the 1st element of the tuple value for each key value pair.
                missing_columns = [col for col in columns if col not in data.columns]
                if missing_columns:
                    print(f"Missing columns {missing_columns} in file for asset: {asset_name}")
                    continue  # Skip to the next asset

                query_cols = ['Date'] + [col for col in columns if col != 'Date']  # Avoid duplicating 'Date'
                data = data[query_cols]
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
                results[asset_name] = data
            except Exception as e:
                print(f"Failed to read or process file for asset {asset_name}. Error: {e}")

        return results






def getPnLFutures(asset, portdf, OHLC, sig, start_date, end_date,notional, instrument_type,assetCount, costfactor, diagnostic):
    """
    :param asset:
    :param ohlc_data:
    :param sig:
    :param start_date:
    :return:
    """
    # TODO: need separte FX, Futures and IRS backtesters. ideally in the BT class, so they can inherit the neccesary data.
    # set up all the params
    close = pd.DataFrame(OHLC['Close'])
    atr = pd.DataFrame(OHLC['ATR'])
    comm = pd.DataFrame(portdf['commission'])
    assetWgt = portdf['rel_asset_weight']
    pv = portdf['pointvalue']
    actclose = pd.DataFrame(OHLC['Adjusted_Close'])
    quotecurr = pd.DataFrame(OHLC['quotecurrency'])
    riskcurr = pd.DataFrame(OHLC['riskcurr'])
    expiry = pd.DataFrame(OHLC['Is_Roll_Day'])
    fxrateQuote = pd.DataFrame(OHLC['quotecurrency'])
    fxrateRisk = pd.DataFrame(OHLC['riskcurr'])
    # if you want to re size at times other than position change or alloc_change
    trtrigger = pd.DataFrame({ 'Folio_alloc': 0}, index=sig.index)
    if 'Resize' in OHLC.columns:
        trtrigger = OHLC['Resize']

    # slippage = getSlippage(close) based on market vol, increase the cost to trade...
    sig = align_time_series(sig, close['Close'], carry_over=True)
    # get rid of nan closes
    sig = pd.DataFrame(sig[close['Close'].notna()])
    folio_alloc = pd.DataFrame({ 'Folio_alloc': assetWgt.loc[0] }, index=sig.index)
    # this gives us a position size which means we target
    div = pd.DataFrame(2*atr*pv.loc[0])
    top = pd.DataFrame((notional*modDF(folio_alloc,fxrateQuote,"mult")))
    # if col names not the same you need to use iloc
    # TODO: build out the IRS and FX BT and then see what is the best way to structure them
    sr = modDF(top, div, "div").dropna()
    siga = align_time_series(sig, sr)
    firstsig = siga.first_valid_index()
    dsig = pd.DataFrame(abs(siga.diff()))
    dsig.loc[firstsig] = 1
    dfolio_alloc = pd.DataFrame(abs(folio_alloc.diff()))
    dfolio_alloc.loc[dfolio_alloc.head(1).index] = 0
    # resize the position
    resize = sr[(dsig.iloc[:,0] + dfolio_alloc.iloc[:,0] + trtrigger.iloc[:,0]) > 0]
    posSize = align_time_series(resize, sig, carry_over=True)
    posSize.columns = sig.columns
    posSizeUnround = sig*posSize
    contracts =np.floor(posSizeUnround)
    # set anything before first trade to zero
    contracts[contracts.index < contracts.first_valid_index()] = 0
    #contracts.loc[firsttrade] = 1
    lagcontracts = contracts.shift(1)
    # expiry
    expiry.columns = contracts.columns
    expiry = align_time_series(expiry, contracts, carry_over=False)
    rolltrade = abs(contracts[expiry ==1]).fillna(0)
    trade = contracts - lagcontracts
    trade.loc[trade.first_valid_index()] = contracts.loc[trade.first_valid_index()]
    trade =trade.dropna()
    if comm.at[0, 'commission'] < 0.05:
        commdf = pd.DataFrame({'commission': comm.at[0, 'commission']*pv.loc[0]*modDF(actclose,fxrateQuote, "mult")}, index=sig.index)
    else:
        commdf = pd.DataFrame({'commission': comm.at[0, 'commission']},
                              index=sig.index)  # this only takes USD lots, you may want to do one for pct bps cost too.
    commissions = ((modDF(abs(trade),rolltrade, "add"))*(modDF(commdf,fxrateRisk, "div"))).dropna()
    ror = (modDF((close-close.shift(1)),lagcontracts,"mult")*pv.loc[0]).dropna()
    tror = modDF(ror,fxrateQuote,"div")
    tror = modDF(tror,commissions, "sub").fillna(0)
    cumtror = tror.cumsum()
    cumtror = align_time_series(cumtror, sig, carry_over=False)
    cumtror.columns = sig.columns
    if diagnostic:
        data = pd.concat([sig,cumtror, contracts,close['Close'], atr], axis = 1)
        data.columns = [sig.columns[0], "TotalUSDRet","Contracts","Close", "Atr"]
    return cumtror







def getPnLIRS(asset, portdf, OHLC, sig, start_date, end_date,notional, instrument_type,assetCount, costfactor, diagnostic):
    """

    :return:
    """
    close = pd.DataFrame(OHLC['Close']) # close here is a daily pnl series total return for the IRS.(based on valuing the swap each day using the yield curve).
    atr = pd.DataFrame(OHLC['ATR'])
    comm = pd.DataFrame(portdf['commission'])
    assetWgt = portdf['rel_asset_weight']
    pv = portdf['pointvalue']
    actclose = pd.DataFrame(OHLC['Adjusted_Close'])
    bid = pd.DataFrame(OHLC['ActBid'])
    ask = pd.DataFrame(OHLC['ActAsk'])
    arets = close.diff().fillna()
    quotecurr = pd.DataFrame(OHLC['quotecurrency'])
    riskcurr = pd.DataFrame(OHLC['riskcurr'])
    dv01 = pd.DataFrame(OHLC['DV01'])
    minTicket = pd.DataFrame(portdf['minTicket'])
    expiry = pd.DataFrame(OHLC['Is_Roll_Day'])
    fxrateQuote = pd.DataFrame(OHLC['quotecurrency'])
    fxrateRisk = pd.DataFrame(OHLC['riskcurr'])
    rollSlippage  = pd.DataFrame(portdf['RolLSlip']) # if na then 1
    # if you want to re size at times other than position change or alloc_change
    trtrigger = pd.DataFrame({ 'Folio_alloc': 0}, index=sig.index)
    if 'Resize' in OHLC.columns:
        trtrigger = OHLC['Resize']

    # slippage = getSlippage(close) based on market vol, increase the cost to trade...
    sig = align_time_series(sig, close['Close'], carry_over=True)
    # get rid of nan closes
    sig = pd.DataFrame(sig[close['Close'].notna()])
    folio_alloc = pd.DataFrame({ 'Folio_alloc': assetWgt.loc[0] }, index=sig.index)
    # EFT backtest will need to find the rets of the ETF
    # the spread cost (which is dependent on the sizing range).
    # the usual commossion cost.
    # the total expense ratio, broken into the daily rate for the year?
    # financing rate, day count adjusted etf financing rate
    # dividends use gross dividends (which is the dividend amount divided by the price of the etf) by quantity by price of etf
    # total comms will be equal to spread, commissions, ter , borrow and financing costs.
    # etf quantity is given as (risk_curr*sig)/(200*sizing_range)
    # spread cost is 0.00045*price*quantity or etf held (based on signal strength
    # the ter is just a fixed cost default is 0.0014
    # default borrow cost is 0.005, each individual one seems unique. only pay borrow on the shorts.
    # the day count is divided by 365.
    # this gives us a position size which means we target
    div = pd.DataFrame(2*atr*pv.loc[0])
    top = pd.DataFrame((notional*modDF(folio_alloc,fxrateQuote,"mult")))
    # if col names not the same you need to use iloc
    # TODO: build out the IRS and FX BT and then see what is the best way to structure them
    sr = modDF(top, div, "div").dropna()
    siga = align_time_series(sig, sr)
    firstsig = siga.first_valid_index()
    dsig = pd.DataFrame(abs(siga.diff()))
    dsig.loc[firstsig] = 1
    dfolio_alloc = pd.DataFrame(abs(folio_alloc.diff()))
    dfolio_alloc.loc[dfolio_alloc.head(1).index] = 0
    # resize the position
    resize = sr[(dsig.iloc[:,0] + dfolio_alloc.iloc[:,0] + trtrigger.iloc[:,0]) > 0]
    posSize = align_time_series(resize, sig, carry_over=True)
    posSize.columns = sig.columns
    posSizeUnround = sig*posSize
    contracts =np.floor(posSizeUnround)
    # set anything before first trade to zero
    contracts[contracts.index < contracts.first_valid_index()] = 0
    #contracts.loc[firsttrade] = 1
    lagcontracts = contracts.shift(1)
    # you will need to have a min size trade so that we don't trade small amounts.
    # expiry
    expiry.columns = contracts.columns
    expiry = align_time_series(expiry, contracts, carry_over=False)
    rolltrade = abs(contracts[expiry ==1]).fillna(0)
    trade = contracts - lagcontracts
    trade.loc[trade.first_valid_index()] = contracts.loc[trade.first_valid_index()]
    trade =trade.dropna()
    if comm.at[0, 'commission'] < 0.05:
        commdf = pd.DataFrame({'commission': comm.at[0, 'commission']*pv.loc[0]*modDF(actclose,fxrateQuote, "mult")}, index=sig.index)
    else:
        commdf = pd.DataFrame({'commission': comm.at[0, 'commission']},
                              index=sig.index)  # this only takes USD lots, you may want to do one for pct bps cost too.
    commissions = comm*dv01*100 # you need a dynamic commsission here with bid ask smoothing etc. expressed in spread points.
    minClearing = minTicket*(abs(trade)>0+ (abs(trade)<=0)) # seems to have been set to zero
    ror = modDF((arets,lagcontracts,"mult")).dropna() # arets includes the dv01 variable, while lagpos is inverse wrt dv01
    tror = modDF(ror,fxrateQuote,"div")
    tror = modDF(tror,commissions, "sub").fillna(0)
    cumtror = tror.cumsum()
    cumtror = align_time_series(cumtror, sig, carry_over=False)
    cumtror.columns = sig.columns
    if diagnostic:
        data = pd.concat([sig,cumtror, contracts,close['Close'], atr], axis = 1)
        data.columns = [sig.columns[0], "TotalUSDRet","Contracts","Close", "Atr"]
    return cumtror