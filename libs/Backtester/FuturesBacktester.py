from libs.Backtester.Backtester import Backtest
from libs.io import getAssetBT
import time
import os
import re
import copy
from libs.utils_setup import get_last_business_day, create_unique_date_folder, create_log_folder
from libs.dates_io import align_time_series
from libs.utils_setup import modDF
from libs.signal_utils import sortino_ratio, sharpe_ratio,skewness, kurtosis
import pandas as pd
import numpy as np
import statsmodels.api as sm


class FuturesBT(Backtest):
    def run_FuturesBT(self, OHLC, sig0, assetCount, costFactor, diagnostic, anger):
        # Align signals and close prices
        close = pd.DataFrame(OHLC['Close'])
        atr = pd.DataFrame(OHLC['ATR'])
        comm = self.portdf['commission'].values[0] #comm = pd.DataFrame(self.portdf['commission'])
        assetWgt = self.portdf['rel_asset_weight']
        actclose = pd.DataFrame(OHLC['Close'])
        quotecurr = pd.DataFrame(OHLC['quotecurrency'])
        riskcurr = pd.DataFrame(OHLC['riskcurr'])
        pv = self.portdf['pointvalue']
        fxrateQuote = pd.DataFrame(OHLC['quotecurrency'])
        fxrateRisk = pd.DataFrame(OHLC['riskcurr'])
        expiry = pd.DataFrame(OHLC['Is_Roll_Day'])
        # if you want to re size at times other than position change or alloc_change
        trtrigger = pd.DataFrame({'Folio_alloc': 0}, index=sig0.index)
        if 'Resize' in OHLC.columns:
            trtrigger = OHLC['Resize']
        elif self.resize:
            trtrigger = copy.deepcopy(close*0)
            trtrigger[trtrigger.index.month.diff() == 1] = 1

        # slippage = getSlippage(close) based on market vol, increase the cost to trade...
        sig0 = align_time_series(sig0, close['Close'], carry_over=True)
        # get rid of nan closes
        sig0 = pd.DataFrame(sig0[close['Close'].notna()])
        folio_alloc = pd.DataFrame({'Folio_alloc': assetWgt.loc[0]}, index=sig0.index)
        # this gives us a position size which means we target
        div = pd.DataFrame(2 * atr)
        top = pd.DataFrame(( modDF((self.notional/assetCount*anger), modDF(folio_alloc, fxrateQuote, "mult"), "mult")))
        # if col names not the same you need to use iloc
        # TODO: build out the IRS and FX BT and then see what is the best way to structure them
        sr = modDF(top, div, "div").dropna()
        siga = align_time_series(sig0, sr)
        firstsig = siga.first_valid_index()
        dsig = pd.DataFrame(abs(siga.diff()))
        dsig.loc[firstsig] = 1
        dfolio_alloc = pd.DataFrame(abs(folio_alloc.diff()))
        dfolio_alloc.loc[dfolio_alloc.head(1).index] = 0
        # resize the position
        resize = sr[(dsig.iloc[:, 0] + dfolio_alloc.iloc[:, 0] + trtrigger.iloc[:, 0]) > 0]
        posSize = align_time_series(resize, sig0, carry_over=True)
        posSize.columns = sig0.columns
        posSizeUnround = sig0 * posSize
        contracts = posSizeUnround
        # set anything before first trade to zero
        contracts[contracts.index < contracts.first_valid_index()] = 0
        # contracts.loc[firsttrade] = 1
        lagcontracts = contracts.shift(1)
        # expiry
        expiry.columns = contracts.columns
        expiry = align_time_series(expiry, contracts, carry_over=False)
        rolltrade = abs(contracts[expiry == 1]).fillna(0)
        trade = contracts - lagcontracts
        trade.loc[trade.first_valid_index()] = contracts.loc[trade.first_valid_index()]
        trade = trade.dropna()
        if comm< 0.05:
            commdf = pd.DataFrame(
                {'commission': comm * pv.loc[0] * modDF(actclose, fxrateQuote, "mult")},
                index=sig0.index)
        else:
            commdf = pd.DataFrame({'commission': comm},
                                  index=sig0.index)  # this only takes USD lots, you may want to do one for pct bps cost too.
        commissions = ((modDF(abs(trade), rolltrade, "add")) * (modDF(commdf, fxrateRisk, "div"))).dropna()
        ror = (modDF((close - close.shift(1)), lagcontracts, "mult") * pv.loc[0]).dropna()
        tror = modDF(ror, fxrateQuote, "div")
        tror = modDF(tror, (commissions*costFactor), "sub").fillna(0)
        cumtror = tror.cumsum()
        cumtror = align_time_series(cumtror, sig0, carry_over=False)
        cumtror.columns = sig0.columns
        if diagnostic:
            data = pd.concat([sig0, cumtror, contracts, close['Close'], atr], axis=1)
            data.columns = [sig0.columns[0], "TotalUSDRet", "Contracts", "Close", "Atr"]
        return cumtror

    def RunPnL(self, assets):
        """

        :param assets:
        :return:
        """
        start_time = time.time()  # Start time
        assetCount = self.getAssetCount(self.ohlc_data)
        folio_results = dict()
        # How to handle a list of sigs, or a dict of signals?
        # asset level first

        for folio in self.strats.keys():
            tot_pnl = 0
            agg_sig = 0
            agg_wgt = 0
            strat_dict = dict()
            strat_stats = dict()
            for asset in assets:
                for strat_name in self.strats[folio]:
                    weight = self.strats[folio][strat_name]
                    sig = self.getSigs(self.sig_folder, asset, strat_name)
                    agg_sig += sig[asset] * weight  # add up signal contribution where needed.
                    agg_wgt += weight
                    # THE TUPLE IS SIG AND WEIGHT?
                    # HOW TO ACCESS THE WEIGHT?
                    # I dont think dict and tuple value will work, how else will you be able to look up the stuff?
                    # assuming this will go through the cols?
                if round(agg_wgt, 5) == 1:
                    pnl = self.run_FuturesBT(OHLC = self.ohlc_data[asset], sig0 = agg_sig, assetCount = assetCount,
                                        costFactor = self.costFactor, diagnostic = self.diagnostic,anger = self.anger)
                    ret = (pnl / self.notional)
                    tot_pnl += ret
                    #strat_stats[folio + '_' + asset]
                    strat_dict[folio + '_' + asset] = [pnl,  self.plStats(ret, agg_sig)]
                else:
                    print("Weights not adding up.")
                folio_results[folio] = strat_dict
            folio_results[folio]['FolioRet'] = 100 + tot_pnl

        end_time = time.time()
        runTime = end_time - start_time
        minutes = int(runTime // 60)
        seconds = runTime % 60
        # Print the runtime
        print('Finished in %d minutes and %.2f seconds' % (minutes, seconds))
        return folio_results