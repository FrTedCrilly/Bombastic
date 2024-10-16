import copy

from libs.Backtester.Backtester import Backtest
from libs.io import getAssetBT
import time
import os
import copy
import re
from libs.utils_setup import get_last_business_day, create_unique_date_folder, create_log_folder
from libs.dates_io import align_time_series
from libs.utils_setup import modDF
from libs.signal_utils import sortino_ratio, sharpe_ratio,skewness, kurtosis
import pandas as pd
import numpy as np
import statsmodels.api as sm
class FXBT(Backtest):
    def run_FXBT(self, asset, OHLC, sig0, assetCount, costFactor,
                 diagnostic, anger):
        """

        :return:
        """
        close = pd.DataFrame(OHLC['Close'])
        atr = pd.DataFrame(OHLC['ATR'])
        comm = self.portdf['commission'].values[0] # pd.DataFrame(self.portdf['commission'])
        assetWgt = self.portdf['rel_asset_weight']
        #
        #actclose = pd.DataFrame(OHLC['Close'])
        #quotecurr = pd.DataFrame(OHLC['quotecurrency'])
        #riskcurr = pd.DataFrame(OHLC['riskcurr'])
        baseDollar = pd.DataFrame(OHLC['baseDollar'])
        termDollar = pd.DataFrame(OHLC['termDollar'])
        #fxrateQuote = pd.DataFrame(OHLC['quotecurrency'])
        #fxrateRisk = pd.DataFrame(OHLC['riskcurr'])
        #pv = 1 / fxrateQuote
        base = asset[:3]
        term = asset[3:]
        carry = pd.DataFrame(OHLC['daily_carry'])
        # Interest per trading day.
        accuraldays = pd.DataFrame(close.index.diff().days, index = close.index)
        carryAccured = modDF(accuraldays, carry, "mult")
        lagcarryAccured = carryAccured.shift(1).fillna(0)
        # for the carry, think about the forward returns and the points in the month wrt spot date and the value date etc (it can change with holidays).
        # then replicate the citi version for forward returns. You can need to build out a column for spot and value date, then get the correct 1 day carry.
        # think about spot return and a carry return?
        spotRet = (close - close.shift(1)) / close # this is due to the convex nature of fx, the further the currency moves
        lnSpotRet = np.log(close).diff()
        # the further you need to revalue it. ie, if you make 50 JPY with USd going from 100 to 150, that is only worth 50/150 USD, ie 33% not 50%. Hence why it makes sense to use log returns for FX.
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
        folio_alloc = pd.DataFrame({'Folio_alloc': assetWgt.iloc[0]}, index=sig0.index)
        div = pd.DataFrame(2 * atr)
        # notional is always in USD, notional needs to be in risk currency, which is the 1/Close.
        if term == "USD":
            FXnotional = self.notional*(1/close['Close']) # now expressed as risk curr
        elif base == "USD":
            FXnotional = self.notional
        else: # non USD cross
            # you need to have the baseUSD rate and the termUSD rate
            FXnotional =  self.notional*(1/baseDollar)

        top = pd.DataFrame(modDF(modDF(FXnotional,assetCount, "div") * anger, folio_alloc, "mult")) # now expressed as quote currency
        # if col names not the same you need to use iloc
        # TODO: build out the IRS and FX BT and then see what is the best way to structure them
        sr = modDF(modDF(top, close['Close'], "mult"), div, "div").dropna()
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
        # forward returns (with carry)
        carryimpact = modDF(lagcarryAccured, lagcontracts, "mult").fillna(0)
        # expiry
        trade = contracts - lagcontracts
        trade.loc[trade.first_valid_index()] = contracts.loc[trade.first_valid_index()]
        trade = trade.dropna()
        commdf = pd.DataFrame({'commission': comm },index=sig0.index) # comm is a pct already
     # this only takes USD lots, you may want to do one for pct bps cost too.
        commissions = ((modDF(commdf, abs(trade) , "mult"))).dropna() # expressed as quote ccy
        rorSpot = (modDF(spotRet, lagcontracts, "mult")).dropna() # pct based return
        #rorSpotNotional = (modDF(close,lagcontracts,"mult") - modDF(close.shift(1), lagcontracts,"mult") ) # in quote ccy and currency MTM
        #rorSpotLn = (modDF(lnSpotRet, lagcontracts, "mult")).dropna() # pct based return
        trorSpot = modDF(rorSpot, (commissions*costFactor), "sub").fillna(0)
        #trorSpotNotional = modDF(rorSpotNotional, (commissions * costFactor), "sub").fillna(0)
        #trorSpotLn = modDF(rorSpotLn, (commissions * costFactor), "sub").fillna(0)
        trorSpot = modDF(trorSpot, termDollar, "mult")
        cumtrorSpot = trorSpot.cumsum()
        cumtrorSpot = align_time_series(cumtrorSpot, sig0, carry_over=False)
        cumtrorSpot.columns = sig0.columns + "_SpotRet"
        rorFwd = modDF(rorSpot, carryimpact, "add")
        trorFwd = modDF(rorFwd, (commissions*costFactor), "sub").fillna(0)
        trorFwd = modDF(trorFwd, termDollar, "mult")
        cumtrorFwd = trorFwd.cumsum()
        cumtrorFwd = align_time_series(cumtrorFwd, sig0, carry_over=False)
        cumtrorFwd.columns = sig0.columns + "_TotalRetIncCarry"
        carry_ret = modDF(cumtrorFwd, cumtrorSpot, "sub")
        if diagnostic:
            data = pd.concat([sig0, cumtrorFwd, cumtrorSpot, carry_ret, contracts, close['Close'], atr], axis=1)
            data.columns = [sig0.columns[0], "TotalUSDRet", "SpotRet", "CarryRet", "Contracts", "ClSose", "Atr"]
        return cumtrorFwd

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
                    pnl = self.run_FXBT(asset, self.ohlc_data[asset], sig0 = agg_sig, assetCount = assetCount,
                                        costFactor = self.costFactor, diagnostic = self.diagnostic, anger = self.anger)
                    ret = pnl / self.notional
                    tot_pnl += ret
                    strat_dict[folio + '_' + asset] = [pnl,  self.plStats(ret, agg_sig)]
                else:
                    print("Weights not adding up.")
                folio_results[folio] = strat_dict
        folio_results[folio]['FolioRet'] = 100 + tot_pnl
        j = folio_results
        self.runStats(j, self.outp)
        end_time = time.time()
        runTime = end_time - start_time
        minutes = int(runTime // 60)
        seconds = runTime % 60
        # Print the runtime
        print('Finished in %d minutes and %.2f seconds' % (minutes, seconds))
        return folio_results
