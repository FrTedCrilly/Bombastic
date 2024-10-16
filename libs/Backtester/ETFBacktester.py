
from libs.Backtester.Backtester import Backtest
import time
import os
import re
import numpy as np
import pandas as pd
class ETFBacktester(Backtest):
    def getPnLETF(asset, portdf, OHLC, sig, start_date, end_date, notional, instrument_type, assetCount, costfactor,
                  diagnostic):
        """

        :return:
        """
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
        trtrigger = pd.DataFrame({'Folio_alloc': 0}, index=sig.index)
        if 'Resize' in OHLC.columns:
            trtrigger = OHLC['Resize']

        # slippage = getSlippage(close) based on market vol, increase the cost to trade...
        sig = align_time_series(sig, close['Close'], carry_over=True)
        # get rid of nan closes
        sig = pd.DataFrame(sig[close['Close'].notna()])
        folio_alloc = pd.DataFrame({'Folio_alloc': assetWgt.loc[0]}, index=sig.index)
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
        div = pd.DataFrame(2 * atr * pv.loc[0])
        top = pd.DataFrame((notional * modDF(folio_alloc, fxrateQuote, "mult")))
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
        resize = sr[(dsig.iloc[:, 0] + dfolio_alloc.iloc[:, 0] + trtrigger.iloc[:, 0]) > 0]
        posSize = align_time_series(resize, sig, carry_over=True)
        posSize.columns = sig.columns
        posSizeUnround = sig * posSize
        contracts = np.floor(posSizeUnround)
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
        if comm.at[0, 'commission'] < 0.05:
            commdf = pd.DataFrame(
                {'commission': comm.at[0, 'commission'] * pv.loc[0] * modDF(actclose, fxrateQuote, "mult")},
                index=sig.index)
        else:
            commdf = pd.DataFrame({'commission': comm.at[0, 'commission']},
                                  index=sig.index)  # this only takes USD lots, you may want to do one for pct bps cost too.
        commissions = ((modDF(abs(trade), rolltrade, "add")) * (modDF(commdf, fxrateRisk, "div"))).dropna()
        ror = (modDF((close - close.shift(1)), lagcontracts, "mult") * pv.loc[0]).dropna()
        tror = modDF(ror, fxrateQuote, "div")
        tror = modDF(tror, commissions, "sub").fillna(0)
        cumtror = tror.cumsum()
        cumtror = align_time_series(cumtror, sig, carry_over=False)
        cumtror.columns = sig.columns
        if diagnostic:
            data = pd.concat([sig, cumtror, contracts, close['Close'], atr], axis=1)
            data.columns = [sig.columns[0], "TotalUSDRet", "Contracts", "Close", "Atr"]
        return cumtror

