import pandas as pd
import numpy as np
import os
from libs.utils_setup import modDF
from libs.dates_io import align_time_series, find_index, rangeDates
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from datetime import datetime, timedelta
from libs.io import seriesMerge
import copy
import re


def getIRS_def(asset, portdf):
    """

    :param asset: the asset in question
    :param port: the csv file
    :return:
    """
    try:
        # Make it iterate over a list of assets and bring back only those.
        # weights of subset will be proportional.
        assetdf = portdf.explode('Asset').query('Asset in @asset').drop_duplicates()
        if assetdf.empty:
            print("Asset doesn't exist in portfolio:", asset)
    except Exception as e:
        # This block executes if an exception occurs in the try block
        print(f"Error with asset '{asset}' in system: {e}")
    assetdf['Currency'] = assetdf['quotecurr']
    return assetdf

def dal_get_ts(start_date, end_date, query = None, asset= None):

    if query == "PLN":
        df = pd.read_csv(r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\PLN_yieldCurve.csv",
                         parse_dates=['Date'])
    else:
        df = pd.read_csv(r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\US_yieldCurve.csv",
                         parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return(df.loc[start_date:end_date])

def get_IRS_OHLC(start, finish, IRS, from_cache=True, use_MACLS=True, AF=None, put_cache=False, swap_mode=['spot'], discount_continuous=False, close_point=10):
    # Placeholder functions for dal.get_ts, get_timeCal, align_time, Rename, etc.
    immdates = getIMMCal(start, finish)

    if use_MACLS:
        CP = str(close_point)
        asset = IRS['ticker'].replace("Currency", "")
        hgh = dal_get_ts(start, finish, f"SCP({CP}):High", asset)
        low = dal_get_ts(start, finish, f"SCP({CP}):Low", asset)
        close = dal_get_ts(start, finish, f"SCP({CP}):Close", asset)
        true_range = dal_get_ts(start, finish, f"SCP({CP}):TrueRange", asset)
        av_true_range = dal_get_ts(start, finish, f"SCP({CP}):AverageTrueRange", asset)
        dvo1 = dal_get_ts(start, finish, f"SCP({CP}):Dvol01(10,10)", asset)
        act_close = dal_get_ts(start, finish, f"SCP({CP}):ActClose", asset)
        bid = dal_get_ts(start, finish, f"SCP({CP}):Bid", asset)
        ask = dal_get_ts(start, finish, f"SCP({CP}):Ask", asset)

        mdates = 1 # IRSMaturityDate(10,10)
        immts = pd.Series(range(1, len(immdates) + 1), index=pd.to_datetime(immdates.index))
        immts.index = immts.index.to_series().interpolate().fillna(0)
        ex = mdates[immts.index]
        expiry = (align_time_series(ex, mdates)).dropna()
        high = hgh.dropna()
        low = low.dropna()
        close = close.dropna()
        true_range = true_range.dropna()
        av_true_range = av_true_range.dropna()
        dvo1 = dvo1.dropna()
        act_close = act_close.dropna()
        bid = bid.align(close, axis=0)[0]
        ask = ask.align(close, axis=0)[0]
        expiry = expiry.dropna()
        ohclac = series_merge(high, low, close, true_range, dvo1, act_close, bid, ask, expiry)
        ohclac.columns = ["HighPrice", "LowPrice", "ClosePrice", "TrueRange", "Dvol01", "ActClosePrice", "ActBidPrice",
                          "ActAskPrice", "Expiry"]
        if put_cache:
            cache_path = os.path.join("CRSRdataFolder", "cache", f"{IRS['macls']}.csv")
            ohclac.to_csv(cache_path)

        intconst = 1.68
        tr_fact = 1
        true_range_int = np.round((ohclac["TrueRange"] * tr_fact + intconst) * 100, 0)
        av_true_range = Rename(av_true_range, "AverageTrueRange", "AverageTrueRange")
        ohclac = series_merge(ohclac, true_range_int, av_true_range)

        dd = seriestats(ohclac)
        dd = dd.to_frame().reset_index()
        dd.columns = ["TradeDate", "Value"]

        dd1 = seriestats(ohclac).to_frame().reset_index()
        dd1.columns = ["Metric", "Stat"]
        dd = pd.concat([dd, dd1], axis=1)

        dd.to_csv(os.path.join("CRSRdataFolder", "cache", "dd.csv"), index=False)

    else:
        date_format = pd.Timestamp(start).strftime("%Y%m%d")
        yc = getYC(pd.to_datetime("2005-01-01"), pd.to_datetime(finish), IRS, saveYC = False, continous = False, curvemax = 120)

        tenor = str(IRS['tenor'].iloc[0]) + 'Y'
        close = yc['sw'][tenor]

        # need to get the data for this
        try:
            high =   close*1.05 # dal_get_ts(start, finish, "PX_LAST", "PX_HIGH", wacs_log)
        except:
            high = close*1.05
        try:
            low =   close*0.95 # dal_get_ts(start, finish, "PX_LAST", "PX_HIGH", wacs_log)
        except:
            low = close*0.95
        try:
            bid = close
        except:
            bid = close
        try:
            ask = close #(dal_get_ts(start, finish, osub("PX_LAST", "PX_ASK", wacs_log)), None)
        except:
            ask = close # try_except(dal_get_ts(start, finish, osub("PX_LAST", "PX_ASK", wacs_log)), None)

        ohclac = seriesMerge([high, low, close, bid, ask, close*0, close*0, close*0, close*0, close*0, close*0, close*0, close*0,close*0, close*0, close*0])
        ohclac.columns = ["ActHighPrice", "ActLowPrice", "ActClosePrice", "ActBidPrice", "ActAskPrice", "carry", "rolldown",
                          "HighPrice", "LowPrice", "ClosePrice", "ret", "Dv01", "TrueRange", "AvgTrueRange", "Expiry", "par"]

        ohclac = ohclac.interpolate().fillna(0)
        ohclac = align_time_series(ohclac, close)
        swap = None
        dFrom = "2020-01-10"
        dTo = "2021-01-10"
        immdates = getIMMCal(start_date=start, end_date = "2045-01-01")
        immexpiry = immdates[0]
        data_store = []
        for dt in ohclac.index:
            if swap is not None:
                exf = swap_valuation(swap1, yc, IRS,dt)
                #print(dt)
                #print(swap['dates']['value_date'])
                pnl = exf['npv'] - entry1['npv']
                accrual_period = (pd.to_datetime(dt) - pd.to_datetime(entry1['swap']['dates']['value_date'])).days
                fwdrate = entry1['swap']['flt']['fwdrate']
                spot = entry1['swap']['flt']['libor']
                carry = (fwdrate - spot)*(accrual_period / 365)
                del entry1
                del swap1
                swap = None
                del entry
            else:
                pnl = 0
                carry = 0
                fwdrate =0
                spot = 0
                accrual_period = 0
            if swap_mode == ["spot"]:
                new_dates = get_swap_dates(dt, yc['zc'],IRS['tenor'].iloc[0], IRS['SettleOff'].iloc[0], None)
                swap = get_swap(new_dates, yc, IRS, coupon = 0, notional = 1,imm_align=False, at_par = True, continous=False)
                swap['flt']['pays'][-1] = swap['flt']['pays'][-1] + 1
                entry = swap_valuation(swap, yc, IRS, dt)
                swap1 =  copy.deepcopy(swap)
                entry1 =  copy.deepcopy(entry)
            elif swap_mode == ["fwdImm"]:
                new_dates = get_swap_dates(dt, yc['zc'], IRS['FloatReset'].iloc[0], IRS['tenor'].iloc[0], settle=7, IMM=True)
                imm_test = find_index(immdates, new_dates['maturity'], x = 1, date_column='date') # find the next IMM date after the maturity
                if imm_test > immexpiry:
                    swap = get_swap(new_dates, yc, IRS, coupon = 0, notional = 1,imm_align=True, at_par = True, continuous=False)
                    immexpiry = imm_test
                entry = swap_valuation(swap, yc, IRS, dt)
            roll = (entry['swap']['flt']['fwdrate'] - entry['swap']['flt']['libor']) * (accrual_period / 365)
            par = entry['swap']['coupon']
            expiry = immexpiry
            data_store.append({"date": dt, "ret": pnl, "carry": carry, "dv01": entry['dv01'], "carryRoll":carry + roll,
                                "par": par, "expiry": expiry})
            print(dt)
            #print(pnl)

        outp = pd.DataFrame(data_store)
        outp.set_index('date', inplace = True)
        # Returns and carry returns per dollar notional of swap
        ohclac['ClosePrice'] = outp['ret'].cumsum()
        ohclac['HighPrice'] = ohclac['ClosePrice'] + (ohclac['ActClosePrice'] - ohclac['ActLowPrice']) * outp['dv01'] # nb actclose is rate not price
        ohclac['LowPrice'] = ohclac['ClosePrice'] + (ohclac['ActHighPrice']- ohclac['ActClosePrice']) * outp['dv01']
        high = ohclac['ActHighPrice']
        low = ohclac['ActLowPrice']
        close = ohclac['ActClosePrice']
        close_lag = close.shift(1)

        # 1. Calculating dd
        dd = np.maximum.reduce([np.abs(high - low), np.abs(high - close_lag), np.abs(close_lag - low)])
        dd[0] = np.abs(high[0] - low[0])

        # 2. Calculating dr0 and trFactor
        dr0 = np.abs(high - low)
        dr10 = dr0.rolling(window=10, min_periods=1).mean()
        trFactor = (dr10 > 0).astype(int) + 2 * (dr10 <= 0).astype(int)
        dr = pd.DataFrame(dd, index = ohclac.index)
        intConst = 1e8
        ohclac['TrueRangeInt'] = round(modDF(modDF(dr, outp['dv01'], 'mult'), trFactor, 'mult') * intConst * 100, 0)
        ohclac['AverageTrueRange'] = ohclac['TrueRangeInt'].rolling(window = 10).mean() / intConst

    return ohclac

# Placeholder functions to mimic the R functions used in swapValuation


def get_swap_float_leg(swap, yc, IRS, value_date = None, imm_align = True):
    annual = get_af(IRS['FloatCal'].iloc[0])

    if value_date is None:
        value_date = swap['dates']['tradeDate']

    # float leg schedule
    if 'flt' not in swap or swap['flt'] is None:
        swap['flt'] = {}
        swap['flt']['dates'] = get_swap_schedule(swap['dates'], yc, IRS['FloatFreq'].iloc[0], imm_align=imm_align)
    else:
        if value_date == swap['dates']['tradeDate']:
            return swap

    x = len(swap['flt']['dates'])
    case = True
    if IRS['FloatReset'].iloc[0] == 0:
        if case:
            weekly = 7
            discount = discount_curve(value_date, swap['flt']['dates'], yc, annual=annual,
                                      continuous=swap['continuous'])

            x = len(swap['flt']['dates'])
            t1 = np.arange(2, x + 1)
            t0 = np.arange(2, x)

            f = discount['fwdrates']
            z = discount['zerorates']

            T = np.arange(x)
            yr0 = T / annual
            intervals = np.diff(T, prepend=0)
            resets0 = intervals / weekly
            compounds_upper = (1 + z[t1 - 1] / resets0[t1]) ** (resets0[t1] * yr0[t1]) / (
                        1 + z[t0 - 1] / resets0[t0]) ** (
                                      resets0[t0] * yr0[t0]) - 1
            compounds_lower = (1 + f * weekly / annual) ** (intervals / weekly) - 1
            compounds = (compounds_lower + compounds_upper) / 2

            err = np.sum(compounds[2:x] - compounds[2:x])
            if err > 0.0001:
                raise ValueError(f"Error exceeds threshold: {err}")

            floatpays = np.zeros(len(swap['flt']['dates']))
            if value_date > swap['dates']['tradeDate']:
                floatpays[1] = swap['flt']['pays'][1]
            else:
                floatpays[1] = swap['notional'] * compounds[1]
                swap['flt']['sibor'] = discount['fwdrates'][1]  # spot fix
                swap['flt']['fwdrate'] = discount['fwdrates'][x - 1]

            rangex = np.arange(3, x + 1)
            floatpays[rangex - 1] = swap['notional'] * compounds[rangex - 1]
            floatpays[-1] += swap['notional']

            swap['valuesdate'] = value_date
            swap['flt']['pays'] = floatpays
            swap['flt']['discount'] = discount
            swap['flt']['final'] = x
    else:
        discount = discount_curve(value_date, swap['flt']['dates'], yc, annual=annual, continuous=swap['continous'])
        fwdrates = discount['fwdrates']
        do_day_count_scalendar = False

        # Initialize float payments array
        floatpays = np.zeros(len(swap['flt']['dates']))

        if not do_day_count_scalendar:
            floatpays = fwdrates * discount['yrs'] * swap['notional']
            if value_date > swap['dates']['tradeDate']:
                # Floating rate fixed prior to valuedate
                floatpays[0] = swap['flt']['pays'][0]
            else:
                swap['flt']['libor'] = fwdrates[1] # does this need to hold the sofr non libor rate as well?
                swap['flt']['fwdrate'] = fwdrates[-1]
            floatpays[-1] = floatpays[-1] + int(swap['notional'])
        else:
            # Evaluate each interval by correct calendar counting
            frac = IRS['FloatFreq'] / 12
            stub0 = get_year_frac(swap['flt']['dates'][0:2], IRS['FloatCal'])
            stub1 = get_year_frac(swap['flt']['dates'][-2:], IRS['FloatCal'])

            if value_date > swap['dates']['tradeDate']:
                # Floating rate fixed prior to
                floatpays[1] = swap['flt']['pays'][1]
            else:
                floatpays[1] = swap['notional'] * fwdrates[1] * stub0['frac']

            rangex = np.arange(1, min(len(swap['flt']['dates']), 3))
            floatpays[rangex] = swap['notional'] * fwdrates[rangex] * frac
            floatpays[-1] = swap['notional'] * fwdrates[-1] * stub1['frac']
            floatpays[-1] += swap['notional']

        swap['dates']['value_date'] = value_date
        swap['flt']['pays'] = floatpays
        swap['flt']['final'] = len(swap['flt']['dates'])
        swap['flt']['discount'] = discount
    return(swap)

def swap_valuation(swap, yc, IRS, value_date=None, fileout=None, from_date=None, to_date=None):
    if value_date is None:
        value_date = swap['dates']['tradeDate']

    discount = discount_curve(value_date, swap['fix']['dates'], yc, bp =0, show = False, annual = get_af(IRS['FixedCal'].iloc[0]), continuous=swap['continous'])
    npv_fix = np.sum(discount['factors'] * swap['fix']['pays'])
    discountbp = discount_curve(value_date, swap['fix']['dates'], yc, bp = 0.01, show = False,  annual = get_af(IRS['FixedCal'].iloc[0]), continuous=swap['continous'])
    npv_fix_bp = np.sum(discountbp['factors'] * swap['fix']['pays'])
    dv01 = npv_fix - npv_fix_bp

    if swap['dates']['tradeDate'] != value_date:
        swap = get_swap_float_leg(swap, yc, IRS, value_date=value_date)

    discount_flt = discount_curve(value_date, swap['flt']['dates'], yc, bp = 0, show = False, annual = get_af(IRS['FloatCal'].iloc[0]), continuous=swap['continous'])
    npv_flt = np.sum( swap['flt']['pays'] * discount_flt['factors'])

    if fileout is not None:
        if pd.to_datetime(value_date) < pd.to_datetime(from_date):
            if pd.to_datetime(value_date) <= pd.to_datetime(to_date):
                schedule = f"Entry date {swap['dates']['tradeDate']} at valuedate {value_date}"
                if swap['fixFltFinal'] == swap['fltFltFinal']:
                    df0 = pd.DataFrame({
                        "fixDates": swap['fixDates'],
                        "fltPays": swap['fltPays'],
                        "fixPays": swap['fixPays'] * discount_flt['Factors'],
                        "fltPays": swap['fltPays'] * discount_flt['Factors'],
                        "Factors": discount_flt['Factors'],
                        "zeroRates": discount_flt['ZeroRates'],
                        "dayCounts": discount_flt['DayCounts'],
                        "fwdRates": discount_flt['FwdRates']
                    })
                    df0.to_csv(fileout, mode='a', index=False, header=not os.path.exists(fileout), sep=',', na_rep='NA')
                else:
                    df = pd.DataFrame({
                        "fixDates": swap['fixDates'],
                        "fltPays": swap['fltPays'],
                        "fixPays": swap['fixPays'] * discount_flt['Factors'],
                        "fltPays": swap['fltPays'] * discount_flt['Factors'],
                        "Factors": discount_flt['Factors'],
                        "zeroRates": discount_flt['ZeroRates'],
                        "dayCounts": discount_flt['DayCounts'],
                        "fwdRates": discount_flt['FwdRates']
                    })
                    df.to_csv(fileout, mode='a', index=False, header=not os.path.exists(fileout), sep=',', na_rep='NA')

    return {
        "npv": round(npv_fix- npv_flt,6),
        "npvFlt": npv_flt,
        "npvFix": npv_fix,
        "dv01": dv01,
        "swap": swap
    }


def get_swap(dates, yc, IRS, coupon=0, notional=1, imm_align=True, at_par=False, continous=False):
    swap = {}
    swap['dates'] = dates
    swap['notional'] = notional
    swap['continous'] = continous

    swap = get_swap_float_leg(swap, yc, IRS, imm_align=imm_align)
    swap['fix'] = {}
    swap['fix']['dates'] = get_swap_schedule(swap['dates'], yc, IRS['FixedFreq'].iloc[0], imm_align= imm_align)
    swap['fix']['discount'] = discount_curve(swap['dates']['tradeDate'], swap['fix']['dates'], yc, annual=get_af(IRS['FixedCal'].iloc[0]), continuous=swap['continous'])

    if at_par:
        float_leg = swap['flt']['pays']
        float_leg[-1] = float_leg[-1] - notional
        par = 100 * np.sum(float_leg*swap['flt']['discount']['factors']) / np.sum(swap['fix']['discount']['yrs'] * swap['fix']['discount']['factors'])
        swap['coupon'] = par
    else:
        swap['coupon'] = coupon

    fixed_pays = np.array(swap['notional'] * swap['fix']['discount']['yrs'])*swap['coupon']/100
    idx = len(fixed_pays)
    fixed_pays[idx-1] += swap['notional']
    swap['fix']['pays'] = fixed_pays
    swap['fix']['final'] = idx

    return swap


def get_swap_schedule(dates, yc, freq, imm_align=True):
    if not imm_align:
        if freq == 1:
            fixed_paydates = pd.to_datetime([find_bd(yc['zc'], date, fwd=True) for date in rangeDates(start=dates['startDate'], end=dates['maturity'],interval='1M', include_start_date =  True)])
            final_date = fixed_paydates[-1]
            if dates['maturity'] > final_date:
                fixed_paydates =  pd.to_datetime(np.append(fixed_paydates, find_bd(yc['zc'],dates['maturity'], fwd=True)))
        elif freq == 3:
            fixed_paydates = pd.to_datetime([find_bd(yc['zc'], date, fwd=True) for date in rangeDates(start=dates['startDate'], end=dates['maturity'],interval='3M', include_start_date =  True)])
            final_date = fixed_paydates[-1]
            if dates['maturity'] > final_date:
                fixed_paydates =  pd.to_datetime(np.append(fixed_paydates, find_bd(yc['zc'],dates['maturity'], fwd=True)))
        elif freq == 6:
            # does the float leg need to avoid having the 1st date entry?
            seqDates = rangeDates(start=dates['startDate'], end=dates['maturity'],interval='6M', include_start_date =  True) # pd.date_range(start=dates['startDate'], end=dates['maturity'], freq='QE')
            fixed_paydates = pd.to_datetime([find_bd(yc['zc'], date, fwd=True) for date in seqDates])
            final_date = fixed_paydates[-1]
            if dates['maturity'] > final_date:
                fixed_paydates = pd.to_datetime(np.append(fixed_paydates, find_bd(yc['zc'],dates['maturity'], fwd=True)))
        elif freq == 12:
            fixed_paydates = pd.to_datetime([find_bd(yc['zc'],date, fwd=True) for date in rangeDates(start=dates['startDate'], end=dates['maturity'],interval='1Y', include_start_date =  True)])
            final_date = fixed_paydates[-1]
            if dates['maturity'] > final_date:
                fixed_paydates =  pd.to_datetime(np.append(fixed_paydates, find_bd(yc['zc'],dates['maturity'], fwd=True)))
        else:
            raise NotImplementedError(f"Please implement rolling calendar for {freq} month schedule")
    else:
        swdt = pd.to_datetime(dates['startDate'])
        fixed_paydates = []
        x = 0
        while True:
            swdt = pd.to_datetime(get_IRS_IMM_date(swdt, freq))  # next coupon date advance by 3, 6, or 12 months
            if swdt <= pd.to_datetime(dates['maturity']):
                fixed_paydates.append(find_bd(yc['zc'],swdt, fwd=True))
                x += 1
            else:
                break
        fixed_paydates[x - 1] = pd.to_datetime(dates['maturity'])
        fixed_paydates = pd.to_datetime(fixed_paydates[:x])

    return fixed_paydates

# Placeholder functions to mimic the R functions used in getSwapSchedule
def get_IRS_IMM_date(swdt, freq):
    # Placeholder function to get the next IMM date
    imm_dates = getIMMCal(start_date="1990-01-01", end_date = "2050-01-01")
    imm_dt = find_index(imm_dates, swdt, x=int(freq / 3))
    return imm_dt

def get_year_frac(dates, IRScal):
    if IRScal == "30/360":
        YR = 360
        mo1 = dates[0].month
        dy1 = dates[0].day
        dy1 = 30 if dy1 == 31 else dy1
        yr1 = dates[0].year

        mo2 = dates[1].month
        dy2 = dates[1].day
        dy2 = 30 if dy2 == 31 else dy2
        yr2 = dates[1].year

        dc = 360 * (yr2 - yr1) + 30 * (mo2 - mo1) + (dy2 - dy1)
    elif IRScal == "ACT/360":
        YR = 360
        dc = (dates[1] - dates[0]).days
    elif IRScal == "ACT/365":
        YR = 365
        dc = (dates[1] - dates[0]).days
    elif IRScal == "ACT/ACT":
        YR = 365
        dc = (dates[1] - dates[0]).days
    else:
        raise ValueError(f"unknown accrual calendar: {IRScal}")

    return {"frac": dc / YR, "daycount": dc}


def find_bd(df, target_date, fwd=True):
    """
    Find the nearest date in the DataFrame index to the target_date.

    Parameters:
    - df (pd.DataFrame): DataFrame with dates as the index.
    - target_date (str or pd.Timestamp): The date to look for.
    - fwd (bool): Direction to search if the date is not found.
                  True for forward, False for backward.

    Returns:
    - pd.Timestamp: The nearest date found in the specified direction.
    """
    # Ensure target_date is a Timestamp
    target_date = pd.to_datetime(target_date)

    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    if target_date < df.index[0]:
        print("Date is before the start of the index ffs.")
        return None
    # Check if target_date is in the DataFrame index
    if target_date in df.index:
        return target_date

    # Find nearest dates
    if fwd:
        nearest_dates = df[df.index > target_date]
        if not nearest_dates.empty:
            return nearest_dates.index[0]
        else:
            if target_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    # Bump to the next Monday
                    target_date += pd.DateOffset(days=(7 - target_date.weekday()))
            return target_date
    else:
        nearest_dates = df[df.index < target_date]
        if not nearest_dates.empty:
            return nearest_dates.index[-1]
        else:
            if target_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                # Bump to the previous Friday
                target_date -= pd.DateOffset(days=(target_date.weekday() - 4))
            return target_date

def get_swap_dates(trade_date, zc,tenor=5, settle=2, IMM=None):
    # tradeDate is rate fixing date
    print(trade_date)
    td = find_bd(zc,trade_date)
    # start date is +2 or 0, or forward = first date of accrual
    if IMM is not None:
        sd = get_IRS_IMM_date(td + pd.Timedelta(days=settle), freq=3)  # use 3 for next IMM
    else:
        sd = find_bd(zc,td + pd.Timedelta(days=settle))

    lf = find_bd(zc,td, fwd=False)
    startquote = zc.index[0]
    if lf < startquote:
        lf = startquote

    if IMM is not None:
        final = get_IRS_IMM_date(sd, freq=tenor * 12)
    else:
        final = find_bd(zc, get_swap_maturity(sd, tenor))

    return {"tradeDate": td, "startDate": sd, "liborFix": lf, "maturity": final}


def getIMMCal(start_date, end_date):
    # Convert string dates to datetime if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    imm_dates = []
    current_date = start_date

    while current_date <= end_date:
        year = current_date.year
        for month in [3, 6, 9, 12]:
            third_wednesday = datetime(year, month, 15)
            while third_wednesday.weekday() != 2:  # 2 corresponds to Wednesday
                third_wednesday += timedelta(days=1)
            if start_date <= third_wednesday <= end_date:
                imm_dates.append(third_wednesday)
        current_date = datetime(year + 1, 1, 1)  # move to the next year

    return imm_dates

def business_day(dt, yc, fwd=True):
    if fwd:
        td = dt
        while True:
            if dt in yc:
                bd = dt
                break
            else:
                dt += pd.Timedelta(days=1)
    else:
        td = dt
        while True:
            if dt in yc:
                bd = dt
                break
            else:
                dt -= pd.Timedelta(days=1)

    w = dt.weekday()
    if w == 6:  # Sunday
        dtb = dt + pd.Timedelta(days=1) if fwd else dt - pd.Timedelta(days=1)
    elif w == 0:  # Monday
        dtb = dt - pd.Timedelta(days=3) if not fwd else dt + pd.Timedelta(days=0)
    else:
        dtb = dt

    return dtb

def rotatevec(x, n=1):
    if n == 0:
        return x
    else:
        return np.concatenate((x[-n:], x[:-n]))


def integer_date(dt):
    mo = dt.month
    dy = dt.day
    yr = dt.year
    ndt = 10000 * yr + 100 * mo + dy
    return ndt


def date_from_integer(dt):
    dt_str = str(dt)
    dt2 = pd.to_datetime(f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:]}")
    return dt2

def interval_idate(d0, d1):
    y0 = int(d0 // 10000)
    y1 = int(d1 // 10000)
    yr = y1 - y0

    mo0 = int(d0 // 100) - (y0 * 100)
    m1 = int(d1 // 100) - (y1 * 100)
    if m1 < mo0:
        yr -= 1
    mo = (m1 - mo0) % 12
    return {"yr": yr, "mo": mo}


def get_af(cal, default=365):
    if cal in ["ACT/360", "30/360"]:
        af = 360
    elif cal in ["ACT/365", "30/365"]:
        af = 365
    else:
        af = default
    return af

def get_swap_rate(swapdates, sw, AF=365, show=False, annual=365):
    valuedate = swapdates[0]
    nd = swapdates[1]
    ro = sw.loc[valuedate].values
    x = np.round((annual / 12) * sw.columns.astype(float)) * 12
    fwdrates = interp1d(x, ro, kind='cubic')(nd)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(x, ro, label='interpol')
        plt.plot(nd, fwdrates, color='red', label='fwdrates')
        plt.legend()
        plt.show()
    return fwdrates

def get_BBIR_swap_code(currency="USD", tenor=10, path=r'C:\Users\edgil\Documents\git_working\Research\config\asset_config\IRS\swap_codes.csv', reload=False):

    IRCodes = pd.read_csv(path, dtype=str)
    m = IRCodes.loc[IRCodes['Currency'] == currency].index[0]
    bbcode = f"{IRCodes.at[m, 'prefix']}{tenor} {IRCodes.at[m, 'suffix']}"
    return bbcode


def align_yc(yc, align_to):
    yc['zc'] = align_time_series(yc['zc'], align_to)
    yc['svp'] = align_time_series(yc['svp'], align_to)
    yc['svfloating'] = align_time_series(yc['svfloating'], align_to)
    yc['ssw'] = align_time_series(yc['ssw'], align_to)
    return yc



def get_swap_maturity(dt, fwdyr=5):
    dt = pd.to_datetime(dt)
    if dt.month == 2 and dt.day == 29:
        dt = dt.replace(day=28)

    months = 12 * fwdyr
    tenor_yrs = months // 12
    dt = dt.replace(year=dt.year + tenor_yrs)
    tenor_month = months % 12
    if tenor_month > 0:
        dt = add_months(dt, tenor_month)
    return dt


def bond_duration(nominal=100, freq=2, yldpct=0, couponpct=0, maturity=0):
    ts = []
    yld = yldpct / 100
    coupon = 1 / freq
    t = maturity
    i = 1
    while t > 0:
        ts.append(t)
        t -= coupon
        i += 1

    npv = nominal / (1 + yld) ** maturity
    npvt = maturity * npv
    for t in ts:
        coup = (nominal * (couponpct / 100 * freq)) / (1 + yld) ** t
        npv += coup
        npvt += t * coup

    dur = npvt / npv
    moddur = npvt / (npv * (1 + (yld / freq)))
    dvo1 = moddur * npv / 100

    ret = {
        'dur': dur,
        'moddur': moddur,
        'npv': npv,
        'dvo1': dvo1
    }
    return ret


def bond_npv(yldpct, couponpct=6, maturity=10, nominal=100, freq=2):
    return bond_duration(nominal, freq, yldpct, couponpct, maturity)['npv']


def ytm(price, tol=1e-6, nominal=100, freq=2, couponpct=6, maturity=10):
    ytm0 = (couponpct + (nominal - price) / maturity) * 200 / (nominal + price)
    result = brentq(lambda yldpct: bond_npv(yldpct, couponpct, maturity, nominal, freq) - price,
                    ytm0 * 0.9, ytm0 * 1.1, xtol=tol)
    return result


def discount_curve(value_date, fwddates, yc, bp=0, show=False, annual=365, continuous=False, flat_projection=True, use_svensson_fit=False):
    nd = (fwddates - value_date).days
    #nd = nd[nd > 0]
    ro = yc['zc'].loc[value_date].values + bp
    x = np.round((annual / 12) * get_tenors( yc['zc'].columns).values)
    yr_float = nd / annual

    if use_svensson_fit:
        jj = 1
        #srates = rates(yc.loc[valuedate, 'svp'].values, yr_float, "spot")
        #zerorates = (srates + bp) / 100
    else:
        interpol = interp1d(x[0], ro, kind='cubic', fill_value='extrapolate')(nd)

        if flat_projection:  # before first pillar will be flat
            flat = np.where(nd < x[0][0])
            if len(flat[0]) > 0:
                interpol[flat] = ro[0]
            zerorates = interpol / 100

    if show:
        import matplotlib.pyplot as plt
        plt.plot(x, ro, label='interpol')
        plt.plot(nd, interpol, color='red', label='zerorates')
        plt.legend()
        plt.show()

    dt = [(fwddates[0] - value_date).days] + list(nd.diff().dropna()) # only ACT/n calendars
    divis = [i/annual  for i in dt]

    if continuous:
        accrual = zerorates * yr_float
        disc_factors = np.exp(-accrual)
        with np.errstate(divide='ignore', invalid='ignore'):
            fwdrates = np.divide(np.diff(-np.log(np.append(1, disc_factors))),
                                 divis,
                                 where=divis != 0,
                                 out=np.zeros_like(divis))
        fwdrates[np.isinf(fwdrates)] = 0
        fwdrates[np.isnan(fwdrates)] = 0
        #fwdrates =fwdrates[fwdrates > 0]
    else:
        disc_factors = 1 / (1 + zerorates) ** yr_float
        with np.errstate(divide='ignore', invalid='ignore'):
            fwdrates = np.divide(((np.append(1, disc_factors) / np.append(disc_factors, 1))[:-1] - 1),
                                 divis,
                                 where=divis != 0,
                                 out=np.zeros_like(divis))
        fwdrates[np.isinf(fwdrates)] = 0
        fwdrates[np.isnan(fwdrates)] = 0
        #fwdrates = fwdrates[fwdrates > 0]

    return {
        'factors': disc_factors,
        'zerorates': zerorates,
        'dayCounts': nd,
        'yrs': divis,
        'fwdrates': fwdrates
    }


def sub(query1, query2):
    # Dummy function to represent the sub in R
    return f"{query1}_{query2}"


def get_tenors(columns):
    months_list = []
    add_cols = []
    for element in columns:
        if 'Y' in element:
            years = int(element.replace('Y', ''))
            months = years * 12
            months_list.append(months)
            add_cols.append(element)
        elif 'M' in element:
            months = int(element.replace('M', ''))
            months_list.append(months)
            add_cols.append(element)

    df = pd.DataFrame([months_list], columns=columns)
    return df
def getYC(start, finish, IRS, saveYC = False, continous = False, curvemax = 120):
    """
    Get the yiled curve pillars for swap valuation.
    :param start: start = "1990-01-01"
    :param finish: finish = "2024-01-02"
    :param IRS:
    :param saveYC:
    :param continous:
    :param curvemax:
    :return:
    """
    irsprogpath = r'C:\Users\edgil\Documents\git_working\Research\config\asset_config\IRS\IRS_pillars.csv'
    dfPillars = pd.read_csv(irsprogpath)
    # if not in pillars, we just query bbg for the 5 year swap (NOK, SEK etc)
    asset = IRS['bbticker'].iloc[0].replace("Curncy", "").replace(" ", "")
    # use the yr swap form the csv file to set a start
    bbid = get_BBIR_swap_code(IRS['Currency'].iloc[0], IRS['tenor'].iloc[0], reload=True)
    wacslq = (bbid + "::'PX_LAST'")
    try:
        checkrange = dal_get_ts(start_date=start, end_date=finish, query = "PLN")
        startO = pd.to_datetime(checkrange.index[0])
    except Exception as e:
        checkrange = None
        startO = None
    if False: #len(dfPillars) > 9000000000
        #curvemax = get_tenors([max(dfPillars['tenor'])])
        #dfPillar1 = dfPillars[dfPillars['Currency'] == IRS['Currency'].iloc[0]]
        #dfPillar1Anchor = dfPillar1[dfPillar1['tenor'] == f'{IRS['tenor'].iloc[0]}'"Y"]
        #wacslq = (dfPillar1Anchor['ticker'].values[0] + "::'PX_LAST'") # make the call to db to get this info.
        try:
            checkrange = dal_get_ts(start_date=start, end_date=finish, signal=wacslq)
            startO = pd.to_datetime(checkrange.index[0])
        except Exception as e:
            checkrange = None
            startO = None
    else:
        None

    if pd.to_datetime(start) < pd.to_datetime(startO):
        start = startO
    # load wacsl rates data
    weekdays = pd.date_range(start=start, end=finish, freq='B')
    nr = len(weekdays)
    ts0 = pd.Series([0] * nr, index=weekdays)
    # IRS here is the row in the csv file corresponding to the correct swap.
    ycdef = getYCSource(IRS, dfPillars)

    if ycdef is not None:
        swap_ts = ts0
        if ycdef.get('swap_tenors') is not None:
            """
            for query in ycdef['wacslq_swap']:
                try:
                    # make a call to bloomberg/database
                    ts1 = dal_get_ts(start_date=start, end_date=finish, signal=query)
                    if ts1.shape[0] < 1:
                        ts1 = ts0 * np.nan
                except Exception:
                    ts1 = ts0 * np.nan
                swap_ts = seriesMerge(swap_ts, ts1, pos="union", how="NA")
            """
        swap_ts = dal_get_ts(start_date=start, end_date=finish, query = "PLN")# swap_ts.iloc[:, 1:] # cheeky placeholder for when you have data
        swap_ts.columns = ycdef['tenors'] # ycdef['swap_tenors']

        """
                rate_ts = ts0
        # this gets the non swaps, i.e. the rates less than 1y.
        if ycdef.get('rate.tenors') is not None:
            for query in ycdef['wacslq.rate']:
                try:
                    ts = dal_get_ts(start_date=start, end_date=finish, signal=query)
                    cumul = ycdef['rate.cumul'][ycdef['wacslq.rate'] == query]
                    if cumul.any():
                        if cumul[0]:
                            fcal = pd.date_range(start=start, end=finish, freq='D')
                            fcal_xts = pd.Series(1, index=fcal)
                            logts = np.log(ts)
                            ts1 = fillDays(logts, fcal_xts)
                            lograte = ts1.diff().fillna(0)
                            rate = (np.exp(lograte * 365) - 1) * 100
                            rate = alignUltimate(rate, ts)
                            rate.iloc[0] = rate.iloc[1]  # backfill
                            ts = rate
                except Exception:
                    ts = ts0 * np.nan
                rate_ts = seriesMerge(rate_ts, ts, pos="union", how="NA")
        rate_ts = rate_ts.iloc[:, 1:]
        rate_ts.columns = ycdef['rate.tenors']
    else:
        print(f"Yield curve undefined for {IRS['Asset']}")
        raise SystemExit
    if ycdef['filling'] == "none":  # raw nofill
        fltq = f"{IRS['floatQuery']}.Index"
        wacslq_float = f"BloombergRequest({fltq},'PX_LAST')"
    else:
        wacslq_float = f"IC({IRS['floatQuery'].replace(' Index', '')}):IrsPricingCurveItem(10)"
        """

    # filter some days with incomplete data
    #raw_curve = pd.concat([rate_ts, swap_ts], axis=1)
    #raw_curve.columns = [col.lower() for col in rate_ts.columns] + [col.upper() for col in swap_ts.columns]
    raw_curve = swap_ts
    # discard days with insufficient data for a curve
    if ycdef['filtering'] == "standard":  # this branch is suitable for current IRS incl Nordic, China
        curve_6mo = raw_curve.iloc[:, :4].notna().sum(axis=1) >= 1  # require <= 6 m quote
        raw_curve = raw_curve[curve_6mo]

        ctenor = raw_curve.columns.get_loc(f"{IRS['tenor'].iloc[0]}Y")  # require quote for target tenor
        c2y = raw_curve.columns.get_loc("2Y")
        c10y = raw_curve.columns.get_loc("10Y")
        curve_10yr = raw_curve.iloc[:, c2y:c10y + 1].notna().sum(axis=1)  # require > 1yr quote

        have_5yr = raw_curve.iloc[:, ctenor].notna()
        ok_rows = (curve_10yr > 1)  & have_5yr
        raw_curve = raw_curve[ok_rows]
    else:  # this may be modified for y/c usage in other instruments
        if len(dfpillars) > 0:
            pillars_selected = dfpillars.loc[dfpillars['tenor'].str.upper().isin(raw_curve.columns.str.upper())]
            ratecols = [i for i, col in enumerate(pillars_selected['type']) if 'ra' in col]
        else:
            ratecols = [i for i, col in enumerate(raw_curve.columns) if 'Y' in col]

        yearcols = [i for i, col in enumerate(raw_curve.columns) if 'Y' in col]

        curve_6mo = raw_curve.iloc[:, ratecols].notna().sum(axis=1) >= 1  # require < 1Y quote
        raw_curve = raw_curve[curve_6mo]

        tenortag = f"{IRS['tenor']}Y"
        tenorcol = raw_curve.columns.get_loc(tenortag)
        curve_10yr = raw_curve.iloc[:, yearcols].notna().sum(axis=1)  # require > 1yr quote

        have_5yr = raw_curve.iloc[:, tenorcol].notna().sum()
        ok_rows = (curve_10yr > 1) | ((curve_10yr == 1) & have_5yr)
        raw_curve = raw_curve[ok_rows]

    nr = raw_curve.shape[0]  # usable pillar data rows

    # interpolate any missing obs so we have 10 yearly swap rates
    swap_cols = [element for element in raw_curve.columns if 'Y' in element]
    swap_tenors_months1 = get_tenors(swap_cols)
    swap_tenors_months = swap_tenors_months1[swap_tenors_months1 <= curvemax].dropna(axis=1)
    rate_cols = [element for element in raw_curve.columns if 'M' in element]
    rate_tenors_months = get_tenors(rate_cols)
    payments_tenors = np.arange(IRS['FixedFreq'].iloc[0], curvemax + IRS['FixedFreq'].iloc[0],
                                IRS['FixedFreq'].iloc[0])  # beyond the target swap maturity is required
    # add payment tenors to swap maturities to be filled by interpolation
    filled_swap_tenors = np.sort(np.unique(np.concatenate((payments_tenors, swap_tenors_months.values[0]))))

    # add payment tenors less than max rate tenor rate series to be filled by interpolation
    rates_fill = payments_tenors[payments_tenors < max(rate_tenors_months.values[0])]
    filled_rate_tenors = np.sort(np.unique(np.concatenate((rate_tenors_months.values[0], rates_fill))))

    rates_n = len(rate_tenors_months.columns)
    swaps_n = len(swap_tenors_months.columns)

    nc = rates_n + swaps_n  # required pillar tenors
    needed_cols = np.concatenate((rate_tenors_months.columns, swap_tenors_months.columns))
    filled_curve = pd.DataFrame(np.zeros((nr, nc)), index=raw_curve.index,
                                columns=needed_cols)
    # Do I need the spline interpolation?
    # its only needed to fill NA values it seems.
    """
    
    for i in range(nr):  # interpolate separately missing swap pillars, and missing rate pillars
        # interpolate any missing rate obs
        data = raw_curve[rate_tenors_months].iloc[i].values
        spline_func = interp1d(rate_tenors_months, data, kind='cubic', fill_value="extrapolate")
        filled_curve.iloc[i, :rates_n] = np.round(spline_func(filled_rate_tenors), 10)

        data = raw_curve.iloc[i, swap_cols].values
        spline_func = interp1d(swap_tenors_months, data, kind='cubic', fill_value="extrapolate")
        filled_curve.iloc[i, rates_n:rates_n + swaps_n] = np.round(spline_func(filled_swap_tenors), 10)

    # other outputs
    sw = filled_curve.iloc[:, rates_n:rates_n + swaps_n]  # nearest to ois rate
    fltrate = filled_curve.iloc[:, rate_cols[0]]

    # choose the pillars for bootstrap
    swap_tenors_use = filled_swap_tenors[filled_swap_tenors > max(filled_rate_tenors)]  # use ois rate preference
    # swap.tenors.use = filled_swap_tenors  # use ois rate preference
    is_swap = [i for i, comment in enumerate(filled_curve.columns) if comment == 's']
    swap_tenors = np.array([float(col) for col in filled_curve.columns[is_swap]])
    swap_cols = is_swap[np.isclose(np.round(swap_tenors_use, 2),
                                   np.round(swap_tenors, 2))]  # these are the swap pillars used for bootstrap

    # case 1 prefer rates
    rate_tenors_use = rate_tenors_months
    # case 2 - prefer swaps
    # rate_tenors_use = rate_tenors_months[rate_tenors_months < min(swap_tenors_use)]
    is_rate = [i for i, comment in enumerate(filled_curve.columns) if comment == 'r']
    rate_tenors = np.array([float(col) for col in filled_curve.columns[is_rate]])
    rate_cols = is_rate[np.isclose(np.round(rate_tenors_use, 2),
                                   np.round(rate_tenors, 2))]  # these are the rate pillars used for bootstrap
    """

    # place holder
    filled_curve = raw_curve[needed_cols].apply(lambda row: interpolate_row(row.values), axis=1, result_type='expand')
    # z.rates will contain bootstrapped curve
    # TODO: understand the zero rate calculation and the diff between the rate and the swap yield.
    # once you have the YC, then you need to understand the swap valuation. PLEASE
    zc = filled_curve # .iloc[:, np.concatenate([rate_cols, swap_cols])]
    zc.columns = np.concatenate((rate_tenors_months.columns, swap_tenors_months.columns))
    z_swap = zc[swap_tenors_months.columns]
    z_rate = zc[rate_tenors_months.columns]
    ztenors = pd.concat([rate_tenors_months,swap_tenors_months], axis = 1)
    bootstrap_cols = np.searchsorted(ztenors.values[0], payments_tenors)  # want only the pillars that align with swap payments
    minterm = 1 # needs to be 2? or greater than 1Y.
    maxterm = len(bootstrap_cols)
    payments_yr = payments_tenors / 12
    accrued_pd = np.diff(np.concatenate([[0], payments_yr]))

    # z.rates should be zero rates up to bootstrap.x, then swaps. now get the implied zero rates from the swaps
    for i in range(nr):
        z_curve = zc.iloc[i, bootstrap_cols].to_numpy()

        for t in range(minterm, maxterm):
            sw = z_curve[t]
            if t > 0: # zero indexed
                rj = z_curve[:t] / 100
                tj = payments_yr[:t]
                dfj = np.exp(-rj * tj) if continous else (1 / (1 + rj)) ** tj  # annualised
                sw_coupons  = sw * accrued_pd[:t]
                discounted_coupons = np.sum(sw_coupons * dfj)
            else:
                discounted_coupons = 0

            if continous:
                z_curve[t] = -100 * np.log((100 - discounted_coupons) / (100 + (sw * accrued_pd[t]))) / payments_yr[t]
            else:
                z_curve[t] = (100 / ((100 - discounted_coupons) / (100 + (sw * accrued_pd[t]))) ** (
                            1 / payments_yr[t])) - 100

        zc.iloc[i, bootstrap_cols] = z_curve

    svensson_p = None
    dosvensson = False
    if dosvensson:
        ycfolder = f"{outDir()}latest/irs/yield/"
        svsfile = f"{ycfolder}{IRS['currency']}_svensson.csv"
        if os.path.exists(svsfile):
            svensson_p = pd.read_csv(svsfile)
            svto = svensson_p.iloc[-1].name
            zcto = z.iloc[-1].name
            if zcto > svto:
                newsv = z.iloc[(z.index > svto)]
                sv = svensson(newsv, z.columns.to_numpy() / 12)
                svensson_p = pd.concat([svensson_p, sv])
        else:
            svensson_p = None
            nr = z.shape[0]
            for i in range(0, nr, 100):
                sv = svensson(z.iloc[i:min(nr, i + 100)], z.columns.to_numpy() / 12)
                svensson_p = pd.concat([svensson_p, sv])
                print(f"computing svensson fit {i}/{nr}")
    z = np.round(zc.astype('float64'), 10)
    fltrate = z[rate_tenors_months.columns[0]] # nearest rate to ois float?
    fltrate = align_time_series(fltrate, z)
    startx = np.min(np.where(~np.isnan(fltrate)))
    dx = fltrate.index[startx]
    startw = np.min(np.where(~np.isnan(z_swap)))
    dw = fltrate.index[startw]
    sq = pd.to_datetime(max([dw, dx]))
    saveYc = False
    if saveYc:
        ycfolder = f"{outDir()}latest/irs/yield/"
        os.makedirs(ycfolder, exist_ok=True)

        if 'svsfile' in locals():
            if svsfile is not None:
                svensson_p.to_csv(svsfile)  # , header=["date"] + list(svensson_p.columns))

        ycffile = f"{ycfolder}{IRS['currency']}_zero.csv"
        z.to_csv(ycffile, header=["date"] + list(z.columns))

        ycffile = f"{ycfolder}{IRS['currency']}_swap.csv"
        SW.to_csv(ycffile, header=["date"] + list(z.columns))

        ycffile = f"{ycfolder}{IRS['currency']}_rawswap.csv"
        raw_curve.to_csv(ycffile, header=["date"] + list(z.columns))

    return {'zc': z, 'sw': z_swap, 'svp': svensson_p, 'floating': fltrate, 'startquote': sq, 'start0': start,
               'wacslq': "data_query_ofTraded_swap", 'pillars': dfPillars}
def getYCSource(IRS,pillars= None):
    """

    :param IRS:
    :param dfPillars:
    :return:
    """
    is_cumul = None
    if (IRS['Asset'].str.contains("CCSWNI").any()):  # china offshore
        tenors = ["7D", "1M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
        is_swap = [False] + [True] * 13
        wacslq = [None] * len(tenors)
        for idx, tenor in enumerate(tenors):
            if re.search("Y", tenor):
                tenorcode = re.sub("Y", "", tenor)
                wacsl = get_BBIR_swap_code(IRS['Currency'].iloc[0], tenorcode)
                wacsl = re.sub(" Curncy", "", wacsl)
                wacslq[idx] = f"IC({wacsl}):IrsPricingCurveItem(10)"
            elif re.search("M", tenor):
                tenorM = ["1", "3", "6"]
                tenorCode = ["A", "C", "F"][tenorM.index(re.sub("M", "", tenor))]
                wacsl = get_BBIR_swap_code(IRS['Currency'].iloc[0], tenorCode)
                wacsl = re.sub(" Curncy", "", wacsl)
                wacslq[idx] = f"IC({wacsl}):IrsPricingCurveItem(10)"
            elif re.search("D", tenor):
                wacsl = re.sub(" Index", "", IRS['OISquery'].iloc[0])
                wacslq[idx] = f"IC({wacsl}):IrsPricingCurveItem(10)"

        terms = [None] * len(tenors)
        for idx, tenor in enumerate(tenors):
            if re.search("Y", tenor):
                terms[idx] = int(re.sub("Y", "", tenor)) * 12
            elif re.search("M", tenor):
                terms[idx] = round(int(re.sub("M", "", tenor)) / 1, 6)
            elif re.search("D", tenor):
                terms[idx] = round(int(re.sub("D", "", tenor)) / 28, 6)
        rate_tenors = [tenors[i] for i in range(len(tenors)) if not is_swap[i]]
        swap_tenors = [tenors[i] for i in range(len(tenors)) if is_swap[i]]
        wacslq_swap = [wacslq[i] for i in range(len(wacslq)) if is_swap[i]]
        wacslq_rate = [wacslq[i] for i in range(len(wacslq)) if not is_swap[i]]

        return {
            'tenors': tenors,
            'terms': terms,
            'is_swap': is_swap,
            'query': wacslq,
            'filtering': "standard",
            'filling': "wacsl",
            'rate_tenors': rate_tenors,
            'wacslq_rate': wacslq_rate,
            'swap_tenors': swap_tenors,
            'wacslq_swap': wacslq_swap
        }
    elif (IRS['Asset'].str.contains("NK").any()) or (IRS['Asset'].str.contains("SK").any()) or (IRS['Asset'].str.contains("NDSW").any()):
        swap_tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y" ,"15Y", "20Y", "30Y"]  # all tenors
        rate_tenors = ["1M", "2M", "3M", "6M"]  # all tenors
        tenors = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y","15Y", "20Y", "30Y"]  # all tenors
        is_swap = [False] * 4 + [True] * 13  # all tenors
        wacslq = []
        import re
        for tenor in tenors:
            if re.search("Y", tenor):
                tenorcode = re.sub("Y", "", tenor)
                wacslq.append(f"BHD('{get_BBIR_swap_code(IRS['Currency'].iloc[0], tenorcode)}','PX_LAST')")
            elif re.search("M", tenor):
                tenorp = re.sub("M", "", tenor)
                oisq = f"{IRS['OISquery'].iloc[0]}{tenorp} Index"
                wacslq.append(f"BloombergRequest('{oisq}','PX_LAST')")
            elif re.search("D", tenor):
                wacslq.append(f"BloombergRequest('{IRS['OISquery'].iloc[0]} Index','PX_LAST')")
        terms = []
        for tenor in tenors:
            if re.search("Y", tenor):
                terms.append(int(re.sub("Y", "", tenor)) * 12)
            elif re.search("M", tenor):
                terms.append(round(int(re.sub("M", "", tenor)) / 1, 6))
            elif re.search("D", tenor):
                terms.append(round(int(re.sub("D", "", tenor)) / 28, 6))

        rate_tenors = [tenors[i] for i in range(len(tenors)) if not is_swap[i]]
        swap_tenors = [tenors[i] for i in range(len(tenors)) if is_swap[i]]
        wacslq_swap = [wacslq[i] for i in range(len(wacslq)) if is_swap[i]]
        wacslq_rate = [wacslq[i] for i in range(len(wacslq)) if not is_swap[i]]

        return {
            'tenors': tenors,
            'terms': terms,
            'is_swap': is_swap,
            'query': wacslq,
            'filtering': "standard",
            'filling': "wacsl",
            'rate_tenors': rate_tenors,
            'wacslq_rate': wacslq_rate,
            'swap_tenors': swap_tenors,
            'wacslq_swap': wacslq_swap
        }

    elif (IRS['Asset'].str.contains("CK").any()) or (IRS['Asset'].str.contains("PZ").any()) or (IRS['Asset'].str.contains("HF").any()):
        # this handle the different conventiosn between swaps it seems...
        swap_tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y"]  # all tenors
        rate_tenors = ["1M", "3M", "6M", "9M"]  # all tenors
        tenors = ["1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y"]  # all tenors
        is_swap = [False] * 4 + [True] * 14  # all tenors ois
        import re
        wacslq = []
        for tenor in tenors:
            if re.search("Y", tenor):
                tenorcode = re.sub("Y", "", tenor)
                wacslq.append(f"BHD('{get_BBIR_swap_code(IRS['Currency'].iloc[0], tenorcode)}','PX_LAST')")
            elif re.search("M", tenor):
                tenorp = re.sub("M", "", tenor)
                oisq = f"{IRS['OISquery'].iloc[0]}{tenorp} Index"
                wacslq.append(f"BloombergRequest('{oisq}','PX_LAST')")
            elif re.search("D", tenor):
                wacslq.append(f"BloombergRequest('{IRS['OISquery'].iloc[0]} Index','PX_LAST')")

        terms = []
        for tenor in tenors:
            if re.search("Y", tenor):
                terms.append(int(re.sub("Y", "", tenor)) * 12)
            elif re.search("M", tenor):
                terms.append(round(int(re.sub("M", "", tenor)) / 1, 6))
            elif re.search("D", tenor):
                terms.append(round(int(re.sub("D", "", tenor)) / 28, 6))
        rate_tenors = [tenors[i] for i in range(len(tenors)) if not is_swap[i]]
        swap_tenors = [tenors[i] for i in range(len(tenors)) if is_swap[i]]
        wacslq_swap = [wacslq[i] for i in range(len(wacslq)) if is_swap[i]]
        wacslq_rate = [wacslq[i] for i in range(len(wacslq)) if not is_swap[i]]

        return {
            'tenors': tenors,
            'terms': terms,
            'is_swap': is_swap,
            'query': wacslq,
            'filtering': "standard",
            'filling': "wacsl",
            'rate_tenors': rate_tenors,
            'wacslq_rate': wacslq_rate,
            'swap_tenors': swap_tenors,
            'wacslq_swap': wacslq_swap
        }
    else:
        if len(pillars) > 0:  # for any irregular tickers and queries wanted by yc - set these up in pillars.csv
            tenors = pillars['tenor'].astype(
                str).tolist()  # ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
            is_swap = pillars['type'].str.contains("SW").astype(bool).tolist()
            is_rate = pillars['type'].str.contains("ra").astype(bool).tolist()
            wacslq = ["BloombergRequest('" + ticker + "','PX_LAST')" for ticker in pillars['ticker'].iloc[0]]

            terms = []
            for i in range(len(pillars)):
                pillar = pillars.iloc[i]
                if not pd.isna(pillar['year']):
                    ter = int(pillar['year']) * 12
                elif not pd.isna(pillar['month']):
                    ter = int(pillar['month'])
                elif not pd.isna(pillar['days']):
                    ter = round(int(pillar['days']) / (365 / 12), 0)
                terms.append(ter)

            rate_tenors = pillars['tenor'][is_rate].tolist()
            is_cumul = pillars['cumul'][is_rate].tolist()
            swap_tenors = pillars['tenor'][is_swap].tolist()
            wacslq_swap = [wacslq[i] for i in range(len(wacslq)) if is_swap[i]]
            wacslq_rate = [wacslq[i] for i in range(len(wacslq)) if is_rate[i]]
        else:
            tenors = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]  # all tenors
            is_swap = [False] * 4 + [True] * 10
            wacslq = []
            for tenor in tenors:
                if re.search("Y", tenor):
                    tenorcode = re.sub("Y", "", tenor)
                    wacslq.append(f"BloombergRequest('{get_BBIR_swap_code(IRS['currency'].iloc[0], tenorcode)}','PX_LAST')")
                elif re.search("M", tenor):
                    tenorp = re.sub("M", "", tenor)
                    oisq = f"{re.sub(' ', tenorp, IRS['OISquery'].iloc[0])} Index"
                    wacslq.append(f"TSFilled(BloombergRequest('{oisq}','PX_LAST'),8,-5)")  # 5 days fill
                elif re.search("D", tenor):
                    wacslq.append(f"TSFilled(BloombergRequest('{IRS['OISquery'].iloc[0]} Index','PX_LAST'),8,-5)")

            terms = []
            for tenor in tenors:
                if re.search("Y", tenor):
                    terms.append(int(re.sub("Y", "", tenor)) * 12)
                elif re.search("M", tenor):
                    terms.append(round(int(re.sub("M", "", tenor)) / 1, 6))
                elif re.search("D", tenor):
                    terms.append(round(int(re.sub("D", "", tenor)) / 28, 6))

        rate_tenors = [tenors[i] for i in range(len(tenors)) if not is_swap[i]]
        swap_tenors = [tenors[i] for i in range(len(tenors)) if is_swap[i]]
        wacslq_swap = [wacslq[i] for i in range(len(wacslq)) if is_swap[i]]
        wacslq_rate = [wacslq[i] for i in range(len(wacslq)) if not is_swap[i]]

        return {
            'tenors': tenors,
            'terms': terms,
            'is_swap': is_swap,
            'query': wacslq,
            'filtering': "optimistic",
            'filling': "none",
            'rate_tenors': rate_tenors,
            'wacslq_rate': wacslq_rate,
            'rate_cumul': is_cumul,
            'swap_tenors': swap_tenors,
            'wacslq_swap': wacslq_swap
        }

def interpolate_row(row):
    """
    usage is df_interpolated = df.apply(lambda row: interpolate_row(row.values), axis=1, result_type='expand')
    :param row:
    :return:
    """
    x = np.array([i for i in range(len(row)) if not np.isnan(row[i])])  # Indices of non-NA values
    y = np.array([row[i] for i in range(len(row)) if not np.isnan(row[i])])  # Non-NA values
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')  # Interpolation function
    return [f(i) if np.isnan(row[i]) else row[i] for i in range(len(row))]  # Interpolate