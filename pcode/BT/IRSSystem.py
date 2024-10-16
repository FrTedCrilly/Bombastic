from libs.irs_utils import *
import pickle
import logging
import os
from libs.io import getFolioFile
# Configure logging
logging.basicConfig(filename='my_app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Example log message
logging.info('Application started')
asset = "PZSW5"
# load assets
configDir = r'C:\Users\edgil\Documents\git_working\Research\config\asset_config'
# Could we make this a class with each function to do something, then inherit the neccessary attributes for each specific system?
baseDir = r'C:\Users\edgil\Documents\OUTDIR'
folio = getFolioFile("IRS", "IRS_EA_data")
foliodf = getIRS_def(asset, folio)
foliodf['datafolder'] = configDir + "IRS_EA_data.csv"
IRS = foliodf
start = "2005-01-03"
finish = "2024-03-04"
yc = getYC(start, finish, IRS, saveYC = False, continous = False, curvemax = 120)
trade_date = "2005-01-10"
ohlc = get_IRS_OHLC(start, finish, IRS)
#trade_date = "2021-04-21"
#newdates = get_swap_dates(trade_date, yc['zc'],tenor=IRS['tenor'].iloc[0], settle=IRS['SettleOff'].iloc[0], IMM=None)
#swap5 = get_swap(newdates,yc, IRS,coupon = 0, notional = 1,imm_align=False,at_par = True, continous= False)
#entry = swap_valuation(swap5,yc,IRS,value_date=swap5['dates']['tradeDate'])
#ex_date = swap5['dates']['tradeDate'] +pd.DateOffset(days = 1)
#exit = swap_valuation(swap5,yc,IRS,value_date=ex_date)