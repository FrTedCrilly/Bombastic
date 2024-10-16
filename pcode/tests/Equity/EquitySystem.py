# Equity Futures
# First day of the rest of your life 17 Feb 2024
from libs.io import  getFolioFile, getQuickData
from libs.utils_setup import get_last_business_day, create_unique_date_folder
from datetime import datetime
import pickle
import logging
import os

# Configure logging
logging.basicConfig(filename='my_app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Example log message
logging.info('Application started')
# load assets
# Could we make this a class with each function to do something, then inherit the neccessary attributes for each specific system?
baseDir = r'C:\Users\edgil\Documents\OUTDIR'
folio = getFolioFile("Equity")
# of course, make these dynamic and callabble.
assets = folio["Asset"]
asset = "^GSCI"
asset_no = 1

start_date = datetime.strptime("1992-01-01", '%Y-%m-%d')
finish  = get_last_business_day()

OHLC = getQuickData(assets, start_date.strftime('%Y-%m-%d'), finish)

outmain = create_unique_date_folder(baseDir, "EquitySystem")

# For audit purposes, save original data which created the values.
with open(os.path.join(outmain, asset + 'data.pkl') , 'wb') as pickle_file:
    pickle.dump(OHLC, pickle_file)
logging.info('Saved price data with pickle')
# next step up the asset params , such as regionforLag
regionForLag = folio["RegionForLag"][asset_no]
Country = folio["Country"][asset_no]
# set up the total return index and the close and actcloses.
close = OHLC["Close"]
actClose = OHLC["Adj Close"]
Volume = OHLC["Volume"]
# Start to call the various components to run the signals.
# first we start with Trend



