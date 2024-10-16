import numpy as np
import pandas as pd

from libs.Backtester.Backtester import Backtest
from libs.Backtester.FuturesBacktester import FuturesBT
from libs.Backtester.FXBacktester import FXBT

from matplotlib import pyplot as plt

# for BT testing
Esig_folder = r"C:\Users\edgil\Documents\OUTDIR\EquitySystem\EquitySystem_2024-03-16"

strats = ['BBands_20_window_1_sd_num_True_reversal', 'BBands_30_window_1_sd_num_True_reversal', 'BBands_40_window_1_sd_num_True_reversal']
systemname, start_date, end_date = "Equity", "2000-01-01", "2021-01-01"

# test the Class methods
BT = Backtest("Equity", sig_folder=sig_folder, strats = strats)
# test the BT funciton
FBT = FuturesBT("Equity", sig_folder=sig_folder, strats = strats,start_date=start_date,end_date = end_date)
# you need to run get SW first, assest specific.
FBT.getSW("SP")
res = FBT.RunPnL(["SP"])
## Test FX function
systemname, start_date, end_date = "FX", "2000-01-01", "2021-01-01"
strats = ['BBands_20_window_1_sd_num_True_reversal', 'BBands_30_window_1_sd_num_True_reversal', 'BBands_40_window_1_sd_num_True_reversal']
FXsig_folder = r"C:\Users\edgil\Documents\OUTDIR\FXSystem\FXSystem_2024-06-17"

FXBT = FXBT("FX", sig_folder=FXsig_folder, strats = strats,start_date=start_date,end_date = end_date)
res = FXBT.getSW("EURUSD")
pnls = FXBT.RunPnL(["EURUSD"])
FXBT.PlotPdf(pnls, FXsig_folder + r'\pdf_res.pdf')


strats = getSigNames(sig_folder) # , ["SP"], columns=None, name_like=None, name_notlike=None)

strat_dict = parse_signals1(strats)
for strat in strats.keys():
    for signame in strats[strat]:
    strat_dict[signame] = 1
Folio = {"Folio": strat_dict}
portdf, ohlc_data = getAssetBT(systemname, "SP")
outp = create_unique_date_folder(sig_folder, folderN)
portdf['rel_asset_weight'] = portdf['weight'] / portdf['weight'].sum()
assetCount = getAssetCount(ohlc_data)
strat_name = "BBands_40_window_7_sd_num_True_reversal"
weight = Folio["Folio"][strat_name]
sig = getSigs(sig_folder, asset_names = "SP", columns = strat_name)
OHLC, instrument_type,assetCount, costfactor,notional =  ohlc_data["SP"], "Future", 1, 1, 100000000
j = getPnLFutures(asset, portdf, OHLC, sig, start_date, end_date,notional, instrument_type,assetCount, costfactor)
###########################################################################
outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
front_df = pd.read_csv(outDir + "SP_front.csv", index_col=0, parse_dates=True)
back_df = pd.read_csv(outDir + "SP_back.csv", index_col=0, parse_dates=True)
# Define parameters
volatility = 0.10  # 10%
risk_free_rate = 0.00  # Assuming 0% for simplification
sharpe_ratios = [0.75] #[0.5, 0.4, 1, 1.5, 0, -0.4]

# Generate date range for business days from 2002 to 2024
dates = pd.bdate_range(start='2002-01-01', end='2024-12-31')

# Prepare a DataFrame to hold the return series
return_series = pd.DataFrame(index=dates)

# Generate return series for each Sharpe Ratio
for sharpe_ratio in sharpe_ratios:
    expected_return = (sharpe_ratio * volatility)/260
    dvol = volatility/np.sqrt(260)
    # Generate random returns around the expected return with specified volatility
    # Assuming normally distributed returns
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(loc=expected_return, scale=dvol, size=len(dates))

    # Adding the series to the DataFrame
    return_series[f'SR_{sharpe_ratio}'] = returns

# Display the first few rows of the DataFrame
print(return_series.head())
pnls = return_series.cumsum()*100 + 100

pnls.to_csv(outDir + "pnls.csv")

pnls = pd.read_csv(outDir + "pnls.csv", index_col=0, parse_dates=True)
#######################################################################
# test Macro Data
#######################################################################
import numpy as np
import pandas as pd
from pcode.BT.Signals.Macro.MacroSignal import MacroSignal
from libs.io import getAssetBT
from libs.io import WriteSig
from libs.utils_setup import create_unique_date_folder
from libs.Backtester.bt_utils import getSigNames, getSigs
baseDir = r'C:\Users\edgil\Documents\OUTDIR'
outmain = create_unique_date_folder(baseDir, "EquitySystem")
fileIO = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
US_ESI = pd.read_csv(fileIO + "ESI_US.csv", index_col=0, parse_dates=True)
macroData = US_ESI['ESI']
tradeSign, start_date , end_date, regionForLag = "+ve", "2010-01-01" , "2020-03-16", "Europe/APAC"
asset = "SP"
portdf, ohlc_data = getAssetBT("Equity", asset = "SP")
macroSig = MacroSignal(varData = macroData, OHLC = ohlc_data[asset]['Close'], tradeSign = tradeSign,
                       start_date = start_date ,  end_date= end_date, uselag=None, regionForLag = regionForLag,
                       assetName = asset, outDir = outmain)

# run a data tran test
macroSig.dataTran(chg=10, short=None, long=None, MA=None, Zscore=False, rmMean=True, pctchg=False,
                 useLog=False, cheat=False)
macroSig.getSignal(pEnter = 0.5, pExit = 0.5, doZ = True, doQ = True, Zwin = 250, Qwin = 250, Zexpand = True,
                  corChg = 10, corEntry = 0.15, corExit= 0.05, corWindow = 250 , corSign = "+ve",
                  Qexpand = True)
j = macroSig.analyze_signal()
kk = macroSig.AnalyseFwdRet()
macroSig.macroParamScan()
outmain = r"C:\Users\edgil\Documents\OUTDIR\EquitySystem\EquitySystem_2024-08-19"
strats = getSigNames(outmain, name_like= "C10")[:5]
sigs = getSigs(outmain, asset_names = "SP", columns = ['C10_ESI_Z750Q750_P5_P10'])
from libs.Backtester.FuturesBacktester import FuturesBT
FBT = FuturesBT( systemname = "Equity", sig_folder = outmain, strats = strats, start_date= "1992-01-01", end_date = "2020-02-01")
snw = FBT.getSW(asset = "SP")
pnls = FBT.RunPnL(["SP"])
pnls['C10_ESI_Z750Q750_P5_P10']['C10_ESI_Z750Q750_P5_P10_SP'][0].plot()
from matplotlib import pyplot as plt


