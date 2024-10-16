from libs.io import WriteSig
import pandas as pd
import numpy as np

class ParamScan:
    def __init__(self, data, output_dir, asset_name, freq = True):
        self.data = data
        self.output_dir = output_dir
        self.asset_name = asset_name
        self.freq = freq
        # the self params allow me to have global VARIABLES within the class and stop code duplication.
        if self.freq:
            self.conversionF = 1
        elif self.freq == "Weekly" or self.freq == "W":
            self.conversionF = 5
        elif self.freq == "Monthly" or self.freq == "M":
            self.conversionF = 22

    def module_scan(self, func, param_grid, func_name):
        from itertools import product
        zipped_lists = list(zip(*param_grid.values()))
        # Generate all combinations of parameter values
        for params in zipped_lists:
            signal_data = func(*params)  # Assuming func returns a Series or DataFrame of signals
            sigPure = signal_data['sigPure']
            # Prepare signal data for saving: ensure it has a 'Date' column and a signal column
            if not isinstance(sigPure, pd.Series):
                sigPure['Date'] = signal_data.index
            else:
                sigPure = sigPure.to_frame()

            # Generate a descriptive name for this batch of signals
            param_name = list(zip(params,param_grid.keys()))
            signal_name = func_name + "_" + "_".join(f"{value}_{name}" for value, name in param_name)
            #signal_name = f"{func_name}_{'_'.join([f'{k}{int(v)}' for k, v in params.items()])}"
            sigPure.columns = [signal_name]  # Rename columns for clarity
            # Use WriteSig to save/update the signal data
            WriteSig(self.output_dir, self.asset_name, sigPure)

    def ma_scan(self, trend_system_func):
        # Define parameter grid for moving average scan
        # clearly assumes daily data.
        short_window = [5, 10, 15, 29, 5,  20, 30, 120, 49,  150, 220]
        long_window =  [20, 40, 60, 90, 50, 60, 90, 180, 199, 300, 600]
        short_window = [int(round(i/self.conversionF,0)) for i in short_window]
        long_window = [int(round(i/self.conversionF,0)) for i in long_window]
        param_grid = {
            'short_window': short_window,
            "long_window": long_window
        }
        self.module_scan(trend_system_func, param_grid, 'MA_Xover')


    def ma_breakout(self, trend_system_func):
        # Define parameter grid for moving average scan
        # clearly assumes daily data.

        short_window = [5, 10, 15, 29, 5, 20, 30, 120, 49, 150, 220]
        short_window = [int(round(i / self.conversionF, 0)) for i in short_window]
        param_grid = {
            'short_window': short_window,
        }
        self.module_scan(trend_system_func, param_grid, 'breakout')

    def HP_ADX(self, trend_system_func):
        # Define parameter grid for moving average scan
        # clearly assumes daily data.
        short_window = [14,       15,     29,     49]
        lamb_window = [16000, 100000, 150000, 200000]
        short_window = [int(round(i / self.conversionF, 0)) for i in short_window]
        param_grid = {
            'short_window': short_window,
            "long_window": lamb_window
        }
        self.module_scan(trend_system_func, param_grid, 'HPADX')

    def BBands(self, trend_system_func):

        sd_num = [1.5, 2, 2.25, 2.5, 3, 4, 5, 6, 7, 8]
        window = [20, 30, 40, 50, 60]
        sd_nums = np.repeat(sd_num, len(window))
        windows = [x for xs in [window]*len(sd_num) for x in xs]
        reversal = [True]*len(windows)
        momomentum = [False]*len(windows)
        param_grid = {
            "window": [int(round(i / self.conversionF, 0)) for i in windows],
            'sd_num': [int(i) for i in sd_nums],
            "reversal" : reversal
        }
        param_grid_mom = {
            "window": [int(round(i / self.conversionF, 0)) for i in windows],
            'sd_num': [int(i) for i in sd_nums],
            "reversal" : momomentum
        }
        self.module_scan(trend_system_func, param_grid, 'BBands')
        # need to run the False case too.
        self.module_scan(trend_system_func, param_grid_mom, 'BBands')




