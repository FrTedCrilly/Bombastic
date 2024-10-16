import time
import numpy as np
import pandas as pd
import c_utils

df = pd.DataFrame({
    'A': np.random.randn(10000),
    'B': np.random.randn(10000)
})

start_time = time.time()
processed_df_z = c_utils.apply_zscore(df.to_numpy(), Zwin=10, Zexpand=False, rmMean=True, cheat=True)
processed_df_z = pd.DataFrame(processed_df_z, columns=df.columns)
cython_time_z = time.time() - start_time

start_time = time.time()
processed_df_q = c_utils.apply_quantile(df.to_numpy(), Qwin=10, Qexpand=False)
processed_df_q = pd.DataFrame(processed_df_q, columns=df.columns)
cython_time_q = time.time() - start_time

print(f"Cython Z-score processing time: {cython_time_z:.4f} seconds")
print(f"Cython Quantile processing time: {cython_time_q:.4f} seconds")