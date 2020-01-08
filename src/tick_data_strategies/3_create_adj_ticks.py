# https://help.yahoo.com/kb/SLN28256.html

import os
import numpy as np
import pandas as pd
import datetime as dt

from multiprocessing import Pool
my_dir = os.getcwd()

ticks_folder = os.path.join(my_dir, "data/4_Ticks")
dividends_folder = os.path.join(my_dir, "data/3_Dividends")

adj_ticks_folder = os.path.join(my_dir, "data/5_AdjTicks")

if os.path.basename(adj_ticks_folder) not in os.listdir(os.path.dirname(adj_ticks_folder)):
    os.mkdir(adj_ticks_folder)

keys = [key[:4] for key in os.listdir(ticks_folder) if not key.startswith(".")]
print(keys)

for key in keys:
    div_keys = [key[:4] for key in os.listdir(dividends_folder) if not key.startswith(".")]
    if key not in div_keys:
        print(key, 'passed')
        pass
    else:
        print('processing ', key)
        dividends = pd.read_csv(os.path.join(dividends_folder, (key + ".ME.csv")), index_col=0, parse_dates=True)
        dividends.sort_index(inplace=True)

        ticks_file = [f for f in os.listdir(ticks_folder) if key in f][0]
        ticks = pd.read_parquet(os.path.join(ticks_folder, ticks_file), engine='fastparquet')
        ticks.index = ticks.date_time

        dividends_due = dividends[((dividends.index < ticks.index[-1]) & (dividends.index > ticks.index[0]))]
        ss = pd.Series(ticks.index).searchsorted(dividends_due.index)
        for_backward_roll = ticks.iloc[(ss - 1)]
        # print(for_backward_roll)

        dividends_due["Coef"] = 0.
        for i in range(len(dividends_due)):
            dividends_due.iloc[i, 1] = 1 - dividends_due.iloc[i, 0] / for_backward_roll.price.iloc[i]

        adj_ticks = ticks.copy()
        for i in range(len(dividends_due)):
            adj_ticks.price.loc[adj_ticks.index < dividends_due.index[i]] *= dividends_due.iloc[i, 1]

        t1 = dt.datetime.now()

        adj_ticks_file = os.path.join(adj_ticks_folder, ticks_file)

        print("saving parquet")
        adj_ticks.reset_index(drop=True).to_parquet(adj_ticks_file, compression='GZIP', engine='fastparquet')
        t2 = dt.datetime.now()
        print(t2 - t1)
