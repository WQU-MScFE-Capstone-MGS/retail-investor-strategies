import os
import numpy as np
import pandas as pd
import datetime as dt

from multiprocessing import Pool

my_dir = os.getcwd()
raw_tick_dir = os.path.join(my_dir, "data/1_RawTicks")
target_folder = os.path.join(my_dir, "data/4_Ticks")

if os.path.basename(target_folder) not in os.listdir(os.path.dirname(target_folder)):
    os.mkdir(target_folder)


def check_breaks(file_names):
    end = dt.datetime.strptime(file_names[0][-10:-4], "%y%m%d")
    breaches = []
    for i in file_names[1:]:
        expected_previous_end = dt.datetime.strptime(i[5:11], "%y%m%d") - dt.timedelta(days=1)
        if not end == expected_previous_end:
            breaches.append([end, expected_previous_end])
        end = dt.datetime.strptime(i[-10:-4], "%y%m%d")

    if breaches:
        print("Breaches: ", breaches)
        return False
    return True


def get_csv(file):
    global source_folder
    file_path = os.path.join(source_folder, file)

    df_temp = pd.read_csv(file_path, sep=';')

    if '<DATE>' not in df_temp.columns:
        df_temp = pd.read_csv(file_path, sep=',')

    assert '<DATE>' in df_temp.columns, "No DATE in columns in " + file

    b = np.arange(len(df_temp)) * 10 ** -6
    delta = np.array(list(map(lambda x: "{:.6f}".format(x)[1:], b)))
    date_time = df_temp['<DATE>'].apply(str) + df_temp['<TIME>'].apply(str) + delta
    dt_temp = pd.to_datetime(date_time, format='%Y%m%d%H%M%S.%f', infer_datetime_format=True)

    # to deal with repeating datetime values
    df_temp['date_time'] = dt_temp
    df_temp_new = df_temp[['date_time', '<LAST>', '<VOL>']]
    df_temp_new.columns = ['date_time', 'price', 'volume']

    return df_temp_new


if __name__ == '__main__':

    keys = [key for key in os.listdir(raw_tick_dir) if not key.startswith(".")]
    print(keys)

    for key in keys:
        print(key)

        source_folder = os.path.join(raw_tick_dir, key)

        files = [file for file in os.listdir(source_folder) if file.startswith(key + '_') and not file.endswith("part")]
        files.sort()

        assert check_breaks(files), "List of files is not uninterrupted"

        t0 = dt.datetime.now()

        with Pool() as pool:  # start 4 worker processes
            it = pool.imap(get_csv, files)
            p = [df for df in it]
            df_new = pd.concat(p, sort=False)

        t1 = dt.datetime.now()
        print(t1 - t0)
        print("Amount of ticks: ", df_new.shape)

        file_name = os.path.join(target_folder, f"{key}_{files[0][5:11]}_{files[-1][-10:-4]}.gzip")

        print("saving parquet")
        df_new.to_parquet(file_name, compression='GZIP', engine='fastparquet')
        t2 = dt.datetime.now()
        print(t2 - t1)
