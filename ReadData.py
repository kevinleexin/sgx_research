import os
import pandas as pd


def extract_contract(filename):
    filename_piece = filename.split("-")
    return filename_piece[3]


def calc_ohlc(time_interval, raw_df):
    """
    This func is to statistic historic base info like as: open, high, low, close
    Output different time interval for OHLC, for example, 30second, 1min, 5min, 30min
    time_interval: 0.5 = 30second, 1 = 1min
    :return:
    """
    raw_df['t'] = pd.to_datetime(raw_df['t'])
    raw_df.set_index('t', inplace=True)

    df = raw_df.dropna(subset=['dp0'])

    ohlc_dict = {
        'dp0': ['first', 'max', 'min', 'last'],
    }

    bar_df = df.resample(time_interval).agg(ohlc_dict)

    bar_df.columns = ['open', 'high', 'low', 'close']

    return bar_df


if __name__ == "__main__":

    data_dir = "data"

    interval = "1min"

    dataframes = []

    for date_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, date_folder)
        #print(date_folder)
        # where to record date of folder
        if os.path.isdir(folder_path):
        
            for filename in os.listdir(folder_path):
                if filename.endswith(".parquet"):

                    file_path = os.path.join(folder_path, filename)
                
                    df = pd.read_parquet(file_path)
                    bar_1min_df = calc_ohlc(interval, df)

                    bar_1min_df.to_csv("./data/" + date_folder + "/" + extract_contract(filename)
                                       + "-" + interval + ".csv")