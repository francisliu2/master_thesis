import matplotlib.pyplot as plt
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like  # path fo is_list_like has been changed
from pandas_datareader import data
import numpy as np
import h5py
import datetime
import os, sys

sys.path.append("../../")
from src.data import get_raw_stock_price_data


def check_data_avaliability(h5_file, start_date, end_date, freq, sort_by_date_ascending):
    start_date_h5 = datetime.datetime.strptime(pd.read_hdf(h5_file, 'start_date').iloc[-1], '%Y-%m-%d')
    end_date_h5 = datetime.datetime.strptime(pd.read_hdf(h5_file, 'end_date').iloc[-1], '%Y-%m-%d')
    freq_h5 = pd.read_hdf(h5_file, key='freq')
    sort_by_date_ascending_h5 = pd.read_hdf(h5_file, key='sort_by_date_ascending')
    error = False
    if start_date_h5 > datetime.datetime.strptime(start_date, '%Y-%m-%d'):
        print('start date does not match')
        error = True

    if end_date_h5 < datetime.datetime.strptime(end_date, '%Y-%m-%d'):
        print('end date does not match')
        error = True

    if freq_h5.item() != freq:
        print('freq does not match')
        error = True

    if sort_by_date_ascending_h5.item() != sort_by_date_ascending:
        print('sort_by_date_ascending does not match')
        error = True

    if error:
        return False
    else:
        return True


def plot_result_np(result_np):
    f = plt.figure(figsize=(16, 9))
    plt.tight_layout()

    f.add_subplot(221)
    plt.plot(result_np[0, :, :])
    plt.title('open')

    f.add_subplot(222)
    plt.plot(result_np[1, :, :])
    plt.title('high')

    f.add_subplot(223)
    plt.plot(result_np[2, :, :])
    plt.title('low')

    f.add_subplot(224)
    plt.plot(result_np[3, :, :])
    plt.title('close')

    plt.show()


def data_preprocessing_1(ticker_list_input=None,
                         path='../../data/raw/',
                         freq = 'daily',  # 'daily', 'weekly', or 'monthly'
                         start_date = '2007-10-01',
                         end_date = '2017-09-29',
                         sort_by_date_ascending = True):
    if ticker_list_input is None:
        ticker_list_input = ['AAPL', 'PG', 'UL', 'INTC', 'NVDA',
                             'QCOM', 'MSFT', 'EBAY', 'CSCO', 'BIDU',
                             'GOOGL']
    ticker_in_raw = os.listdir(path)

    # check if stock data exist
    ticker_list = []
    for ticker in ticker_list_input:
        print("working on", ticker)
        if ticker + '.h5' in ticker_in_raw:
            if check_data_avaliability(path + ticker + '.h5', start_date, end_date, freq,
                                       sort_by_date_ascending) == True:
                ticker_list.append(path + ticker + '.h5')
            else:
                get_raw_stock_price_data.update_price_data(path, ticker, start_date, end_date)
        else:
            print(ticker, "is not in raw data folder")
            get_raw_stock_price_data.download_price_data(path, ticker, freq, start_date, end_date,
                                                         sort_by_date_ascending)
            ticker_list.append(path + ticker + '.h5')

    datelist = pd.Series(pd.date_range(start_date, end_date)).apply(lambda x: str(x)[:10])
    date_pd = pd.DataFrame({'formatted_date': datelist})

    open_df = date_pd
    high_df = date_pd
    low_df = date_pd
    close_df = date_pd

    for ticker_path in ticker_list:
        if check_data_avaliability(ticker_path, start_date, end_date, freq, sort_by_date_ascending) == False:
            print(ticker_path, 'is not complete for this request')
        ticker_name = ticker_path.replace('.h5', '').replace(path, '')
        price_df = pd.read_hdf(ticker_path, key='df')

        open_df = pd.merge(open_df, price_df[['open', 'formatted_date']], on='formatted_date', how='left')
        open_df.rename(columns={'open': ticker_name}, inplace=True)

        high_df = pd.merge(high_df, price_df[['high', 'formatted_date']], on='formatted_date', how='left')
        high_df.rename(columns={'high': ticker_name}, inplace=True)

        low_df = pd.merge(low_df, price_df[['low', 'formatted_date']], on='formatted_date', how='left')
        low_df.rename(columns={'low': ticker_name}, inplace=True)

        close_df = pd.merge(close_df, price_df[['close', 'formatted_date']], on='formatted_date', how='left')
        close_df.rename(columns={'close': ticker_name}, inplace=True)

    open_df.fillna(0, inplace=True)
    high_df.fillna(0, inplace=True)
    low_df.fillna(0, inplace=True)
    close_df.fillna(0, inplace=True)

    # Remove non-trading days, which is the day that none of the stock/assets are traded that day
    open_df = open_df[open_df.sum(axis=1) != 0]
    high_df = high_df[high_df.sum(axis=1) != 0]
    low_df = low_df[low_df.sum(axis=1) != 0]
    close_df = close_df[close_df.sum(axis=1) != 0]

    open_np = np.array(open_df.iloc[:, 1:])
    high_np = np.array(high_df.iloc[:, 1:])
    low_np = np.array(low_df.iloc[:, 1:])
    close_np = np.array(close_df.iloc[:, 1:])
    result_np = np.stack([open_np, high_np, low_np, close_np], axis=0)

    # print(np.mean(result_np[0, :, :] == open_np),
    #       np.mean(result_np[1, :, :] == high_np),
    #       np.mean(result_np[2, :, :] == low_np),
    #       np.mean(result_np[3, :, :] == close_np))

    return result_np



if __name__ == "__main__":
    path = '../../data/raw/'
    freq = 'daily'  # 'daily', 'weekly', or 'monthly'
    start_date = '2007-10-01'
    end_date = '2017-09-29'
    sort_by_date_ascending = True


    result_np =  data_preprocessing_1()
    print('The resulting data has shape', str(result_np.shape) + ',',
          'and has', result_np.size - np.count_nonzero(result_np), 'missing data.')



    np.save('../../data/processed/20181004_1.npy', result_np, allow_pickle=True, fix_imports=True)
