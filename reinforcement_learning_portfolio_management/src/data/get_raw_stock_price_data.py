import pandas as pd
import h5py
import datetime
import os
from yahoofinancials import YahooFinancials

def clean_stock_data(stock_data_list):
    new_list = []
    for rec in stock_data_list:
        if 'type' not in rec.keys():
            new_list.append(rec)
    return new_list

def download_price_data(path, ticker, freq, start_date, end_date, sort_by_date_ascending):
    yf = YahooFinancials(ticker)
    price_data = clean_stock_data(yf.get_historical_stock_data(start_date, end_date, freq)[ticker]['prices'])
    pd_price_data = pd.DataFrame(price_data)
    print("The downloaded price data has shape" + str(pd_price_data.shape))

    retrieved_at = updated_at = str(datetime.datetime.now())[:10]

    if sort_by_date_ascending:
        pd_price_data.sort_values('date', ascending=True, inplace=True)
        pd_price_data.reset_index(inplace=True, drop=True)

    file = path + ticker + '.h5'
    pd_price_data.to_hdf(file, key='df', mode='w')
    print(str(ticker) + "'s" + " price data is saved.")

    pd.Series(ticker).to_hdf(file, key='ticker')
    pd.Series(freq).to_hdf(file, key='freq')
    pd.Series(pd_price_data['formatted_date'].iloc[0]).to_hdf(file, key='start_date')
    pd.Series(pd_price_data['formatted_date'].iloc[-1]).to_hdf(file, key='end_date')
    pd.Series(sort_by_date_ascending).to_hdf(file, key='sort_by_date_ascending')
    pd.Series(start_date).to_hdf(file, key='start_date')
    pd.Series(end_date).to_hdf(file, key='end_date')
    pd.Series(retrieved_at).to_hdf(file, key='retrieved_at')
    pd.Series(updated_at).to_hdf(file, key='updated_at')


def update_price_data(path, ticker, _start_date='2009-10-01', _end_date=str(datetime.datetime.now())[:10]):
    file = path + ticker + '.h5'
    start_date_DT = datetime.datetime.strptime(pd.read_hdf(file, 'start_date').iloc[-1], '%Y-%m-%d')
    end_date_DT = datetime.datetime.strptime(pd.read_hdf(file, 'end_date').iloc[-1], '%Y-%m-%d')
    _start_date_DT = datetime.datetime.strptime(_start_date, '%Y-%m-%d')
    _end_date_DT = datetime.datetime.strptime(_end_date, '%Y-%m-%d')

    retrieved_at = pd.read_hdf(file, 'retrieved_at')
    updated_at = str(datetime.datetime.now())[:10]
    sort_by_date_ascending = pd.read_hdf(file, 'sort_by_date_ascending').iloc[-1]
    freq = pd.read_hdf(file, 'freq').iloc[-1]


    df = pd.read_hdf(file, 'df')

    if _start_date_DT < start_date_DT:
        yf = YahooFinancials(ticker)
        END = str(start_date_DT - datetime.timedelta(days=1))[:10]
        price_data = clean_stock_data(yf.get_historical_stock_data(_start_date, END, freq)[ticker]['prices'])
        pd_price_data = pd.DataFrame(price_data)
        print("The downloaded price data has shape" + str(pd_price_data.shape))
        df = df.append(pd_price_data, ignore_index=True)

    if _end_date_DT > end_date_DT:
        yf = YahooFinancials(ticker)
        START = str(end_date_DT + datetime.timedelta(days=1))[:10]
        price_data = clean_stock_data(yf.get_historical_stock_data(START, _end_date, freq)[ticker]['prices'])
        pd_price_data = pd.DataFrame(price_data)
        print("The downloaded price data has shape" + str(pd_price_data.shape))
        df = df.append(pd_price_data, ignore_index=True)

    if sort_by_date_ascending:
        df.sort_values('date', ascending=True, inplace=True)
        df.reset_index(inplace=True, drop=True)

    df.to_hdf(file, key='df', mode='w')
    pd.Series(df['formatted_date'].iloc[0]).to_hdf(file, key='start_date')
    pd.Series(df['formatted_date'].iloc[-1]).to_hdf(file, key='end_date')
    pd.Series(freq).to_hdf(file, key='freq')
    pd.Series(sort_by_date_ascending).to_hdf(file, key='sort_by_date_ascending')
    pd.Series(retrieved_at).to_hdf(file, key='retrieved_at')
    pd.Series(updated_at).to_hdf(file, key='updated_at')


    updated_at = str(datetime.datetime.now())[:10]
    pd.Series(updated_at).to_hdf(file, key='updated_at')
    print(str(ticker) + "'s" + " price data is updated")

if __name__ == "__main__":
    ticker_list = ['AAPL', 'PG', 'UL', 'FB', 'NVDA',
                   'QCOM', 'TSLA', 'EBAY', 'CSCO', 'BIDU', 'GOOGL'] # In future, there will be a config file. NASDAQ-100 https://en.wikipedia.org/wiki/NASDAQ-100
    # Arguements
    path = '../../data/raw/'
    freq = 'daily'  # 'daily', 'weekly', or 'monthly'
    start_date = '2007-10-01'
    end_date = '2017-10-01'
    sort_by_date_ascending = True

    price_data_in_raw = os.listdir(path)
    for TICKER in ticker_list:
        if TICKER + '.h5' not in price_data_in_raw:
           download_price_data(path, TICKER, freq = freq, start_date=start_date, end_date=end_date, sort_by_date_ascending=True)
        else:
            update_price_data(path, TICKER, start_date, end_date)



