import numpy as np
import pandas as pd
import requests 
import json
import time
import pickle

def date2unixtime(date):
    return round(time.mktime(time.strptime(date, '%Y-%m-%d')))
# time.struct_time.index

def get_coin_data(name):
    coin = name
    start = date2unixtime('2014-01-01')
    end = date2unixtime('2018-12-31')
    period = '1800' #30 mins interval
    url = ('https://poloniex.com/public?command=returnChartData&'+
           'currencyPair=BTC_%s&'%(coin)+
           'start=%s&'%(start)+
           'end=%s&'%(end)+
           '&period=%s'%(period))
    c = requests.get(url).content #to get coins data
    j = json.loads(c.decode('utf-8')) #bytes to json 
    df = pd.DataFrame(j)
    return df

coin_list = ['ETH', #Ethereum
             'XRP', #XRP
             'BCH', #Bitcoin Cash
             'EOS', #EOS
             'LTC', #Litecoin
             'ZEC', #Zcash
             'XMR', #Monero
             'ETC', #Ethereum Classic
             'DOGE', #Doge
             'NEOS', #NeosCoin
             'OMG', #OmiseGO
             'LSK'] #Lisk
#no stellar
no_data = ['XLM', 
'MIOTA',  
'NEO',
'USDT', #Tether USDT:USD = 1:1
'ADA', #Cardano
'TRX'] #TRON

temp = []
n=0
for coin in coin_list:
    temp.append(get_coin_data(coin))
#     time.sleep(2)
    n+=1
    print(n)
    
# pickle.dump(temp, open( "temp_coin_data.p", "wb" ) )
# temp = pickle.load(open('temp_coin_data.p', 'rb'))
for table in temp:
    table.set_index('date', inplace=True)
    
for n,j in enumerate(coin_list):
    temp[n].columns = ([temp[n].columns[i] + '_' + j for i in range(len(temp[0].columns))])
    
L = []
for layer in ['open', 'high', 'low']:
    filter_col = [col for col in result if col.startswith(layer)]
    L.append(np.array(result[filter_col].tail(17520))) #get the data of last year 365*24*2 , 30 mins interval
#     result.tail(17520) from 1503234000 to 1534768200 
#     from GMT: Sunday, August 20, 2017 1:00:00 PM 
#     to GMT: Monday, August 20, 2018 12:30:00 PM


data_to_go = np.stack(L)

data_to_go.shape
# pickle.dump(data_to_go, open( "coin_chart_1.p", "wb" ) )
