import numpy as np
import pandas as pd
import os, sys
sys.path.append('../src/data/')
import data_preprocessing_from_yahoo_finance as dp
import matplotlib.pyplot as plt

class myenv:
    def __init__(self, result_np, total_steps=100):
        self.step = 0 # start from step 0
        self.total_steps = total_steps # how many days to trade on
        self.price_window = int(result_np.shape[1]-self.total_steps) # How many data points to look at in each step
        self.comission_rate = 0.05
        
        self.result_np = result_np
        
        # Initialize Price Tensor
        self.cash_price = np.ones(result_np.shape[0:2])
        self.all_prices = np.concatenate((self.cash_price[:,:,None], result_np), axis=2) 
        self.all_prices_normalized = self.all_prices[:,1:,:]/self.all_prices[:,:-1,:]
        
        # Backtest
        self.portfolio_size = []
        self.portfolio_return = []
        self.sharpe_ratio = 0
    
    def reset(self):
        weight = np.zeros(self.result_np.shape[2]+1) # initial weight
        weight[0] = 1
        self.step = 0 
        start_prices = self.all_prices_normalized[:,self.step:self.step+self.price_window,:]
        step_portfolio_size = np.sum(weight*start_prices[3,-1,:]) # Assume trade on closing price of the next trading day 
        self.portfolio_size.append(step_portfolio_size)
        return weight, start_prices, step_portfolio_size

    def next_step(self, weight):
        self.step += 1 
        next_prices = self.all_prices_normalized[:,self.step:self.step+self.price_window,:]
        step_portfolio_size = np.sum(weight*next_prices[3,-1,:]) # Assume trade on closing price of the next trading day

        if self.step == self.total_steps+1:
            self.end_game()
            return False
        
        self.portfolio_size.append(step_portfolio_size)
        self.portfolio_return.append((step_portfolio_size - self.portfolio_size[-2])/self.portfolio_size[-2])
        return weight, next_prices, step_portfolio_size
    
    def end_game(self):
        expected_return = np.mean(self.portfolio_return)
        std_return = np.std(self.portfolio_return)
        self.sharpe_ratio = expected_return / std_return