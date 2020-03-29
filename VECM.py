import os
import sys
import warnings
from datetime import date

import pandas as pd

import numpy as np
from numpy.linalg import LinAlgError

import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller

from sklearn.metrics import mean_squared_error

from scipy.stats import probplot, moment

from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

# fitler statsmodel convergence warning ; already handled
# warnings.simplefilter('once', category=UserWarning)
from warnings import filterwarnings
filterwarnings('ignore')
warnings.simplefilter('ignore')

# helper functions
    

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    
    ############################################## DECLARE FUNCTIONS ################################################ 
    def is_order_1(data, confidence_level=0.05):
        '''
        method to check if data (series) is integrated order 1
        '''
        adf_initial = tsa.adfuller(data)
        differenced_data = np.diff(data)
        adf_after_differencing = tsa.adfuller(differenced_data)
    
        if adf_initial[1] > confidence_level and adf_after_differencing[1] <= confidence_level:
            return True
        else:
            return False


    def is_cointegrated_pair(series_1, series_2, confidence_level=0.05):
        '''
        method to check if two series are cointegrated, at the declared confidence level
    
        '''
        if is_order_1(series_1) and is_order_1(series_2):
            try:
                _, pvalue, _ = tsa.coint(series_1, series_2)  # try johansen cointegration next
                if pvalue <= confidence_level:
                    return True
            except:
                return False
        else:
            return False
    
    
    def is_serially_correlated(residuals, confidence_level=0.05):
        '''
        helper method to check if residuals are serially correlated, using Box Test
    
        '''
        p_values = sm.stats.acorr_ljungbox(residuals)[1]
        
        if np.any(p_values > confidence_level):
            return True
        else:
            return False
    
    
    def get_pairs(lst_of_futures):
        '''
        helper method to return a list of possible pairs from a given list
        '''
        pairs = [(lst_of_futures[i], lst_of_futures[j]) for i in range(len(lst_of_futures)) for j in range(i+1, len(lst_of_futures))]
        return pairs
    
    
    def get_cointegrated_pairs(pairs, lookback_length=252):
        '''
        helper method to return cointegrated pairs from list of possible pairs
    
        '''
        cointegrated_pairs = []
        for pair in pairs:
            future_1 = CLOSE[-lookback_length:, pair[0]]
            future_2 = CLOSE[-lookback_length:, pair[1]]
            
            if is_cointegrated_pair(future_1, future_2):
                cointegrated_pairs.append(pair)
        
        return cointegrated_pairs
     ############################################## END DECLARE FUNCTIONS ################################################
    
    markets = settings['markets']
    train_length = settings['train_length']
    pos = np.zeros(len(markets), dtype=np.float)
    pairs = get_pairs([i[0] for i in enumerate(markets)])
    cointegrated_pairs = get_cointegrated_pairs(pairs)  # returns the indexes of pairs that are cointegrated
    print(f'Cointegrated pairs: {cointegrated_pairs}')
    
    # we look for cointegrated pairs everyday
    for pair in cointegrated_pairs:
        futures_1_index = pair[0]
        futures_2_index = pair[1]
        
        futures_1_name = markets[futures_1_index]
        futures_2_name = markets[futures_2_index]
        
        if settings['is_part_of_cointegrated_pair'][futures_1_name] or settings['is_part_of_cointegrated_pair'][futures_2_name]:
            continue
        
        print(f'checking {futures_1_name} and {futures_2_name} now')
        
        futures_1_close_price = CLOSE[-252:, futures_1_index]
        futures_2_close_price = CLOSE[-252:, futures_2_index]
        
        futures_1_differenced = np.diff(futures_1_close_price, n=1)
        futures_2_differenced = np.diff(futures_2_close_price, n=1)
        
        # fit VECM model
        data = np.transpose([futures_1_differenced, futures_2_differenced])
        #print(f'data for {futures_1_name} and {futures_2_name}')
        #print(data)
        
        vecm = tsa.VECM(endog=data, k_ar_diff=2, coint_rank=1)  # TODO: optimize selection of k_ar_diff by using AIC method on VAR model
        try:
            vecm_fit = vecm.fit()
        except LinAlgError:
            print(f'failed to fit model for {futures_1_name} and {futures_2_name}')
            continue
        
        # get model residuals
        futures_1_residuals = [residuals[0] for residuals in vecm_fit.resid]
        futures_2_residuals = [residuals[1] for residuals in vecm_fit.resid]
        
        # check if model is valid
        futures_1_is_valid = not is_serially_correlated(futures_1_residuals)
        futures_2_is_valid = not is_serially_correlated(futures_2_residuals)
        
        # if futures_1_is_valid and futures_2_is_valid:
        #     settings['is_part_of_cointegrated_pair'][futures_1_name] = True
        #     settings['is_part_of_cointegrated_pair'][futures_2_name] = True
        
        futures_1_is_valid = True
        futures_2_is_valid = True
        
        if futures_1_is_valid and futures_2_is_valid:
            print(f'generating signal for {futures_1_name} and {futures_2_name}')
            alpha = vecm_fit.alpha
            beta = np.transpose(vecm_fit.beta)
            
            # long run relationship
            futures_1_today_price = futures_1_close_price[-1]
            futures_2_today_price = futures_2_close_price[-1]
            long_run = np.dot(beta, [futures_1_today_price, futures_2_today_price])
            print(f'long run: {long_run}')
            
            # not sure about this part for now
            if long_run > 0:
                # short both
                pos[futures_1_index] = -1
                pos[futures_2_index] = -1
                
            if long_run < 0:
                # long both
                pos[futures_1_index] = 1
                pos[futures_2_index] = 1            
    
    return pos, settings
        
def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # Futures Contracts

    currency_futures = ['F_AD', 'F_BP', 'F_CD', 'F_DX', 'F_EC', 'F_JY', 'F_MP', 
                    'F_SF', 'F_LR', 'F_ND', 'F_RR', 'F_RF', 'F_RP', 'F_TR']
    
    index_futures = ['F_ES', 'F_MD', 'F_NQ', 'F_RU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_LX', 'F_VX', 'F_AE', 'F_DM',
                    'F_AH', 'F_DZ', 'F_FB', 'F_FM', 'F_FY', 'F_NY', 'F_PQ', 'F_SH', 'F_SX', 'F_GD']

    bond_futures = ['F_FV', 'F_TU', 'F_TY', 'F_US']
    
    settings['markets']  = currency_futures  # test on currency futures first
    settings['beginInSample'] = '20171017'  # backtesting starts settings[lookback] days after period
    settings['endInSample'] = '20200131'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    settings['train_length'] = 252  # the amt of data we fit the model on
    
    is_part_of_cointegrated_pair_dict = {}
    for future in settings['markets']:
       is_part_of_cointegrated_pair_dict[future] = False
      
    settings['is_part_of_cointegrated_pair'] = is_part_of_cointegrated_pair_dict
    
    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
