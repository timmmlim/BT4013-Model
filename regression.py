# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:15:25 2020

@author: MengYuan
"""

### Quantiacs Trading System Template
# This program may take several minutes
# import necessary Packages

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, USA_UNR, USA_CPI, USA_PP, USA_RSM, USA_IP, OI, P, R, RINFO, exposure, equity, settings):
    """ This system uses linear regression to allocate capital into the desired equities"""

    # Get parameters from setting
    nMarkets = len(settings['markets'])
    lookback = settings['lookback']
    dimension = settings['dimension']
    threshold = settings['threshold']
    long = settings['threshold']
    short = settings['threshold'] * 5

    pos = np.zeros(nMarkets, dtype=np.float)
    trends = np.zeros(nMarkets, dtype=np.float)

    poly = PolynomialFeatures(degree=dimension)
    x2 = USA_UNR
    x3 = USA_CPI
    x4 = USA_PP
    
    res = []
    
    for market in range(nMarkets):
        reg = linear_model.LinearRegression()
        x1_train = poly.fit_transform(np.arange(lookback).reshape(-1, 1)) # 504*(dimension+1)
        x2_train = poly.fit_transform(x2.reshape(-1, 1))
        x3_train = poly.fit_transform(x3.reshape(-1, 1))
        x4_train = poly.fit_transform(x4.reshape(-1, 1))

#        print(x1_train.size,x2_train.size)
#        print(x1_train)
        x_train = np.concatenate((x1_train,x2_train,x3_train,x4_train),axis=1)
        y_train = CLOSE[:, market]                                      #504
#        print(x_train[0:5])

        try:
            reg.fit(x_train, y_train)            
#            y_pred = reg.predict(poly.fit_transform(np.array([[lookback]])))
            
            x1_test = poly.fit_transform(np.array([[lookback]]))
            x2_test = [x2_train[lookback-1]]
            x3_test = [x3_train[lookback-1]]
            x4_test = [x4_train[lookback-1]]

            test = np.concatenate((x1_test,x2_test,x3_test,x4_test),axis=1)
#            print(test[0:5])
            y_pred = reg.predict(test)
            res.append(y_pred - CLOSE[-1, market])
#            print(x_train.size, y_train.size, y_pred.size)
            trend = (y_pred - CLOSE[-1, market]) / CLOSE[-1, market]
            
            if trend[0] < 0:
                if abs(trend[0]) < short:
                    trend[0] = 0
            if trend[0] > 0:
                if trend[0] < long:
                    trend[0] = 0
                    
#            if abs(trend[0]) < threshold:
#                trend[0] = 0
                    
                
            pos[market] = np.sign(trend)
            trends[market] = trend

        # for NaN data set position to 0
        except ValueError:
            pos[market] = .0
            
#    print(res)
#    for market in range(nMarkets):
#        pos[market] = trends[market]/max(np.abs(trends))
#    pos = trends

    return pos, settings


def mySettings():
    """ Define your trading system settings here """

    settings = {}

    # Futures Contracts
    # does not include F_VX
    settings['markets'] = ['CASH', 'F_AD', 'F_AE', 'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA',
                           'F_CC', 'F_CD', 'F_CF', 'F_CL', 'F_CT', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_DZ', 'F_EB',
                           'F_EC', 'F_ED', 'F_ES', 'F_F',  'F_FB', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY',
                           'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_HG', 'F_HO', 'F_HP', 'F_JY', 'F_KC', 'F_LB', 'F_LC',
                           'F_LN', 'F_LQ', 'F_LR', 'F_LU', 'F_LX', 'F_MD', 'F_MP', 'F_ND', 'F_NG', 'F_NQ', 'F_NR',
                           'F_NY', 'F_O',  'F_OJ', 'F_PA', 'F_PL', 'F_PQ', 'F_RB', 'F_RF', 'F_RP', 'F_RR', 'F_RU',
                           'F_RY', 'F_S',  'F_SB', 'F_SF', 'F_SH', 'F_SI', 'F_SM', 'F_SS', 'F_SX', 'F_TR', 'F_TU',
                           'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VT', 'F_VW', 'F_W', 'F_XX', 'F_YM',
                           'F_ZQ']
    
#    settings['markets'] = ['CASH', 'F_PA', 'F_ED', 'F_GC', 'F_TY', 'F_US', 'F_FV', 'F_NR', 'F_TU']

    settings['threshold'] = 0.01
    settings['dimension'] = 2
#    settings['beginInSample'] = '20180101'
#    settings['endInSample'] = '20200327'
    settings['beginInSample'] = '20171001'  # backtesting starts settings[lookback] days after period
    settings['endInSample'] = '20191231'
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
