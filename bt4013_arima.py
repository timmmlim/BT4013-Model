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
from statsmodels.stats.diagnostic import het_arch

from sklearn.metrics import mean_squared_error

from scipy.stats import probplot, moment

from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

# fitler statsmodel convergence warning ; already handled
# warnings.simplefilter('once', category=UserWarning)
from warnings import filterwarnings
filterwarnings('ignore')
warnings.simplefilter('ignore')

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

    ############################################## DECLARE FUNCTIONS ################################################

    def get_best_p_d_q(test_results={}, best_model_criteria="BIC"):
        # convert to dataframe
        test_results_df = pd.DataFrame(test_results).T
        test_results_df.columns = ['AIC', 'BIC', 'convergence', 'stationarity']
        test_results_df.index.names = ['p', 'd', 'q']
        test_results_df = test_results_df.reset_index()

        # find p,d,q that gives the minimum criteria
        index = np.argmin(test_results_df[best_model_criteria])
        best_p = test_results_df.iloc[index]['p']
        best_d = test_results_df.iloc[index]['d']
        best_q = test_results_df.iloc[index]['q']

        return int(best_p), int(best_d), int(best_q)
    
    def get_best_p_q(test_results={}, best_model_criteria="BIC"):
        # helper function to get best order for ARCH model
        # convert to dataframe
        test_results_df = pd.DataFrame(test_results).T
        test_results_df.columns = ['AIC', 'BIC']
        test_results_df.index.names = ['p', 'q']
        test_results_df = test_results_df.reset_index()

        # find p,d,q that gives the minimum criteria
        index = np.argmin(test_results_df[best_model_criteria])
        best_q = test_results_df.iloc[index]['q']
        best_p = test_results_df.iloc[index]['p']

        return int(best_p), int(best_q)

    def get_best_arima_model(data, max_p=3, max_d=2, max_q=3, criteria="BIC"):
        """

        helper function to find the best order for our model

        """
        # test results to keep track of the metrics
        test_results = {}

        # iterate through range of possible orders
        for p in range(max_p):
            for q in range(max_q):
                for d in range(max_d):
                    aic, bic = 10000, 10000
                    if p == 0 and q == 0:
                        continue
                    convergence_error = stationarity_error = 0
    
                    try:
                        model = tsa.ARIMA(endog=data, order=(p, d, q)).fit(optimized=False)
                        aic = model.aic
                        bic = model.bic
                    except LinAlgError:
                        convergence_error += 1
                    except ValueError:
                        stationarity_error += 1

                    test_results[(p, d, q)] = [aic,
                                               bic,
                                               convergence_error,
                                               stationarity_error]
        # Get the best order
        best_p, best_d, best_q = get_best_p_d_q(test_results=test_results,
                                      best_model_criteria=criteria)
        print(f"Best p: {best_p}, Best d: {best_d}, Best q: {best_q}")
        
        # Fit ARMA with best order
        best_model = tsa.ARMA(endog=data, order=(best_p, best_q)).fit()
        print(f"Model Params: {best_model.params}")
        
        return best_model
    
    def is_serially_correlated(residuals, confidence_level=0.05):
        '''
        helper method to check if residuals are serially correlated, using Box Test
        null hypothesis: variables are iid (no serial correlation)
    
        '''
        lags = int(np.log(len(residuals)))
        p_values = sm.stats.acorr_ljungbox(residuals, lags=lags)[1]
        
        if np.any(p_values <= confidence_level):
            return True
        else:
            return False
    
    def is_arch_effect_present(residuals, confidence_level=0.05):
        '''
        helper method to check if arch effect is present
        arch effect is defined as serial correlation not being present among residuals
        but squared residuals show dependence
        '''
        is_resid_correlated = is_serially_correlated(residuals, confidence_level)
        is_resid_squared_correlated = is_serially_correlated(residuals ** 2, confidence_level)
        return (~is_resid_correlated and is_resid_squared_correlated)
        

    def get_best_arch_model(arima_model, max_q=6, max_p=6):
        test_results = {}
        for p in range(max_p):
            for q in range(max_q):
                model = arch_model(mean='Zero', y=arima_model.resid, q=q, p=p, vol='GARCH').fit()
                
                aic = model.aic
                bic = model.bic
                
                test_results[q] = [aic, bic]
                
        # get best order
        best_p, best_q = get_best_p_q(test_results, best_model_criteria='BIC')
        best_model = arch_model(mean='Zero', y=arima_model.resid, p=best_p, q=best_q, vol='GARCH', rescale=True).fit()
        
        return best_model
 ############################################## END DECLARE FUNCTIONS ################################################

    # Get parameters from setting
    markets = settings['markets']
    lookback = settings['lookback']
    threshold = settings['threshold']
    traded_days_count = settings["TradingDay"]  # traded_days to indicate when should we re-run the model


    pos = np.zeros(len(markets), dtype=np.float)

    # For each market
    for i, market in enumerate(markets):

        ##### i converted the prices to pandas, because i find it easier to work with, not sure if it slows down the shit #####
        try:            
            window = 10
            CURR_CLOSE = pd.Series(CLOSE[-window:, i]).dropna()
            DAILY_RETURN = CURR_CLOSE.pct_change().dropna()

            # calculate log returns
            LOG_DAILY_RETURN = np.log(DAILY_RETURN + 1)
            
            retrain_period = 10  # period to retrain the model
            if traded_days_count % retrain_period == 0:
                settings['forecasted_returns'][market] = np.zeros(retrain_period)
                print(f'Retraining model for {market}')
                settings['is_valid_model'][market] = False
                settings["TrainedCounts"] = settings["TrainedCounts"] + 1
                max_p = 5
                max_q = 5
                max_d = 2
                
                arima_model = get_best_arima_model(data=LOG_DAILY_RETURN, # LOG_RETURN IS SERIES
                                            max_p=max_p,
                                            max_d=max_d,
                                            max_q=max_q)
                
                mu_pred = arima_model.forecast(steps=retrain_period)[0]
                var_pred = 0
                
                garch_forecast = None
                if is_arch_effect_present(arima_model.resid, confidence_level=0.15):
                    # if ARCH effect present, fit GARCH model
                    garch_model = get_best_arch_model(arima_model)
                
                    # check if model is valid
                    settings['is_valid_model'][market] = is_serially_correlated(garch_model.resid)
                    garch_forecast = garch_model.forecast(horizon=retrain_period).mean.iloc[-1]  / garch_model.scale # overwrite None variable
                    print(f'GARCH forecast for mean: {garch_forecast}')
                    
                else:
                    settings['is_valid_model'][market] = is_serially_correlated(arima_model.resid)
                                
                if garch_forecast:
                    settings['forecasted_returns'][market] = garch_forecast
                else:
                    settings['forecasted_returns'][market] = mu_pred  # use arima pred
                
            # get the forecasts from the model
            print(f'Day in cycle: {traded_days_count % retrain_period}')
            model_forecast = settings['forecasted_returns'][market][traded_days_count % retrain_period]
            predicted_returns = np.exp(model_forecast) - 1
            print(f'model prediction: {model_forecast}')
            print(f"predicted_returns for {market}: {predicted_returns}")
            
            # check if forecast exceeds threshold
            if abs(predicted_returns) < threshold:
                predicted_returns = 0
            
            # check if model passes box test
            if not settings['is_valid_model'][market]:
                predicted_returns = 0

            # update position in the given market (i.e buy / sell)
            pos[i] = np.sign(predicted_returns)
        
        except ValueError:
            pos[i] = 0
            
    settings['TradingDay'] = traded_days_count + 1
    
    return pos, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings = {}

    # Futures Contracts

    currency_futures = ['F_AD', 'F_BP', 'F_CD', 'F_DX', 'F_EC', 'F_JY', 'F_MP', 
                    'F_SF', 'F_LR', 'F_ND', 'F_RR', 'F_RF', 'F_RP', 'F_TR']
    
    index_futures = ['F_ES', 'F_MD', 'F_NQ', 'F_RU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_LX', 'F_VX', 'F_AE', 'F_DM',
                    'F_AH', 'F_DZ', 'F_FB', 'F_FM', 'F_FY', 'F_NY', 'F_PQ', 'F_SH', 'F_SX', 'F_GD']

    bond_futures = ['F_FV', 'F_TU', 'F_TY', 'F_US']
        
    #settings['markets']  = currency_futures + ['CASH']
    settings['markets'] = currency_futures

    settings['lookback'] = 504
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20171001'  # backtesting starts settings[lookback] days after period
    settings['endInSample'] = '20191231'

    settings['threshold'] = 0.001  # probably need to set lower, since the forecasted values are all close to 0
    settings["TradingDay"] = 0
    settings["TrainedCounts"] = 0
    settings["forecasted_returns"] = {}
    settings['is_valid_model'] = {}
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts

    results = runts(__file__)
