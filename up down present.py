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
import pickle
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
        best_model = tsa.ARIMA(endog=data, order=(best_p, best_d, best_q)).fit()
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
        for p in range(1, max_p):
            for q in range(max_q):
                print(p, q)
                model = arch_model(mean='Zero', y=arima_model.resid, q=q, p=p, vol='GARCH', rescale=True).fit()
                
                aic = model.aic
                bic = model.bic
                
                test_results[(p, q)] = [aic, bic]
                
        # get best order
        best_p, best_q = get_best_p_q(test_results, best_model_criteria='BIC')
        best_model = arch_model(mean='Zero', y=arima_model.resid, p=best_p, q=best_q, vol='GARCH', rescale=True).fit()
        
        return best_model
    
    def get_one_step_forecast(model, new_data):
        """Input:
            model: Either ARIMA or GARCH
            new_data: the entire new log returns data
        """
        try:
            if str(type(model)) == "<class 'arch.univariate.base.ARCHModelResult'>":  # means garch model
                # update arima to get residual
                arima_model = settings['arima_model'][market]
                old_params = arima_model.params
                old_order = (len(arima_model.arparams), len(arima_model.maparams))
                new_arima_model = tsa.ARMA(new_data, order = old_order).fit(start_params= old_params, max_iter = 0)  # assume ARMA, d=0
    
                # update garch model
                old_garch_params = model.params
                new_updated_residual = new_arima_model.resid
                new_model = arch_model(mean='Zero', y=new_updated_residual,vol='GARCH', rescale=True).fit(starting_values = old_garch_params,
                                                                               update_freq = 0, 
                                                                               disp = 'off')
                ## Question: but variance? or mean
                forecast = new_model.forecast(horizon=1).mean.iloc[-1] / new_model.scale # forecast using the original params 
    
            else:
                # update arima to get residual
                old_params = model.params
                old_order = (len(model.arparams), len(model.maparams))
                new_arima_model = tsa.ARMA(new_data, order = old_order).fit(start_params= old_params, max_iter = 0)
                forecast, _, _ = new_arima_model.forecast(steps=1)
                forecast = forecast[0]
        
        except:
            forecast = 0 
        
        return forecast

    def get_pre_trained_forecast(lst_of_models, new_data):
        forecast_sum = 0
        for model in lst_of_models:
            model_forecast = get_one_step_forecast(model, new_data)
            model_forecast = np.exp(model_forecast) - 1  # transform back to daily return
            forecast_sum += model_forecast
        avg_forecast = forecast_sum/len(lst_of_models)
        return avg_forecast

 ############################################## END DECLARE FUNCTIONS ################################################

    # Get parameters from setting
    markets = settings['markets']
    lookback = settings['lookback']
    threshold = settings['threshold']
    traded_days_count = settings["TradingDay"]  # traded_days to indicate when should we re-run the model

    pos = np.zeros(len(markets), dtype=np.float)

    # For each market
    for i, market in enumerate(markets):

        try:            
            window = 10
            CURR_CLOSE = pd.Series(CLOSE[-window:, i]).dropna()
            DAILY_RETURN = CURR_CLOSE.pct_change().dropna()

            # calculate log returns
            LOG_DAILY_RETURN = np.log(DAILY_RETURN + 1)

            retrain_period = 10  # period to retrain the model
            if traded_days_count % retrain_period == 0:
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
                settings['arima_model'][market] = arima_model
                
                if is_arch_effect_present(arima_model.resid, confidence_level=0.1):
                    # if ARCH effect present, fit GARCH model
                    garch_model = get_best_arch_model(arima_model)
                    # check if model is valid
                    
                    settings['model'][market] = garch_model
                    settings['is_valid_model'][market] = not is_serially_correlated(garch_model.resid)
                        
                else:
                    settings['is_valid_model'][market] = not is_serially_correlated(arima_model.resid)
                    settings['model'][market] = arima_model
                    
            # get the forecasts from the model
            model = settings['model'][market]
            present_forecast = get_one_step_forecast(model, LOG_DAILY_RETURN)
            print(f'one step forecast: {present_forecast}')
            present_forecast = np.exp(present_forecast) - 1  # transform back to return
            predicted_returns = present_forecast  # init variable first
            print(f'present forecast: {market} {present_forecast}')
            
            if not settings['is_valid_model'][market]:
                present_forecast = 0
                
            settings['forecasts_by_model']['present'][market] += [present_forecast]
                
            # get forecasts from pre-trained models
            high_models = [model for v in settings['high_models'][market].values() for model in v]  # cos i segregated the arima and garch models in preprocessing
            low_models = [model for v in settings['low_models'][market].values() for model in v] 
            
            high_forecast = get_pre_trained_forecast(high_models, LOG_DAILY_RETURN)  # returns forecast for daily return
            low_forecast = get_pre_trained_forecast(low_models, LOG_DAILY_RETURN)
            print(f'high forecast: {high_forecast}')
            print(f'low forecast: {low_forecast}')
            
            settings['forecasts_by_model']['high'][market] += [high_forecast]
            settings['forecasts_by_model']['low'][market] += [low_forecast]
            
            if settings['TradingDay'] > 5:
                # assume that 'best' is the model that forecasted the direction correctly
                last_5_days_direction = np.array(np.sign(LOG_DAILY_RETURN[-5:]))
                high_forecast_list = np.sign(settings['forecasts_by_model']['high'][market][-6:-1])
                low_forecast_list = np.sign(settings['forecasts_by_model']['low'][market][-6:-1])
                present_forecast_list = np.sign(settings['forecasts_by_model']['present'][market][-6:-1])
                
                score_array = [0, 0, 0]
                for n in range(len(last_5_days_direction)):
                    observed = last_5_days_direction[n]
                    if present_forecast_list[n] == observed:
                        score_array[0] += 1
                    if high_forecast_list[n] == observed:
                        score_array[1] += 1
                    if low_forecast_list[n] == observed:
                        score_array[2] += 1
                        
                print(f'score array: {score_array}')
                    
                # return the highest score
                max_index = score_array.index(max(score_array))
                if max_index == 0:
                    predicted_returns = present_forecast
                elif max_index == 1:
                    predicted_returns = high_forecast
                else:
                    predicted_returns = low_forecast
                
                # check if forecast exceeds threshold
                print(f'predicted returns: {predicted_returns}')
                if abs(predicted_returns) < threshold:
                    print('returns below threshold')
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

    # currency_futures = ['F_AD', 'F_BP', 'F_CD', 'F_DX', 'F_EC', 'F_JY', 'F_MP', 
    #                 'F_SF', 'F_LR', 'F_ND', 'F_RR', 'F_RF', 'F_RP', 'F_TR']
    
    # index_futures = ['F_ES', 'F_MD', 'F_NQ', 'F_RU', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_LX', 'F_VX', 'F_AE', 'F_DM',
    #                 'F_AH', 'F_DZ', 'F_FB', 'F_FM', 'F_FY', 'F_NY', 'F_PQ', 'F_SH', 'F_SX', 'F_GD']

    # bond_futures = ['F_FV', 'F_TU', 'F_TY', 'F_US']
    settings['markets'] = ['F_GC', 'F_US']
    #settings['markets'] = ['F_PA', 'F_ED', 'F_GC', 'F_US', 'F_NR', 'F_TU']
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
    settings['model'] = {}
    settings['arima_model'] = {}
    settings['forecasts_by_model'] = {'high': {}, 'low': {}, 'present': {}}
    
    # init settings['forecasts_by_model]
    for market in settings['markets']:
        for key in settings['forecasts_by_model'].keys():
            settings['forecasts_by_model'][key][market] = []
    
    # read pickle
    objects = []
    pickle_file_name = 'fitted_time_series.pickle'
    with (open(pickle_file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    
    settings['high_models'] = objects[0]
    settings['low_models'] = objects[1]
    
    # structure of settings['high_models']
    # dict{k=future code, v = dict{k=arima/garch, v=list of models}}
    
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts

    results = runts(__file__)
