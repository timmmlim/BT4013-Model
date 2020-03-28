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

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

    ############################################## DECLARE FUNCTIONS ################################################

    def get_best_p_q(test_results={}, best_model_criteria="BIC"):
        # get the best p and best q
        test_results_df = pd.DataFrame(test_results).T
        
        test_results_df.columns = ['RMSE', 'AIC',
                                'BIC', 'convergence', 'stationarity']
        test_results_df.index.names = ['p', 'q']
        test_results_df = test_results_df.reset_index(level=[0,1])
        # print(f'test_results_df:{test_results_df}')
        index = np.argmin(
            test_results_df[best_model_criteria])
        best_p = test_results_df.iloc[index]['p']
        best_q = test_results_df.iloc[index]['q']
        # return integer values
        return int(best_p), int(best_q)

    # Data consists of 504 days over here, automatically replaced by runts
    def re_train_model(data,
                       train_size,
                       max_p=5,
                       max_q=5):
        """

        We can look back 504 days
        if train_size = 400 with rolling window of step size 1 means:
        For each p:q, we have (504 - 400) = 104 prediction -> 104 models for validation 

        """
        import warnings
        warnings.simplefilter('ignore')
        # test results to keep track of the metrics
        test_results = {}
        # this is for test data, so not wrong to do it here
        y_true = data.iloc[train_size:]
        
        # try 3 x 3 because prof said any order more than 2 is suspicious already
        for p in range(max_p):
            for q in range(max_q):
                # print(f"P,Q:{p,q}")
                aic, bic = [], []
                if p == 0 and q == 0:
                    continue
                convergence_error = stationarity_error = 0
                y_pred = []
                for T in range(train_size, len(data)):  # step size 1
                    # print(f'len(data):{len(data)}')
                    train_set = data.iloc[T-train_size:T]
                    try:
                        model = tsa.ARMA(endog=train_set, order=(p, q)).fit(optimized=False)
                    except LinAlgError:
                        convergence_error += 1
                    except ValueError:
                        stationarity_error += 1

                    forecast, _, _ = model.forecast(steps=1)
                    y_pred.append(forecast[0])
                    aic.append(model.aic)
                    bic.append(model.bic)

                result = (pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
                          .replace(np.inf, np.nan)
                          .dropna())
                
                # i think shouldn't use future predictions to evaluate model RMSE
                # AIC/BIC is evaluating how well the model fits on the train data
                rmse = np.sqrt(mean_squared_error(
                    y_true=result.y_true, y_pred=result.y_pred))

                test_results[(p, q)] = [rmse,
                                        np.mean(aic),
                                        np.mean(bic),
                                        convergence_error,
                                        stationarity_error]
        # Set model selection criteria
        best_model_criteria = "BIC"
        # Get the best order
        best_p, best_q = get_best_p_q(test_results=test_results,
                                      best_model_criteria=best_model_criteria)
        
        print(f"Best p: {best_p}, Best q: {best_q}")
        
        # Fit ARMA with best order
        best_model = tsa.ARMA(endog=data, order=(best_p, best_q)).fit()
        print(f"Model Params: {best_model.params}")
        
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
            CURR_CLOSE = pd.Series(CLOSE[:, i]).dropna()

            DAILY_RETURN = CURR_CLOSE.pct_change().dropna()

            # calculate log returns
            LOG_DAILY_RETURN = np.log(DAILY_RETURN + 1)
            
            # retrain our model every 20 days
            if traded_days_count % 20 == 0:
                print(f'Retraining model for {market}')
                
                settings["TrainedCounts"] = settings["TrainedCounts"] + 1
                train_size = 450
                max_p = 3
                max_q = 3

                best_model = re_train_model(data=LOG_DAILY_RETURN, # LOG_RETURN IS SERIES
                                            train_size=train_size,
                                            max_p=max_p,
                                            max_q=max_q)
                
                # forecast the results for the next 20 days
                forecast_next_20 = best_model.forecast(steps=20)[0]
                settings['forecasted_returns'][market] = forecast_next_20                
                print(f'Forecasted returns for {market}: {forecast_next_20}')
                
            # get the forecasts from the model
            model_forecast = settings['forecasted_returns'][market][traded_days_count % 20]
            predicted_returns = np.exp(model_forecast) - 1
            print(f'model prediction: {model_forecast}')
            print(f"predicted_returns for {market}: {predicted_returns}")
            
            if abs(predicted_returns) < threshold:
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

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets'] = ['CASH', 'F_TU', 'F_FV', 'F_TY', 'F_NQ', 'F_US']

    settings['lookback'] = 504
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20171017'
    settings['endInSample'] = '20200131'

    settings['threshold'] = 0.01
    settings["TradingDay"] = 0
    settings["TrainedCounts"] = 0
    settings["curr_best_model"] = {}
    settings["forecasted_returns"] = {}
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts

    results = runts(__file__)
