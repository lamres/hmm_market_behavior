import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
import datetime
import seaborn as sns

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, date_rules, time_rules, get_datetime)

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# Normalized st. deviation
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)

def add_to_log(file, text):
    with open(file, 'a') as f:
      f.write(text + '\n') 

def initialize(context):
    context.asset = symbol('btc_usd')
    context.leverage = 1.0

    context.std_period = 10
    context.ma_period = 10
    context.price_deviation_period = 10
    context.volume_deviation_period = 10
    context.n_periods = 5 + int(np.max([context.std_period, context.ma_period,
        context.price_deviation_period, context.volume_deviation_period]))
    context.tf = '1440T'

    context.model = joblib.load('quandl_BITFINEX_BTCUSD_final_model.pkl')
    context.cols_features = ['last_return', 'std_normalized', 'ma_ratio', 'price_deviation', 'volume_deviation']
    context.long_states = [2]
    context.random_states = [1]
    context.short_states = [0]

    context.set_commission(maker = 0.001, taker = 0.002)
    context.set_slippage(slippage = 0.0005)
    context.set_benchmark(context.asset)

def handle_data(context, data):
    current_date = get_datetime().date()
    current_time = get_datetime().time()

    # Just one time in a day (first minute)
    if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
        prices = pd.DataFrame()
        volumes = pd.DataFrame()

        try:
            prices = data.history(context.asset,
                fields = 'price',
                bar_count = context.n_periods,
                frequency = context.tf)

            volumes = data.history(context.asset,
                fields = 'volume',
                bar_count = context.n_periods,
                frequency = context.tf)
        except:
            print('NO DATA')

        if prices.shape[0] == context.n_periods and volumes.shape[0] == context.n_periods:
            features = pd.DataFrame()
            features['price'] = prices
            features['volume'] = volumes
            features['last_return'] = features['price'].pct_change()
            features['std_normalized'] = features['price'].rolling(context.std_period).apply(std_normalized)
            features['ma_ratio'] = features['price'].rolling(context.ma_period).apply(ma_ratio)
            features['price_deviation'] = features['price'].rolling(context.price_deviation_period).apply(values_deviation)
            features['volume_deviation'] = features['volume'].rolling(context.volume_deviation_period).apply(values_deviation)

            state = context.random_states[0]
            if features.dropna().shape[0] == (context.n_periods - context.ma_period + 1):
                state = int(context.model.predict(features[context.cols_features].dropna())[-1])
            else:
                print('PROBLEM: features dataframe is too small')

            print('State on ' + str(current_date) + ' ' + str(current_time) + ': ' + str(state))
        
            print('Amount on ' + str(current_date) + ' ' + str(current_time) + ': ' + str(context.portfolio.positions[context.asset].amount))
            print(prices.dropna())
            print(volumes.dropna())

            if context.portfolio.positions[context.asset].amount <= 0 and state in context.long_states:
                print('LONG on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, 1.0 * context.leverage)
                context.best_price_ts = data.current(context.asset, 'close')

            if context.portfolio.positions[context.asset].amount != 0 and state in context.random_states:
                print('CLOSE on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, 0.0)

            if context.portfolio.positions[context.asset].amount >= 0 and state in context.short_states:
                print('SHORT on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, -1.0 * context.leverage)   
                context.best_price_ts = data.current(context.asset, 'close')   

            record(price = prices[-1],
                state = state,
                amount = context.portfolio.positions[context.asset].amount)

    
def analyze(context, perf):
    sns.set()

    # Summary output
    print("Total return: " + str(perf.algorithm_period_return[-1]))
    print("Sortino coef: " + str(perf.sortino[-1]))
    print("Max drawdown: " + str(np.min(perf.max_drawdown[-1])))
    print("alpha: " + str(perf.alpha[-1]))
    print("beta: " + str(perf.beta[-1]))

    f = plt.figure(figsize = (7.2, 7.2))

    # Plot return
    ax1 = f.add_subplot(211)
    ax1.plot(perf.algorithm_period_return, 'blue')
    ax1.plot(perf.benchmark_period_return, 'red')
    ax1.legend()
    ax1.set_title("Returns")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')

    # Plot state
    ax2 = f.add_subplot(212, sharex = ax1)
    ax2.plot(perf.state, 'grey')
    ax2.set_title("State")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')

    plt.tight_layout()
    plt.show()


run_algorithm(
    capital_base = 100000,
    data_frequency = 'minute',
    initialize = initialize,
    handle_data = handle_data,
    analyze = analyze,
    exchange_name = 'bitfinex',
    quote_currency = 'usd',
    start = pd.to_datetime('2018-1-1', utc = True),
    end = pd.to_datetime('2019-5-22', utc = True))

