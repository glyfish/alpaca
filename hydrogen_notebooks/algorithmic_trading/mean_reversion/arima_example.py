# %%
# Example taken from Time Series Forecasting with Python page 212

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima
from statsmodels.tsa.arima_model import ARIMA as pyarima

pyplot.style.use(config.glyfish_style)

wd = os.getcwd()

# %%

def plot(df):
    figure, axis = pyplot.subplots(nrows=4, ncols=2, figsize=(10, 8))
    for i, axis in enumerate(axis.flatten()):
        data = df[df.columns[i]]
        axis.plot(data)
        axis.set_title(df.columns[i], fontsize=12)
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    pyplot.tight_layout(pad=1.0)
    config.save_post_asset(figure, "mean_reversion", "machine_learning_plus_macro_example_timeseries")


def date_parser(date):
    return pandas.datetime.strptime("201"+date, "%Y-%m")

# %%

filepath = os.path.join(wd, "data", "examples", "shampoo-sales.csv")
df = pandas.read_csv(filepath, parse_dates=['Month'], index_col='Month', date_parser=date_parser)
