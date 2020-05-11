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

def plot(df, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(10, 8))
    figure.autofmt_xdate()
    data = df[df.columns[0]]
    axis.plot(data)
    axis.set_title(df.columns[0])
    axis.set_xlabel(df.index.name)
    axis.set_ylabel(df.columns[0])
    pyplot.tight_layout(pad=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def date_parser(date):
    return pandas.datetime.strptime("201"+date, "%Y-%m")

# %%

filepath = os.path.join(wd, "data", "examples", "shampoo-sales.csv")
df = pandas.read_csv(filepath, parse_dates=['Month'], index_col='Month', date_parser=date_parser)

# %%

title = "Shampoo Sales"
plot_name = "arima_example_shampoo_sales"
plot(df, title, plot_name)
