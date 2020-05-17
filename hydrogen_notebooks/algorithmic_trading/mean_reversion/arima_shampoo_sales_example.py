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
    axis.set_title(title)
    axis.set_xlabel(df.index.name, labelpad=10)
    axis.set_ylabel(df.columns[0], labelpad=10)
    pyplot.tight_layout(pad=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def date_parser(date):
    return pandas.datetime.strptime("201"+date, "%Y-%m")

# %%

filepath = os.path.join(wd, "data", "examples", "shampoo-sales.csv")
sales = pandas.read_csv(filepath, parse_dates=['Month'], index_col='Month', date_parser=date_parser)

# %%

title = "Shampoo Sales"
plot_name = "arima_example_shampoo_sales"
plot(sales, title, plot_name)

# %%

pandas.plotting.autocorrelation_plot(sales)

# %%

samples = sales['Sales'].to_numpy()
title = f"Shampoo Sales ACF"
plot_name = "arima_example_shampoo_sales_acf"
max_lag = 36
arima.acf_plot(title, samples, max_lag, plot_name)

# %%

sales_diff = sales.diff()[1:]
sales_diff.columns = ["Sales Difference"]

title = "Shampoo Sales First Difference"
plot_name = "arima_example_shampoo_sales_first_difference"
plot(sales_diff, title, plot_name)

# %%

fit_sales = sales
fit_sales.index = fit_sales.index.to_period("M")
model = pyarima(fit_sales, order=(5, 1, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())
