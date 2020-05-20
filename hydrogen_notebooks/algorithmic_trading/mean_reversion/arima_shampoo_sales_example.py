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

def timeseries_comparison_plot(samples, labels, title, plot_name):
    nplot, nsample = samples.shape
    ymin = numpy.amin(samples)
    ymax = numpy.amax(samples)
    figure, axis = pyplot.subplots(sharex=True, figsize=(12, 9))
    axis.set_title(title)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$x_t$")
    axis.set_ylim([ymin, ymax])
    axis.set_xlim([0.0, nsample])
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        axis.plot(time, samples[i], label=labels[i])
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot_name)

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

l = 25
train_sales, test_sales = sales[:l], sales[l:]
train_sales.index = train_sales.index.to_period("M")
model = pyarima(train_sales, order=(5, 1, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%

residuals = pandas.DataFrame(model_fit.resid)
residuals.plot()

# %%

residuals.plot(kind="kde")

# %%

residuals.describe()

# %%

history = train_sales.values.flatten()
test = test_sales.values.flatten()
predictions = numpy.array([])
for i in range(len(test)):
    model = pyarima(history, order=(5, 1, 0))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast()
    predictions = numpy.append(predictions, forecast[0])
    history = numpy.append(history, test[i])

# %%

title = "Shampoo Sales Rolling Prediction"
plot_name = "arima_example_shampoo_sales_rolling_prediction"
labels = ["Test", "Prediction"]
timeseries_comparison_plot(numpy.array([test, predictions]), labels, title, plot_name)
