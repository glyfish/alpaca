# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion
from lib import regression as reg
from scipy.stats import chi2
from statsmodels.tsa import stattools

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
# Example of single variance hypothesis test

npts = 1001
xmax = 300.0
x = numpy.linspace(0.0001, xmax, npts)
k = 145

# %%

title = r"$\chi^2$"+f" PDF from scipy, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
plot = f"chi_squared_test_chi_squared_{k}_pdf"
reg.distribution_plot(reg.chi_squared_pdf(k)(x), x, title, ylabel, xlabel, plot)

# %%

title = r"$\chi^2$"+f" CDF, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
plot = f"chi_squared_test_chi_squared_{k}_cdf"
reg.distribution_plot(reg.chi_squared_cdf(k)(x), x, title, ylabel, xlabel, plot)

# %%

sx2 = 308.56
σ2 = 225.0
t = k*sx2/σ2
t
reg.chi_squared_cdf(k)(113.65)
reg.chi_squared_cdf(k)(180.23)
reg.chi_squared_cdf(k)(174.3)
1.0-reg.chi_squared_cdf(k)(t)
1.0-chi2.cdf(t, k)
