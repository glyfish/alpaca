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
import statsmodels.api as sm
from lib import var

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')

pyplot.style.use(config.glyfish_style)

# %%

def arma_generate_sample(φ, δ, n):
    φ = numpy.r_[1, -φ]
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n)

def arma_estimate_parameter(samples, order):
    model = sm.tsa.ARMA(samples, order).fit(trend='nc', disp=0)
    return model.params

# %%

n = 10000
φ = numpy.array([0.3, 0.4])
δ = numpy.array([])
xt = arma_generate_sample(φ, δ, n)

# %%

p = numpy.array2string(φ, precision=2, separator=',')
d = numpy.array2string(δ, precision=2, separator=',')
plot_name = "arma_2_0_python_simulation_timeseries"
title = f"ARMA(2,0) Simulation: φ={p}, δ={d}"

params = arma_estimate_parameter(xt, (2,0))

reg.timeseries_plot(xt, params, δ, 1000, title, plot_name)

# %%

plot_name = "arma_2_0_python_simulation_autocorrelation"
title = f"ARMA(2,0) Autocorrelation Coefficient: φ={p}, δ={d}"
reg.autocorrelation_plot(xt, 60, title, [-0.05, 1.0], plot_name)

# %%

n = 10000
φ = numpy.array([0.3, 0.4])
δ = numpy.array([0.2, 0.3])
xt = arma_generate_sample(φ, δ, n)

# %%

p = numpy.array2string(φ, precision=2, separator=',')
d = numpy.array2string(δ, precision=2, separator=',')
plot_name = "arma_2_2_1_python_simulation_timeseries"
title = f"ARMA(2,2) Simulation: φ={p}, δ={d}"

params = arma_estimate_parameter(xt, (2,2))

reg.timeseries_plot(xt, params[:2], params[2:], 1000, title, plot_name)

# %%

plot_name = "arma_2_2_1_python_simulation_autocorrelation"
title = f"ARMA(2,2) Autocorrelation Coefficient: φ={p}, δ={d}"
reg.autocorrelation_plot(xt, 60, title, [-0.05, 1.0], plot_name)

# %%

n = 10000
φ = numpy.array([-0.3, -0.4])
δ = numpy.array([-0.2, -0.3])
xt = arma_generate_sample(φ, δ, n)

# %%

p = numpy.array2string(φ, precision=2, separator=',')
d = numpy.array2string(δ, precision=2, separator=',')
plot_name = "arma_2_2_2_python_simulation_timeseries"
title = f"ARMA(2,2) Simulation: φ={p}, δ={d}"

params = arma_estimate_parameter(xt, (2,2))

reg.timeseries_plot(xt, params[:2], params[2:], 1000, title, plot_name)

# %%

plot_name = "arma_2_2_2_python_simulation_autocorrelation"
title = f"ARMA(2,2) Autocorrelation Coefficient: φ={p}, δ={d}"
reg.autocorrelation_plot(xt, 60, title, [-1.0, 1.0], plot_name)

# %%

n = 10000
φ = numpy.array([0.3, 0.4, 0.2])
δ = numpy.array([0.2, 0.3])
xt = arma_generate_sample(φ, δ, n)

# %%

p = numpy.array2string(φ, precision=2, separator=',')
d = numpy.array2string(δ, precision=2, separator=',')
plot_name = "arma_3_2_python_simulation_timeseries"
title = f"ARMA(3,2) Simulation: φ={p}, δ={d}"

params = arma_estimate_parameter(xt, (3,2))

reg.timeseries_plot(xt, params[:3], params[3:], 1000, title, plot_name)

# %%

plot_name = "arma_3_2_python_simulation_autocorrelation"
title = f"ARMA(3,2) Autocorrelation Coefficient: φ={p}, δ={d}"
reg.autocorrelation_plot(xt, 60, title, [-0.1, 1.0], plot_name)
