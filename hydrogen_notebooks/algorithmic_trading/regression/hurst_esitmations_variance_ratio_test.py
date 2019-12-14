
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm
from lib import regression as reg
from lib import stats
import statsmodels.api as sm

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def s_period_variance(x, s):
    t = len(x) - 1
    μ = (x[t] - x[0]) / t
    m = s*(t - s + 1.0)/(1.0 - s/t)
    σ = 0.0
    for i in range(s, t+1):
        σ += (x[i] - x[i-s] - μ*s)**2
    return σ / m

def vr_statistic(x, s):
    t = len(x) - 1
    var_s = s_period_variance(x, s)
    var_1 = s_period_variance(x, 1)
    vr = var_s/var_1
    θ = 2.0*(2.0*s - 1.0)*(s - 1.0)/(3.0*s*t)
    return (vr - 1.0)/numpy.sqrt(θ)

def vr_plot(s_var, s, h, title, plot_name):
    var = s**(2.0*h)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"Variance")
    axis.set_xlabel(r"$s$")
    axis.set_title(title)
    axis.loglog(s_var, s, label=r"$\sigma^2(s)$", marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75)
    axis.loglog(var, s, label=r"$s^{2H}$")
    axis.legend(bbox_to_anchor=[0.8, 0.8])
    config.save_post_asset(figure, "regression", plot_name)

def plot(sample, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"t")
    axis.set_title(title)
    axis.plot(t, sample, lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

# %%

npts = 100
fs = numpy.logspace(numpy.log10(10.0), numpy.log10(1000.0), npts)
s_vals = list(dict.fromkeys([int(v) for v in fs]))

# %%

H = 0.5
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)

# %%

title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
plot_name =f"variance_ration_fbm_H_{H}"
plot(samples, time, title, plot_name)

# %%

s_var = [s_period_variance(samples, s) for s in s_vals]

s_var
