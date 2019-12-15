
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import scipy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm
from lib import regression as reg
from lib import stats


wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def s_period_variance(x, s):
    t = len(x) - 1
    μ = (x[t] - x[0]) / t
    m = (t - s + 1.0)*(1.0 - s/t)
    σ = 0.0
    for i in range(s, t+1):
        σ += (x[i] - x[i-s] - μ*s)**2
    return σ / m

def vr_statistic(x, s):
    t = len(x) - 1
    var_s = s_period_variance(x, s)
    var_1 = s_period_variance(x, 1)
    vr = var_s/(s*var_1)
    θ = 2.0*(2.0*s - 1.0)*(s - 1.0)/(3.0*s*t)
    return (vr - 1.0)/numpy.sqrt(θ)

def vr_plot(s_var, s, h, title, plot_name):
    var = s**(2.0*h)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"Variance")
    axis.set_xlabel(r"$s$")
    axis.set_title(title)
    axis.loglog(s, s_var, label=r"$\sigma^2(s)$", marker='o', markersize=5.0, zorder=10, linestyle="None", markeredgewidth=1.0, alpha=0.75)
    axis.loglog(s, var, label=r"$s^{2H}$", zorder=5)
    axis.legend(bbox_to_anchor=[0.8, 0.8])
    config.save_post_asset(figure, "regression", plot_name)

def plot(samples, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"t")
    axis.set_title(title)
    axis.plot(t, samples, lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

def vr_test_plot(samples, s, sig_level, title, plot_name):
    x_vals = numpy.linspace(-8.0, 8.0, 100)
    y_vals = [scipy.stats.norm.cdf(x, 0.0, 1.0) for x in x_vals]
    left_critical_value = scipy.stats.norm.ppf(sig_level/2.0, 0.0, 1.0)
    right_critical_value = -left_critical_value
    vr_test_stat = vr_statistic(samples, s)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$CDF$")
    axis.set_xlabel(r"s")
    axis.set_title(title)
    axis.set_ylim([0.0, 1.0])
    axis.plot(x_vals, y_vals)
    axis.plot([left_critical_value, left_critical_value], [0.0, 1.0], color='red', label="Left Critical Value")
    axis.plot([right_critical_value, right_critical_value], [0.0, 1.0], color='black', label="Right Critical Value")
    axis.plot([vr_test_stat, vr_test_stat], [0.0, 1.0], color='green', label="t-Statistic")
    axis.legend()
    config.save_post_asset(figure, "regression", plot_name)

# %%

npts = 50
fs = numpy.logspace(numpy.log10(10.0), numpy.log10(1000.0), npts)
s_vals = numpy.array(list(dict.fromkeys([int(v) for v in fs])))

# %%

H = 0.5
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)

# %%

title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_fbm_H_{H}"
plot(samples, time, title, plot_name)

# %%

s_var = numpy.array([s_period_variance(samples, s) for s in s_vals])

# %%

title = f"Variance Ratio: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_H_{H}"
vr_plot(s_var, s_vals, H, title, plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)

# %%

title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_fbm_H_{H}"
plot(samples, time, title, plot_name)

# %%

s_var = numpy.array([s_period_variance(samples, s) for s in s_vals])

# %%

title = f"Variance Ratio: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_H_{H}"
vr_plot(s_var, s_vals, H, title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)

# %%

title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_fbm_H_{H}"
plot(samples, time, title, plot_name)

# %%

s_var = numpy.array([s_period_variance(samples, s) for s in s_vals])

# %%

title = f"Variance Ratio: Δt={Δt}, H={H}"
plot_name =f"variance_ratio_H_{H}"
vr_plot(s_var, s_vals, H, title, plot_name)

# %%

H = 0.5
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, 1000, α, title, plot_name)

 # %%

H = 0.55
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, 1000, α, title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, 1000, α, title, plot_name)

# %%

H = 0.52
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, 1000, α, title, plot_name)

# %%

H = 0.48
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, 1000, α, title, plot_name)
