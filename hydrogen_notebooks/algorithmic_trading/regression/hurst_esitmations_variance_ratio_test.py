
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import stats as scipy_stats
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm
from lib import regression as reg
from lib import stats


wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def hetero_delta_factor(x, j):
    t = len(x) - 1
    μ = (x[t] - x[0]) / t
    factor = 0.0
    for i in range(j+1, t):
        f1 = (x[i] - x[i-1] - μ)**2
        f2 = (x[i-j] - x[i-j-1] - μ)**2
        factor += f1*f2
    return factor / s_period_variance(x, 1)**2

def hetero_theta_factor(x, s):
    t = len(x) - 1
    μ = (x[t] - x[0]) / t
    factor = 0.0
    for j in range(1, s):
        delta = hetero_delta_factor(x, j)
        factor += delta*(2.0*(s-j)/s)**2
    return factor/t**2

def hetero_vr_statistic(x, s):
    t = len(x) - 1
    var_s = s_period_variance(x, s)
    var_1 = s_period_variance(x, 1)
    vr = var_s/(s*var_1)
    θ = hetero_theta_factor(x, s)
    return (vr - 1.0)/numpy.sqrt(θ)

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
    y_vals = [scipy_stats.norm.cdf(x, 0.0, 1.0) for x in x_vals]
    left_critical_value = scipy_stats.norm.ppf(sig_level/2.0, 0.0, 1.0)
    right_critical_value = -left_critical_value
    vr_test_stat = vr_statistic(samples, s)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$CDF$")
    axis.set_xlabel(r"s")
    axis.set_title(title)
    axis.set_ylim([-0.05, 1.05])
    axis.plot(x_vals, y_vals)
    axis.plot([left_critical_value, left_critical_value], [0.0, 1.0], color='red', label="Left Critical Value")
    axis.plot([right_critical_value, right_critical_value], [0.0, 1.0], color='black', label="Right Critical Value")
    axis.plot([vr_test_stat, vr_test_stat], [0.0, 1.0], color='green', label="t-Statistic")
    axis.legend()
    config.save_post_asset(figure, "regression", plot_name)

def multi_vr_test_plot(samples, s, sig_level, title, plot_name):
    nvals = len(s)
    x_vals = numpy.linspace(-8.0, 8.0, 100)
    y_vals = [scipy_stats.norm.cdf(x, 0.0, 1.0) for x in x_vals]
    left_critical_value = scipy_stats.norm.ppf(sig_level/2.0, 0.0, 1.0)
    right_critical_value = -left_critical_value
    vr_test_stat = []
    for i in range(nvals):
        vr_test_stat.append(vr_statistic(samples, s[i]))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$CDF$")
    axis.set_xlabel(r"s")
    axis.set_title(title)
    axis.set_ylim([-0.05, 1.05])
    axis.plot(x_vals, y_vals)
    axis.plot([left_critical_value, left_critical_value], [0.0, 1.0], color='red', label="Left Critical Value")
    axis.plot([right_critical_value, right_critical_value], [0.0, 1.0], color='black', label="Right Critical Value")
    for i in range(nvals):
        axis.plot([vr_test_stat[i], vr_test_stat[i]], [0.0, 1.0], label=f"s={s[i]}")
    axis.legend()
    config.save_post_asset(figure, "regression", plot_name)

def multi_hetero_vr_test_plot(samples, s, sig_level, title, plot_name):
    nvals = len(s)
    x_vals = numpy.linspace(-8.0, 8.0, 100)
    y_vals = [scipy_stats.norm.cdf(x, 0.0, 1.0) for x in x_vals]
    left_critical_value = scipy_stats.norm.ppf(sig_level/2.0, 0.0, 1.0)
    right_critical_value = -left_critical_value
    vr_test_stat = []
    for i in range(nvals):
        vr_test_stat.append(hetero_vr_statistic(samples, s[i]))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$CDF$")
    axis.set_xlabel(r"s")
    axis.set_title(title)
    axis.set_ylim([-0.05, 1.05])
    axis.plot(x_vals, y_vals)
    axis.plot([left_critical_value, left_critical_value], [0.0, 1.0], color='red', label="Left Critical Value")
    axis.plot([right_critical_value, right_critical_value], [0.0, 1.0], color='black', label="Right Critical Value")
    for i in range(nvals):
        axis.plot([vr_test_stat[i], vr_test_stat[i]], [0.0, 1.0], label=f"s={s[i]}")
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
s = 1000

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}, s={s}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.55
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = 1000

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}, s={s}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.45
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = 1000

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}, s={s}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.3
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = 1000

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}, s={s}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.6
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = 1000

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}, s={s}"
plot_name =f"variance_ratio_test_H_{H}"
vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.5
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [100, 1000, 10000]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}"
multi_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.45
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [100, 1000, 10000]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}"
multi_vr_test_plot(samples, s, α, title, plot_name)


# %%

H = 0.55
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [100, 1000, 10000]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}"
multi_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.6
Δt = 1.0
npts = 2**16
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [100, 1000, 10000]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}"
multi_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.45
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}_2"
multi_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.55
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}_2"
multi_vr_test_plot(samples, s, α, title, plot_name)


# %%

H = 0.5
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_test_lag_scan_H_{H}_2"
multi_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.5
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedastic Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.6
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedasticity Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.55
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedasticity Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.4
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedasticity Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.45
Δt = 1.0
Δt = 1.0
npts = 2**10
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedasticity Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)

# %%

H = 0.5
Δt = 1.0
Δt = 1.0
npts = 2**14
samples = bm.fbm_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
α = 0.05
s = [2, 10, 20]

# %%

title = f"Heteroscedastic Variance Ratio Test: Δt={Δt}, H={H}, α={α}"
plot_name =f"variance_ratio_hetero_test_lag_scan_H_{H}_2"
multi_hetero_vr_test_plot(samples, s, α, title, plot_name)
