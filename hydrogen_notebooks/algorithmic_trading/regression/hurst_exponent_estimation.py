# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def aggregated_process(samples, m):
    n = len(samples)
    d = int(n / m)
    agg = numpy.zeros(d)
    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k] / m
    return agg

def aggregated_variance(samples, npts, m_min):
    n = len(samples)
    m_vals = numpy.logspace(numpy.log10(m_min), numpy.log10(n/10.0), npts)
    agg_var = numpy.zeros(npts)
    for i in range(npts):
        m = int(m_vals[i])
        agg = aggregated_process(samples, m)
        agg_mean = numpy.mean(agg)
        for k in range(len(agg)):
            agg_var[l] += (agg[k] - agg_mean)**2/(d - 1)
    return agg_var

def agg_var_plot(agg_var, m_min, npts, title, plot_name):
    n = len(samples)
    m_vals = numpy.logspace(numpy.log10(m_min), numpy.log10(n/10.0), npts)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$Var(X^{m})$")
    axis.set_xlabel(r"$m$")
    axis.set_title(title)
    axis.loglog(m_vals, agg_var)
    config.save_post_asset(figure, "regression", plot_name)

# %%

Δt = 1.0
npts = 2**16

# %%

H = 0.4
samples = bm.fbm_fft(H, Δt, npts)
title = f"FFT Fractional Brownian Motion: Δt={Δt}, H={H}"
plot =f"fbm_fft_H_{H}"
bm.plot(samples, time, title, plot)

# %%

m = 100
agg = aggregated_process(samples, m)
nagg = len(agg)
agg_time = numpy.linspace(0.0, float(m*nagg)*Δt, nagg)
title = f"Aggregated Process: H={H}, m={m}"
plot_name = f"fbm_aggregated_process_H_{H}_{npts}"
bm.plot(agg, agg_time, title, plot)

# %%

nagg = 500
title = f"Aggregated Variance: H={H}"
plot_name = f"fbm_aggregated_variance_H_{H}_{npts}"
agg_var = aggregated_variance(samples, nagg, 100.0)
agg_var_plot(agg_var, 100.0, nagg, title, plot_name)

# %%

H = 0.8
samples = bm.fbm_fft(H, Δt, npts)
title = f"FFT Fractional Brownian Motion: Δt={Δt}, H={H}"
plot =f"fbm_fft_H_{H}"
bm.plot(samples, time, title, plot)

# %%
