# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import math
from scipy import special
from scipy.stats import chi2
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def brownian_noise(μ, σ, n):
    return numpy.random.normal(μ, σ, n)

def donsker_brownian_motion(μ, bn, t, n):
    if t == 0.0:
        return 0.0
    m = math.floor(n*t)
    w = 0.0
    for i in range(m):
        w += bn[i]
    return (w - μ * n * t)/numpy.sqrt(n)

def donsker_brownian_motion_plot(μ, σ, nvals):
    title = f"Donsker CLT μ={μ}, σ={σ}"
    plot_name = f"donsker_μ-{μ}_σ-{σ}_n-{nvals[-1]}"
    nplot = len(nvals)
    bn = brownian_noise(μ, σ, nvals[-1])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$W_n(t)$")
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    for i in range(nplot):
        t = numpy.linspace(0.0, 1.0, nvals[i]+1)
        w = [donsker_brownian_motion(μ, bn, t[j], nvals[i]) for j in range(nvals[i]+1)]
        axis.plot(t, w, label=f"n={nvals[i]}")
    axis.legend(bbox_to_anchor=(0.25, 0.3))
    config.save_post_asset(figure, "regression", plot_name)

# %%

nvals = [50, 100, 500, 1000]

# %%

μ = 0.0
σ = 1.0
donsker_brownian_motion_plot(μ, σ, nvals)


# %%

μ = 0.0
σ = 2.0
donsker_brownian_motion_plot(μ, σ, nvals)

# %%

μ = 1.0
σ = 1.0
donsker_brownian_motion_plot(μ, σ, nvals)
