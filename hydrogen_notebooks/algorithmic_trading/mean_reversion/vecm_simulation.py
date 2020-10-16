# %%

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
import statsmodels.api as sm
import scipy

pyplot.style.use(config.glyfish_style)

# %%

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def comparison_plot(title, samples, α, β, labels, box_pos, plot):
    nplot, nsamples = samples.shape
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_xlabel(r"$t$")
    axis.set_xlim([0, nsamples-1])

    params = []
    d = ", "
    nα, _ = α.shape
    nβ, _ = β.shape
    for i in range(nα):
        params.append(f"$α_{{{i+1}}}$=[{d.join([format(elem, '2.2f') for elem in numpy.array(α[i]).flatten()])}]")
    for i in range(nβ):
        params.append(f"$β_{{{i+1}}}$=[{d.join([format(elem, '2.2f') for elem in numpy.array(β[i]).flatten()])}]")
    params_string = "\n".join(params)
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(box_pos[0], box_pos[1], params_string, fontsize=15, bbox=bbox, transform=axis.transAxes)

    for i in range(nplot):
        axis.plot(range(nsamples), samples[i].T, label=labels[i], lw=1)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def vecm_generate_sample(α, β, a, Ω, nsample):
    n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsample)))
    εt = numpy.matrix(multivariate_normal_sample(numpy.zeros(n), Ω, nsample))
    for i in range(2, nsample):
        Δxt1 = xt[:,i-1] - xt[:,i-2]
        Δxt = α*β*xt[:,i-1] + a*Δxt1 + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return xt

# %%

nsample = 1000
α = numpy.matrix([-0.5, 0.0, 0.0]).T
β = numpy.matrix([1.0, -0.5, -0.5])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 1 Cointegrationg Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_1"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.1], plot)

# %%

nsample = 1000
α = numpy.matrix([-0.5, 0.0, 0.0]).T
β = numpy.matrix([1.0, -0.5, 0.0])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 1 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_2"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.6, 0.1], plot)

# %%

nsample = 1000
α = numpy.matrix([-0.5, 0.0, 0.0]).T
β = numpy.matrix([1.0, 0.0, -0.5])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 1 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_3"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.6, 0.1], plot)

# %%

nsample = 1000
α = numpy.matrix([[-0.5, -0.5],
                  [-0.5, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, -0.5, -0.5],
                  [-0.5, 1.0, -0.5]])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_4"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.15], plot)

# %%

nsample = 1000
α = numpy.matrix([[-0.5, 0.0],
                  [0.0, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, 0.0, -0.5],
                  [0.0, 1.0, -0.5]])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_5"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.6, 0.15], plot)

# %%

nsample = 1000
α = numpy.matrix([[-0.5, 0.0],
                  [0.0, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, 0.0, -0.5],
                  [0.0, 1.0, -0.5]])
a = numpy.matrix([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_6"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.15], plot)

# %%

nsample = 1000
α = numpy.matrix([[-0.5, -0.5],
                  [0.0, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, 0.0, -0.5],
                  [0.0, 1.0, -0.5]])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_7"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.15], plot)

# %%

nsample = 1000
α = numpy.matrix([[-0.5, -0.0],
                  [0.0, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, 0.0, -0.5],
                  [-1.0, 1.0, -0.5]])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_comparison_8"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.15], plot)
