import numpy
from matplotlib import pyplot
from lib import config
from scipy import special

pyplot.style.use(config.glyfish_style)

# %%

def chi_squared_pdf(x, k):
    return x**(k/2.0 - 1.0) * numpy.exp(-x/2.0) / (2.0**(k/2.0)*special.gamma(k/2.0))

def chi_squared_cdf(x, k):
    return special.gammainc(k/2.0, x/2.0)

def chi_squared_tail(x, k):
    return 1.0 - chi_squared_cdf(x, k)

# Plots

def distribution_multiplot(fx, x, labels, ylabel, lengend_location, ylim, title, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"$x$")
    axis.set_title(title)
    axis.set_ylim(ylim)
    for i in range(nplot):
        axis.plot(x, fx[i], label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "regression", plot_name)

def hypothesis_region_plot(fx, x, acceptance_level, title, ylabel, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel("Probability")
    axis.set_xlabel(r"$x$")
    axis.set_title(title)
    axis.plot(x, fx, label=ylabel)
    axis.plot([x[0], x[-1]], [acceptance_level, acceptance_level], label=f"Acceptance: {acceptance_level}")
    axis.legend(bbox_to_anchor=[0.8, 0.8])
    config.save_post_asset(figure, "regression", plot_name)

def distribution_plot(fx, x, title, ylabel, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"$x$")
    axis.set_title(title)
    axis.plot(x, fx)
    config.save_post_asset(figure, "regression", plot_name)
