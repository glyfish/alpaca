import numpy
from matplotlib import pyplot
from lib import config
from scipy import special

pyplot.style.use(config.glyfish_style)

# %%

def normal(σ=1.0, μ=0.0):
    def f(x):
        ε = (x - μ)**2/(2.0*σ**2)
        return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)
    return f

def chi_squared_pdf(k):
    def f(x):
        return x**(k/2.0 - 1.0) * numpy.exp(-x/2.0) / (2.0**(k/2.0)*special.gamma(k/2.0))
    return f

def chi_squared_cdf(k):
    def f(x):
        return special.gammainc(k/2.0, x/2.0)
    return f

def chi_squared_tail(k):
    def f(x):
        return 1.0 - special.gammainc(k/2.0, x/2.0)
    return f

def student_t_pdf(n):
    def f(x):
        return (1.0/numpy.sqrt(numpy.pi*n))*(special.gamma((n+1)/2.0)/special.gamma(n/2.0))*(x**2/n + 1.0)**(-(n+1)/2)
    return f

def student_t_cdf(n):
    def f(x):
        y = n/(x**2 + n)
        if x > 0:
            return 1.0 - 0.5*special.betainc(n/2, 0.5, y)
        else:
            return 0.5*special.betainc(n/2, 0.5, y)
    return f

def student_t_tail(n):
    def f(x):
        y = n/(x**2 + n)
        if x > 0:
            return 0.5*special.betainc(n/2, 0.5, y)
        else:
            return 1.0 - 0.5*special.betainc(n/2, 0.5, y)
    return f

# Plots

def pdf_samples(pdf, samples, title, ylabel, xlabel, plot, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, 50, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    axis.plot(xrange, sample_distribution, label=f"Target PDF", zorder=6)
    axis.legend(bbox_to_anchor=(0.75, 0.9))
    config.save_post_asset(figure, "regression", plot)

def distribution_multiplot(fx, x, labels, ylabel, xlabel, lengend_location, ylim, title, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_ylim(ylim)
    for i in range(nplot):
        axis.plot(x, fx[i], label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "regression", plot_name)

def distribution_comparission_multiplot(fx, gx, x, labels, ylabel, xlabel, lengend_location, ylim, title, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_ylim(ylim)
    for i in range(nplot):
        axis.plot(x, fx[i], label=labels[i])
    axis.plot(x, gx, label=labels[-1], lw=3.0, color='black')
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "regression", plot_name)

def hypothesis_region_plot(fx, x, ylabel, xlabel, acceptance_level, title, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.plot(x, fx, label=ylabel)
    axis.plot([x[0], x[-1]], [acceptance_level, acceptance_level], label=f"Acceptance: {acceptance_level}")
    axis.legend(bbox_to_anchor=[0.8, 0.8])
    config.save_post_asset(figure, "regression", plot_name)

def distribution_plot(fx, x, title, ylabel, xlabel, plot_name):
    nplot = len(fx)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.plot(x, fx)
    config.save_post_asset(figure, "regression", plot_name)
