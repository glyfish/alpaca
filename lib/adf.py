import numpy
from matplotlib import pyplot
from lib import config
from scipy import special

pyplot.style.use(config.glyfish_style)

# %%

def scaled_brownian_noise(n):
    return numpy.random.normal(0.0, 1.0/numpy.sqrt(n), n)

def brownian_motion(bn, t):
    return sum(bn[:t])

def unit_normal(t):
    return numpy.exp(-t**2)/numpy.sqrt(2.0*numpy.pi)

def modified_chi_squared(x):
    return  2.0*numpy.exp(-(2.0*x+1.0)/2.0) / numpy.sqrt(2.0*numpy.pi*(2.0*x+1.0))

# stochastic integral simulation
# \int_0^1{B(s)dB(s)}

def stochastic_integral_ensemble_1(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_1(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_1(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n):
        val += brownian_motion(bn, i-1)*bn[i]
    return val

# Analytic Solution of integral 1
# \frac{1}{2}[B^2(1) - 1]

def stochastic_integral_solution_1(n):
    return 0.5*(numpy.random.normal(0.0, 1.0, n)**2 - 1.0)

# stochastic integral simulation
# \int_0^1{B^2(s)ds}

def stochastic_integral_ensemble_2(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_2(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_2(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return val/n

# stochastic integral simulation
# \sqrt{\int_0^1{B^2(s)ds}}

def stochastic_integral_ensemble_3(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_3(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return numpy.sqrt(val/n)

# Dickey-Fuller Statisti distribution
# \frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_0^1{B^2(s)ds}}

def df_test_statistic_ensemble(n, nsample):
    vals = numpy.zeros(nsample)
    numerator = stochastic_integral_solution_1(nsample)
    for i in range(nsample):
        vals[i] = numerator[i] / stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

# Plots

def noise_plot(samples, time, plot_name):
    nsamples = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(f"Scaled Brownian Noise, nsamples = {nsamples}")
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "regression", plot_name)

def distribution_comparison_plot(pdf, samples, title, plot, label=None, xrange=None, ylimit=None, bins=50, title_offset=1.0):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_T(t)$")
    axis.set_xlabel(r"t")
    axis.set_title(title, y=title_offset)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, bins, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    if label is None:
        label=f"Target PDF"
    axis.plot(xrange, sample_distribution, label=label, zorder=6)
    axis.legend(bbox_to_anchor=(0.75, 0.9))
    config.save_post_asset(figure, "regression", plot)

def distribution_histogram(samples, title, plot, xrange=None, ylimit=None, bins=50, title_offset=1.0):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_T(t)$")
    axis.set_xlabel(r"t")
    axis.set_title(title, y=title_offset)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, bins, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    config.save_post_asset(figure, "regression", plot)

def pdf_histogram(samples, range, nbins=50):
    return numpy.histogram(samples, bins=nbins, range=range, density=True)

def cdf_histogram(x, pdf):
    npoints = len(pdf)
    cdf = numpy.zeros(npoints)
    for i in range(npoints):
        dx = x[i+1] - x[i]
        cdf[i] = numpy.sum(pdf[:i])*dx
    return cdf

def histogram_plot(x, f, title, ylabel, plot, title_offset=1.0):
    width = 0.9*(x[1]-x[0])
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"t")
    axis.set_title(title, y=title_offset)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    axis.bar(x, f, align='center', width=width, zorder=10)
    config.save_post_asset(figure, "regression", plot)

def adf_statistic(samples, σ=1.0):
    nsample = len(samples)
    delta_numerator = 0.0
    var = 0.0
    for i in range(1, nsample):
        delta = samples[i] - samples[i-1]
        delta_numerator += samples[i-1] * delta
        var += samples[i-1]**2

    return delta_numerator / (numpy.sqrt(var)*σ**2)
