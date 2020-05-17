import numpy
import statsmodels.api as sm
from matplotlib import pyplot
from lib import config

def arima_generate_sample(φ, δ, d, n):
    samples = arma_generate_sample(φ, δ, n)
    for _ in range(d):
        samples = numpy.cumsum(samples)
    return samples

def sample_difference(samples):
    n = len(samples)
    diff = numpy.zeros(n-1)
    for i in range(n-1):
        diff[i] = samples[i+1] - samples[i]
    return diff

def arma_generate_sample(φ, δ, n):
    φ = numpy.r_[1, -φ]
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n)

def ma_generate_sample(δ, n):
    φ = numpy.array(1.0)
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n)

def ar_generate_sample(φ, n):
    φ = numpy.r_[1, -φ]
    δ = numpy.array([1.0])
    return sm.tsa.arma_generate_sample(φ, δ, n)

def arma_estimate_parameters(samples, order):
    model = sm.tsa.ARMA(samples, order).fit(trend='nc', disp=0)
    return model.params

def partial_autocorrelation(x, max_lag):
    pacf, _ = sm.regression.yule_walker(x, order=max_lag, method='mle')
    return pacf

def autocorrelation(x):
    n = len(x)
    x_shifted = x - x.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    h_fft = numpy.conj(x_fft) * x_fft
    ac = numpy.fft.ifft(h_fft)
    return ac[0:n]/ac[0]

def timeseries_comparison_plot(samples, params, tmax, title, plot_name):
    nplot, nsample = samples.shape
    ymin = numpy.amin(samples)
    ymax = numpy.amax(samples)
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, tmax-1, tmax)
    for i in range(nplot):
        stats=f"μ={format(numpy.mean(samples[i]), '2.2f')}\nσ={format(numpy.std(samples[i]), '2.2f')}"
        bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
        axis[i].text(0.05, 0.75, stats, fontsize=15, bbox=bbox, transform=axis[i].transAxes)
        axis[i].text(0.7, 0.75, params[i], fontsize=15, bbox=bbox, transform=axis[i].transAxes)
        axis[i].set_ylabel(r"$x_t$")
        axis[i].set_ylim([ymin, ymax])
        axis[i].set_xlim([0.0, tmax])
        axis[i].plot(time, samples[i,:tmax], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def acf_pcf_plot(title, samples, ylim, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    acf = numpy.real(autocorrelation(samples))[:max_lag]
    pacf = partial_autocorrelation(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim(ylim)
    axis.plot(range(max_lag), acf, label="ACF")
    axis.plot(range(1, max_lag+1), pacf, label="PACF")
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def acf_plot(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    acf = numpy.real(autocorrelation(samples))[:max_lag]
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(max_lag), acf)
    axis.plot([0.0, max_lag], [0.0, 0.0], lw=4.0, color='black')
    config.save_post_asset(figure, "mean_reversion", plot)

def pcf_plot(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    pacf = partial_autocorrelation(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(1, max_lag+1), pacf)
    axis.plot([0.0, max_lag], [0.0, 0.0], lw=4.0, color='black')
    config.save_post_asset(figure, "mean_reversion", plot)

# ADF Test

def df_test(series):
    adfuller(series, 'nc')

def adf_test(series):
    adfuller(series, 'c')

def adf_test_with_trend(series):
    adfuller(series, 'ct')

def adfuller(series, test_type):
    adf_result = stattools.adfuller(series, regression=test_type)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    isStationary = adf_result[0] < adf_result[4]["5%"]
    print(f"Is Stationary at 5%: {isStationary}")
    print("Critical Values")
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))
