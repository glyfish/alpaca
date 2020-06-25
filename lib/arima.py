import numpy
import statsmodels.api as sm
import statsmodels
from matplotlib import pyplot
from lib import config

# Sample generation

def sample_difference(samples):
    n = len(samples)
    diff = numpy.zeros(n-1)
    for i in range(n-1):
        diff[i] = samples[i+1] - samples[i]
    return diff

def arima_generate_sample(φ, δ, d, n, σ=1.0):
    assert d <= 2, "d must equal 1 or 2"
    samples = arma_generate_sample(φ, δ, n, σ)
    if d == 1:
        return numpy.cumsum(samples)
    else:
        for i in range(2, n):
            samples[i] = samples[i] + 2.0*samples[i-1] - samples[i-2]
        return samples

def arma_generate_sample(φ, δ, n, σ=1.0):
    φ = numpy.r_[1, -φ]
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

def ma_generate_sample(δ, n, σ=1.0):
    φ = numpy.array(1.0)
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

def ar_generate_sample(φ, n, σ=1.0):
    φ = numpy.r_[1, -φ]
    δ = numpy.array([1.0])
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

def ecm_sample_generate(arima_params, ecm_params, n):
    δ = arima_params["δ"]
    φ = arima_params["φ"]
    d = arima_params["d"]
    xt = arima_generate_sample(φ, δ, d, n)
    if "δ" in ecm_params:
        δ = ecm_params["δ"]
    else:
        δ = 0.0
    if "α" in ecm_params:
        α = ecm_params["α"]
    else:
        α = 0.0
    if "σ" in ecm_params:
        σ = ecm_params["σ"]
    else:
        σ = 1.0
    λ = ecm_params["λ"]
    β = ecm_params["β"]
    γ = ecm_params["γ"]
    yt = numpy.zeros(n)
    ξt = numpy.random.normal(0.0, σ, n)
    for i in range(1, n):
        Δxt = xt[i] - xt[i-1]
        Δyt = δ + γ*Δxt + λ*(yt[i-1] - α - β*xt[i-1]) + ξt[i]
        yt[i] = Δyt + yt[i-1]
    return xt, yt

# Parameter estimation

def arima_estimate_parameters(samples, order):
    model = statsmodels.tsa.arima_model.ARIMA(samples, order=order)
    return model.fit(disp=False)

def arma_estimate_parameters(samples, order):
    return statsmodels.tsa.arima_model.ARMA(samples, order).fit(trend='nc', disp=0)

def yule_walker(x, max_lag):
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

def ols_estimate(xt, yt):
    xt = sm.add_constant(xt)
    model = sm.OLS(yt, xt)
    results = model.fit()
    print(results.summary())
    return results.params, results.rsquared, results.bse

def ecm_estimate_parameters(xt, yt, α, β):
    n = len(xt)
    εt = yt - α - β*xt
    Δxt = sample_difference(xt)
    Δyt = sample_difference(yt)
    return ols_estimate(numpy.transpose(numpy.array([Δxt, εt[:n-1]])), Δyt)

# ACF-PACF

def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

def pacf(samples, nlags):
    return sm.tsa.stattools.pacf(samples, nlags=nlags, method="ywunbiased")

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

def acf_yule_walker_pcf_plot(title, samples, ylim, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    acf_values = acf(samples, max_lag)
    pacf_values = yule_walker(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim(ylim)
    axis.plot(range(max_lag+1), acf_values, label="ACF")
    axis.plot(range(1, max_lag+1), pacf_values, label="PACF")
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def acf_pcf_plot(title, samples, ylim, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    acf_values = acf(samples, max_lag)
    pacf_values = pacf(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim(ylim)
    axis.plot(range(max_lag+1), acf_values, label="ACF")
    axis.plot(range(max_lag+1), pacf_values, label="PACF")
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def acf_plot(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    acf_values = acf(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(max_lag+1), acf_values)
    axis.plot([0.0, max_lag], [0.0, 0.0], lw=4.0, color='black')
    config.save_post_asset(figure, "mean_reversion", plot)

def pcf_plot(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    pacf_values = pacf(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(max_lag+1), pacf_values)
    axis.plot([0.0, max_lag], [0.0, 0.0], lw=4.0, color='black')
    config.save_post_asset(figure, "mean_reversion", plot)

# ADF Test

def df_test(series):
    return adfuller_report(series, 'nc')

def adf_report(series):
    return adfuller_report(series, 'c')

def adf_report_with_trend(series):
    return adfuller_report(series, 'ct')

def adf_test(samples):
    return adfuller_test(series, 'c')

def adfuller_test(series, test_type):
    adf_result = sm.tsa.stattools.adfuller(series, regression=test_type)
    return adf_result[0] < adf_result[4]["5%"]

def adfuller_report(series, test_type):
    adf_result = sm.tsa.stattools.adfuller(series, regression=test_type)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    isStationary = adf_result[0] < adf_result[4]["5%"]
    print(f"Is Stationary at 5%: {isStationary}")
    print("Critical Values")
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))
    return adf_result[0] < adf_result[4]["5%"]
