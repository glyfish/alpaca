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
from statsmodels.tsa.vector_ar import vecm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

pyplot.style.use(config.glyfish_style)

# %%

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def comparison_plot(title, df, α, β, labels, box_pos, plot):
    samples = data_frame_to_samples(df)
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

def scatter_matrix_plot(title, df, plot_name):
    nsample, nvar = df.shape
    vars = df.columns
    samples = data_frame_to_samples(df)
    figure, axis = pyplot.subplots(nvar, nvar, figsize=(9, 9))
    figure.subplots_adjust(wspace=0.1, hspace=0.1)
    axis[0, int(nvar/2)].set_title(title, y=1.05)
    for i in range(nvar):
        yt = samples[i].T
        axis[i,0].set_ylabel(vars[i])
        for j in range(nvar):
            if j != 0:
                axis[i,j].set_yticklabels([])
            if i == nvar - 1:
                axis[i,j].set_xlabel(vars[j])
            else:
                axis[i,j].set_xticklabels([])
            xt = samples[j].T
            axis[i,j].set_ylim([numpy.amin(yt), numpy.amax(yt)])
            axis[i,j].set_xlim([numpy.amin(xt), numpy.amax(xt)])
            axis[i,j].plot(xt, yt, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5)

    config.save_post_asset(figure, "mean_reversion", plot_name)

def acf_pcf_plot(title, df, max_lag, plot):
    samples = data_frame_to_samples(df)
    vars = df.columns
    nplot, n = samples.shape
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)

    for i in range(nplot):
        data = numpy.squeeze(numpy.array(samples[i]))
        acf_values = acf(data, max_lag)
        pacf_values = pacf(data, max_lag)
        if i == nplot - 1:
            axis[i].set_xlabel("Time Lag (τ)")
        axis[i].set_ylabel(vars[i])
        axis[i].set_xlim([-0.1, max_lag])
        axis[i].set_ylim([-1.1, 1.1])
        axis[i].plot(range(max_lag+1), acf_values, label="ACF")
        axis[i].plot(range(max_lag+1), pacf_values, label="PACF")
        axis[i].legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

# Implementation from Reduced Rank Regression For the Multivariate Linear Model
def covariance(x, y):
    _, n = x.shape
    cov = x[:,0]*y[:,0].T
    for i in range(1, n):
        cov += x[:,i]*y[:,i].T
    return cov/float(n)

def ols_residual(x, y):
    a = simple_multivariate_ols(x, y)
    return y-a*x

def vecm_anderson_form(samples):
    Δ = numpy.diff(samples)
    y = Δ[:,0:-1]
    z = Δ[:,1:]
    x = samples[:,1:-1]
    return y, x, z

def johansen_statistic(ρ2, n, r):
    m = len(ρ2)
    λ = numpy.log(numpy.ones(m-r)-ρ2[r:])
    return -n * numpy.sum(λ)

def johansen_statistic_critical_value(p, m, r):
    return scipy.stats.chi2.ppf(p, (m-r)**2)

def johansen_coint_theory(df, report=True):
    samples = data_frame_to_samples(df)
    m, n = samples.shape

    y, x, z = vecm_anderson_form(samples)

    x_star = ols_residual(z, x)
    y_star = ols_residual(z, y)

    d_star =  simple_multivariate_ols(z, y)

    Σxx = covariance(x_star, x_star)
    Σyy = covariance(y_star, y_star)
    Σxy = covariance(x_star, y_star)
    Σyx = covariance(y_star, x_star)

    sqrt_Σyy = numpy.matrix(scipy.linalg.sqrtm(Σyy))
    sqrt_Σyy_inv = numpy.matrix(numpy.linalg.inv(sqrt_Σyy))
    Σyy_inv = numpy.matrix(numpy.linalg.inv(Σyy))
    Σxx_inv = numpy.matrix(numpy.linalg.inv(Σxx))

    R = sqrt_Σyy_inv*Σyx*Σxx_inv*Σxy*sqrt_Σyy_inv

    ρ2, M = numpy.linalg.eig(R)
    idx = ρ2.argsort()[::-1]
    ρ2 = ρ2[idx]
    M = M[:,idx]

    rank = None
    for r in range(m):
        cv = johansen_statistic_critical_value(0.99, m, r)
        l = johansen_statistic(ρ2, n, r)
        if rank is None:
            print(f"Critical Value: {cv}, Trace Statistic: {l}")
        if l < cv:
            rank = r
            break

    α = sqrt_Σyy*M[:,:rank]
    β = M[:,:rank].T*sqrt_Σyy_inv*Σyx*numpy.matrix(numpy.linalg.inv(Σxx))

    if report:
        print(f"Rank={rank}")
        print("Eigen Values\n", ρ2)
        print("Eigen Vectors\n", M)
        print("α\n", α)
        print("β\n", β)

    if rank is None:
        print("Reduced Rank Solution Does Not Exist")
        return None

    return ρ2[:rank], M[:,:rank], α, β

# scipy implementation
def johansen_coint(df, report=True):
    samples = data_frame_to_samples(df)
    m, _  = samples.shape

    df = pandas.DataFrame(samples.T)
    result = vecm.coint_johansen(df, 0, 1)

    l = result.lr1
    cv = result.cvt

    # 0: 90%  1:95% 2: 99%
    rank = None
    for r in range(m):
        if report:
            print(f"Critical Value: {cv[r, 2]}, Trace Statistic: {l[r]}")
        if l[r] < cv[r, 2]:
            rank = r
            break

    ρ2 = result.eig
    M = numpy.matrix(result.evec)

    if report:
        print(f"Rank={rank}")
        print("Eigen Values\n", ρ2)
        print("Eigen Vectors\n", M)

    if rank is None:
        print("Reduced Rank Solution Does Not Exist")
        return None

    return ρ2[:rank], M[:,:rank]

# Data generation
def vecm_generate_sample(α, β, a, Ω, nsample):
    n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsample)))
    εt = numpy.matrix(multivariate_normal_sample(numpy.zeros(n), Ω, nsample))
    for i in range(2, nsample):
        Δxt1 = xt[:,i-1] - xt[:,i-2]
        Δxt = α*β*xt[:,i-1] + a*Δxt1 + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return samples_to_data_frame(xt)

def multivariate_test_sample(a, n, σ):
    m, _ = a.shape
    t = numpy.linspace(0.0, 10.0, n)
    x = numpy.matrix(t)
    for i in range(1, m):
        x = numpy.concatenate((x, numpy.matrix([t**(1/(i+1))])))
    ε = numpy.matrix(multivariate_normal_sample(numpy.zeros(m), σ*numpy.eye(m), n))
    y = a*x + ε.T
    return x, y

# Utilities
def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

def pacf(samples, nlags):
    return sm.tsa.stattools.pacf(samples, nlags=nlags, method="ywunbiased")

def simple_multivariate_ols(x, y):
    return covariance(y, x) * numpy.linalg.inv(covariance(x, x))

def multiple_ols(df, formula):
    return smf.ols(formula=formula, data=df).fit()

# Regression

def cointgration_params_estimate(df, rank):
    vars = df.columns
    result = []
    for i in range(rank):
        formula = vars[0] + " ~ " +  " + ".join(vars[1:])
        vars = numpy.roll(vars, -1)
        result.append(multiple_ols(df, formula))
    return result

# Data trandformations
def samples_to_data_frame(samples):
    m, n = samples.shape
    columns = [f"x{i+1}" for i in range(m)]
    index = (pandas.date_range(pandas.Timestamp.now(tz="UTC"), periods=n) - pandas.Timedelta(days=n)).normalize()
    df = pandas.DataFrame(samples.T, columns=columns, index=index)
    return df

def data_frame_to_samples(df):
    return numpy.matrix(df.to_numpy()).T

def difference(df):
    return df.diff().dropna()

# Statistical tests
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

def adf_report(series):
    return adfuller_report(series, 'c')

def adf_test(samples):
    return adfuller_test(series, 'c')

def residual_adf_test(df, params, report=False):
    nres, _ = params.shape
    samples = data_frame_to_samples(df)
    result = []
    for i in range(nres):
        p = -numpy.matrix(numpy.concatenate((params[i,:i], params[i,i+1:]))/params[i,i])
        s = numpy.concatenate((samples[:i], samples[i+1:]))
        x = p*s
        y = numpy.squeeze(numpy.asarray(samples[i,:]))
        εt = y - numpy.squeeze(numpy.asarray(x))
        if report:
            result.append(adf_report(εt))
        else:
            result.append(adf_test(εt))
    return result

def sample_adf_test(df, report=True):
    results = []
    for c in df.columns:
        samples = df[c].to_numpy()
        if report:
            print(f">>> ADF Test Result for: {c}")
            results.append(adf_report(samples))
            print("")
        else:
            results.append(adf_test(samples))
    return results

def causality_matrix(df, maxlag, cv=0.05, report=False):
    vars = df.columns
    nvars = len(vars)
    results = pandas.DataFrame(numpy.zeros((nvars, nvars)), columns=vars, index=vars)
    for col in vars:
        for row in vars:
            test_result = grangercausalitytests(df[[row, col]], maxlag=maxlag, verbose=False)
            pvals = [round(test_result[i+1][0]["ssr_ftest"][1], 2) for i in range(maxlag)]
            result = numpy.min(pvals) <= cv
            if report:
                print(f"{col} causes {row}: {result}")
            results.loc[row, col] = result
    return results

# %%
# Test multivariate regression

n = 10000
σ = 1.0
a = numpy.matrix([[1.0, 2.0, 1.0],
                  [4.0, 1.0, -2.0],
                  [-2.0, 1.0, 5.0]])
x, y = multivariate_test_sample(a, n, σ)
simple_multivariate_ols(x, y)

# %%

result = LinearRegression().fit(x.T, y.T)
result.coef_

# %%

df = samples_to_data_frame(numpy.concatenate((y[0], x)))
result = multiple_ols(df, "x1 ~ x2 + x3 + x4")
print(result.summary())

df = samples_to_data_frame(numpy.concatenate((y[1], x)))
result = multiple_ols(df, "x1 ~ x2 + x3 + x4")
print(result.summary())

df = samples_to_data_frame(numpy.concatenate((y[2], x)))
result = multiple_ols(df, "x1 ~ x2 + x3 + x4")
print(result.summary())

# %%
# Test one cointegration vector with one cointegration vector

nsample = 1000
α = numpy.matrix([-0.5, 0.0, 0.0]).T
β = numpy.matrix([1.0, -0.5, -0.5])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = "Trivariate VECM 1 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_analysis_samples"
df = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, df, α.T, β, labels, [0.1, 0.1], plot)

# %%

johansen_coint_theory(df)

# %%

johansen_coint(df)

# %%

results = cointgration_params_estimate(df, 1)
print(results[0].summary())

# %%

sample_adf_test(df)

# %%

title = "Trivariate VECM 1 Cointegrating Vector First Difference"
labels = [r"$Δx_1$", r"$Δx_2$", r"$Δx_3$"]
plot = "vecm_analysis_samples_diff_1"
df_diff_1 = difference(df)
comparison_plot(title, df_diff_1, α.T, β, labels, [0.1, 0.8], plot)

# %%

sample_adf_test(df_diff_1)

# %%

result = VAR(df_diff_1).select_order(maxlags=15)
result.ics
result.selected_orders

# %%

result = VECM(df, k_ar_diff=1, coint_rank=1, deterministic="nc").fit()
result.coint_rank
α = result.alpha
β = result.beta.T
result.gamma

# %%

residual_adf_test(df, β, report=True)

# %%

title = "Trivariate VECM 1 Cointegrating Vector ACF-PCF"
plot = "vecm_analysis_acf_pcf_1"
max_lag = 9
acf_pcf_plot(title, df, max_lag, plot)

# %%

df.corr()

# %%

df.cov()

# %%

title = "Trivariate VECM 1 Cointegrating Vector Scatter Matrix"
plot = "vecm_analysis_scatter_matrix_1"
scatter_matrix_plot(title, df, plot)

# %%

df_diff_1.corr()

# %%

df_diff_1.cov()

# %%

title = "Trivariate VECM 1 Cointegrating Vector Difference Scatter Matrix"
plot = "vecm_analysis_differencescatter_matrix_1"
scatter_matrix_plot(title, df_diff_1, plot)

# %%

causality_matrix(df_diff_1, 2, cv=0.05)
