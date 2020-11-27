import numpy
from matplotlib import pyplot
from lib import config
import scipy
import pandas

from statsmodels.tsa.vector_ar import vecm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

# Plots
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
        if i == 0:
            axis[i].legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

# Statistical Tests
def johansen_coint(df, report=False):
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

    return rank

def adfuller_test(samples, regression):
    adf_result = sm.tsa.stattools.adfuller(samples, regression=regression)
    return adf_result[0] < adf_result[4]["5%"]

def adfuller_report(samples, regression):
    adf_result = sm.tsa.stattools.adfuller(samples, regression=regression)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    isStationary = adf_result[0] < adf_result[4]["5%"]
    print(f"Is Stationary at 5%: {isStationary}")
    print("Critical Values")
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))
    return adf_result[0] < adf_result[4]["5%"]

def adf_report(samples, regression='c'):
    return adfuller_report(samples, regression)

def adf_test(samples, regression='c'):
    return adfuller_test(samples, regression)

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

def sample_adf_test(df, report=False):
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

# Utilities
def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

def pacf(samples, nlags):
    return sm.tsa.stattools.pacf(samples, nlags=nlags, method="ywunbiased")

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

# Parameter Estimation
def cointgration_params_estimate(df, rank):
    vars = df.columns
    result = []
    for i in range(rank):
        formula = vars[0] + " ~ " +  " + ".join(vars[1:])
        vars = numpy.roll(vars, -1)
        result.append(multiple_ols(df, formula))
    return result

def simple_multivariate_ols(x, y):
    return covariance(y, x) * numpy.linalg.inv(covariance(x, x))

def multiple_ols(df, formula):
    return smf.ols(formula=formula, data=df).fit()

def vecm_estimate(df, maxlags, rank, deterministic="nc", report=False):
    result = VECM(df, k_ar_diff=maxlags, coint_rank=rank, deterministic=deterministic).fit()
    if report:
        print(f"Rank={result.coint_rank}")
        print(f"α={result.alpha}")
        print(f"β={result.beta}")
        print(f"γ={result.gamma}")
    return result

def aic_order(df, maxlags):
    return VAR(df).select_order(maxlags=maxlags).selected_orders['aic']

# Transformations
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

def log(df):
    samples = data_frame_to_samples(df)
    return samples_to_data_frame(numpy.log(samples))

def exp(df):
    samples = data_frame_to_samples(df)
    return samples_to_data_frame(numpy.exp(samples))
