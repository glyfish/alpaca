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

# Simulation
def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def vecm_generate_sample(α, β, a, Ω, nsample):
    n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsample)))
    εt = numpy.matrix(multivariate_normal_sample(numpy.zeros(n), Ω, nsample))
    for i in range(2, nsample):
        Δxt1 = xt[:,i-1] - xt[:,i-2]
        Δxt = α*β*xt[:,i-1] + a*Δxt1 + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return samples_to_data_frame(xt)

# Estimation
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
    if report:
        ρ2 = result.eig
        M = numpy.matrix(result.evec)
        print(f"Rank={rank}")
        print("Eigen Values\n", ρ2)
        print("Eigen Vectors\n", M)

    if rank is None and report:
        print("Reduced Rank Solution Does Not Exist")

    return rank

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

def residual_adf_test(df, params, report=True):
    samples = data_frame_to_samples(df)
    x = numpy.matrix(params[1:])*samples[1:,:]
    y = numpy.squeeze(numpy.asarray(samples[0,:]))
    εt = y - params[0] - numpy.squeeze(numpy.asarray(x))
    if report:
        return arima.adf_report(εt)
    else:
        return arima.adf_test(εt)

def sample_adf_test(df, report=True):
    results = []
    for c in df.columns:
        samples = df[c].to_numpy()
        if report:
            print(f">>> ADF Test Result for: {c}")
            results.append(arima.adf_report(samples))
            print("")
        else:
            results.append(arima.adf_test(samples))
    return results

def multiple_ols(df, formula):
    return smf.ols(formula=formula, data=df).fit()

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

title = "Trivariate VECM 1 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_estimation_comparison_trivariate_1"
df = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, df, α.T, β, labels, [0.6, 0.2], plot)

# %%

title = "Trivariate VECM 1 Cointegrating Vector First Difference"
labels = [r"$Δx_1$", r"$Δx_2$", r"$Δx_3$"]
plot = "vecm_estimation_samples_diff_1"
df_diff_1 = difference(df)
comparison_plot(title, df_diff_1, α.T, β, labels, [0.1, 0.8], plot)

# %%

rank = johansen_coint(df, report=True)

# %%

sample_adf_test(df)

# %%

result = VAR(df_diff_1).select_order(maxlags=15)
result.ics
result.selected_orders
lag_order = result.selected_orders['aic']

# %%

result = VECM(df, k_ar_diff=lag_order, coint_rank=rank, deterministic="nc").fit()
result.coint_rank
result.alpha
result.beta
result.gamma

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

title = "Trivariate VECM 2 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_estimation_comparison_trivariate_2"
df = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, df, α.T, β, labels, [0.6, 0.2], plot)

# %%

title = "Trivariate VECM 2 Cointegrating Vector First Difference"
labels = [r"$Δx_1$", r"$Δx_2$", r"$Δx_3$"]
plot = "vecm_estimation_samples_diff_2"
df_diff_1 = difference(df)
comparison_plot(title, df_diff_1, α.T, β, labels, [0.6, 0.1], plot)

# %%

sample_adf_test(df)

# %%

rank = johansen_coint(df, report=True)

# %%

result = VAR(df_diff_1).select_order(maxlags=15)
result.ics
result.selected_orders
lag_order = result.selected_orders['aic']

# %%

result = VECM(df, k_ar_diff=lag_order, coint_rank=rank, deterministic="nc").fit()
result.coint_rank
result.alpha
result.beta
result.gamma
