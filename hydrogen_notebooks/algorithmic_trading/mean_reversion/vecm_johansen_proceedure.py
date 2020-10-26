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
import scipy
import datetime

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

def multivariate_test_sample(a, n, σ):
    m, _ = a.shape
    t = numpy.linspace(0.0, 10.0, n)
    x = numpy.matrix([t])
    for i in range(1, m):
        x = numpy.concatenate((x, numpy.matrix([t**(1/(i+1))])))
    ε = numpy.matrix(multivariate_normal_sample(numpy.zeros(m), σ*numpy.eye(m), n))
    y = a*x + ε.T
    return x, y

def covariance(x, y):
    _, n = x.shape
    cov = x[:,0]*y[:,0].T
    for i in range(1, n):
        cov += x[:,i]*y[:,i].T
    return cov/float(n)

def multivariate_ols(x, y):
    return covariance(y, x) * numpy.linalg.inv(covariance(x, x))

def ols_residual(x, y):
    a = multivariate_ols(x, y)
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

# def sort_eigen_values_vectors(vals, vecs):
#     n = len(vals)
#     for i in range(n)
#
# def johansen_estimate(samples):
#     m, n = samples.shape
#
#     y, x, z = vecm_anderson_form(samples)
#
#     x_star = ols_residual(z, x)
#     y_star = ols_residual(z, y)
#
#     d_star =  multivariate_ols(z, y)
#
#     Σxx = covariance(x_star, x_star)
#     Σyy = covariance(y_star, y_star)
#     Σxy = covariance(x_star, y_star)
#     Σyx = covariance(y_star, x_star)
#
#     sqrt_Σyy = numpy.matrix(scipy.linalg.sqrtm(Σyy))
#     sqrt_Σyy_inv = numpy.matrix(numpy.linalg.inv(sqrt_Σyy))
#     Σyy_inv = numpy.matrix(numpy.linalg.inv(Σyy))
#     Σxx_inv = numpy.matrix(numpy.linalg.inv(Σxx))
#
#     R = sqrt_Σyy_inv*Σyx*numpy.matrix(numpy.linalg.inv(Σxx))*Σxy*sqrt_Σyy_inv
#
#     ρ2, M = numpy.linalg.eig(R)
#
#     for r in len(ρ2):
#         crit = johansen_statistic(ρ2, n, r)

# %%

n = 10000
σ = 1.0
a = numpy.matrix([[1.0, 2.0, 1.0],
                  [4.0, 1.0, -2.0],
                  [-2.0, 1.0, 5.0]])
x, y = multivariate_test_sample(a, n, σ)
multivariate_ols(x, y)

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

title = "VECM 1 Cointegrating Vector"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_estimation_1"
samples = vecm_generate_sample(α, β, a, Ω, nsample)

# %%

comparison_plot(title, samples, α.T, β, labels, [0.1, 0.1], plot)

# %%

y, x, z = vecm_anderson_form(samples)

x_star = ols_residual(z, x)
y_star = ols_residual(z, y)

d_star =  multivariate_ols(z, y)

Σxx = covariance(x_star, x_star)
Σyy = covariance(y_star, y_star)
Σxy = covariance(x_star, y_star)
Σyx = covariance(y_star, x_star)

sqrt_Σyy = numpy.matrix(scipy.linalg.sqrtm(Σyy))
sqrt_Σyy_inv = numpy.matrix(numpy.linalg.inv(sqrt_Σyy))
Σyy_inv = numpy.matrix(numpy.linalg.inv(Σyy))
Σxx_inv = numpy.matrix(numpy.linalg.inv(Σxx))

R = sqrt_Σyy_inv*Σyx*Σxx_inv*Σxy*sqrt_Σyy_inv
R1 = Σyy_inv*Σyx*Σxx_inv*Σxy

ρ2, M = numpy.linalg.eig(R)

numpy.linalg.eig(R)
numpy.linalg.eig(R1)

johansen_statistic(ρ2, nsample, 1)
johansen_statistic_critical_value(0.95, 3, 1)

α = sqrt_Σyy*M
β = M.T*sqrt_Σyy_inv*Σyx*numpy.matrix(numpy.linalg.inv(Σxx))

# %%

df = pandas.DataFrame(samples.T)
result = vecm.coint_johansen(df, 0, 1)

result.lr1
result.cvt

result.lr2
result.cvm

result.eig
numpy.matrix(result.evec)

# %%

title = "VECM 1 Cointegrating Vector Anderson Form X"
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = "vecm_estimation_anderson_form_x_1"
comparison_plot(title, x, α.T, β, labels, [0.6, 0.1], plot)

# %%

title = "VECM 1 Cointegrating Vector Anderson Form αβX"
labels = [r"$αβx_1$", r"$αβx_2$", r"$αβx_3$"]
plot = "vecm_estimation_anderson_form_αβx_1"
comparison_plot(title, α*β*x, α.T, β, labels, [0.6, 0.1], plot)

# %%

title = "VECM 1 Cointegrating Vector Anderson Form Y"
labels = [r"$y_1$", r"$y_2$", r"$y_3$"]
plot = "vecm_estimation_anderson_form_y_1"
comparison_plot(title, y, α.T, β, labels, [0.6, 0.1], plot)

# %%

title = "VECM 1 Cointegrating Vector Anderson Form Z"
labels = [r"$z_1$", r"$z_2$", r"$z_3$"]
plot = "vecm_estimation_anderson_form_z_1"
comparison_plot(title, z, α.T, β, labels, [0.6, 0.1], plot)
