# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import stats
from lib import config

pyplot.style.use(config.glyfish_style)

# %%

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def timeseries_plot(samples, ylabel, title, plot_name):
    nplot, nsample = samples.shape
    ymin = numpy.amin(samples)
    ymax = numpy.amax(samples)
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        stats=f"μ={format(numpy.mean(samples[i]), '2.2f')}\nσ={format(numpy.std(samples[i]), '2.2f')}"
        bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
        axis[i].text(0.05, 0.75, stats, fontsize=15, bbox=bbox, transform=axis[i].transAxes)
        axis[i].set_ylabel(ylabel[i])
        axis[i].set_ylim([ymin, ymax])
        axis[i].set_xlim([0.0, nsample])
        axis[i].plot(time, samples[i], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def var_simulate(x0, μ, φ, Ω, n):
    m, l = x0.shape
    xt = numpy.zeros((m, n))
    ε = multivariate_normal_sample(μ, Ω, n)
    for i in range(l):
        xt[:,i] = x0[:,i]
    for i in range(l, n):
        xt[:,i] = μ + ε[i]
        for j in range(l):
            t1 = φ[j]*numpy.matrix(xt[:,i-j-1]).T
            t2 = numpy.squeeze(numpy.array(t1), axis=1)
            xt[:,i] += t2
    return xt

def phi_companion_form(φ):
    l, n, _ = φ.shape
    p = φ[0]
    for i in range(1,l):
        p = numpy.concatenate((p, φ[i]), axis=1)
    for i in range(1, n):
        if i == 1:
            r = numpy.eye(n)
        else:
            r = numpy.zeros((n, n))
        for j in range(1,l):
            if j == i - 1:
                r = numpy.concatenate((r, numpy.eye(n)), axis=1)
            else:
                r = numpy.concatenate((r, numpy.zeros((n, n))), axis=1)
        p = numpy.concatenate((p, r), axis=0)
    return numpy.matrix(p)

def mean_companion_form(μ):
    n = len(μ)
    p = numpy.zeros(n**2)
    p[:n] = μ
    return numpy.matrix([p]).T

def omega_companion_form(ω):
    n, _ = Ω.shape
    p = numpy.zeros((n**2, n**2))
    p[:n, :n] = ω
    return numpy.matrix(p)

def vec(m):
    _, n = m.shape
    v = numpy.matrix(numpy.zeros(n**2)).T
    for i in range(n):
        d = i*n
        v[d:d+n] = m[:,i]
    return v

def unvec(v):
    n2, _ = v.shape
    n = int(numpy.sqrt(n2))
    m = numpy.matrix(numpy.zeros((n, n)))
    for i in range(n):
        d = i*n
        m[:,i] = v[d:d+n]
    return m

def stationary_mean(φ, μ):
    Φ = phi_companion_form(φ)
    Μ = mean_companion_form(μ)
    n, _ = Φ.shape
    tmp = numpy.matrix(numpy.eye(n)) - Φ
    return numpy.linalg.inv(tmp)*Μ

def stationary_covariance_matrix(φ, ω):
    Ω = omega_companion_form(ω)
    Φ = phi_companion_form(φ)
    n, _ = Φ.shape
    eye = numpy.matrix(numpy.eye(n**2))
    tmp = eye - numpy.kron(Φ, Φ)
    inv_tmp = numpy.linalg.inv(tmp)
    vec_var = inv_tmp * vec(Ω)
    return unvec(vec_var)

def eigen_values(φ):
    Φ = phi_companion_form(φ)
    λ, _ = numpy.linalg.eig(Φ)
    return λ

# %%

φ = numpy.array([
        numpy.matrix([[1.0, 0.5],
                     [0.5, 1.0]]),
        numpy.matrix([[0.5, 0.3],
                     [0.2, 0.1]])
])
phi_companion_form(φ)

# %%

φ = numpy.array([
        numpy.matrix([[1.0, 0.5, 2.0],
                      [0.5, 1.0, 3.0],
                      [0.5, 1.0, 3.0]]),
        numpy.matrix([[2.0, 3.0, 4.0],
                      [7.0, 6.0, 5.0],
                      [8.0, 9.0, 10.0]]),
        numpy.matrix([[7.0, 8.0, 9.0],
                      [12.0, 11.0, 10.0],
                      [13.0, 14.0, 15.0]])
])
phi_companion_form(φ)

# %%

μ = [1.0, 2.0]
mean_companion_form(μ)

# %%

Ω = numpy.matrix([[1.0, 0.5], [0.5, 2.0]])
omega_companion_form(Ω)

# %%

m = numpy.matrix([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
vec(m)

# %%

unvec(vec(m))

# %%

m = numpy.matrix([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
ones = numpy.matrix(numpy.ones((3,3)))
numpy.kron(m, ones)

# %%

numpy.kron(ones, m)

# %%

μ = [1.0, 2.0]
φ = numpy.array([
        numpy.matrix([[1.0, 0.5],
                     [0.5, 1.0]]),
        numpy.matrix([[0.5, 0.3],
                     [0.2, 0.1]])
])
stationary_mean(φ, μ)

# %%

φ = numpy.array([
        numpy.matrix([[0.5, 0.0],
                     [0.0, 0.75]]),
        numpy.matrix([[0.0, 0.0],
                     [0.0, 0.0]])
])
phi_companion_form(φ)
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
stationary_covariance_matrix(φ, ω)

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.5, 0.0],
                     [0.0, 0.75]]),
        numpy.matrix([[0.0, 0.0],
                     [0.0, 0.0]])
])
eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 5000
xt = var_simulate(x0, μ, φ, ω, n)

# %%
# Diagonalize companion form of φ

Φ = phi_companion_form(φ)
λ, v = numpy.linalg.eig(Φ)
v = numpy.matrix(v)
numpy.linalg.inv(v)*Φ*v

# %%

M = stationary_mean(φ, μ)
Σ = stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_simulation_1_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.5, 0.25],
                     [0.25, 0.75]]),
        numpy.matrix([[0.0, 0.0],
                     [0.0, 0.0]])
])
eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 5000
xt = var_simulate(x0, μ, φ, ω, n)

# %%

M = stationary_mean(φ, μ)
Σ = stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_simulation_2_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.0, 0.0],
                     [0.0, 0.0]]),
        numpy.matrix([[0.5, 0.25],
                     [0.25, 0.75]])
])
eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 5000
xt = var_simulate(x0, μ, φ, ω, n)

# %%

M = stationary_mean(φ, μ)
Σ = stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_simulation_3_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.1],
                      [0.1, 0.3]]),
        numpy.matrix([[0.2, 0.25],
                     [0.25, 0.3]])
])
eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 5000
xt = var_simulate(x0, μ, φ, ω, n)

# %%

M = stationary_mean(φ, μ)
Σ = stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_simulation_3_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)

# %%

μ = [1.0, 1.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.1],
                      [0.1, 0.3]]),
        numpy.matrix([[0.2, 0.25],
                     [0.25, 0.3]])
])
eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 50000
xt = var_simulate(x0, μ, φ, ω, n)

# %%

M = stationary_mean(φ, μ)
Σ = stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_simulation_4_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)
