import numpy
from matplotlib import pyplot
from lib import config

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def timeseries_plot(samples, tmax, ylabel, title, plot_name):
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
        axis[i].set_ylabel(ylabel[i])
        axis[i].set_ylim([ymin, ymax])
        axis[i].set_xlim([0.0, tmax])
        axis[i].plot(time, samples[i,:tmax], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def autocorrelation_plot(title, samples, γt, ylim, plot):
    max_lag = len(γt)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-1.0, max_lag])
    axis.set_ylim(ylim)
    ac = autocorrelation(samples)
    axis.plot(range(max_lag), numpy.real(ac[:max_lag]), marker='o', markersize=10.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, label="Simulation", zorder=6)
    axis.plot(range(max_lag), γt, lw="2", label=r"$γ_{\tau}$", zorder=5)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def cross_correlation_plot(title, x, y, γt, ylim, plot):
    max_lag = len(γt)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    cc = cross_correlation(x, y)
    axis.set_xlim([-1.0, max_lag])
    axis.set_ylim(ylim)
    axis.plot(range(max_lag), numpy.real(cc[:max_lag]), marker='o', markersize=10.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, label="Simulation", zorder=6)
    axis.plot(range(max_lag), γt, lw="2", label=r"$γ_{\tau}$", zorder=5)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def plot_data_frame(df, tmax, plot_name):
    _, nplot = df.shape
    if nplot > 4:
        nrows = int(nplot / 2)
        ncols = 2
    else:
        nrows = nplot
        ncols = 1
    figure, axis = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
    for i, axis in enumerate(axis.flatten()):
        data = df[df.columns[i]]
        axis.plot(data[:tmax], lw=1)
        axis.set_title(df.columns[i], fontsize=12)
        axis.tick_params(axis="x", labelsize=10)
        axis.tick_params(axis="y", labelsize=10)
    pyplot.tight_layout(pad=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def time_series_to_data_frame(columns, series):
    n = len(columns)
    d = {}
    for i in range(n):
        d[columns[i]] = series[i]
    return pandas.DataFrame(d)

def var_simulate(x0, μ, φ, Ω, n):
    m, l = x0.shape
    xt = numpy.zeros((m, n))
    ε = multivariate_normal_sample(μ, Ω, n)
    for i in range(l):
        xt[:,i] = x0[:,i]
    for i in range(l, n):
        xt[:,i] = ε[i]
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
    n, _ = ω.shape
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

def stationary_autocovariance_matrix(φ, ω, n):
    t = numpy.linspace(0, n-1, n)
    Φ = phi_companion_form(φ)
    Σ = stationary_covariance_matrix(φ, ω)
    l, _ = Φ.shape
    γ = numpy.zeros((n, l, l))
    γ[0] = numpy.matrix(numpy.eye(l))
    for i in range(1,n):
        γ[i] = γ[i-1]*Φ
    for i in range(n):
        γ[i] = Σ*γ[i].T
    return γ

def eigen_values(φ):
    Φ = phi_companion_form(φ)
    λ, _ = numpy.linalg.eig(Φ)
    return λ

def autocorrelation(x):
    n = len(x)
    x_shifted = x - x.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    h_fft = numpy.conj(x_fft) * x_fft
    ac = numpy.fft.ifft(h_fft)
    return ac[0:n]/ac[0]

def cross_correlation(x, y):
    n = len(x)
    x_shifted = x - x.mean()
    y_shifted = y - y.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    y_padded = numpy.concatenate((y_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    y_fft = numpy.fft.fft(y_padded)
    h_fft = numpy.conj(x_fft)*y_fft
    cc = numpy.fft.ifft(h_fft)
    return cc[0:n] / float(n)

def yt_parameter_estimation_form(xt):
    l, n = xt.shape
    yt = xt[:,l-1:n-1]
    for i in range(2,l+1):
        yt = numpy.concatenate((yt, xt[:,l-i:n-i]), axis=0)
    return yt

def theta_parameter_estimation(xt):
    l, n = xt.shape
    yt = yt_parameter_estimation_form(xt)
    m, _ = yt.shape
    yy = numpy.matrix(numpy.zeros((m, m)))
    xy = numpy.matrix(numpy.zeros((l, m)))
    for i in range(l, n):
        x = numpy.matrix(xt[:,i]).T
        y = numpy.matrix(yt[:,i-l]).T
        yy += y*y.T
        xy += x*y.T
    return xy*numpy.linalg.inv(yy)

def split_theta(theta):
    l, _ = theta.shape
    return numpy.split(theta, l, axis=1)

def omega_parameter_estimation(xt, theta):
    l, n = xt.shape
    yt = yt_parameter_estimation_form(xt)
    omega = numpy.matrix(numpy.zeros((l, l)))
    for i in range(l, n):
        x = numpy.matrix(xt[:,i]).T
        y = numpy.matrix(yt[:,i-l]).T
        term = x - theta*y
        omega += term*term.T
    return omega / float(n-l)
