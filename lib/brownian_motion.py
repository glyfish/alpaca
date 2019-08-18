import numpy
from matplotlib import pyplot
from lib import config
from lib import stats

pyplot.style.use(config.glyfish_style)

# Fractional Brownian Motion variance and Autocorrelation

def fbm_variance(H, n):
    return n**(2.0*H)

def fbm_covariance(H, s, n):
    return 0.5*(n**(2.0*H) + s**(2.0*H) - numpy.abs(n - s)**(2.0*H))

def fbn_autocorrelation(H, n):
    return 0.5*((n-1.0)**(2.0*H) + (n+1.0)**(2.0*H) - 2.0*n**(2.0*H))

def fbn_autocorrelation_large_n(H, n):
    return H*(2.0*H - 1.0)*n**(2.0*H - 2.0)

def brownian_noise(n):
    return numpy.random.normal(0.0, 1.0, n)

def fb_motion_riemann_sum(H, Δt, n, B1=None, B2=None):
    b = int(numpy.ceil(n**(1.5)))
    if B1 is None or B2 is None:
        B1 = brownian_noise(b)
        B2 = brownian_noise(n+1)
    if len(B1) != b or len(B2) != n + 1:
        raise Exception(f"B1 should have length {b} and B2 should have length {n+1}")
    Z = numpy.zeros(n+1)
    for i in range(1, n+1):
        bn = int(numpy.ceil(i**(1.5)))
        C = 0.0
        for k in range(-bn, i):
            if k < 0:
                Z[i] += ((float(i) - float(k))**(H - 0.5) - (-k)**(H - 0.5))*B1[k]
                C += ((1.0 - float(k)/float(i))**(H - 0.5) - (-float(k)/float(i))**(H - 0.5))**2
            elif k > 0:
                Z[i] += ((float(i) - float(k))**(H - 0.5))*B2[k]
        C += 1.0/(2.0*H)
        Z[i] = Z[i]*Δt**(H - 0.5)/numpy.sqrt(C)
    return Z

def fbn_autocorrelation_matrix(H, n):
    γ = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(n+1):
            if i != j :
                γ[i,j] = fbn_autocorrelation(H, numpy.abs(i-j))
            else:
                γ[i,j] = 1.0
    return γ

def cholesky_decompose(H, n):
    l = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(i+1):
            if j == 0 and i == 0:
                l[i,j] = 1.0
            elif j == 0:
                l[i,j] = fbn_autocorrelation(H, i) / l[0,0]
            elif i == j:
                l[i,j] = numpy.sqrt(l[0,0] - numpy.sum(l[i,0:i]*l[i,0:i].T))
            else:
                l[i,j] = (fbn_autocorrelation(H, i - j) - numpy.sum(l[i,0:j]*l[j,0:j].T)) / l[j,j]
    return l

def fbn_cholesky(H, Δt, n, dB=None, L=None):
    if dB is None:
        dB = brownian_noise(n+1)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dB = numpy.matrix(dB)
    if L is None:
        R = fbn_autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    return numpy.squeeze(numpy.asarray(L*dB.T))

def fbm_cholesky(H, Δt, n, dB=None, L=None):
    if dB is None:
        dB = brownian_noise(n+1)
    if L is None:
        R = fbn_autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dZ = fbn_cholesky(H, Δt, n, L, dB)
    Z = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

# Brownian Motion Simulations

def brownian_motion_from_noise(dB):
    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i - 1] + dB[i]
    return B

def brownian_motion(Δt, n):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return samples

def brownian_motion_with_drift(μ, σ, Δt, n):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ * Δ * numpy.sqrt(Δt)) + (μ * Δt)
    return samples

def geometric_brownian_motion(μ, σ, s0, Δt, n):
    samples = brownian_motion_with_drift(μ, σ, Δt, n)
    return s0*numpy.exp(samples)

# Plots

def comparison_multiplot(samples, time, labels, lengend_location, title, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1, label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def multiplot(samples, time, text_pos, title, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples[:,-1]), '2.2f')}\nσ={format(numpy.std(samples[:,-1]), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def plot(samples, time, title, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def autocor_coef(title, samples, Δt, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, Δt*max_lag])
    axis.set_ylim([-1.05, 1.0])
    ac = stats.autocorrelate(samples)
    axis.plot(Δt*numpy.array(range(max_lag)), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "brownian_motion", plot)

def autocor(title, samples, Δt, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, Δt*max_lag])
    axis.set_ylim([-1.05, 1.0])
    ac = stats.autocorrelate(samples)
    axis.plot(Δt*numpy.array(range(max_lag)), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "brownian_motion", plot)
