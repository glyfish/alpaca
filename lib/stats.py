import numpy

def ensemble_mean(samples):
    nsim, npts = samples.shape
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j,i] / float(nsim)
    return mean

def ensemble_std(samples):
    mean = ensemble_mean(samples)
    nsim, npts = samples.shape
    std = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            std[i] += (samples[j,i] - mean[i])**2 / float(nsim)
    return numpy.sqrt(std)

def ensemble_autocorrelation(samples):
    nsim, npts = samples.shape
    ac_avg = numpy.zeros(npts)
    for j in range(nsim):
        ac = autocorrelate(samples[j]).real
        for i in range(npts):
            ac_avg[i] += ac[i]
    return ac_avg / float(nsim)

def cummean(samples):
    nsample = len(samples)
    mean = numpy.zeros(nsample)
    mean[0] = samples[0]
    for i in range(1, nsample):
        mean[i] = (float(i) * mean[i - 1] + samples[i])/float(i + 1)
    return mean

def cumsigma(samples):
    nsample = len(samples)
    mean = cummean(samples)
    var = numpy.zeros(nsample)
    var[0] = samples[0]**2
    for i in range(1, nsample):
        var[i] = (float(i) * var[i - 1] + samples[i]**2)/float(i + 1)
    return numpy.sqrt(var-mean**2)

def cum_covaraince(x, y):
    nsample = len(x)
    meanx = cummean(x)
    meany = cummean(y)
    cov = numpy.zeros(nsample)
    cov[0] = x[0]*y[0]
    for i in range(1, nsample):
        cov[i] = (float(i) * cov[i - 1] + x[i] * y[i])/float(i + 1)
    return cov - meanx * meany

def autocorrelate(x):
    n = len(x)
    x_shifted = x - x.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    h_fft = numpy.conj(x_fft) * x_fft
    ac = numpy.fft.ifft(h_fft)
    return ac[0:n]/ac[0]

def autocorrelate_sum(x, max_lag):
    n = len(x)
    x_shifted = x - x.mean()
    ac = numpy.zeros(n)
    for t in range(max_lag):
        for k in range(0, n - t):
            ac[t] += x_shifted[k] * x_shifted[k + t]
    return ac/ac[0]
