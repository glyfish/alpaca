# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import special
from scipy.stats import t as student_t
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def t_distribution_simulation(n, μ, σ, ntrials):
    t = numpy.zeros(ntrials)
    for k in range(ntrials):
        samples = numpy.random.normal(μ, σ, n)
        var = reg.bias_corrected_var(samples)
        t[k] = (numpy.mean(samples) - μ)/numpy.sqrt(var/n)
    return t

# %%

npts = 1001
xmax = 6.0
x = numpy.linspace(-xmax, xmax, npts)

# %%

n = 3
title = f"Student-t PDF from scipy, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(student_t.pdf(x, n), x, title, ylabel, xlabel, "student_t_test_3_pdf_scipi")

# %%

n = 3
title = f"Student-t CDF from scipy, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(student_t.cdf(x, n), x, title, ylabel, xlabel, "student_t_test_3_cdf_scipi")

# %%

n = 3
title = f"Student-t PDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(reg.student_t_pdf(n)(x), x, title, ylabel, xlabel, "student_t_test_3_pdf")

# %%

n = 3
title = f"Student-t CDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_cdf(n)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_3_cdf")

# %%

n = 3
title = f"Student-t Tails CDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_tail(n)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_3_tail_cdf")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t PDF, for Range of Degrees of Freedom Compared to Unit Normal"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
pdf = [reg.student_t_pdf(n)(x) for n in n_vals]
reg.distribution_comparission_multiplot(pdf, reg.normal()(x), x, labels, ylabel, xlabel, [0.33, 0.8], [0.0, 0.42], title, "student_t_test_pdf_scan")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t CDF, for Range of Degrees of Freedom"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
cdf = []
for n in n_vals:
    cdf.append([reg.student_t_cdf(n)(i) for i in x])

reg.distribution_multiplot(cdf, x, labels, ylabel, xlabel, [0.33, 0.8], [0.0, 1.05], title, "student_t_test_cdf_scan")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t Tail CDF, for Range of Degrees of Freedom Compared to Unit Normal"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
cdf = []
for n in n_vals:
    cdf.append([reg.student_t_tail(n)(i) for i in x])

reg.distribution_multiplot(cdf, x, labels, ylabel, xlabel, [0.33, 0.5], [0.0, 1.05], title, "student_t_test_tail_cdf_scan")


# %%

n = 3
μ = 5.0
σ = 4.0
ntrials = 1000

t = t_distribution_simulation(n, μ, σ, ntrials)

# %%

numpy.mean(t)
numpy.var(t)
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
title = f"t-Test Sampled Distribution {ntrials} Trails, {n} Degrees of Freedom"
reg.pdf_samples(reg.student_t_pdf(n), t, title, ylabel, xlabel, f"t_test_example_0_{n}_simulation", xrange=numpy.arange(-10.5, 10.5, 0.01), nbins=200)

# %%

n = 5
μ = 5.0
σ = 4.0
ntrials = 1000

t = t_distribution_simulation(n, μ, σ, ntrials)

# %%

numpy.mean(t)
numpy.var(t)

ylabel = r"$f(t;3)$"
xlabel = r"$t$"
title = f"t-Test Sampled Distribution {ntrials} Trails, {n} Degrees of Freedom"
reg.pdf_samples(reg.student_t_pdf(n), t, title, ylabel, xlabel, f"t_test_example_0_{n}_simulation", xrange=numpy.arange(-10.5, 10.5, 0.01), nbins=50)

# %%

n = 20
μ = 5.0
σ = 4.0
ntrials = 1000

t = t_distribution_simulation(n, μ, σ, ntrials)

# %%

numpy.mean(t)
numpy.var(t)

ylabel = r"$f(t;3)$"
xlabel = r"$t$"
title = f"t-Test Sampled Distribution {ntrials} Trails, {n} Degrees of Freedom"
reg.pdf_samples(reg.student_t_pdf(n), t, title, ylabel, xlabel, f"t_test_example_0_{n}_simulation", xrange=numpy.arange(-10.5, 10.5, 0.01), nbins=25)

# %%

n = 3
μ = 5.0
σ = 4.0
ntrials = 100000

t = t_distribution_simulation(n, μ, σ, ntrials)

# %%

title = f"t-Test Cumulative μ {ntrials} Trails, {n} Degrees of Freedom"
plot = f"t_test_example_0_{n}_cumulative_μ"
reg.cumulative_mean_plot(t, 0.0, title, plot, legend_pos=[0.3, 0.5])

# %%

title = f"t-Test Cumulative σ {ntrials} Trails, {n} Degrees of Freedom"
plot = f"t_test_example_0_{n}_cumulative_σ"
var = n/(n-2.0)
reg.cumulative_var_plot(t, var, title, plot, legend_pos=[0.3, 0.7])

# %%

n = 20
μ = 5.0
σ = 4.0
ntrials = 100000

t = t_distribution_simulation(n, μ, σ, ntrials)

# %%

title = f"t-Test Cumulative μ {ntrials} Trails, {n} Degrees of Freedom"
plot = f"t_test_example_0_{n}_cumulative_μ"
reg.cumulative_mean_plot(t, 0.0, title, plot, legend_pos=[0.7, 0.8])

# %%

title = f"t-Test Cumulative σ {ntrials} Trails, {n} Degrees of Freedom"
plot = f"t_test_example_0_{n}_cumulative_σ"
var = n/(n-2.0)
reg.cumvar(t)
reg.cumulative_var_plot(t, var, title, plot, legend_pos=[0.7, 0.5])

# %%
# Latte Problem (Single Sample Test)

μ=4.0
x_bar=4.6
s = 0.22
n = 25
df=n-1

t = numpy.sqrt(n)*(x_bar-μ)/s
t
reg.student_t_tail(df)(2.064)
reg.student_t_tail(df)(-2.064)
reg.student_t_tail(df)(t)
reg.student_t_tail(df)(1.71)
reg.student_t_tail(df)(-1.71)

# %%

title = f"Student-t Tail CDF, Number of Degrees of Freedom: {df}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_tail(df)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_example_1_tail_cdf")


# %%
# Boise-LA Salaries (Paired Two Sample Test)

boise_salaries = numpy.array([53047.0, 49958.0, 41974.0, 44366.0, 40470.0, 36963.0])
la_salaries = numpy.array([62490.0, 58850.0, 49445.0, 52263.0, 47674.0, 43542.0])
n = len(la_salaries)

df= n - 1
d = la_salaries - boise_salaries
d
numpy.mean(d)
numpy.sqrt(reg.bias_corrected_var(d))
t = numpy.sqrt(n)*numpy.mean(d) / numpy.sqrt(reg.bias_corrected_var(d))
t

reg.student_t_tail(df)(2.571)
reg.student_t_tail(df)(-2.571)
reg.student_t_tail(df)(t)

reg.student_t_tail(df)(2.01)

# %%

title = f"Student-t Tail CDF, Number of Degrees of Freedom: {df}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_tail(df)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_example_2_tail_cdf")
