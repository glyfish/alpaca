# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from scipy import stats
from lib import config

pyplot.style.use(config.glyfish_style)

# %%

def brownian_noise(n):
    return numpy.random.normal(0.0, 1.0, n)

def bivatiate_pdf(x, y, μ, Ω):
    pos = numpy.empty(x.shape+(2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    return stats.multivariate_normal.pdf(pos, μ, Ω)

def bivatiate_pdf_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def cholesky_decompose(Ω):
    return numpy.linalg.cholesky(Ω)

def bivatiate_pdf_mesh(μ, Ω, n):
    σ = max(Ω[0][0], Ω[1][1])
    δ = 6.0*σ/n
    x1 = -3.0*σ + μ[0]
    x2 = 3.0*σ + μ[0]
    y1 = -3.0*σ + μ[1]
    y2 = 3.0*σ + μ[1]
    x, y = numpy.mgrid[x1:x2:δ, y1:y2:δ]
    return (x, y)

def bivatiate_pdf_contour_plot(μ, Ω, n, contour_values, plot_name):
    x, y = bivatiate_pdf_mesh(μ, Ω, n)
    f = bivatiate_pdf(x, y, μ, Ω)
    figure, axis = pyplot.subplots(figsize=(9, 9))
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    σ = max(Ω[0][0], Ω[1][1])
    x1 = -3.2*σ + μ[0]
    x2 = 3.2*σ + μ[0]
    y1 = -3.2*σ + μ[1]
    y2 = 3.2*σ + μ[1]
    axis.set_xlim([x1, x2])
    axis.set_ylim([y1, y2])
    title = f"Bivariate Normal Distribution: γ={format(Ω[0][1], '2.2f')}, " + \
             r"$σ_x$=" + f"{format(Ω[0][0], '2.2f')}, " + r"$σ_y$=" + \
             f"{format(Ω[1][1], '2.2f')}"
    axis.set_title(title)
    contour = axis.contour(x, y, f, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def bivatiate_pdf_samples_plot(p, q, μ, Ω, n, contour_values, plot_name):
    x, y = bivatiate_pdf_mesh(μ, Ω, n)
    f = bivatiate_pdf(x, y, μ, Ω)
    figure, axis = pyplot.subplots(figsize=(9, 9))
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    σ = max(Ω[0][0], Ω[1][1])
    x1 = -3.2*σ + μ[0]
    x2 = 3.2*σ + μ[0]
    y1 = -3.2*σ + μ[1]
    y2 = 3.2*σ + μ[1]
    axis.set_xlim([x1, x2])
    axis.set_ylim([y1, y2])
    title = f"Bivariate Normal Distribution: γ={format(Ω[0][1], '2.2f')}, " + \
             r"$σ_x$=" + f"{format(Ω[0][0], '2.2f')}, " + r"$σ_y$=" + \
             f"{format(Ω[1][1], '2.2f')}"
    axis.set_title(title)
    _, _, _, image = axis.hist2d(p, q, normed=True, bins=bins, cmap=config.alternate_color_map)
    contour = axis.contour(x, y, f, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    figure.colorbar(image)
    config.save_post_asset(figure, "mean_reversion", plot_name)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.0], [0.0, 1.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_0.0_0.0_1.0_1.0_0.0"

bivatiate_pdf_contour_plot(μ, Ω, n, [0.01, 0.05, 0.1, 0.15, 0.2], plot_name)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.0], [0.0, 2.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_0.0_0.0_1.0_2.0_0.0"

bivatiate_pdf_contour_plot(μ, Ω, n, [0.001, 0.025, 0.05, 0.075, 0.1], plot_name)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.5], [0.5, 1.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_0.0_0.0_1.0_1.0_0.5"

bivatiate_pdf_contour_plot(μ, Ω, n, [0.01, 0.025, 0.05, 0.075, 0.1, 0.125], plot_name)

# %%

μ = [1.0, 1.0]
Ω = [[1.0, 0.5], [0.5, 1.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_1.0_1.0_1.0_1.0_0.5"

bivatiate_pdf_contour_plot(μ, Ω, n, [0.01, 0.025, 0.05, 0.075, 0.1, 0.125], plot_name)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.5], [0.5, 2.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_0.0_0.0_1.0_2.0_0.5"

bivatiate_pdf_contour_plot(μ, Ω, n,  [0.001, 0.025, 0.05, 0.075, 0.1], plot_name)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.5], [0.5, 1.0]]
n = 100
plot_name = "var_simulation_bivatiate_gaussian_samples_0.0_0.0_1.0_1.0_0.5"

samples = bivatiate_pdf_sample(μ, Ω, n)

# bivatiate_pdf_samples_plot(μ, Ω, n, [0.01, 0.025, 0.05, 0.075, 0.1, 0.125], plot_name)
