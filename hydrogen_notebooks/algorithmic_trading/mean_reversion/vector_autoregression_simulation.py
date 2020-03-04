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

def pdf(μ1, μ2, σ1, σ2, γ):
    def f(x):
        return stats.multivariate_normal.pdf(x, [μ1, μ2], [[σ1, γ], [γ, σ2]])
    return f

def cholesky_decompose(Ω):
    return numpy.linalg.cholesky(Ω)

def pdf_mesh(μ1, μ2, σ1, σ2, γ):
    npts = 500
    σ=max(σ1, σ2)
    x1 = numpy.linspace(-σ*3.0, σ*3.0, npts)
    x2 = numpy.linspace(-σ*3.0, σ*3.0, npts)
    f = pdf(μ1, μ2, σ1, σ2, γ)
    x1_grid, x2_grid = numpy.meshgrid(x1, x2)
    f_x1_x2 = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f_x1_x2[i, j] = f([x1_grid[i,j], x2_grid[i,j]])
    return (x1_grid, x2_grid, f_x1_x2)

def contour_plot(μ1, μ2, σ1, σ2, γ, contour_values, plot_name):
    npts = 500
    x1_grid, x2_grid, f_x1_x2 = pdf_mesh(μ1, μ2, σ1, σ2, γ)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    σ=max(σ1, σ2)
    axis.set_xlim([-3.2*σ, 3.2*σ])
    axis.set_ylim([-3.2*σ, 3.2*σ])
    title = f"Bivariate Normal Distribution: γ={format(γ, '2.2f')}, " + \
             r"$σ_u$=" + f"{format(σ1, '2.2f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.2f')}"
    axis.set_title(title)
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, post, plot_name)

# %%
