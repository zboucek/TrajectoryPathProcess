#!/usr/bin/env python
# coding: utf-8

# Apparatus for PseudoSpectral Collocation method based on Trefethen, L. N. (2000).
# Spectral Methods in MATLAB. In Spectral Methods in MATLAB. Society for Industrial
# and Applied Mathematics. https://doi.org/10.1137/1.9780898719598

import numpy as np
import numpy.matlib


def diff_matrix(N):
    """
    Calculate differentiation matrix for N-degree polynomial
    cheb.m, p. 54, from Trefethen, L. N. (2000). 
    Spectral Methods in MATLAB. In Spectral Methods in MATLAB. 
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719598
    rewritten from Matlab to NumPy
    """

    # nx~N, deg(p)~N - 1  # number of collocation points in degree of polynomial - 1
    x = np.sort(np.cos(np.pi * np.arange(N + 1) / N))
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * np.power((-1), np.arange(N + 1))
    X = np.matlib.repmat(x.reshape(x.shape[0], 1), 1, N + 1)
    dX = X - X.T
    D = (c.reshape(c.shape[0], 1) * (1 / c)) / (dX + (np.eye(N + 1)))
    D = D - np.diag(np.sum(D.T, 0))
    return D, x


def clencurt(N):
    """
    Calculate Clenshaw-Curtis integral weights for N-degree polynomial
    clencurt.m, p. 128, from Trefethen, L. N. (2000). 
    Spectral Methods in MATLAB. In Spectral Methods in MATLAB. 
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719598
    rewritten from Matlab to NumPy
    """

    # N = N - 1  # number of collocation points is degree of polynomial - 1
    theta = (np.pi * np.arange(N + 1) / N)
    x = np.sort(np.cos(theta))
    theta = theta.reshape(theta.shape[0], 1)
    w = np.zeros((1, N + 1))
    ii = np.arange(1, N)
    v = np.ones((N - 1, 1))
    if np.mod(N, 2) == 0:
        w[0, 0] = 1 / (N * N - 1)
        w[0, -1] = w[0, 0]
        for k in np.arange(1, N / 2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        v = v - np.cos(N * theta[ii]) / (N * N - 1)
    else:
        w[0, 0] = 1 / (N * N)
        w[0, -1] = w[0, 0]
        for k in np.arange(1, (N - 1) / 2 + 1):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
    w[0, ii] = np.transpose(2 * v / N)
    return w[0], x


def cheb_scale(col, dom):
    """
    Rescaling of chebyshev points, weights and differential matrix to an arbitrary interval
    """

    shift = 0.5 * (dom[0] + dom[1])
    scale = 0.5 * (dom[1] - dom[0])

    x = scale * col[0] + shift
    w = col[1] * scale
    D = col[2] / scale

    return x, w, D


def cheb_scaled(N, dom):
    """
    Return scaled chebyshev points, weights and differential matrix to an arbitrary interval dom
    """
    col = cheb(N)
    x, w, D = cheb_scale(col, dom)

    return x, w, D


def cheb(N):
    """
    Calculate Gauss-Lobatto grid points, differentiation matrix and integral weights
    """

    D, xx = diff_matrix(N)
    ww, __ = clencurt(N)

    return xx, ww, D
