#!/usr/bin/env python
# coding: utf-8

# Apparatus for PseudoSpectral Collocation method based on Trefethen, L. N. (2000).
# Spectral Methods in MATLAB. In Spectral Methods in MATLAB. Society for Industrial
# and Applied Mathematics. https://doi.org/10.1137/1.9780898719598

import numpy as np
import numpy.matlib


def legendre_diff_matrix(N):
    """
    Calculate differentiation matrix for N-degree polynomial
    gauss.m, p. 54, from Trefethen, L. N. (2000). 
    Spectral Methods in MATLAB. In Spectral Methods in MATLAB. 
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719598
    rewritten from Matlab to NumPy
    """

    # nx~N, deg(p)~N - 1  # number of collocation points in degree of polynomial - 1
    x = legendre_points(N)
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * np.power((-1), np.arange(N + 1))
    X = np.matlib.repmat(x.reshape(x.shape[0], 1), 1, N + 1)
    dX = X - X.T
    D = (c.reshape(c.shape[0], 1) * (1 / c)) / (dX + (np.eye(N + 1)))
    D = D - np.diag(np.sum(D.T, 0))
    return D, x

def legendre_points(N):
    beta = 0.5/np.sqrt(1 - (2*np.arange(1,N-1))**(-2.0)) 
    T = np.diag(beta,1) + np.diag(beta,-1)
    D,V = np.linalg.eig(T)
    x = D
    i = np.argsort(x)
    x = x[i]
    x = np.append(np.append(-1, x), 1.0)
    
    return x

def gauss_lobatto_weights(N):
    """
    Calculate Gauss integral weights for N-degree polynomial
    gauss.m, p. 128, from Trefethen, L. N. (2000). 
    Spectral Methods in MATLAB. In Spectral Methods in MATLAB. 
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719598
    rewritten from Matlab to NumPy
    """
    w_gl = np.zeros(N+1)
    # N = N - 1  # number of collocation points is degree of polynomial - 1
    beta = 0.5/np.sqrt(1 - (2*np.arange(1,N-1))**(-2.0)) 
    T = np.diag(beta,1) + np.diag(beta,-1)
    D,V = np.linalg.eig(T)
    x = D
    i = np.argsort(x)
    x = x[i]
    x = np.append(np.append(-1, x), 1.0)
    # Interior weights from Legendre gauss quadrature
    w = 2*(V[0,i]**2)
    w_gl[1:-1] = w
    # End points have weight 2/(N(N+1))
    w_gl[0] = 2/(N*(N+1))
    w_gl[-1] = w[0]
    
    return w_gl, x


def legendre_scale(col, dom):
    """
    Rescaling of chebyshev points, weights and differential matrix to an arbitrary interval
    """

    shift = 0.5 * (dom[0] + dom[1])
    scale = 0.5 * (dom[1] - dom[0])

    x = scale * col[0] + shift
    w = col[1] * scale
    D = col[2] / scale

    return x, w, D


def legendre_scaled(N, dom):
    """
    Return scaled legendre points, weights and differential matrix to an arbitrary interval dom
    """
    col = legendre(N)
    x, w, D = legendre_scale(col, dom)

    return x, w, D


def legendre(N):
    """
    Calculate legendre grid points, differentiation matrix and integral weights
    """

    D, xx = legendre_diff_matrix(N)
    ww, __ = gauss_lobatto_weights(N)

    return xx, ww, D
