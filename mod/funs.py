# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

#region imports
from IPython import get_ipython

# noinspection PyBroadException
try:
    _magic = get_ipython().run_line_magic
    _magic("load_ext", "autoreload")
    _magic("autoreload", "2")
except:
    pass

# noinspection PyUnresolvedReferences
import datetime as dt
# noinspection PyUnresolvedReferences
import glob
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import pprint
# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import cartopy as crt
# noinspection PyUnresolvedReferences
import matplotlib as mpl
# noinspection PyUnresolvedReferences
import matplotlib.colors
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import seaborn as sns
# noinspection PyUnresolvedReferences
import xarray as xr
# noinspection PyUnresolvedReferences
#import bnn_tools.bnn_array


#endregion

def unstack_day(ds):
    return (
        ds
        .assign_coords(t=(lambda d: d['time']))
        .assign_coords(day=(lambda d: d['time'].dt.floor('1D')))
        .assign_coords(
            hour=(lambda d: (d['time'] - d['day']) / pd.Timedelta('1H')))
        .set_index({'time': ['day', 'hour']})
        .unstack('time')
    )


def single_day(day,ds):
    d0 = day - pd.Timedelta('1D')
    d1 = day + pd.Timedelta('2D')
    return (
        ds
        .loc[{'time': slice(d0, d1)}]
        .assign_coords(day=(lambda d_: xr.full_like(d_['time'], day)))
        .assign_coords(
            hour=(lambda d: (d['time'] - d['day']) / pd.Timedelta('1H'))
        )
        .set_index({'time': ['day', 'hour']})
        .unstack('time')
    )

def unstack_2day(ds):
    days = np.unique(ds['time'].dt.floor('D'))
    dss = []
    for d in days:
        dss.append(single_day(d, ds))
    return xr.concat(dss, dim='day')


import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt




# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean) ** 2 / 2 / stddev ** 2)


# Define three-Gaussian function
def three_gaussians(x, *m):
    return (
            gaussian(x, m[0], m[1], m[2]) +
            gaussian(x, m[3], m[4], m[5]) +
            gaussian(x, m[6], m[7], m[8])
    )


def fit_gaussians(logn, mi, ma, A, B, C, bw):
    """
    Fits three Gaussians to the logn distribution and calculates m12 and m23.

    Parameters:
        logn (array-like): An array of values representing the distribution.
        mi (float): The minimum value for the x-axis.
        ma (float): The maximum value for the x-axis.

    Returns:
        Tuple of two floats: m12 and m23.
    """

    # Estimate kernel density function
    kern = stats.gaussian_kde(logn, bw_method=bw)

    # Generate x and y values for plotting
    x = np.linspace(mi, ma, 200)
    y = kern(x)

    # Fit three Gaussians to the data
    res = curve_fit(
        three_gaussians, x, y, p0=A,
        bounds=(B, C)
        )

    (a1, c1, s1, a2, c2, s2, a3, c3, s3,) = res[0]

    # Calculate m12 and m23
    c12 = (c1 + c2) / 2
    c23 = (c2 + c3) / 2

    # Plot the results
    plt.plot(x, three_gaussians(x, *res[0]), label='fit', lw=2, ls='--',alpha=.5, c='C0')
    plt.plot(x, gaussian(x, a1, c1, s1, ), label='g1',c='C1')
    plt.plot(x, gaussian(x, a2, c2, s2, ), label='g2',c='C2')
    plt.plot(x, gaussian(x, a3, c3, s3, ), label='g3',c='C4')
    # plt.plot(x, y, label='KDE')
    plt.axvline(c12, label='c12', c='k', ls=':',alpha=.5)
    plt.axvline(c23, label='c23', c='k', ls=':',alpha=.5)

    plt.legend()
    # plt.show()

    # Return m12 and m23
    return c12, c23, res[0]


