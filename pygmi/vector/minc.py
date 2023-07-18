# -----------------------------------------------------------------------------
# Name:        minc.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2021 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
Minimum Curvature Gridding Routine.

Based on the work by:

Briggs, I. C., 1974, Machine contouring using minimum curvature, Geophysics
vol. 39, No. 1, pp. 39-48
"""

from operator import itemgetter
import numpy as np
from numba import jit
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


def minc(x, y, z, dxy, showlog=print, extent=None, bdist=None,
         maxiters=100):
    """
    Minimum Curvature Gridding.

    Parameters
    ----------
    x : numpy array
        1D array with x coordinates.
    y : numpy array
        1D array with y coordinates.
    z : numpy array
        1D array with z coordinates.
    dxy : float
        Cell x and y dimension.
    showlog : function, optional
        Routine to show text messages. The default is print.
    extent : list, optional
        Extent defined as (left, right, bottom, top). The default is None.
    bdist : float, optional
        Blanking distance in units of cell. The default is None.
    maxiters : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    u : numpy array
        2D numpy array with gridding z values.

    """
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)

    filt = np.isnan(x) | np.isnan(y) | np.isnan(z)
    if filt.max() == True:
        filt = ~filt
        x = x[filt]
        y = y[filt]
        z = z[filt]

    # Set extent
    if extent is None:
        extent = [x.min(), x.max(), y.min(), y.max()]

    extent = np.array(extent)

    showlog('Setting up grid...')

    # Add buffer
    extent[0] -= dxy*3
    extent[1] += dxy*3
    extent[2] -= dxy*3
    extent[3] += dxy*3

    rows = int((extent[3] - extent[2])/dxy+1)
    cols = int((extent[1] - extent[0])/dxy+1)

    dxy1 = dxy
    for mult in [4, 2, 1]:
        dxy = dxy1*mult

    # Make new start grid

    xxx = np.linspace(extent[0], extent[1], cols)
    yyy = np.linspace(extent[2], extent[3], rows)
    xxx, yyy = np.meshgrid(xxx, yyy)

    points = np.transpose([x.flatten(), y.flatten()])

    showlog('Creating nearest neighbour starting value...')

    u = griddata(points, z, (xxx, yyy), method='nearest')
    u = u[::-1]

    # define new grid
    ufixed = np.zeros((rows, cols), dtype=bool)

    x2 = x.flatten()
    y2 = y.flatten()
    z2 = z.flatten()

    showlog('Organizing input data...')

    crds, blist = morg(x2, y2, z2, extent, dxy, rows, cols)

    coords = {}
    excludedpnts = 0
    for k, val in enumerate(crds):
        iint, jint, r, zval = val

        iint = int(iint)
        jint = int(jint)
        b = blist[k]
        if b is None:
            bmax = np.inf
        elif None in b:
            excludedpnts += 1
            continue
        else:
            bmax = np.abs(b).max()

        if (iint, jint) not in coords:
            coords[iint, jint] = []
        coords[iint, jint].append([bmax, r, zval, b])

    if excludedpnts > 0:
        showlog(str(excludedpnts)+' point(s) excluded.')
    # Choose only the closest coordinate per cell
    ijxyz = []
    for key in coords:
        iint, jint = key
        coords[key].sort(key=itemgetter(1))
        _, r, zval, _ = coords[key][0]
        if r < 0.05:
            u[iint, jint] = zval
            ufixed[iint, jint] = True
            continue
        if 1 < iint < rows-2 and 1 < jint < cols-2:
            coords[key].sort()
            _, _, zval, b = coords[key][0]
            ijxyz.append([iint, jint, zval, b])

    showlog('Creating minimum curvature grid...')
    uold = np.zeros((rows, cols))

    # mean error per cell
    errdiff1 = np.abs(u-uold)
    errdiff = np.sum(errdiff1)/(rows*cols)
    errold = errdiff*1.1  # Starting 'old' error.
    errstd = errdiff1.std()*2.5

    iters = 0
    while errdiff < errold and iters < maxiters:
        iters += 1
        uold = u.copy()
        errold = errdiff

        if iters < 2:
            for vals in ijxyz:
                i, j, w, b = vals
                tmp = off_grid(uold, i, j, w, b)

                if (abs(tmp-uold[i, j]) > errdiff1[i, j]+errstd and iters > 1):
                    ufixed[i, j] = False
                else:
                    ufixed[i, j] = True
                    u[i, j] = tmp

        u = mcurv(u, ufixed)

        errdiff1 = np.abs(u-uold)
        errstd = errdiff1.std()*2.5
        errdiff = np.sum(errdiff1)/(rows*cols)
        showlog(f'Solution Error: {errdiff:.5f}')

        if errdiff > errold:
            u = uold
            showlog('Solution diverging. Stopping...')
            break
    showlog('Finished!')

    u = np.ma.array(u)

    # Trim buffer
    u = u[3:-3, 3:-3]
    ufixed = ufixed[3:-3, 3:-3]

    if bdist is not None:
        dist = distance_transform_edt(np.logical_not(ufixed))
        mask = (dist > bdist)
        u = np.ma.array(u, mask=mask)

    return u


@jit(nopython=True)
def u_normal(u, i, j):
    """
    Minimum curvature smoothing for normal cases.

    It is defined as:

    u[i+2, j] + u[i, j+2] + u[i-2, j] + u[i, j-2] +
    2*(u[i+1, j+1] + u[i-1, j+1] + u[i+1, j-1] + u[i-1, j-1]) -
    8*(u[i+1, j]+u[i-1, j]+u[i, j+1]+u[i, j-1]) + 20*u[i, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.
    i : int
        Current row.
    j : int
        Current Column.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[i+2, j] + u[i, j+2] + u[i-2, j] + u[i, j-2] +
            2*(u[i+1, j+1] + u[i-1, j+1] + u[i+1, j-1] + u[i-1, j-1]) -
            8*(u[i+1, j]+u[i-1, j]+u[i, j+1]+u[i, j-1]))/20
    return uij


@jit(nopython=True)
def u_edge(u, i):
    """
    Minimum curvature smoothing for edges.

    It is defined as:

    u[i-2, j] + u[i+2, j] + u[i, j+2] + u[i-1, j+1] + u[i+1, j+1] -
    4*(u[i-1, j] + u[i, j+1] + u[i+1, j]) + 7*u[i, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.
    i : int
        Current row.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[i-2, 0] + u[i+2, 0] + u[i, 2] + u[i-1, 1] + u[i+1, 1] -
            4*(u[i-1, 0] + u[i, 1] + u[i+1, 0]))/7
    return uij


@jit(nopython=True)
def u_one_row_from_edge(u, i):
    """
    Minimum curvature smoothing for one row from edge.

    It is defined as:

    u[i-2, j] + u[i+2, j] + u[i, j+2] +
    2*(u[i-1, j+1] + u[i+1, j+1]) + u[i-1, j-1]+u[i+1, j-1] -
    8*([i-1, j]+u[i, j+1]+u[i+1, j]) - 4*u[i, j-1] + 19*u[i, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.
    i : int
        Current row.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[i-2, 1] + u[i+2, 1] + u[i, 3] +
            2*(u[i-1, 2] + u[i+1, 2]) + u[i-1, 0] + u[i+1, 0] -
            8*(u[i-1, 1]+u[i, 2]+u[i+1, 1]) - 4*u[i, 0])/19
    return uij


@jit(nopython=True)
def u_corner(u):
    """
    Minimum curvature smoothing for corner point.

    It is defined as:

    2*u[i, j]+u[i, j+2] + u[i+2, j] - 2*(u[i, j+1] + u[i+1, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[0, 2] + u[2, 0] - 2*(u[0, 1] + u[1, 0]))/2
    return uij


@jit(nopython=True)
def u_next_to_corner(u):
    """
    Minimum curvature smoothing for next to corner.

    It is defined as:

    u[i, j+2] + u[i+2, j] + u[i-1, j+1] + u[i+1, j-1] + 2*u[i+1, j+1] -
    8*(u[i, j+1] + u[i+1, j]) - 4*([i, j-1]+u[i-1, j]) + 18*u[i, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[1, 3] + u[3, 1] + u[0, 2] + u[2, 0] + 2*u[2, 2] -
            8*(u[1, 2] + u[2, 1]) - 4*(u[1, 0]+u[0, 1]))/18
    return uij


@jit(nopython=True)
def u_edge_next_to_corner(u):
    """
    Minimum curvature smoothing for edge next to corner.

    It is defined as:

    u[i, j+2] + u[i+1, j+1] + u[i-1, j+1] + u[i+2, j] - 2*u[i-1, j] -
    4*(u[i+1, j] + u[i, j+1]) + 6*u[i, j] = 0

    Parameters
    ----------
    u : numpy array
        2D grid of z values.

    Returns
    -------
    uij : float
        Smoothed value to replace in master grid.

    """
    uij = -(u[2, 3] + u[3, 2] + u[1, 2] + u[4, 1] - 2*u[1, 1] -
            4*(u[3, 1] + u[2, 2]))/6
    return uij


def off_grid(u, i, j, wn, b):
    """
    Node value calculation when data value is too far from node.

    Parameters
    ----------
    u : numpy array
        2D grid of z values.
    i : int
        Current row.
    j : TYPE
        Current Column.
    wn : float
        Data value.
    b : list
        List of b values for calculation.

    Returns
    -------
    uij : TYPE
        DESCRIPTION.

    """
    ba, bb, bc, bd, be = b

    ba1, ba2, ba3, ba4, ba5 = ba
    bb1, bb2, bb3, bb4, bb5 = bb
    bc1, bc2, bc3, bc4, bc5 = bc
    bd1, bd2, bd3, bd4, bd5 = bd
    be1, be2, be3, be4, be5 = be

    uij = ((4*ba1*u[i + 1, j - 1] + 4*ba2*u[i, j - 1] + 4*ba3*u[i - 1, j] +
            4*ba4*u[i - 1, j + 1] + 4*ba5*wn

            + bb1*u[i - 1, j]
            - bb1*u[i, j - 1] - bb2*u[i - 1, j - 1] + bb2*u[i - 1, j]
            + bb3*u[i - 1, j] - bb3*u[i - 2, j] + bb4*u[i - 1, j]
            - bb4*u[i - 2, j + 1] - bb5*wn + bb5*u[i - 1, j]

            + bc1*u[i + 1, j] - bc1*u[i + 2, j - 1] - bc2*u[i + 1, j - 1]
            + bc2*u[i + 1, j] + bc3*u[i + 1, j] + bc4*u[i + 1, j]
            - bc4*u[i, j + 1] - bc5*wn + bc5*u[i + 1, j]

            - bd1*u[i + 1, j - 2] + bd1*u[i, j - 1] + bd2*u[i, j - 1]
            - bd2*u[i, j - 2] - bd3*u[i - 1, j - 1] + bd3*u[i, j - 1]
            - bd4*u[i - 1, j] + bd4*u[i, j - 1] - bd5*wn + bd5*u[i, j - 1]

            - be1*u[i + 1, j] + be1*u[i, j + 1] + be2*u[i, j + 1]
            - be3*u[i - 1, j + 1] + be3*u[i, j + 1] - be4*u[i - 1, j + 2]
            + be4*u[i, j + 1] - be5*wn + be5*u[i, j + 1]) /
           (4*ba1 + 4*ba2 + 4*ba3 + 4*ba4 + 4*ba5 + bc3 + be2))

    return uij


@jit(nopython=True)
def get_b(e5, n5):
    """
    Get b values for input data.

    Calculates the b values based on the distance between the data point and
    the nearest node. Distances are expressed in units of cell.

    Parameters
    ----------
    e5 : float
        x distance error.
    n5 : float
        y distance error.

    Returns
    -------
    b1 : float
        b1 value.
    b2 : float
        b2 value.
    b3 : float
        b3 value.
    b4 : float
        b4 value.
    b5 : float
        b5 value.

    """
    d2 = e5 + n5 + 1
    d1 = d2*(n5 + e5)

    if d1 == 0.0 or d2 == 0:
        return None

    b1 = (-e5**2 + 2*e5*n5 - e5 + n5**2 + n5)/d1
    b2 = 2*(e5 - n5 + 1)/d2
    b3 = 2*(-e5 + n5 + 1)/d2
    b4 = (e5**2 + 2*e5*n5 + e5 - n5**2 - n5)/d1
    b5 = 4/d1

    return b1, b2, b3, b4, b5


@jit(nopython=True)
def mcurv(u, ufixed):
    """
    Minimum curvature smoothing.

    This routine smooths the data between fixed data nodes.

    Parameters
    ----------
    u : numpy array
        2D grid of z values.
    ufixed : numpy array
        2D grid of fixed node values.

    Returns
    -------
    u : numpy array
        2D grid of z values.

    """
    rows, cols = u.shape
    for i in range(rows):
        for j in range(cols):
            if ufixed[i, j] == True:
                continue

            if 1 < i < rows-2 and 0 < j < cols-2:
                u[i, j] = u_normal(u, i, j)

            elif j == 0 and 1 < i < rows-2:
                u[i, j] = u_edge(u, i)
            elif j == cols-1 and 1 < i < rows-2:
                u[i, j] = u_edge(u[:, ::-1], i)
            elif i == 0 and 1 < j < cols-2:
                u[i, j] = u_edge(u.T, j)
            elif i == rows-1 and 1 < j < cols-2:
                u[i, j] = u_edge(u.T[:, ::-1], j)

            elif j == 1 and 1 < i < rows-2:
                u[i, j] = u_one_row_from_edge(u, i)
            elif j == cols-2 and 1 < i < rows-2:
                u[i, j] = u_one_row_from_edge(u[:, ::-1], i)
            elif i == 1 and 1 < j < cols-2:
                u[i, j] = u_one_row_from_edge(u.T, j)
            elif i == rows-2 and 1 < j < cols-2:
                u[i, j] = u_one_row_from_edge(u.T[:, ::-1], j)

            elif i == 0 and j == 0:
                u[i, j] = u_corner(u)
            elif i == 0 and j == cols-1:
                u[i, j] = u_corner(u[:, ::-1])
            elif i == rows-1 and j == 0:
                u[i, j] = u_corner(u[::-1])
            elif i == rows-1 and j == cols-1:
                u[i, j] = u_corner(u[::-1, ::-1])

            elif i == 1 and j == 1:
                u[i, j] = u_next_to_corner(u)
            elif i == 1 and j == cols-2:
                u[i, j] = u_next_to_corner(u[:, ::-1])
            elif i == rows-2 and j == 1:
                u[i, j] = u_next_to_corner(u[::-1])
            elif i == rows-2 and j == cols-2:
                u[i, j] = u_next_to_corner(u[::-1, ::-1])

            elif i == 2 and j == 1:
                u[i, j] = u_edge_next_to_corner(u)
            elif i == 1 and j == 2:
                u[i, j] = u_edge_next_to_corner(u.T)
            elif i == 2 and j == cols-1:
                u[i, j] = u_edge_next_to_corner(u[:, ::-1])
            elif i == 1 and j == cols-2:
                u[i, j] = u_edge_next_to_corner(u[:, ::-1].T)
            elif i == rows-1 and j == 2:
                u[i, j] = u_edge_next_to_corner(u[::-1])
            elif i == rows-2 and j == 1:
                u[i, j] = u_edge_next_to_corner(u[::-1].T)
            elif i == rows-1 and j == cols-2:
                u[i, j] = u_edge_next_to_corner(u[::-1, ::-1])
            elif i == rows-2 and j == cols-1:
                u[i, j] = u_edge_next_to_corner(u[::-1, ::-1].T)
    return u


@jit(nopython=True)
def morg(x2, y2, z2, extent, dxy, rows, cols):
    """
    Organise coordinates and calculate b values.

    Parameters
    ----------
    x2 : numpy array
        1D array with x coordinates.
    y2 : numpy array
        1D array with y coordinates.
    z2 : numpy array
        1D array with z coordinates.
    extent : list
        Extent defined as (left, right, bottom, top).
    dxy : float
        Cell x and y dimension.
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    coords : list
        List containing iint, jint, r and zval.
    b : list
        List of b values.

    """
    coords = []
    b = []
    for k, zval in enumerate(z2):
        i = (extent[3] - y2[k])/dxy
        j = (x2[k]-extent[0])/dxy

        iint = round(i)
        jint = round(j)

        # Ignore values outside extents
        if iint < 0 or jint < 0 or iint >= rows-1 or jint >= cols-1:
            continue

        n5 = i-iint
        e5 = j-jint

        r = np.sqrt(e5**2+n5**2)
        if r >= 0.75:
            continue

        if r < 0.05:
            coords.append([iint, jint, r, zval])
            b.append(None)

            continue

        blist = [get_b(i-iint, j-jint),
                 get_b(i-(iint-1), j-jint),
                 get_b(i-(iint+1), j-jint),
                 get_b(i-iint, j-(jint-1)),
                 get_b(i-iint, j-(jint+1))]

        b.append(blist)

        coords.append([iint, jint, r, zval])

    return coords, b


def _testfn():
    """Test routine."""
    import sys
    from PyQt5 import QtWidgets
    import matplotlib.pyplot as plt
    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)

    ifile = r'c:\Workdata\vector\Line Data\MAGARCHIVE.XYZ'

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = 'Geosoft XYZ (*.xyz)'
    IO.settings(True)

    dat = IO.outdata['Vector']

    x = dat.geometry.x.to_numpy()
    y = dat.geometry.y.to_numpy()
    z = dat['Column 8'].to_numpy()

    dxy = 125

    extent = np.array([x.min(), x.max(), y.min(), y.max()])

    odat = minc(x, y, z, dxy, extent=extent, bdist=4)

    vmin = odat.mean()-2*odat.std()
    vmax = odat.mean()+2*odat.std()

    plt.figure(dpi=150)
    plt.imshow(odat, extent=extent, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    _testfn()
