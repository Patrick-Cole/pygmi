# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
"""Class for raster data types and conversion routines."""

import warnings
import numpy as np


def numpy_to_pygmi(data):
    """
    Convert an MxN numpy array into a PyGMI data object.

    Parameters
    ----------
    data : numpy array
        MxN array

    Returns
    -------
    tmp : Data
        PyGMI raster dataset
    """
    if data.ndim != 2:
        warnings.warn('Error: you need 2 dimensions')
        return None
    tmp = Data()
    if np.ma.isMaskedArray(data):
        tmp.data = data
    else:
        tmp.data.data = data
    tmp.ydim, tmp.xdim = data.shape

    return tmp


def pygmi_to_numpy(tmp):
    """
    Convert a PyGMI data object into an MxN numpy array.

    Parameters
    ----------
    tmp : Data
        PyGMI raster dataset

    Returns
    -------
    numpy array
        MxN numpy array
    """
    return np.array(tmp.data)


class Data():
    """
    PyGMI Data Object.

    Attributes
    ----------
    data : numpy masked array
        array to contain raster data
    xdim : float
        x-dimension of grid cell
    ydim : float
        y-dimension of grid cell
    dataid : str
        band name or id
    nullvalue : float
        grid null or nodata value
    wkt : str
        projection information
    units : str
        description of units to be used with color bars
    """

    def __init__(self):
        self.data = np.ma.array([])
        self.extent = (0, 1, -1, 0)  # left, right, bottom, top
        self.xdim = 1.0
        self.ydim = 1.0
        self.dataid = ''
        self.nullvalue = 1e+20
        self.wkt = ''
        self.units = ''
        self.isrgb = False
        self.metadata = {'Cluster': {}}

    def get_gtr(self):
        """
        Ger gtr.

        Returns
        -------
        gtr : tuple
            tuple containing the gtr as (left, xdim, 0, top, 0., -ydim)

        """
        gtr = (self.extent[0], self.xdim, 0.0, self.extent[-1], 0.0,
               -self.ydim)

        return gtr

    def extent_from_gtr(self, gtr):
        """
        Import extent, xdim and ydim from a gtr list.

        Parameters
        ----------
        gtr : list
            gtr list.

        Returns
        -------
        None.

        """
        rows, cols = self.data.shape

        self.xdim = gtr[1]
        self.ydim = -gtr[5]

        left = gtr[0]
        top = gtr[3]
        right = left + self.xdim*cols
        bottom = top - self.ydim*rows

        self.extent = (left, right, bottom, top)
