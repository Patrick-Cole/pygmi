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
""" Class for raster data types and conversion routines"""

import numpy as np


def numpy_to_pygmi(data):
    """
    Converts an MxN numpy array into a PyGMI data object

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
        print("Error: you need 2 dimensions")
        return
    tmp = Data()
    if np.ma.isMaskedArray(data):
        tmp.data = data
    else:
        tmp.data.data = data
    tmp.ydim, tmp.xdim = data.shape

    return tmp


def pygmi_to_numpy(tmp):
    """
    Converts a PyGMI data object into an MxN numpy array

    Parameters
    ----------
    tmp : Data
        PyGMI raster dataset

    Returns
    -------
    array : numpy array
        MxN numpy array
    """
    return np.array(tmp.data)


class Data(object):
    """
    PyGMI Data Object

    Attributes
    ----------
    data : numpy masked array
        array to contain raster data
    tlx : float
        Top Left X coordinate of raster grid
    tly : float
        Top Left Y coordinate of raster grid
    xdim : float
        x-dimension of grid cell
    ydim : float
        y-dimension of grid cell
    nrofbands : int
        number of raster bands
    dataid : str
        band name or id
    rows : int
        number of rows for each raster grid/band
    cols : int
        number of columns for each raster grid/band
    nullvalue : float
        grid null or nodata value
    norm : dictionary
        normalized data
    gtr : tuple
        projection information
    wkt : str
        projection information
    units : str
        description of units to be used with color bars
    """
    def __init__(self):
        self.data = np.ma.array([])
        self.tlx = 0.0  # Top Left X coordinate
        self.tly = 0.0  # Top Left Y coordinate
        self.xdim = 1.0
        self.ydim = 1.0
        self.nrofbands = 1
        self.dataid = ''
        self.rows = -1
        self.cols = -1
        self.nullvalue = 1e+20
        self.norm = {}
        self.gtr = (0.0, 1.0, 0.0, 0.0, -1.0)
        self.wkt = ''
        self.units = ''
