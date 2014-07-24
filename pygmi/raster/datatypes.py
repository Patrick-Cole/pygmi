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
""" Class for data types """

# pylint: disable=E1101, C0103
import numpy as np


def numpy_to_pygmi(data):
    """ Converts an MxN numpy array into a PyGMI data object """
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
    """ Converts a PyGMI data object into an MxN numpy array  """
    return np.array(tmp.data)


class Data(object):
    """ Data Object """
    def __init__(self):
        self.data = np.ma.array([])
        self.tlx = 0.0  # Top Left X coordinate
        self.tly = 0.0  # Top Left Y coordinate
        self.xdim = 1.0
        self.ydim = 1.0
        self.nrofbands = 1
        self.bandid = ""
        self.rows = -1
        self.cols = -1
        self.nullvalue = -9999.0
        self.norm = {}
        self.gtr = (0.0, 1.0, 0.0, 0.0, -1.0)
        self.wkt = ''
        self.units = ''