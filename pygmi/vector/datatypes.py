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


class PData(object):
    """ Class for point data """
    def __init__(self):
        self.xdata = np.array([])
        self.ydata = np.array([])
        self.zdata = np.array([])
        self.dataid = ""


class VData(object):
    """ Class for Line data """
    def __init__(self):
        # Since each line can have variable amount of points, it is best to
        # store in a list
        self.crds = []
        self.attrib = [{}]
        self.dataid = ""
        self.dtype = ""
