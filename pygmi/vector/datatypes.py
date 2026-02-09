# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2025 Council for Geoscience
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
"""Class for data types."""

import numpy as np

from pygmi.raster.datatypes import Data


class VoxModel():
    """
    Voxel Model Data.

    This is the main data structure for voxel data

    Attributes
    ----------
    data : numpy masked array
        Voxel data.
    origin : list
        Origin coordinates as x, y, z
    spacing : list
        Spacing in x, y and z directions.
    """

    def __init__(self):
        self.data = None
        self.origin = [0, 0, 0]
        self.spacing = [1, 1, 1]
