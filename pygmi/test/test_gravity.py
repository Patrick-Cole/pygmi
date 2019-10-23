# -----------------------------------------------------------------------------
# Name:        test_cluster.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
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
These are tests. Run pytest on this file from within this directory to do
the tests .
"""

import os
import sys
from PyQt5 import QtWidgets
import numpy as np
import pytest
from pygmi.grav import iodefs, dataprep

APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


def test_process_data():
    """ test process gravity data """

    idir = os.path.dirname(os.path.realpath(__file__))

    grvfile = os.path.join(idir, r'data\GravityCG5.txt')
    gpsfile = os.path.join(idir, r'data\GravityDGPS.csv')

# Import Data
    IO = iodefs.ImportCG5()
    IO.get_cg5(grvfile)
    IO.get_gps(gpsfile)
    IO.settings(True)

# Process Data
    PD = dataprep.ProcessData()
    PD.indata = IO.outdata
    PD.settings(True)

    datout = PD.outdata['Line']

    boug = datout.data['0.0']['BOUGUER']

    datout2 = np.array([2467.97473763, 2467.98598368, 2468.20238129,
                        2468.19436119, 2468.33253955, 2468.27440309,
                        2468.43977943, 2468.48208232, 2468.50404003,
                        2468.56558788, 2468.53271366, 2468.43134379,
                        2468.53404498, 2468.66827408, 2468.4295316,
                        2468.77538352, 2468.76984603, 2468.87797134,
                        2468.72045414, 2468.82416749, 2468.99515481,
                        2468.88027135, 2468.88301386, 2468.85303658,
                        2468.93303902])

    np.testing.assert_array_almost_equal(datout2, boug)


if __name__ == "__main__":
    test_process_data()
