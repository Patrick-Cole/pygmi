# -----------------------------------------------------------------------------
# Name:        test_gravity.py (part of PyGMI)
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

from pygmi.grav import iodefs, dataprep

APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


def test_process_data():
    """test process gravity data."""

    idir = os.path.dirname(os.path.realpath(__file__))

    grvfile = os.path.join(idir, r'testdata\GravityCG5.txt')
    gpsfile = os.path.join(idir, r'testdata\GravityDGPS.csv')

# Import Data
    IO = iodefs.ImportCG5(None)
    IO.get_cg5(grvfile)
    IO.get_gps(gpsfile)
    IO.settings(True)

# Process Data
    PD = dataprep.ProcessData()
    PD.indata = IO.outdata
    PD.settings(True)

    datout = PD.outdata['Line']

    boug = datout['Gravity']['BOUGUER']
    boug = np.array(boug)

    datout2 = np.array([-716.5271207830797, -716.5167383494435,
                        -716.3040775379834, -716.3141448945184,
                        -716.1779168502552, -716.2380602844478,
                        -716.0745685905054, -716.0335678754984,
                        -716.0125048061484, -715.9501174250662,
                        -715.9810255366001, -716.0802093258817,
                        -715.9757727522261, -715.8399624210209,
                        -716.0774210031059, -715.7307423337603,
                        -715.7356179855093, -715.62703373732,
                        -715.7836328320539, -715.6793901363467,
                        -715.5077653510672, -715.622045673794,
                        -715.6187909466202, -715.6485139794081,
                        -715.5684009646598])

    np.testing.assert_array_almost_equal(datout2, boug)


if __name__ == "__main__":
    test_process_data()
