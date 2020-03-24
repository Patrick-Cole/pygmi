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

    datout2 = np.array([-716.54569535, -716.53438794, -716.32502705,
                        -716.33644437, -716.19896636, -716.25688484,
                        -716.09051822, -716.04696756, -716.02400454,
                        -715.95874222, -715.98742539, -716.08405924,
                        -715.97707272, -715.83871245, -716.07362109,
                        -715.72504247, -715.72801816, -715.61688398,
                        -715.77093313, -715.66479048, -715.49094074,
                        -715.60267113, -715.59686646, -715.62371456,
                        -715.54105161])

    np.testing.assert_array_almost_equal(datout2, boug)


if __name__ == "__main__":
    test_process_data()
