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

    boug = datout.data['0.0']['BOUGUER']

    datout2 = np.array([-716.52712078, -716.51673835, -716.30407754,
                        -716.31414489, -716.17791685, -716.23806028,
                        -716.07456859, -716.03356788, -716.01250481,
                        -715.95011743, -715.98102554, -716.08020933,
                        -715.97577275, -715.83996242, -716.07742100,
                        -715.73074233, -715.73561799, -715.62703374,
                        -715.78363283, -715.67939014, -715.50776535,
                        -715.62204567, -715.61879095, -715.64851398,
                        -715.56840096])

    np.testing.assert_array_almost_equal(datout2, boug)


if __name__ == "__main__":
    test_process_data()
