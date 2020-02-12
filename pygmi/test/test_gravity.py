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

    datout2 = np.array([53.94162922, 53.95201165, 54.16467246, 54.15460511,
                        54.29083315, 54.23068972, 54.39418141, 54.43518212,
                        54.45624519, 54.51863257, 54.48772446, 54.38854067,
                        54.49297725, 54.62878758, 54.39132900, 54.73800767,
                        54.73313201, 54.84171626, 54.68511717, 54.78935986,
                        54.96098465, 54.84670433, 54.84995905, 54.82023602,
                        54.90034904])

    np.testing.assert_array_almost_equal(datout2, boug)


if __name__ == "__main__":
    test_process_data()
