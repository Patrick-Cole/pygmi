# -----------------------------------------------------------------------------
# Name:        test_mag.py (part of PyGMI)
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
the tests.
"""

import sys
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pytest

from pygmi.raster.datatypes import Data
from pygmi.mag import dataprep
from pygmi.mag import igrf, tiltdepth

APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


def test_tilt1():
    """test tilt angle."""
    datin = np.ma.array([[1, 2], [1, 2]])
    t12 = [[-1.2626272556789115, 1.2626272556789115],
           [-1.2626272556789115, 1.2626272556789115]]
    th2 = [[0.0, 0.0],
           [0.0, 0.0]]
    t22 = [[0.0, 0.0],
           [0.0, 0.0]]
    ta2 = [[1.8572654805528055e-17, 1.8572654805528055e-17],
           [1.8572654805528055e-17, 1.8572654805528055e-17]]
    tdx2 = [[0.30816907111598496, 0.30816907111598496],
            [0.30816907111598496, 0.30816907111598496]]
    t1, th, t2, ta, tdx = dataprep.tilt1(datin, 90, 0)

    np.testing.assert_array_equal(t1, t12)
    np.testing.assert_array_equal(th, th2)
    np.testing.assert_array_equal(t2, t22)
    np.testing.assert_array_equal(ta, ta2)
    np.testing.assert_array_equal(tdx, tdx2)


def test_rtp():
    """Test RTP."""
    datin = Data()
    datin.data = np.ma.array([[1, 2], [1, 2]])
    dat2 = [[0.7212671143002998, 1.9651600796627182],
            [1.060458126573062, 1.8041542185243205]]

    dat = dataprep.rtp(datin, 60, 30)

    np.testing.assert_array_equal(dat.data, dat2)


def test_IGRF():
    """Tests IGRF Calculation."""
    dat = Data()
    dat.data = np.ma.array([[29000., 29000.], [29000., 29000.]],
                           mask=[[0, 0], [0, 0]])
    dat.extent = (25, 25, -28, -27)  # left, right, bottom, top
    dat.dataid='mag'

    datin2 = Data()
    datin2.data = np.ma.array([[0., 0.], [0., 0.]], mask=[[0, 0], [0, 0]])

    datin2.extent = (25, 25, -28, -27)  # left, right, bottom, top
    datin2.dataid='dtm'

    dat2 = [[940.640983, 864.497698],
            [1164.106631, 1079.494023]]

    tmp = igrf.IGRF()
    tmp.indata = {'Raster': [dat, datin2]}
    tmp.dateedit.setDate(QtCore.QDate(2000, 1, 1))
    tmp.dsb_alt.setValue(0.)
    tmp.settings(True)

    dat = tmp.outdata['Raster'][-1].data

    np.testing.assert_array_almost_equal(dat, dat2)


def test_tilt():
    """test tilt depth."""

    datin = Data()
    datin.data = np.ma.array([[0, 0, .1, .5, 1],
                              [0, .1, .5, 1, .5],
                              [.1, .5, 1, .5, .1],
                              [.5, 1, .5, .1, 0],
                              [1, .5, .1, 0, 0]])

    tmp = tiltdepth.TiltDepth(None)
    tmp.indata = {'Raster': [datin]}
    tmp.dsb_dec.setValue(0.)
    tmp.dsb_inc.setValue(90.)
    tmp.settings(True)
    tmp.change_band1()

    datout2 = tmp.depths

    datout = np.array([[3.93612464, -1.99438548, 1., 0.32962923],
                       [3.49438548, -2.49438548, 1., 0.34958333],
                       [2.99438548, -2.99438548, 1., 0.34958333],
                       [2.49438548, -3.49438548, 1., 0.34958333],
                       [1.99438548, -3.93612464, 1., 0.32962923],
                       [1.48759916, -2.48888969, 2., 0.36542720],
                       [1.98888969, -1.98888969, 2., 0.36451351],
                       [2.48888969, -1.48759916, 2., 0.36542720]])

    np.testing.assert_array_almost_equal(datout2, datout)


if __name__ == "__main__":
    # pytest.main(['test_mag.py::test_IGRF'])
    test_IGRF()
