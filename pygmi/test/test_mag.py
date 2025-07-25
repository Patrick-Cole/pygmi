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

import numpy as np
from pyproj.crs import CRS

from pygmi.raster.datatypes import Data
from pygmi.mag import dataprep
from pygmi.mag import igrf, tiltdepth


def test_tilt1():
    """test tilt angle."""
    datin = Data()
    datin.data = np.ma.array([[1, 2], [1, 2]])
    datin.set_transform(10, 100, 10, 100)
    # t12 = [[-1.2626272556789115, 1.2626272556789115],
    #        [-1.2626272556789115, 1.2626272556789115]]
    t12 = [[-0.9172433099297409, 0.9172433099297405],
           [-0.9172433099297409, 0.9172433099297405]]

    th2 = [[0.0, 0.0],
           [0.0, 0.0]]
    t22 = [[0.0, 0.0],
           [0.0, 0.0]]
    # ta2 = [[1.8572654805528055e-17, 1.8572654805528055e-17],
    #        [1.8572654805528055e-17, 1.8572654805528055e-17]]
    ta2 = [[3.7229941089390133e-17, 3.722994108939016e-17],
           [3.7229941089390133e-17, 3.722994108939016e-17]]
    # tdx2 = [[0.30816907111598496, 0.30816907111598496],
    #         [0.30816907111598496, 0.30816907111598496]]

    tdx2 = [[0.6535530168651558, 0.6535530168651561],
            [0.6535530168651558, 0.6535530168651561]]

    t1, th, t2, ta, tdx, _, _ = dataprep.tilt1(datin, 90, 0)

    np.testing.assert_array_almost_equal(t1, t12)
    np.testing.assert_array_almost_equal(th, th2)
    np.testing.assert_array_almost_equal(t2, t22)
    np.testing.assert_array_almost_equal(ta, ta2)
    np.testing.assert_array_almost_equal(tdx, tdx2)


def test_rtp():
    """Test RTP."""
    datin = Data()
    datin.data = np.ma.array([[1, 2], [1, 2]])
    datin.set_transform(1, 1000, 1, 1000)
    dat2 = [[0.7212671143002998, 1.9651600796627182],
            [1.060458126573062, 1.8041542185243205]]

    dat = dataprep.rtp(datin, 60, 30)

    np.testing.assert_array_almost_equal(dat.data, dat2)


def test_IGRF():
    """Tests IGRF Calculation."""
    dat = Data()
    dat.data = np.ma.array([[29000., 29000.], [29000., 29000.]],
                           mask=[[0, 0], [0, 0]])

    dat.set_transform(1, 25, 1, -27)
    dat.crs = CRS.from_epsg(4326)
    dat.dataid = 'mag'

    datin2 = Data()
    datin2.data = np.ma.array([[0., 0.], [0., 0.]], mask=[[0, 0], [0, 0]])

    datin2.set_transform(1, 25, 1, -27)
    datin2.crs = CRS.from_epsg(4326)
    datin2.dataid = 'dtm'

    dat2 = [[940.640983, 864.497698],
            [1164.106631, 1079.494023]]

    sdate = 2000.0027322404371
    odata, _, _, _ = igrf.calc_igrf(datin2, sdate, sen_alt=0.,
                                    igrfonly=False)

    dat = dat.data - odata[0].data

    np.testing.assert_array_almost_equal(dat, dat2)


def test_tilt():
    """test tilt depth."""

    datin = Data()
    datin.data = np.ma.array([[0, 0, .1, .5, 1],
                              [0, .1, .5, 1, .5],
                              [.1, .5, 1, .5, .1],
                              [.5, 1, .5, .1, 0],
                              [1, .5, .1, 0, 0]])

    datin.set_transform(1, 1000, 1, 1000)
    tmp = tiltdepth.tiltdepth(datin)
    del tmp['geometry']

    datout2 = tmp.to_numpy()
    datout = np.array([[1003.9160151982273, 998.0138508793193,
                        1, 0.31992562131484004],
                       [1002.9861491206807, 997.0138508793193,
                        1, 0.3437593399011954],
                       [1001.9861491206807, 996.0839848017727,
                        1, 0.31992562131484004],
                       [1001.0839848017727, 996.9861491206807,
                        2, 0.31992562131484004],
                       [1002.0138508793193, 997.9861491206807,
                        2, 0.3437593399011954],
                       [1003.0138508793193, 998.9160151982273,
                        2, 0.31992562131484004]])

    np.testing.assert_array_almost_equal(datout2, datout)


if __name__ == "__main__":
    test_tilt1()
