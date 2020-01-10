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
the tests.
"""

import sys
from PyQt5 import QtWidgets
import numpy as np
from pygmi.raster.datatypes import Data
from pygmi.clust import cluster, crisp_clust, fuzzy_clust

APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


def test_cluster():
    """test cluster."""

    dat1 = Data()
    dat1.data = np.ma.identity(3)
    dat1.data.mask = np.zeros([3, 3])

    dat2 = Data()
    dat2.data = np.ma.ones([3, 3])
    dat2.data.mask = np.zeros([3, 3])

    tmp = cluster.Cluster(None)
    tmp.indata = {'Raster': [dat1, dat2, dat1]}
    tmp.spinbox_minclusters.setValue(2)
    tmp.spinbox_maxclusters.setValue(2)
    tmp.settings(True)

    datout2 = tmp.outdata['Cluster'][0].data.data
    datout = np.array([[1, 2, 2],
                       [2, 1, 2],
                       [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout-3)

    np.testing.assert_array_equal(datout2, datout)


def test_crisp():
    """test crisp cluster."""

    dat1 = Data()
    dat1.data = np.ma.identity(3)
    dat1.data.mask = np.zeros([3, 3])

    dat2 = Data()
    dat2.data = np.ma.ones([3, 3])
    dat2.data.mask = np.zeros([3, 3])

    tmp = crisp_clust.CrispClust(None)
    tmp.indata = {'Raster': [dat1, dat2]}
    tmp.spinbox_minclusters.setValue(2)
    tmp.spinbox_maxclusters.setValue(2)
    tmp.settings(True)

    datout2 = tmp.outdata['Cluster'][0].data.data
    datout = np.array([[1, 2, 2],
                       [2, 1, 2],
                       [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout-3)

    np.testing.assert_array_equal(datout2, datout)


def test_fuzzy():
    """test fuzzy cluster."""

    dat1 = Data()
    dat1.data = np.ma.identity(3)
    dat1.data.mask = np.zeros([3, 3])
    dat1.data.data[0, 0] = 1.1

    dat2 = Data()
    dat2.data = np.ma.ones([3, 3])
    dat2.data.mask = np.zeros([3, 3])

    tmp = fuzzy_clust.FuzzyClust(None)
    tmp.indata = {'Raster': [dat1, dat2]}
    tmp.spinbox_minclusters.setValue(2)
    tmp.spinbox_maxclusters.setValue(2)
    tmp.combobox_alg.setCurrentIndex(0)
    tmp.settings(True)

    datout2 = tmp.outdata['Cluster'][0].data.data
    datout = np.array([[1, 2, 2],
                       [2, 1, 2],
                       [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout-3)

    np.testing.assert_array_equal(datout2, datout)


if __name__ == "__main__":
    test_fuzzy()
