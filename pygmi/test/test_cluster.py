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

import os

import numpy as np
import psutil

from pygmi.clust import cluster, crisp_clust, fuzzy_clust, segmentation
from pygmi.raster.datatypes import Data

os.environ["LOKY_MAX_CPU_COUNT"] = str(psutil.cpu_count(logical=False))


def test_cluster():
    """test cluster."""

    dat1 = Data()
    dat1.data = np.ma.identity(3)
    dat1.data.mask = np.zeros([3, 3])
    dat1.set_transform(1, 0, 1, 0)

    dat2 = Data()
    dat2.data = np.ma.ones([3, 3])
    dat2.data.mask = np.zeros([3, 3])
    dat2.set_transform(1, 0, 1, 0)

    data = [dat1, dat2, dat1]

    datout2 = cluster.cluster(data, cltype="K-Means", min_cluster=2, max_cluster=2)

    datout2 = datout2[0].data.data
    datout = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout - 3)

    np.testing.assert_array_equal(datout2, datout)


def test_crisp():
    """test crisp cluster."""

    dat1 = Data()
    dat1.data = np.ma.identity(3)
    dat1.data.mask = np.zeros([3, 3])

    dat2 = Data()
    dat2.data = np.ma.ones([3, 3])
    dat2.data.mask = np.zeros([3, 3])

    data = [dat1, dat2]
    datout2 = crisp_clust.crispclust(data, min_cluster=2, max_cluster=2)

    datout2 = datout2[0].data.data
    datout = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout - 3)

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

    data = [dat1, dat2]
    datout2 = fuzzy_clust.fuzzyclust(data, min_cluster=2, max_cluster=2)

    datout2 = datout2[0].data.data
    datout = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
    if datout2[0, 0] == 2:
        datout = np.abs(datout - 3)

    np.testing.assert_array_equal(datout2, datout)


def test_segment():
    """Test image segmentation."""
    dat1 = Data()
    dat1.data = np.ma.zeros([3, 3])
    dat1.data.mask = np.zeros([3, 3])
    dat1.data.data[:2, 0] = 1.1

    dat2 = Data()
    dat2.data = np.ma.zeros([3, 3])
    dat2.data.mask = np.zeros([3, 3])
    dat2.data.data[:2, 0] = 1.1

    data = [dat1, dat2]

    data1 = []
    for i in data:
        data1.append(None)
        data1[-1] = 255 * (i.data - i.data.min()) / np.ma.ptp(i.data)

    data1 = np.array(data1)
    data1 = np.moveaxis(data1, 0, -1)

    datout2 = segmentation.segment1(data1, scale=10)

    datout = np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]])

    np.testing.assert_array_equal(datout2, datout)


if __name__ == "__main__":
    test_segment()
