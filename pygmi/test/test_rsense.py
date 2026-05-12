# -----------------------------------------------------------------------------
# Name:        test_rsense.py (part of PyGMI)
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

from pygmi.rsense import change, iodefs, ratios, transforms


def test_change():
    """test change detection."""
    idir = os.path.dirname(os.path.realpath(__file__))

    file1 = os.path.join(idir, r"testdata/rsense1.tif")
    file2 = os.path.join(idir, r"testdata/rsense2.tif")
    file3 = os.path.join(idir, r"testdata/change.tif")

    ilist = [
        "Difference",
        "Mean",
        "Standard Deviation",
        "Coefficient of Variation",
        "Spectral Angle Mapper",
    ]

    bands, tnames, filelist = iodefs.files_to_rastermeta([file1, file2])

    dat = change.calc_change(filelist, ilist)

    dat2 = iodefs.get_data(file3)

    for i in dat:
        for j in dat2:
            if i.dataid == j.dataid:
                np.testing.assert_array_almost_equal(i.data, j.data)
                break


def test_pca():
    """test PCA."""
    idir = os.path.dirname(os.path.realpath(__file__))

    file1 = os.path.join(idir, r"testdata/rsense1.tif")
    file3 = os.path.join(idir, r"testdata/pca.tif")

    dat1 = iodefs.get_data(file1)

    dat = transforms.pca_calc(dat1, 5)

    dat2 = iodefs.get_data(file3)

    for i in dat[0]:
        for j in dat2:
            if i.dataid == j.dataid:
                np.testing.assert_array_almost_equal(np.abs(i.data), np.abs(j.data))
                break


def test_mnf():
    """test MNF."""
    idir = os.path.dirname(os.path.realpath(__file__))

    file1 = os.path.join(idir, r"testdata/rsense1.tif")
    file3 = os.path.join(idir, r"testdata/mnf.tif")

    dat1 = iodefs.get_data(file1)

    dat = transforms.mnf_calc(dat1, ncmps=5)

    dat2 = iodefs.get_data(file3)

    for i in dat[0]:
        for j in dat2:
            if i.dataid == j.dataid:
                np.testing.assert_array_almost_equal(np.abs(i.data), np.abs(j.data))
                break


def test_ratios():
    """test ratios."""
    idir = os.path.dirname(os.path.realpath(__file__))

    file1 = os.path.join(idir, r"testdata/rsense1.tif")
    file3 = os.path.join(idir, r"testdata/ratio.tif")

    dat1 = iodefs.get_data(file1)

    rlist = [
        r"B4/B2 Iron Oxide",
        r"B4/B3 Ferric Iron Fe3+",
        r"B11/B4 Gossan",
        r"B11/B12 Laterite or Alteration",
        r"B12/B8+B3/B4 Ferrous Iron Fe2+",
    ]

    dat = ratios.calc_ratios(dat1, rlist)

    dat2 = iodefs.get_data(file3)

    for i in dat:
        for j in dat2:
            if i.dataid == j.dataid:
                np.testing.assert_array_almost_equal(i.data, j.data)
                break


if __name__ == "__main__":
    test_mnf()
