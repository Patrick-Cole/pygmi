# -----------------------------------------------------------------------------
# Name:        test_raster.py (part of PyGMI)
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
import glob
import sys
import tempfile
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pytest
from pygmi.raster.datatypes import Data
from pygmi.raster import cooper, dataprep, equation_editor, ginterp, graphs
from pygmi.raster import igrf, iodefs, normalisation, smooth, tiltdepth

APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


def test_gradients():
    """test directional derivative."""
    datin = [[1, 2], [1, 2]]
    dat2 = [[-1, -1], [-1, -1]]
    dat = cooper.gradients(datin, 0, 1, 1)
    np.testing.assert_array_equal(dat, dat2)


def test_dratio():
    """test derivative ratio."""
    datin = [[1, 2], [1, 2]]
    dat2 = [[-np.pi/2, -np.pi/2], [-np.pi/2, -np.pi/2]]
    dat = cooper.derivative_ratio(datin, 90, 1)
    np.testing.assert_array_equal(dat, dat2)


def test_viz():
    """test vizibility."""
    datin = np.ma.array([[1, 2], [1, 2]])
    vtot2 = [[4., 4.], [4., 4.]]
    vstd2 = [[0.5345224838248488, 0.5345224838248488],
             [0.5345224838248488, 0.5345224838248488]]
    vsum2 = [[2.613125929752753, 2.613125929752753],
             [2.613125929752753, 2.613125929752753]]
    vtot, vstd, vsum = cooper.visibility2d(datin, 1, 0)

    np.testing.assert_array_equal(vtot, vtot2)
    np.testing.assert_array_equal(vstd, vstd2)
    np.testing.assert_array_equal(vsum, vsum2)


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
    t1, th, t2, ta, tdx = cooper.tilt1(datin, 90, 0)

    np.testing.assert_array_equal(t1, t12)
    np.testing.assert_array_equal(th, th2)
    np.testing.assert_array_equal(t2, t22)
    np.testing.assert_array_equal(ta, ta2)
    np.testing.assert_array_equal(tdx, tdx2)


def test_rtp():
    """test rtp."""
    datin = Data()
    datin.data = np.ma.array([[1, 2], [1, 2]])
    dat2 = [[0.7212671143002998, 1.9651600796627182],
            [1.060458126573062, 1.8041542185243205]]

    dat = dataprep.rtp(datin, 60, 30)

    np.testing.assert_array_equal(dat.data, dat2)


def test_check_dataid():
    """test check dataid."""
    datin = [Data(), Data()]

    dat = dataprep.check_dataid(datin)
    assert dat[0].dataid == '(1)'
    assert dat[1].dataid == '(2)'


def test_trimraster():
    """test trim raster."""
    datin = Data()
    datin.data = np.ma.masked_equal([[0, 0, 0, 0],
                                     [0, 1, 2, 0],
                                     [0, 1, 2, 0],
                                     [0, 0, 0, 0]], 0)
    datin.nullvalue = 0

    dat2 = [[1, 2],
            [1, 2]]

    dat = dataprep.trim_raster([datin])
    np.testing.assert_array_equal(dat[0].data, dat2)


def test_quickgrid():
    """test quick grid."""
    dat2 = [[1, 1],
            [2, 1.3333333333333333]]

    x = np.array([1, 2, 1])
    y = np.array([1, 1, 2])
    z = np.array([1, 1, 2])

    dat = dataprep.quickgrid(x, y, z, 1)
    np.testing.assert_array_equal(dat, dat2)


def test_equation():
    """tests equation editor."""
    datin = Data()
    datin.data = np.ma.array([[1., 2.], [1., 2.]])
    datout = datin.data*2

    tmp = equation_editor.EquationEditor()
    tmp.indata = {'Raster': [datin, datin]}
    tmp.equation = 'i0+i1'
    tmp.settings(True)

    np.testing.assert_array_equal(tmp.outdata['Raster'][0].data, datout)


def test_hmode():
    """tests hmode."""
    datin = [1, 2, 3, 3, 4, 5, 6]
    dat = equation_editor.hmode(datin)
    dat2 = 3.0000384467512493
    assert dat == dat2


def test_aspect():
    """tests aspect."""

    data = np.array([[0, 1, 2, 1],
                     [0, 1, 2, 1],
                     [0, 1, 2, 1],
                     [0, 1, 2, 1]])

    dat2 = [[[270.0, 270.0, -1.0, 90.0],
             [270.0, 270.0, -1.0, 90.0],
             [270.0, 270.0, -1.0, 90.0],
             [270.0, 270.0, -1.0, 90.0]],
            [[0.5, 1., 0., -0.5],
             [0.5, 1., 0., -0.5],
             [0.5, 1., 0., -0.5],
             [0.5, 1., 0., -0.5]],
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]]

    dat = ginterp.aspect2(data)

    np.testing.assert_array_equal(dat, dat2)


def test_shader():
    """tests shader."""

    data = np.array([[0, 1, 2, 1],
                     [0, 1, 2, 1],
                     [0, 1, 2, 1],
                     [0, 1, 2, 1]])

    dat2 = [[0.7626513511471404, 0.7599169088331246, 0.7653668647301795,
             0.7680632445003582],
            [0.7626513511471404, 0.7599169088331246, 0.7653668647301795,
             0.7680632445003582],
            [0.7626513511471404, 0.7599169088331246, 0.7653668647301795,
             0.7680632445003582],
            [0.7626513511471404, 0.7599169088331246, 0.7653668647301795,
             0.7680632445003582]]

    cell = 100.
    phi = -np.pi/4.
    theta = np.pi/4.
    alpha = .0

    dat = ginterp.currentshader(data, cell, theta, phi, alpha)
    np.testing.assert_array_equal(dat, dat2)


def test_histcomp():
    """tests histogram compaction."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[0.0, 1.0, 1.9921875, 1.0],
            [0.0, 1.0, 1.9921875, 1.0],
            [0.0, 1.0, 1.9921875, 1.0],
            [0.0, 1.0, 1.9921875, 1.0]]

    dat = ginterp.histcomp(data)
    np.testing.assert_array_equal(dat, dat2)


def test_histeq():
    """tests histogram equalisation."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[0., 10922.66666667, 32768., 10922.66666667],
            [0., 10922.66666667, 32768., 10922.66666667],
            [0., 10922.66666667, 32768., 10922.66666667],
            [0., 10922.66666667, 32768., 10922.66666667]]

    dat = ginterp.histeq(data)
    np.testing.assert_array_almost_equal(dat, dat2)


def test_img2rgb():
    """tests img to rgb."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[[0, 0, 128, 255],
             [122, 255, 125, 255],
             [132, 0, 0, 255],
             [122, 255, 125, 255]],
            [[0, 0, 128, 255],
             [122, 255, 125, 255],
             [132, 0, 0, 255],
             [122, 255, 125, 255]],
            [[0, 0, 128, 255],
             [122, 255, 125, 255],
             [132, 0, 0, 255],
             [122, 255, 125, 255]],
            [[0, 0, 128, 255],
             [122, 255, 125, 255],
             [132, 0, 0, 255],
             [122, 255, 125, 255]]]

    dat = ginterp.img2rgb(data)
    np.testing.assert_array_equal(dat, dat2)


def test_norm():
    """tests norm2."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[0., 0.5, 1., 0.5],
            [0., 0.5, 1., 0.5],
            [0., 0.5, 1., 0.5],
            [0., 0.5, 1., 0.5]]

    dat = ginterp.norm2(data)
    np.testing.assert_array_equal(dat, dat2)


def test_norm255():
    """tests norm255."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[1, 128, 255, 128],
            [1, 128, 255, 128],
            [1, 128, 255, 128],
            [1, 128, 255, 128]]

    dat = ginterp.norm255(data)
    np.testing.assert_array_equal(dat, dat2)


def test_corr2d():
    """tests corr2d."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = 1.

    dat = graphs.corr2d(data, data)
    np.testing.assert_array_equal(dat, dat2)


@pytest.fixture
def smalldata():
    """Small test dataset."""
    dat = Data()
    dat.data = np.ma.array([[29000., 29000.], [29000., 29000.]],
                           mask=[[0, 0], [0, 0]])
    dat.extent = (25, 25, -28, -27)  # left, right, bottom, top

    return dat


def test_IGRF(smalldata):
    """Tests IGRF Calculation."""
    datin2 = Data()
    datin2.data = np.ma.array([[0., 0.], [0., 0.]], mask=[[0, 0], [0, 0]])

    datin2.extent = (25, 25, -28, -27)  # left, right, bottom, top

    dat2 = [[940.640983, 864.497698],
            [1164.106631, 1079.494023]]

    tmp = igrf.IGRF()
    tmp.indata = {'Raster': [smalldata, datin2]}
    tmp.dateedit.setDate(QtCore.QDate(2000, 1, 1))
    tmp.dsb_alt.setValue(0.)
    tmp.settings(True)

    dat = tmp.outdata['Raster'][-1].data

    np.testing.assert_array_almost_equal(dat, dat2)


@pytest.mark.parametrize("ext, drv", [('.bil', 'EHdr'), ('.tif', 'GTiff'),
                                      ('.ers', 'ERS'), ('.hdr', 'ENVI'),
                                      ('.grd', 'GSBG'), ('.sdat', 'SAGA'),
                                      ('.img', 'HFA')])
def test_io_gdal(smalldata, ext, drv):
    """Tests IO for gdal files."""
    ofile = tempfile.gettempdir() + '\\iotest'+ext

    iodefs.export_gdal(ofile, [smalldata], drv)
    dat2 = iodefs.get_raster(ofile)

    # Cleanup files
    for i in glob.glob(tempfile.gettempdir() + '\\iotest*'):
        os.unlink(i)

    np.testing.assert_array_equal(smalldata.data, dat2[0].data)


def test_io_ascii(smalldata):
    """Tests IO for ascii files."""
    ofile = tempfile.gettempdir() + '\\iotest.asc'

    tmp = iodefs.ExportData(None)
    tmp.ifile = ofile
    tmp.export_ascii([smalldata])

    dat2 = iodefs.get_ascii(ofile)

    # Cleanup files
    for i in glob.glob(tempfile.gettempdir() + '\\iotest*'):
        os.unlink(i)

    np.testing.assert_array_equal(smalldata.data, dat2[0].data)


def test_io_xyz(smalldata):
    """Tests IO for xyz files."""
    ofile = tempfile.gettempdir() + '\\iotest.xyz'

    tmp = iodefs.ExportData(None)
    tmp.ifile = ofile
    tmp.export_ascii_xyz([smalldata])

    dat2 = iodefs.get_raster(ofile)

    # Cleanup files
    for i in glob.glob(tempfile.gettempdir() + '\\iotest*'):
        os.unlink(i)

    np.testing.assert_array_equal(smalldata.data, dat2[0].data)


def test_normalisation():
    """Tests for normalisation."""

    datin = Data()
    datin.data = np.ma.array([[1., 2.], [1., 2.]])

    tmp = normalisation.Normalisation(None)
    tmp.indata = {'Raster': [datin, datin]}
    tmp.radiobutton_interval.setChecked(True)
    tmp.settings(True)
    datout = np.ma.array([[0., 1.], [0., 1.]])

    np.testing.assert_array_equal(tmp.outdata['Raster'][0].data, datout)

    tmp.radiobutton_mean.setChecked(True)
    tmp.settings(True)
    datout = np.ma.array([[-1., 1.], [-1., 1.]])

    np.testing.assert_array_equal(tmp.outdata['Raster'][0].data, datout)

    tmp.radiobutton_median.setChecked(True)
    tmp.settings(True)
    datout = np.ma.array([[-1., 1.], [-1., 1.]])

    np.testing.assert_array_equal(tmp.outdata['Raster'][0].data, datout)

    tmp.radiobutton_8bit.setChecked(True)
    tmp.settings(True)

    datout = np.ma.array([[0., 255.], [0., 255.]])

    np.testing.assert_array_equal(tmp.outdata['Raster'][0].data, datout)


def test_smooth():
    """Tests for smoothing."""
    datin = Data()
    datin.data = np.ma.ones([7, 7])
    datin.data[5, 5] = 2

    tmp = smooth.Smooth(None)
    tmp.indata = {'Raster': [datin]}

    tmp.radiobutton_2dmean.setChecked(True)
    tmp.radiobutton_box.setChecked(True)
    tmp.choosefilter()
    tmp.settings(True)
    datout2 = tmp.outdata['Raster'][0].data.data

    datout = np.array([[0.36, 0.48, 0.6, 0.6, 0.6, 0.48, 0.36],
                       [0.48, 0.64, 0.8, 0.8, 0.8, 0.64, 0.48],
                       [0.6, 0.8, 1., 1., 1., 0.8, 0.6],
                       [0.6, 0.8, 1., 1.04, 1.04, 0.84, 0.64],
                       [0.6, 0.8, 1., 1.04, 1.04, 0.84, 0.64],
                       [0.48, 0.64, 0.8, 0.84, 0.84, 0.68, 0.52],
                       [0.36, 0.48, 0.6, 0.64, 0.64, 0.52, 0.4]])

    np.testing.assert_array_almost_equal(datout2, datout)

    tmp.radiobutton_disk.setChecked(True)
    tmp.choosefilter()
    tmp.settings(True)
    datout2 = tmp.outdata['Raster'][0].data.data

    datout = np. array([[0.30379747, 0.36708861, 0.43037975, 0.44303797,
                         0.43037975, 0.37974684, 0.3164557],
                        [0.36708861, 0.44303797, 0.53164557, 0.5443038,
                         0.53164557, 0.46835443, 0.39240506],
                        [0.43037975, 0.53164557, 0.62025316, 0.63291139,
                         0.62025316, 0.5443038, 0.4556962],
                        [0.44303797, 0.5443038, 0.63291139, 0.63291139,
                         0.63291139, 0.55696203, 0.46835443],
                        [0.43037975, 0.53164557, 0.62025316, 0.63291139,
                         0.62025316, 0.5443038, 0.4556962],
                        [0.37974684, 0.46835443, 0.5443038, 0.55696203,
                         0.5443038, 0.48101266, 0.40506329],
                        [0.3164557, 0.39240506, 0.4556962, 0.46835443,
                         0.4556962, 0.40506329, 0.34177215]])

    np.testing.assert_array_almost_equal(datout2, datout)

    tmp.radiobutton_gaussian.setChecked(True)
    tmp.choosefilter()
    tmp.settings(True)

    datout = np.array([[0.25999671, 0.38869512, 0.50989872, 0.50989872,
                        0.50989872, 0.50989872, 0.38120031],
                       [0.38869512, 0.58109927, 0.76229868, 0.76229868,
                        0.76229868, 0.76229868, 0.56989453],
                       [0.50989872, 0.76229868, 1., 1., 1., 1., 0.74760005],
                       [0.50989872, 0.76229868, 1., 1., 1., 1., 0.74760005],
                       [0.50989872, 0.76229868, 1., 1., 1.06370574,
                        1.06499268, 0.81130578],
                       [0.50989872, 0.76229868, 1., 1., 1.06499268,
                        1.06630562, 0.81259272],
                       [0.38120031, 0.56989453, 0.74760005, 0.74760005,
                        0.81130578, 0.81259272, 0.62261157]])
    datout2 = tmp.outdata['Raster'][0].data.data

    np.testing.assert_array_almost_equal(datout2, datout)

    tmp.radiobutton_2dmedian.setChecked(True)
    tmp.radiobutton_box.setChecked(True)
    tmp.choosefilter()
    tmp.settings(True)
    datout2 = tmp.outdata['Raster'][0].data.data

    datout = np.array([[1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.]])

    np.testing.assert_array_almost_equal(datout2, datout)

    tmp.radiobutton_disk.setChecked(True)
    tmp.choosefilter()
    tmp.settings(True)
    datout2 = tmp.outdata['Raster'][0].data.data

    np.testing.assert_array_almost_equal(datout2, datout)


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
    test_rtp()
