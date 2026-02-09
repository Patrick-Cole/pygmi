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

import numpy as np
from pyproj.crs import CRS
import pytest

from pygmi.raster.datatypes import Data
from pygmi.raster import cooper, dataprep, equation_editor, ginterp, graphs
from pygmi.raster import normalisation, smooth
from pygmi.raster.misc import aspect2, check_dataid


def test_gradients():
    """test directional derivative."""
    datin = [[1, 2], [1, 2]]
    dat2 = [[-1, -1], [-1, -1]]
    dat = cooper.gradients(datin, 0, 1, 1)
    np.testing.assert_array_equal(dat, dat2)


def test_dratio():
    """test derivative ratio."""
    datin = [[1, 2], [1, 2]]
    dat2 = [[-np.pi / 2, -np.pi / 2], [-np.pi / 2, -np.pi / 2]]
    dat = cooper.derivative_ratio(datin, 90, 1)
    np.testing.assert_array_equal(dat, dat2)


def test_thgrad():
    """test total horizontal gradient."""
    datin = [[1, 2], [1, 2]]
    dat2 = [[0.1, 0.1], [0.1, 0.1]]
    dat = cooper.thgrad(datin, 10, 10)
    np.testing.assert_array_equal(dat, dat2)


def test_vertical():
    """test vertical derivative."""
    datin = Data()
    datin.data = np.ma.array([[1, 2], [1, 2]])
    datin.set_transform(10, 100, 10, 100)
    # dat2 = np.array([[-0.90757121, 0.90757121],
    #                  [-0.90757121, 0.90757121]])
    dat2 = np.array([[-0.15707963267948966, 0.15707963267948966],
                     [-0.15707963267948966, 0.15707963267948966]])
    dat = dataprep.verticalp(datin)
    np.testing.assert_array_almost_equal(dat, dat2)


def test_viz():
    """test visibility."""
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


def test_check_dataid():
    """test check dataid."""
    datin = [Data(), Data()]

    dat = check_dataid(datin)
    assert dat[0].dataid == '(1)'
    assert dat[1].dataid == '(2)'


def test_trimraster():
    """test trim raster."""
    datin = Data()
    datin.data = np.ma.masked_equal([[0, 0, 0, 0],
                                     [0, 1, 2, 0],
                                     [0, 1, 2, 0],
                                     [0, 0, 0, 0]], 0)
    datin.nodata = 0

    dat2 = [[1, 2],
            [1, 2]]

    dat = dataprep.trim_raster([datin])
    np.testing.assert_array_equal(dat[0].data, dat2)


def test_equation():
    """tests equation editor."""
    datin = Data()
    datin.data = np.ma.array([[1., 2.], [1., 2.]])
    datout = datin.data * 2

    indata = [datin, datin]
    eq = 'i0+i1'
    outdata = equation_editor.eqedit(indata, eq)

    np.testing.assert_array_equal(outdata[0].data, datout)


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

    adeg, dzdx, dzdy = aspect2(data)
    dat = [adeg, dzdx, dzdy]

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
    phi = -np.pi / 4.
    theta = np.pi / 4.
    alpha = .0

    dat = ginterp.currentshader(data, cell, theta, phi, alpha)
    np.testing.assert_array_almost_equal(dat, dat2)


def test_histcomp():
    """tests histogram compaction."""

    data = np.ma.array([[0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1],
                        [0, 1, 2, 1]])

    dat2 = [[0.0, 1.0, 2., 1.0],
            [0.0, 1.0, 2., 1.0],
            [0.0, 1.0, 2., 1.0],
            [0.0, 1.0, 2., 1.0]]

    dat, _, _ = ginterp.histcomp(data)
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
    """tests img to RGB."""

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
    dat.dataid = 'test'
    dat.data = np.ma.array([[29000., 29000.], [29000., 29000.]],
                           mask=[[0, 0], [0, 0]])
    dat.set_transform(1, 25, 1, -27)
    dat.crs = CRS.from_epsg(4326)

    return dat


def test_normalisation():
    """Tests for normalisation."""

    datin = Data()
    datin.data = np.ma.array([[1., 2.], [1., 2.]])
    indata = [datin, datin]

    outdata = normalisation.norm(indata, 'interval')
    datout = np.ma.array([[0., 1.], [0., 1.]])

    np.testing.assert_array_equal(outdata[0].data, datout)

    outdata = normalisation.norm(indata, 'mean')
    datout = np.ma.array([[-1., 1.], [-1., 1.]])

    np.testing.assert_array_equal(outdata[0].data, datout)

    outdata = normalisation.norm(indata, 'median')
    datout = np.ma.array([[-1., 1.], [-1., 1.]])

    np.testing.assert_array_equal(outdata[0].data, datout)

    outdata = normalisation.norm(indata, '8bit')
    datout = np.ma.array([[0., 255.], [0., 255.]])

    np.testing.assert_array_equal(outdata[0].data, datout)


def test_smooth():
    """Tests for smoothing."""
    datin = np.ma.ones([7, 7])
    datin[5, 5] = 2

    datout2 = smooth.mov_win_filt(datin, 'box', '2D Mean')
    datout = np.array([[0.36, 0.48, 0.6, 0.6, 0.6, 0.48, 0.36],
                       [0.48, 0.64, 0.8, 0.8, 0.8, 0.64, 0.48],
                       [0.6, 0.8, 1., 1., 1., 0.8, 0.6],
                       [0.6, 0.8, 1., 1.04, 1.04, 0.84, 0.64],
                       [0.6, 0.8, 1., 1.04, 1.04, 0.84, 0.64],
                       [0.48, 0.64, 0.8, 0.84, 0.84, 0.68, 0.52],
                       [0.36, 0.48, 0.6, 0.64, 0.64, 0.52, 0.4]])
    np.testing.assert_array_almost_equal(datout2, datout)

    datout2 = smooth.mov_win_filt(datin, 'disc', '2D Mean')
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

    datout2 = smooth.mov_win_filt(datin, 'gaussian', '2D Mean')
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
    np.testing.assert_array_almost_equal(datout2, datout)

    datout2 = smooth.mov_win_filt(datin, 'box', '2D Median')
    datout = np.array([[1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1.]])
    np.testing.assert_array_almost_equal(datout2, datout)

    datout2 = smooth.mov_win_filt(datin, 'disc', '2D Median')
    np.testing.assert_array_almost_equal(datout2, datout)


def test_agc():
    """Tests for AGC."""
    datin = np.ma.ones([7, 7])
    datin[2:-2, 2:-2] = 2

    datout2 = cooper.agc(datin, 3, 'mean')

    datout = np.array([[1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 0, 2.25, 1.50, 2.25, 0, 1e+20],
                       [1e+20, 0, 1.50, 1, 1.50, 0, 1e+20],
                       [1e+20, 0, 2.25, 1.50, 2.25, 0, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20]])

    np.testing.assert_array_almost_equal(datout2, datout)

    datout2 = cooper.agc(datin, 3, 'median')

    datout = np.array([[1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 0, 1, 1, 1, 0, 1e+20],
                       [1e+20, 0, 1, 1, 1, 0, 1e+20],
                       [1e+20, 0, 1, 1, 1, 0, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20]])

    np.testing.assert_array_almost_equal(datout2, datout)

    datout2 = cooper.agc(datin, 3, 'rms')

    datout = np.array([[1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 0, 1.5, 1.22474487, 1.5, 0, 1e+20],
                       [1e+20, 0, 1.22474487, 1, 1.22474487, 0, 1e+20],
                       [1e+20, 0, 1.5, 1.22474487, 1.5, 0, 1e+20],
                       [1e+20, 0, 0, 0, 0, 0, 1e+20],
                       [1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20, 1e+20]])

    np.testing.assert_array_almost_equal(datout2, datout)


if __name__ == "__main__":
    test_vertical()
