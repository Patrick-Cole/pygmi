# -----------------------------------------------------------------------------
# Name:        pfmod.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
""" These are tests. Run this file from within this directory to do the
tests """

import pdb
import sys
import nose
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import pygmi.raster.dataprep as dp
from pygmi.raster.iodefs import get_raster
from pygmi.misc import PTime
import numpy as np
# import pygmi.pfmod.pfmod as pfmod


def test():
    """ Test Routine """
    pass
#    pfmod.test()

# Raster Dataprep tests

def tests():
    """ Tests to debug """
    app = QtWidgets.QApplication(sys.argv)

    ttt = PTime()
    aaa = dp.GroupProj('Input Projection')

    ttt.since_last_call()
    pdb.set_trace()

    points = np.random.rand(1000000, 2)
    values = dp.func(points[:, 0], points[:, 1])

    dat = dp.quickgrid(points[:, 0], points[:, 1], values, .001, numits=-1)

    plt.imshow(dat)
    plt.colorbar()
    plt.show()


def tests_rtp():
    """ Tests to debug RTP """
    import matplotlib.pyplot as plt

    datrtp = get_raster(r'C:\Work\Programming\pygmi\data\RTP\South_Africa_EMAG2_diffRTP_surfer.grd')
    dat = get_raster(r'C:\Work\Programming\pygmi\data\RTP\South_Africa_EMAG2_TMI_surfer.grd')
    dat = dat[0]
    datrtp = datrtp[0]
    incl = -65.
    decl = -22.

    dat2 = dp.rtp(dat, incl, decl)

    plt.subplot(2, 1, 1)
    plt.imshow(dat.data, vmin=-1200, vmax=1200)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(dat2.data, vmin=-1200, vmax=1200)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # doctest.testmod(pygmi.raster)
    nose.run()
