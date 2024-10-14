# -----------------------------------------------------------------------------
# Name:        topo.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2024 Council for Geoscience
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
"""Calculate topographic correction for satellite data."""

import os
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt

from pygmi.raster.misc import norm2, lstack, aspect2
from pygmi.raster.iodefs import get_raster
from pygmi import menu_default
from pygmi.misc import BasicModule


class TopoCorrect(BasicModule):
    """Calculate topographic correction."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lw_indices = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton('Invert Selection')
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.change')
        lbl_ratios = QtWidgets.QLabel('Indices:')

        self.lw_indices.setSelectionMode(self.lw_indices.MultiSelection)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Topographic Correction')

        gl_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gl_main.addWidget(self.lw_indices, 1, 1, 1, 1)
        gl_main.addWidget(btn_invert, 2, 0, 1, 2)

        gl_main.addWidget(helpdocs, 6, 0, 1, 1)
        gl_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.lw_indices.clicked.connect(self.set_selected_indices)
        btn_invert.clicked.connect(self.invert_selection)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'RasterFileList' not in self.indata:
            self.showlog('No batch file list detected.')
            return False

        self.setindices()

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.lw_indices)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        flist = self.indata['RasterFileList']

        datfin = c_correction(flist, ilist, showlog=self.showlog,
                             piter=self.piter)

        if not datfin:
            return False

        self.outdata['Raster'] = datfin

        return True


def c_correction(data, dem, azimuth, zenith, showlog=print, piter=iter):
    """
    Calculate C correction.

    Parameters
    ----------
    data : PyGMI Data type
        Data to be corrected.
    dem : PyGMI Data type
        DEM data used in correction.
    azimuth : float
        Solar azimuth.
    zenith : float
        Solar zenith.
    showlog : function, optional
        Display information. The default is print.        
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    data2 : TYPE
        DESCRIPTION.

    """

    adeg, dzdx, dzdy = aspect2(dem.data)
    # slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    px, py = np.gradient(dem.data, dem.xdim)
    slope = np.sqrt(px ** 2 + py ** 2)

    slope_deg = np.degrees(np.arctan(slope))

    Z = np.deg2rad(zenith)
    a = np.deg2rad(azimuth)
    ap = np.deg2rad(adeg)
    s = np.deg2rad(slope_deg)
    sz = Z

    cosi = np.cos(Z)*np.cos(s)+np.sin(Z)*np.sin(s)*np.cos(a-ap)

    cossz = np.cos(sz)

    # C
    data2 = []
    for Lt in data:
        Lh = Lt.copy()

        x = cosi.flatten()
        y = Lt.data.flatten()
        m, b = np.polyfit(x, y, 1)

        c = b/m

        Lh.data = Lt.data*(cossz+c)/(cosi+c)
        data2.append(Lh)

    return data2


def _testfn2():
    """Test."""
    ifile = r"D:\Landslides\oneclip.tif"
    zenith = 42.7956361279988
    azimuth = 44.8154655713449

    data = get_raster(ifile)

    dem = data.pop(-1)

    adeg, dzdx, dzdy = aspect2(dem.data)
    # slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    px, py = np.gradient(dem.data, dem.xdim)
    slope = np.sqrt(px ** 2 + py ** 2)

    slope_deg = np.degrees(np.arctan(slope))

    Z = np.deg2rad(zenith)
    a = np.deg2rad(azimuth)
    ap = np.deg2rad(adeg)
    s = np.deg2rad(slope_deg)
    h = 1-s/np.pi
    sz = Z

    cosi = np.cos(Z)*np.cos(s)+np.sin(Z)*np.sin(s)*np.cos(a-ap)
    # i = np.arccos(cosi)

    cossz = np.cos(sz)

    # cos

    # data2 = []
    # for Lt in data:
    #     Lh = Lt.copy()
    #     Lh.data = Lt.data*cossz/cosi
    #     data2.append(Lh)


    # C

    data2 = []
    for Lt in data:
        Lh = Lt.copy()

        x = cosi.flatten()
        y = Lt.data.flatten()
        m, b = np.polyfit(x, y, 1)

        c = b/m

        Lh.data = Lt.data*(cossz+c)/(cosi+c)
        data2.append(Lh)


    # Plot
    for dat in [data, data2]:
        plt.figure(dpi=200)

        red = dat[3].data
        green = dat[2].data
        blue = dat[1].data

        rmin, rmax = .1, .2
        gmin, gmax = .1, .2
        bmin, bmax = .1, .2

        img = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)

        img[:, :, 0] = norm2(red, rmin, rmax)*255
        img[:, :, 1] = norm2(green, gmin, gmax)*255
        img[:, :, 2] = norm2(blue, bmin, bmax)*255

        plt.imshow(img, extent=dat[0].extent)
        plt.show()


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt
    from pygmi.rsense.iodefs import ImportBatch

    idir = r'E:\WorkProjects\ST-2020-1339 Landslides\change\ratios'
    idir = r'E:\WorkProjects\ST-2020-1339 Landslides\change\mosaic\ratios'
    os.chdir(r'E:\\')

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = TopoCorrect()
    tmp1.idir = idir
    tmp1.get_sfile(True)
    tmp1.settings()


if __name__ == "__main__":
    _testfn()
