# -----------------------------------------------------------------------------
# Name:        rsense/dataprep.py (part of PyGMI)
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
"""Data preparation for satellite data."""

import os
import sys
import glob
import platform
# from contextlib import redirect_stdout
from subprocess import Popen, PIPE
import numpy as np
from PyQt5 import QtWidgets, QtCore

from pygmi.raster.misc import lstack, aspect2
from pygmi.raster.iodefs import get_raster
from pygmi import menu_default
from pygmi.misc import BasicModule


class TopoCorrect(BasicModule):
    """Calculate topographic correction."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmb_dem = QtWidgets.QComboBox()
        self.dsb_azi = QtWidgets.QDoubleSpinBox()
        self.dsb_zen = QtWidgets.QDoubleSpinBox()
        self.le_azi = QtWidgets.QLineEdit('0.0')
        self.le_zen = QtWidgets.QLineEdit('0.0')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.change')

        lbl_dem = QtWidgets.QLabel('Digital Elevation Model:')
        lbl_azi = QtWidgets.QLabel('Solar Azimuth:')
        lbl_zen = QtWidgets.QLabel('Solar Zenith:')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Topographic Correction')

        gl_main.addWidget(lbl_dem, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_dem, 1, 1, 1, 1)
        gl_main.addWidget(lbl_azi, 2, 0, 1, 1)
        gl_main.addWidget(self.le_azi, 2, 1, 1, 1)
        gl_main.addWidget(lbl_zen, 3, 0, 1, 1)
        gl_main.addWidget(self.le_zen, 3, 1, 1, 1)

        gl_main.addWidget(helpdocs, 6, 0, 1, 1)
        gl_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

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
        if 'Raster' not in self.indata:  # and 'RasterFileList' not in self.indata:
            self.showlog('No Satellite Data')
            return False

        data = self.indata['Raster']

        self.cmb_dem.clear()

        demused = 'None'
        azimuth = None
        zenith = None

        for i in data:
            self.cmb_dem.addItem(i.dataid)
            rmeta = i.metadata['Raster']
            if 'DEM' in rmeta:
                demused = rmeta['DEM']
                azimuth = rmeta['Solar Azimuth']
                zenith = rmeta['Solar Zenith']

        if demused != 'None':
            self.showlog('This dataset already has a topographic correction '
                         'applied.')
            return False

        if azimuth is not None:
            self.le_azi.setText(azimuth)
            self.le_zen.setText(zenith)

        if not nodialog:
            tmp = self.exec()
            if tmp == 0:
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
        data = []
        dem = None
        for i in self.indata['Raster']:
            if i.dataid == self.cmb_dem.currentText():
                dem = i
            else:
                data.append(i)

        data = lstack(data, self.piter, showlog=self.showlog)
        dem = lstack(data+[dem], self.piter, showlog=self.showlog,
                     masterid=data[0].dataid)

        dem = data.pop(-1)
        azimuth = float(self.le_azi.text())
        zenith = float(self.le_zen.text())

        datfin = c_correction(data, dem, azimuth, zenith, showlog=self.showlog,
                              piter=self.piter)

        if not datfin:
            return False

        self.outdata['Raster'] = datfin

        return True


class Sen2Cor(BasicModule):
    """Calculate topographic correction."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.is_import = True
        self.le_sdir = QtWidgets.QLineEdit('')
        self.le_sen2cor = QtWidgets.QLineEdit('')
        self.pb_sen2cor = QtWidgets.QPushButton(' Sen2Cor Directory')
        self.pb_sdir = QtWidgets.QPushButton(' Sentinel-2 L1C .SAFE Directory')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.change')

        pixmapi = QtWidgets.QStyle.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)

        self.pb_sdir.setIcon(icon)
        self.pb_sdir.setStyleSheet('text-align:left;')
        self.pb_sen2cor.setIcon(icon)
        self.pb_sen2cor.setStyleSheet('text-align:left;')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Sen2Cor - Sentinel 2 Amospheric Correction')

        gl_main.addWidget(self.pb_sen2cor, 1, 0, 1, 1)
        gl_main.addWidget(self.le_sen2cor, 1, 1, 1, 1)
        gl_main.addWidget(self.pb_sdir, 2, 0, 1, 1)
        gl_main.addWidget(self.le_sdir, 2, 1, 1, 1)

        gl_main.addWidget(helpdocs, 6, 0, 1, 1)
        gl_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.pb_sdir.pressed.connect(self.get_sdir)
        self.pb_sen2cor.pressed.connect(self.get_sen2cor)

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
        if 'RasterFileList' in self.indata:
            self.pb_sdir.hide()
            self.le_sdir.hide()
        else:
            self.pb_sdir.show()
            self.le_sdir.show()

        if not nodialog:
            tmp = self.exec()
            if tmp == 0:
                return False

        self.acceptall()

        return True

    def get_sdir(self, nodialog=False):
        """Get the satellite directory."""
        if not nodialog:
            idir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Sentinel 2 L1A Data Directory')

            if not idir:
                return False

            if 'L1C' not in os.path.basename(idir):
                self.showlog('Error: not L1C data.')
                self.le_sdir.setText('')
                return False

        self.le_sdir.setText(idir)

        return True

    def get_sen2cor(self, nodialog=False):
        """Get the sen2cor directory."""
        if not nodialog:
            idir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Sen2Cor Directory')

            if not idir:
                return False

            sen2cor = os.path.join(idir, 'L2A_Process')
            if platform.system() == 'Windows':
                sen2cor += '.bat'
            if not os.path.exists(sen2cor):
                self.showlog('Could not find L2A_process file in this '
                             'location')
                self.le_sen2cor.setText('')
                return False

        self.le_sen2cor.setText(idir)

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
        sen2cor = os.path.join(self.le_sen2cor.text(), 'L2A_Process')
        if platform.system() == 'Windows':
            sen2cor += '.bat'
        if 'RasterFileList' in self.indata:
            sdirs = [i.filename for i in self.indata['RasterFileList']]
            sdirs = [i for i in sdirs if 'MTD_MSIL1C.xml' in i]
            if not sdirs:
                self.showlog('No extracted L1C data found.')
                return False
            sdirs = [os.path.dirname(i) for i in sdirs]
        else:
            sdirs = [self.le_sdir.text()]
        l2agip = os.path.join(os.path.dirname(__file__), 'L2A_GIPP.xml')

        for sdir in self.piter(sdirs):
            with Popen([sen2cor, sdir, '--GIP_L2A='+l2agip], stdout=PIPE,
                       text=True) as proc:
                for line in proc.stdout:
                    self.showlog(line.rstrip())

            odir = glob.glob(os.path.dirname(sdir)+'//*L2A*.SAFE')
            sdate = os.path.basename(sdir).split('_')[2]
            odir = [i for i in odir if sdate in i][-1]

            self.showlog(f'Output is saved in the {odir} directory')

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
    adeg, _, _ = aspect2(dem.data)

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
    for Lt in piter(data):
        Lh = Lt.copy()

        x = cosi.flatten()
        y = Lt.data.flatten()
        m, b = np.polyfit(x, y, 1)

        c = b/m

        Lh.data = Lt.data*(cossz+c)/(cosi+c)
        data2.append(Lh)

    return data2


def _testfn2():
    """Test routine sen2cor."""
    from pygmi.rsense.iodefs import ImportBatch

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportBatch()
    tmp1.idir = r'D:\Landslides\L1C'
    tmp1.get_sfile(True)
    tmp1.settings()

    dat = tmp1.outdata

    tmp = Sen2Cor()
    tmp.indata = dat
    # tmp.le_sdir.setText(r"D:\Landslides\L1C\S2B_MSIL1C_20220329T073609_N0400_R092_T36JTN_20220329T094612.SAFE")
    tmp.le_sen2cor.setText(r'C:\Sen2Cor-02.12.03-win64')
    tmp.settings()


def _testfn():
    """Test routine topo."""
    ifile1 = r"D:\Landslides\JTNdem.tif"
    ifile2 = r"D:\Landslides\GeoTiff\S2B_T36JTN_R092_20220428_stack.tif"
    ifile2 = r"D:\Landslides\test.tif"

    dat1 = get_raster(ifile1)
    dat2 = get_raster(ifile2)

    dat = dat1+dat2

    # dat3 = lstack(dat1+dat2, dxy=10, commonmask=True)

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = TopoCorrect()
    tmp1.indata['Raster'] = dat
    tmp1.settings()


if __name__ == "__main__":
    _testfn2()
