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
from subprocess import Popen, PIPE
import numpy as np
from PyQt6 import QtWidgets

from pygmi.raster.misc import lstack, aspect2
from pygmi.raster.iodefs import get_raster
from pygmi.misc import BasicModule


class TopoCorrect(BasicModule):
    """
    GUI to calculate topographic correction.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

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
        self.buttonbox.htmlfile = 'rsense.dm.topo'
        gl_main = QtWidgets.QGridLayout(self)

        lbl_dem = QtWidgets.QLabel('Digital Elevation Model:')
        lbl_azi = QtWidgets.QLabel('Solar Azimuth:')
        lbl_zen = QtWidgets.QLabel('Solar Zenith:')

        self.setWindowTitle('Topographic Correction')

        gl_main.addWidget(lbl_dem, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_dem, 1, 1, 1, 1)
        gl_main.addWidget(lbl_azi, 2, 0, 1, 1)
        gl_main.addWidget(self.le_azi, 2, 1, 1, 1)
        gl_main.addWidget(lbl_zen, 3, 0, 1, 1)
        gl_main.addWidget(self.le_zen, 3, 1, 1, 1)

        gl_main.addWidget(self.buttonbox, 6, 0, 1, 2)

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

        data = lstack(data, piter=self.piter, showlog=self.showlog)
        dem = lstack(data + [dem], piter=self.piter, showlog=self.showlog,
                     masterid=data[0].dataid)

        dem = dem.pop(-1)
        azimuth = float(self.le_azi.text())
        zenith = float(self.le_zen.text())

        datfin = c_correction(data, dem, azimuth, zenith, showlog=self.showlog,
                              piter=self.piter)

        if not datfin:
            return False

        self.outdata['Raster'] = datfin

        return True


class Sen2Cor(BasicModule):
    """
    GUI to calculate atmospheric correction using Sen2Cor.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

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
        self.buttonbox.htmlfile = 'rsense.dm.sen2cor'
        gl_main = QtWidgets.QGridLayout(self)

        pixmapi = QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)

        self.pb_sdir.setIcon(icon)
        self.pb_sdir.setStyleSheet('text-align:left;')
        self.pb_sen2cor.setIcon(icon)
        self.pb_sen2cor.setStyleSheet('text-align:left;')

        self.setWindowTitle('Sen2Cor - Sentinel 2 Atmospheric Correction')

        gl_main.addWidget(self.pb_sen2cor, 1, 0, 1, 1)
        gl_main.addWidget(self.le_sen2cor, 1, 1, 1, 1)
        gl_main.addWidget(self.pb_sdir, 2, 0, 1, 1)
        gl_main.addWidget(self.le_sdir, 2, 1, 1, 1)

        gl_main.addWidget(self.buttonbox, 6, 0, 1, 2)

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
            with Popen([sen2cor, sdir, '--GIP_L2A=' + l2agip], stdout=PIPE,
                       text=True) as proc:
                for line in proc.stdout:
                    self.showlog(line.rstrip())

            odir = glob.glob(os.path.dirname(sdir) + '//*L2A*.SAFE')
            sdate = os.path.basename(sdir).split('_')[2]
            odir = [i for i in odir if sdate in i][-1]

            self.showlog(f'Output is saved in the {odir} directory')

        return True


def c_correction(data, dem, azimuth, zenith, *, showlog=print, piter=iter):
    """
    Calculate C correction.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        Data to be corrected.
    dem : pygmi.raster.datatypes.Data
        DEM data used in correction.
    azimuth : float
        Solar azimuth in degrees.
    zenith : float
        Solar zenith in degrees.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    data2 : list
        List of c-corrected data arrays.

    """
    showlog('Calculating topographic c-correction...')
    adeg, _, _ = aspect2(dem.data)

    px, py = np.gradient(dem.data, dem.xdim)

    slope = np.ma.sqrt(px ** 2 + py ** 2)
    # slope_deg = np.degrees(np.ma.arctan(slope))
    s = np.ma.arctan(slope)

    Z = np.deg2rad(zenith)
    a = np.deg2rad(azimuth)
    ap = np.deg2rad(adeg)
    # s = np.deg2rad(slope_deg)

    # del px, py, slope, slope_deg, adeg
    del px, py, slope, adeg

    cosi = np.cos(Z) * np.cos(s) + np.sin(Z) * np.sin(s) * np.cos(a - ap)

    cossz = np.cos(Z)

    # C
    data2 = []
    for Lt in piter(data):
        Lh = Lt.copy()

        mask = np.logical_or(cosi.mask, Lt.data.mask)

        x = np.ma.masked_where(mask, cosi)
        x = x.compressed()

        y = np.ma.masked_where(mask, Lt.data)
        y = y.compressed()

        m, b = np.polyfit(x, y, 1)
        c = b / m

        print(f'zenith:{zenith} azimuth:{azimuth} c:{c}')
        # plt.figure(dpi=200)
        # plt.plot(x, y, '.')
        # trendpoly = np.poly1d((m, b))
        # plt.plot(x, trendpoly(x))
        # plt.title(c)
        # plt.show()

        Lh.data = Lt.data * (cossz + c) / (cosi + c)
        Lh.set_mask(mask)

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
    import matplotlib.pyplot as plt
    from pygmi.raster.misc import norm2
    from pygmi.misc import frm

    ifile1 = r"D:\Landslides\old\JTNdem.tif"
    ifile2 = r"D:\Landslides\GeoTiff\S2B_T36JTN_R092_20220428_stack.tif"
    # ifile2 = r"D:\Landslides\test.tif"

    dat1 = get_raster(ifile1)
    dat2 = get_raster(ifile2)
    dat = dat1 + dat2

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = TopoCorrect()
    tmp1.indata['Raster'] = dat
    tmp1.settings()

    # ifile = r"D:\Landslides\oneclip.tif"
    # zenith = 42.7956361279988
    # azimuth = 44.8154655713449

    # data = get_raster(ifile)
    # dem = data.pop(-1)

    # data = lstack(data)
    # dem = lstack(data+[dem], masterid=data[0].dataid)
    # dem = dem.pop(-1)

    # data2 = c_correction(data, dem, azimuth, zenith)

    # for dat in [data, data2]:
    #     plt.figure(dpi=200)
    #     ax = plt.gca()

    #     red = dat[3].data
    #     green = dat[2].data
    #     blue = dat[1].data

    #     rmin, rmax = .1, .2
    #     gmin, gmax = .1, .2
    #     bmin, bmax = .1, .2

    #     img = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)

    #     img[:, :, 0] = norm2(red, rmin, rmax)*255
    #     img[:, :, 1] = norm2(green, gmin, gmax)*255
    #     img[:, :, 2] = norm2(blue, bmin, bmax)*255

    #     plt.imshow(img, extent=dat[0].extent)

    #     ax.set_xlabel('Eastings')
    #     ax.set_ylabel('Northings')

    #     ax.xaxis.set_major_formatter(frm)
    #     ax.yaxis.set_major_formatter(frm)

    #     plt.show()

    # for i, _ in enumerate(data):

    #     dat = data[i]
    #     dat2 = data2[i]

    #     plt.figure(dpi=200)
    #     ax = plt.subplot(121)

    #     vmin, vmax = dat.get_vmin_vmax()
    #     plt.imshow(dat.data, vmin=vmin, vmax=vmax)

    #     ax = plt.subplot(122)

    #     vmin, vmax = dat2.get_vmin_vmax()
    #     plt.imshow(dat2.data, vmin=vmin, vmax=vmax)

    #     plt.show()


def _testfn3():
    """Test routine topo."""
    from pygmi.raster.dataprep import mosaic
    from pygmi.rsense.iodefs import get_data
    from pygmi.raster.iodefs import export_raster
    from pygmi.raster.reproj import data_reproject

    ddir = r'D:\Landslides\DEM'
    sdir = r"D:\Landslides\L2A"

    ifiles = glob.glob(sdir + '/S2B_MSIL2A*')

    icnt = 0
    for bfile in ifiles:
        print(icnt)
        bname = os.path.basename(bfile)
        print(bname)
        tmp = {}
        dem = mosaic(tmp, idir=ddir, bfile=bfile, res=10)[0]

        data = get_data(bfile)

        dat2 = []
        for i in data:
            if 'central' in i.dataid:
                i.data = i.data.astype(np.float32)
                dat2.append(i)
        data = dat2
        del dat2

        ofile = f'D:/Landslides/test/{bname}.tif'
        export_raster(ofile, data, compression='DEFLATE')

        azimuth = None
        zenith = None

        for i in data:
            rmeta = i.metadata['Raster']
            if 'DEM' in rmeta:
                azimuth = rmeta['Solar Azimuth']
                zenith = rmeta['Solar Zenith']

        zenith = float(zenith)
        azimuth = float(azimuth)
        dem = data_reproject(dem, data[0].crs)

        data = lstack(data, commonmask=True)
        data = lstack(data + [dem], masterid=data[0].dataid, commonmask=True)

        for i in data:
            i.data = i.data.astype(np.float32)
        dem = data.pop(-1)

        data = c_correction(data, dem, azimuth, zenith)

        ofile = f'D:/Landslides/test/{bname}_tc.tif'
        export_raster(ofile, data, compression='DEFLATE')

        del data, dem


if __name__ == "__main__":
    _testfn2()
