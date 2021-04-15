# -----------------------------------------------------------------------------
# Name:        tiltdepth.py (part of PyGMI)
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
"""
Tilt Depth Routine.

Based on work by EH Stettler

References
----------
Salem et al., 2007, Leading Edge, Dec,p1502-5
"""

import os
from math import pi
import numpy as np

from PyQt5 import QtWidgets
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from numba import jit
import pygmi.raster.cooper as cooper
import pygmi.raster.dataprep as dataprep
import pygmi.menu_default as menu_default
import pygmi.misc as misc
from pygmi.raster.datatypes import Data
from pygmi.vector.dataprep import quickgrid
from pygmi.misc import frm


class TiltDepth(QtWidgets.QDialog):
    """
    Primary class for the Tilt Depth.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    self.mmc : FigureCanvas
        main canvas containing the image
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.units = {}
        self.X = None
        self.Y = None
        self.Z = None
        self.depths = None
        self.cbar = cm.get_cmap('jet')
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.x0 = None
        self.x1 = None
        self.x2 = None
        self.y0 = None
        self.y1 = None
        self.y2 = None

        self.figure = Figure()
        self.mmc = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.cbox_band1 = QtWidgets.QComboBox()
        self.cbox_cbar = QtWidgets.QComboBox(self)
        self.dsb_inc = QtWidgets.QDoubleSpinBox()
        self.dsb_dec = QtWidgets.QDoubleSpinBox()
        self.btn_apply = QtWidgets.QPushButton('Calculate Tilt Depth')
        self.btn_save = QtWidgets.QPushButton('Save Depths to Text File')
        self.pbar = misc.ProgressBar()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        helpdocs = menu_default.HelpButton('pygmi.raster.tiltdepth')
        label2 = QtWidgets.QLabel('Band to perform Tilt Depth:')
        labelc = QtWidgets.QLabel('Color Bar:')
        label_inc = QtWidgets.QLabel('Inclination of Magnetic Field:')
        label_dec = QtWidgets.QLabel('Declination of Magnetic Field:')

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_inc.setValue(-67.)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        self.dsb_dec.setValue(-17.)

        vbl_raster = QtWidgets.QVBoxLayout()
        hbl_all = QtWidgets.QHBoxLayout(self)
        vbl_right = QtWidgets.QVBoxLayout()

        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                       QtWidgets.QSizePolicy.Expanding)
        tmp = sorted(cm.datad.keys())
        self.cbox_cbar.addItem('jet')
        self.cbox_cbar.addItems(tmp)

        self.setWindowTitle('Tilt Depth Interpretation')

        vbl_raster.addWidget(label2)
        vbl_raster.addWidget(self.cbox_band1)
        vbl_raster.addWidget(labelc)
        vbl_raster.addWidget(self.cbox_cbar)
        vbl_raster.addWidget(label_inc)
        vbl_raster.addWidget(self.dsb_inc)
        vbl_raster.addWidget(label_dec)
        vbl_raster.addWidget(self.dsb_dec)
        vbl_raster.addWidget(self.btn_apply)
        vbl_raster.addWidget(self.pbar)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_save)
        vbl_raster.addWidget(helpdocs)
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.cbox_cbar.currentIndexChanged.connect(self.change_cbar)
        self.btn_apply.clicked.connect(self.change_band1)
        self.btn_save.clicked.connect(self.save_depths)

    def save_depths(self):
        """
        Save depths.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if self.depths is None:
            return False

        ext = 'Text File (*.csv)'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent,
                                                            'Save File',
                                                            '.', ext)
        if filename == '':
            return False

        os.chdir(os.path.dirname(filename))
        np.savetxt(filename, self.depths, delimiter=',',
                   header='x, y, id, depth')

        QtWidgets.QMessageBox.information(self.parent, 'Information',
                                          'Save completed!')

        return True

    def change_cbar(self):
        """
        Change the color map for the color bar.

        Returns
        -------
        None.

        """
        zout = self.indata['Raster'][0]
        txt = str(self.cbox_cbar.currentText())

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.contour(self.X, self.Y, self.Z, [0])
        self.axes.contour(self.X, self.Y, self.Z, [45], linestyles='dashed')
        self.axes.contour(self.X, self.Y, self.Z, [-45], linestyles='dashed')

        cmap = cm.get_cmap(txt)
        cmap2 = np.array([cmap(i) for i in range(cmap.N)])
        low = int(cmap.N*(45/180))
        high = int(cmap.N*(135/180))
        cmap2[low:high] = cmap2[int(cmap.N/2)]

        cmap3 = cm.colors.ListedColormap(cmap2)
        ims = self.axes.imshow(self.Z, extent=zout.extent, cmap=cmap3)

        if self.x0 is not None:
            i = 0
            self.axes.plot(self.x1[i], self.y1[i], 'oy')
            self.axes.plot(self.x0[i], self.y0[i], 'sy')
            self.axes.plot(self.x2[i], self.y2[i], 'oy')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.colorbar(ims, format=frm)

        self.figure.canvas.draw()

    def change_band1(self):
        """
        Routine which occurs when apply button is pressed.

        Returns
        -------
        None.

        """
        txt = str(self.cbox_band1.currentText())

        self.btn_apply.setText('Calculating...')
        QtWidgets.QApplication.processEvents()
        self.btn_apply.setEnabled(False)

        for i in self.indata['Raster']:
            if i.dataid == txt:
                self.tiltdepth(i)
                self.change_cbar()

        self.btn_apply.setEnabled(True)
        self.btn_apply.setText('Calculate Tilt Depth')
        QtWidgets.QApplication.processEvents()

    def settings(self, nodialog=False):
        """
        Entry point.

        This is called when the used double clicks the routine from the
        main PyGMI interface.

        Parameters
        ----------
        test : bool, optional
            Parameter indicating testing. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' not in self.indata:
            return False

        self.indata['Raster'] = dataprep.merge(self.indata['Raster'])

        data = self.indata['Raster']
        blist = []
        for i in data:
            blist.append(i.dataid)

        self.cbox_band1.clear()
        self.cbox_band1.addItems(blist)

        if nodialog is False:
            self.show()
        QtWidgets.QApplication.processEvents()

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

#        projdata['ftype'] = '2D Mean'

        return projdata

    def tiltdepth(self, data):
        """
        Calculate tilt depth.

        Output is stored in self.outdata.

        Parameters
        ----------
        data : PyGMI Data.
            PyGMI raster dataset.

        Returns
        -------
        None.

        """
        self.pbar.setValue(0)
        self.pbar.setMaximum(4)

    # RTP
        inc = self.dsb_inc.value()
        dec = self.dsb_dec.value()

        zout = dataprep.rtp(data, inc, dec)

    # Tilt
        self.pbar.setValue(1)

        nr, nc = zout.data.shape
        dy, dx = np.gradient(zout.data)
        dxtot = np.sqrt(dx**2+dy**2)
        dz = cooper.vertical(zout.data)
        t1 = np.arctan(dz/dxtot)

        self.pbar.setValue(2)
        # A negative number implies we are straddling 0

    # Contour tilt
        x = zout.extent[0] + np.arange(nc)*zout.xdim+zout.xdim/2
        y = zout.extent[-1] - np.arange(nr)*zout.ydim-zout.ydim/2

        X, Y = np.meshgrid(x, y)
        Z = np.rad2deg(t1)
        self.X = X
        self.Y = Y
        self.Z = Z

        cnt0 = self.axes.contour(X, Y, Z, [0])
        cnt45 = self.axes.contour(X, Y, Z, [45], alpha=0)
        cntm45 = self.axes.contour(X, Y, Z, [-45], alpha=0)

        self.pbar.setValue(3)

        gx0, gy0, cgrad0, cntid0 = vgrad(cnt0)
        gx45, gy45, _, _ = vgrad(cnt45)
        gxm45, gym45, _, _ = vgrad(cntm45)

        g0 = np.transpose([gx0, gy0])

        self.pbar.setValue(4)

        dmin1 = []
        dmin2 = []

        for i, j in self.pbar.iter(g0):
            dmin1.append(distpc(gx45, gy45, i, j, 0))
            dmin2.append(distpc(gxm45, gym45, i, j, 0))

        dx1 = gx45[dmin1] - gx0
        dy1 = gy45[dmin1] - gy0

        dx2 = gxm45[dmin2] - gx0
        dy2 = gym45[dmin2] - gy0

        grad = np.arctan2(dy1, dx1)*180/pi
        grad[grad > 90] -= 180
        grad[grad < -90] += 180
        gtmp1 = np.abs(90-np.abs(grad-cgrad0))

        grad = np.arctan2(dy2, dx2)*180/pi
        grad[grad > 90] -= 180
        grad[grad < -90] += 180
        gtmp2 = np.abs(90-np.abs(grad-cgrad0))

        gtmp = np.logical_and(gtmp1 <= 10, gtmp2 <= 10)

        gx0 = gx0[gtmp]
        gy0 = gy0[gtmp]
        cntid0 = cntid0[gtmp]
        dx1 = dx1[gtmp]
        dy1 = dy1[gtmp]
        dx2 = dx2[gtmp]
        dy2 = dy2[gtmp]

        dist1 = np.sqrt(dx1**2+dy1**2)
        dist2 = np.sqrt(dx2**2+dy2**2)

        dist = np.min([dist1, dist2], 0)

        self.x0 = gx0
        self.x1 = dx1+gx0
        self.x2 = dx2+gx0
        self.y0 = gy0
        self.y1 = dy1+gy0
        self.y2 = dy2+gy0

        self.depths = np.transpose([gx0, gy0, cntid0.astype(int), dist])

        tmp = quickgrid(gx0, gy0, dist, data.xdim,
                        showprocesslog=self.showprocesslog)

        mask = np.ma.getmaskarray(tmp)
        gdat = tmp.data

        dat = Data()
        dat.data = np.ma.masked_invalid(gdat[::-1])
        dat.data.mask = mask[::-1]
        dat.nullvalue = dat.data.fill_value
        dat.dataid = data.dataid
        dat.xdim = data.xdim
        dat.ydim = data.xdim
        dat.extent = [gx0.min(), gx0.max(), gy0.min(), gy0.max()]

        self.outdata['Raster'] = [dat]


@jit(nopython=True, nogil=True)
def distpc(dx, dy, dx0, dy0, dcnt):
    """
    Find closest distances.

    Parameters
    ----------
    dx : numpy array
        X array.
    dy : numpy array
        Y array.
    dx0 : float
        X point to measure distance from.
    dy0 : float
        Y point to measure distance from.
    dcnt : int
        Starting index to measure distance from.

    Returns
    -------
    dcnt : int
        Index of closest distance found in x and y arrays.

    """
    num = dx.size
    dmin = (dx0-dx[dcnt])**2+(dy0-dy[dcnt])**2

    for i in range(num):
        dist = (dx0-dx[i])**2+(dy0-dy[i])**2
        if dmin > dist:
            dcnt = i
            dmin = dist

    return dcnt


def vgrad(cnt):
    """
    Get contour gradients at vertices.

    Parameters
    ----------
    cnt : axes.contour
        Output from Matplotlib's axes.contour.

    Returns
    -------
    gx : numpy array
        X gradients.
    gy : numpy array
        Y gradients.
    cgrad : numpy array
        Contour gradient.
    cntid : numpy array
        Contour index.

    """
    gx = []
    gy = []
    dx2 = []
    dy2 = []
    cntid = []

    n = 0
    for cntvert in cnt.allsegs[0]:
        n += 1

        dx = np.diff(cntvert[:, 0])
        dy = np.diff(cntvert[:, 1])

        cntid.extend([n]*dx.size)

        gx.extend((cntvert[:, 0][:-1] + dx/2).tolist())
        gy.extend((cntvert[:, 1][:-1] + dy/2).tolist())
        dx2.extend(dx)
        dy2.extend(dy)

    cgrad = np.arctan2(dy2, dx2)
    cgrad = np.rad2deg(cgrad)
    cgrad[cgrad > 90] -= 180.
    cgrad[cgrad < -90] += 180.

    return np.array(gx), np.array(gy), cgrad, np.array(cntid)
