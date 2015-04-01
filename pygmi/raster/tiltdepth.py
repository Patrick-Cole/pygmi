# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
Tilt Depth Routine
Based on work by EH Stettler

References:
    Salem et al., 2007, Leading Edge, Dec,p1502-5
"""

import os
import numpy as np
import pygmi.raster.cooper as cooper
import matplotlib.pyplot as plt
import pygmi.raster.iodefs as iodefs
import numexpr as ne
import pygmi.raster.dataprep as dataprep
from math import atan2, pi

from PyQt4 import QtGui
import matplotlib
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as \
    NavigationToolbar
import pygmi.menu_default as menu_default
import pdb


class TiltDepth(QtGui.QDialog):
    """
    This is the primary class for the Tilt Depth.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    self.mmc : MyMplCanvas, FigureCanvas
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
        self.cbar = cm.jet

        self.figure = Figure()
        self.mmc = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.cbox_band1 = QtGui.QComboBox()
        self.cbox_cbar = QtGui.QComboBox(self)
        self.dsb_inc = QtGui.QDoubleSpinBox()
        self.dsb_dec = QtGui.QDoubleSpinBox()
        self.btn_apply = QtGui.QPushButton()
        self.btn_save = QtGui.QPushButton()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        helpdocs = menu_default.HelpButton('pygmi.raster.tiltdepth')
        label2 = QtGui.QLabel()
        labelc = QtGui.QLabel()
        label_inc = QtGui.QLabel()
        label_dec = QtGui.QLabel()

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_inc.setValue(-67.)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        self.dsb_dec.setValue(-17.)

        vbl_raster = QtGui.QVBoxLayout()
        hbl_all = QtGui.QHBoxLayout(self)
        vbl_right = QtGui.QVBoxLayout()

        mpl_toolbar = NavigationToolbar(self.mmc, self)
        spacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum,
                                   QtGui.QSizePolicy.Expanding)
        tmp = sorted(cm.datad.keys())
        self.cbox_cbar.addItem('jet')
        self.cbox_cbar.addItems(tmp)

        self.setWindowTitle("Tilt Depth Interpretation")
        label2.setText('Band to perform Tilt Depth:')
        labelc.setText('Color Bar:')
        label_inc.setText("Inclination of Magnetic Field:")
        label_dec.setText("Declination of Magnetic Field:")
        self.btn_apply.setText('Recalculate Tilt Depth')
        self.btn_save.setText('Save Depths to Text File')

        vbl_raster.addWidget(label2)
        vbl_raster.addWidget(self.cbox_band1)
        vbl_raster.addWidget(labelc)
        vbl_raster.addWidget(self.cbox_cbar)
        vbl_raster.addWidget(label_inc)
        vbl_raster.addWidget(self.dsb_inc)
        vbl_raster.addWidget(label_dec)
        vbl_raster.addWidget(self.dsb_dec)
        vbl_raster.addWidget(self.btn_apply)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_save)
        vbl_raster.addWidget(helpdocs)
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.cbox_cbar.currentIndexChanged.connect(self.change_cbar)
        self.cbox_band1.currentIndexChanged.connect(self.change_band1)
        self.btn_apply.clicked.connect(self.change_band1)
        self.btn_save.clicked.connect(self.save_depths)

    def save_depths(self):
        ext = "Text File (*.csv)"

        filename = QtGui.QFileDialog.getSaveFileName(self.parent,
                                                     'Save File', '.', ext)
        if filename == '':
            return False

        os.chdir(filename.rpartition('/')[0])
        np.savetxt(filename, self.depths, delimiter=',', header='x, y, depth')

    def change_cbar(self):
        """ Change the color map for the color bar """
        zout = self.indata['Raster'][0]
        txt = str(self.cbox_cbar.currentText())

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.axes.contour(self.X, self.Y, self.Z, [0])
        self.axes.contour(self.X, self.Y, self.Z, [45], linestyles='dashed')
        self.axes.contour(self.X, self.Y, self.Z, [-45], linestyles='dashed')

        cmap = plt.get_cmap(txt)
        cmap2 = np.array([cmap(i) for i in range(cmap.N)])
        low = int(cmap.N*(45/180))
        high = int(cmap.N*(135/180))
        cmap2[low:high] = cmap2[int(cmap.N/2)]

        cmap3 = matplotlib.colors.ListedColormap(cmap2)
        ims = self.axes.imshow(self.Z, extent=dataprep.dat_extent(zout),
                               cmap=cmap3)

#        self.axes.contourf(self.X, self.Y, self.Z, [-45, 45], cmap=cmap3)

        self.figure.colorbar(ims)

        self.figure.canvas.draw()

    def change_band1(self):
        """ Combo box to change the first band"""
        txt = str(self.cbox_band1.currentText())

        self.btn_apply.setText('Calculating...')
        QtGui.QApplication.processEvents()
        self.btn_apply.setEnabled(False)

        for i in self.indata['Raster']:
            if i.dataid == txt:
                self.tiltdepth(i)
                self.change_cbar()

        self.btn_apply.setEnabled(True)
        self.btn_apply.setText('Recalculate Tilt Depth')
        QtGui.QApplication.processEvents()

    def settings(self):
        """ This is called when the used double clicks the routine from the
        main PyGMI interface"""
        if 'Raster' not in self.indata:
            return

        self.indata['Raster'] = dataprep.merge(self.indata['Raster'])

        data = self.indata['Raster']
        blist = []
        for i in data:
            blist.append(i.dataid)

        self.cbox_band1.currentIndexChanged.disconnect()
        self.cbox_band1.clear()
        self.cbox_band1.addItems(blist)

        self.cbox_band1.currentIndexChanged.connect(self.change_band1)

        self.change_band1()

        self.show()
        QtGui.QApplication.processEvents()

        return True

    def tiltdepth(self, data):
        """ Calculate tilt depth """

    # RTP
        inc = self.dsb_inc.value()
        dec = self.dsb_dec.value()

        zout = dataprep.rtp(data, inc, dec)

    # Tilt
        nr, nc = zout.data.shape
        dy, dx = np.gradient(zout.data)
        dxtot = np.sqrt(dx**2+dy**2)
        dz = cooper.vertical(zout.data)
        t1 = np.arctan(dz/dxtot)
        # A negative number implies we are straddling 0

    # Contour tilt
        x = zout.tlx + np.arange(nc)*zout.xdim+zout.xdim/2
        y = zout.tly - np.arange(nr)*zout.ydim-zout.ydim/2
        X, Y = np.meshgrid(x, y)
        Z = np.rad2deg(t1)
        self.X = X
        self.Y = Y
        self.Z = Z

        cnt0 = self.axes.contour(X, Y, Z, [0])
        cnt45 = self.axes.contour(X, Y, Z, [45], alpha=0)
        cntm45 = self.axes.contour(X, Y, Z, [-45], alpha=0)
#        self.axes.contourf(X, Y, Z, [-45, 45])

        gx0, gy0, cgrad0 = vgrad(cnt0)
        gx45, gy45, cgrad45 = vgrad(cnt45)
        gxm45, gym45, cgradm45 = vgrad(cntm45)

        pairs0 = []
        pairs1 = []
        pairs2 = []

        g0 = np.transpose([gx0, gy0, cgrad0])

        dx = gx45
        dy = gy45
        dxm = gxm45
        dym = gym45

        for i, j, k in g0:
            dist = ne.evaluate('(dx-i)**2+(dy-j)**2')
            dmin = np.nonzero(dist == dist.min())[0]
            dx1 = dx[dmin]-i
            dy1 = dy[dmin]-j
            dist1 = dx1**2+dy1**2

            grad = atan2(dy1, dx1)*180/pi
            if grad > 90:
                grad -= 180
            elif grad < -90:
                grad += 180
            gtmp = abs(90-abs(grad-k))
            if gtmp > 10:
                continue

            dist = ne.evaluate('(dxm-i)**2+(dym-j)**2')
            dmin = np.nonzero(dist == dist.min())[0]
            dx2 = dxm[dmin]-i
            dy2 = dym[dmin]-j
            dist2 = dx2**2+dy2**2

            grad = atan2(dy2, dx2)*180/pi
            if grad > 90:
                grad -= 180
            elif grad < -90:
                grad += 180
            gtmp = abs(90-abs(grad-k))
            if gtmp > 10:
                continue

            pairs0.append([i, j])
            pairs1.append([dx1, dy1, dist1])
            pairs2.append([dx2, dy2, dist2])

        pairs0 = np.array(pairs0)
        pairs1 = np.array(pairs1)
        pairs2 = np.array(pairs2)

        dist1 = np.sqrt(pairs1[:, 2])
        dist2 = np.sqrt(pairs2[:, 2])

        dist = (dist1+dist2)/2
#        dist = np.min([dist1, dist2], 0)

#        gx0f = pairs0[:, 0]
#        gy0f = pairs0[:, 1]
#        gx45f = pairs1[:, 0]
#        gy45f = pairs1[:, 1]
#        gxm45f = pairs2[:, 0]
#        gym45f = pairs2[:, 1]

#        pdb.set_trace()
        self.depths = np.hstack([pairs0, dist])


def main():
    """ Main """
    ifile = r'C:\Work\Programming\pygmi\data\magmicrolevel.ers'

    data = iodefs.get_raster(ifile)
    data = data[0]

    x, y, depth = tiltdepth(data, -17, -67)


def vgrad(cnt):
    """ Gets contour gradients at vertices """

    gx = np.array([])
    gy = np.array([])
    dx2 = np.array([])
    dy2 = np.array([])

    for i in cnt.collections[0].get_paths():
        cntvert = i.vertices

        dx = cntvert[:, 0][1:] - cntvert[:, 0][:-1]
        dy = cntvert[:, 1][1:] - cntvert[:, 1][:-1]

        gx = np.append(gx, cntvert[:, 0][:-1] + dx/2)
        gy = np.append(gy, cntvert[:, 1][:-1] + dy/2)
        dx2 = np.append(dx2, dx)
        dy2 = np.append(dy2, dy)

    cgrad = np.arctan2(dy2, dx2)
    cgrad = np.rad2deg(cgrad)
    cgrad[cgrad > 90] -= 180.
    cgrad[cgrad < -90] += 180.

    return gx, gy, cgrad


def tiltdepth(data, dec, inc):
    """ Calculate tilt depth """
# RTP
    zout = dataprep.rtp(data, inc, dec)

# Tilt
    nr, nc = zout.data.shape
    dy, dx = np.gradient(zout.data)
    dxtot = np.sqrt(dx**2+dy**2)
    dz = cooper.vertical(zout.data)
    t1 = np.arctan(dz/dxtot)
    # A negative number implies we are straddling 0


# Contour tilt
    x = zout.tlx + np.arange(nc)*zout.xdim
    y = zout.tly - np.arange(nr)*zout.ydim
    X, Y = np.meshgrid(x, y)
    Z = np.rad2deg(t1)

    cnt0 = plt.contour(X, Y, Z, [0])
    cnt45 = plt.contour(X, Y, Z, [45], alpha=0)
    cntm45 = plt.contour(X, Y, Z, [-45], alpha=0)
    plt.contourf(X, Y, Z, [-45, 45])

#    plt.imshow(zout.data, extent=dataprep.dat_extent(zout))
    plt.imshow(Z, extent=dataprep.dat_extent(zout))
    plt.colorbar()

    gx0, gy0, cgrad0 = vgrad(cnt0)
    gx45, gy45, cgrad45 = vgrad(cnt45)
    gxm45, gym45, cgradm45 = vgrad(cntm45)

    pairs0 = []
    pairs1 = []
    pairs2 = []

    g0 = np.transpose([gx0, gy0, cgrad0])

    dx = gx45
    dy = gy45
    dxm = gxm45
    dym = gym45

    for i, j, k in g0:
        dist = ne.evaluate('(dx-i)**2+(dy-j)**2')
        dmin = np.nonzero(dist == dist.min())[0]
        dx1 = dx[dmin]
        dy1 = dy[dmin]
        dist1 = dx1**2+dy1**2

        grad = atan2(dy1-j, dx1-i)*180/pi
        if grad > 90:
            grad -= 180
        elif grad < -90:
            grad += 180
        gtmp = abs(90-abs(grad-k))
        if gtmp > 10:
            continue

        dist = ne.evaluate('(dxm-i)**2+(dym-j)**2')
        dmin = np.nonzero(dist == dist.min())[0]
        dx2 = dxm[dmin]
        dy2 = dym[dmin]
        dist2 = dx2**2+dy2**2

        grad = atan2(dy2-j, dx2-i)*180/pi
        if grad > 90:
            grad -= 180
        elif grad < -90:
            grad += 180
        gtmp = abs(90-abs(grad-k))
        if gtmp > 10:
            continue

        pairs0.append([i, j])
        pairs1.append([dx1, dy1, dist1])
        pairs2.append([dx2, dy2, dist2])

    pairs0 = np.array(pairs0)
    pairs1 = np.array(pairs1)
    pairs2 = np.array(pairs2)

    dist1 = pairs1[:, 2]
    dist2 = pairs2[:, 2]

    dist = np.min([dist1, dist2], 0)

    gx0f = pairs0[:, 0]
    gy0f = pairs0[:, 1]
    gx45f = pairs1[:, 0]
    gy45f = pairs1[:, 1]
    gxm45f = pairs2[:, 0]
    gym45f = pairs2[:, 1]

    plt.axes().set_aspect('equal')

    i = 200

#    plt.plot(gx0, gy0, '.')
#    plt.plot(gx45, gy45, '.')
#    plt.plot(gxm45, gym45, '.')

    plt.plot(gx45f[i], gy45f[i], 'o')
    plt.plot(gx0f[i], gy0f[i], 's')
    plt.plot(gxm45f[i], gym45f[i], 'o')

    plt.show()

    return gx0, gy0, dist


if __name__ == "__main__":
    main()
