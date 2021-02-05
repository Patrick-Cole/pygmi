# -----------------------------------------------------------------------------
# Name:        tab_prof.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""Core Mask Routines."""

import copy
import os
import sys

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import cm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from pygmi.misc import frm
from pygmi.raster.iodefs import get_raster, export_gdal
from pygmi.misc import ProgressBarText


class CoreMask(QtWidgets.QDialog):
    """Widget class to call the main interface."""

    def __init__(self, parent=None):
        super().__init__(parent)

        if parent is None:
            self.showprocesslog = print
            self.showtext = print
        else:
            self.showprocesslog = parent.showprocesslog
            self.showtext = parent.showtext

        self.parent = parent
        self.class_index = None
        self.mask_index = None
        self.indata = {}

        self.mmc = MyMplCanvas(self)
        self.mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.hs_overview = MySlider()
        self.combo_overview = QtWidgets.QComboBox()

        self.sb_profile_linethick = QtWidgets.QSpinBox()
        self.lw_prof_defs = QtWidgets.QListWidget()

        self.SVCkernel = QtWidgets.QComboBox()
        self.pb_classify = QtWidgets.QPushButton('Classify Data')
        self.pb_savemask = QtWidgets.QPushButton('Save Mask')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.lw_prof_defs.setFixedWidth(220)

        self.hs_overview.setMaximum(100)
        self.hs_overview.setProperty('value', 0)
        self.hs_overview.setOrientation(QtCore.Qt.Horizontal)

        self.sb_profile_linethick.setMinimum(1)
        self.sb_profile_linethick.setMaximum(1000)
        self.sb_profile_linethick.setPrefix('Line Thickness: ')
        self.sb_profile_linethick.setValue(self.mmc.mywidth)

        lbl_class = QtWidgets.QLabel('SVC kernel:')
        self.SVCkernel.addItems(['rbf', 'linear', 'poly'])

        self.pb_savemask.setEnabled(False)

# Set groupboxes and layouts
        gridlayout = QtWidgets.QGridLayout(self)

        hl_pics = QtWidgets.QHBoxLayout()
        hl_pics.addWidget(self.combo_overview)
        hl_pics.addWidget(self.hs_overview)

        vl_plots = QtWidgets.QVBoxLayout()
        vl_plots.addWidget(self.mpl_toolbar)
        vl_plots.addWidget(self.mmc)
        vl_plots.addLayout(hl_pics)

        vl_tools = QtWidgets.QVBoxLayout()
        vl_tools.addWidget(self.lw_prof_defs)
        vl_tools.addWidget(self.sb_profile_linethick)
        vl_tools.addWidget(lbl_class)
        vl_tools.addWidget(self.SVCkernel)
        vl_tools.addWidget(self.pb_classify)
        vl_tools.addWidget(self.pb_savemask)

        gridlayout.addLayout(vl_plots, 0, 0, 8, 1)
        gridlayout.addLayout(vl_tools, 0, 1, 8, 1)

    # Buttons etc
        self.sb_profile_linethick.valueChanged.connect(self.setwidth)
        self.lw_prof_defs.currentItemChanged.connect(self.change_defs)

        self.hs_overview.valueChanged.connect(self.pic_overview2)
        self.combo_overview.currentIndexChanged.connect(self.pic_overview)
        self.pb_classify.pressed.connect(self.classify)
        self.pb_savemask.pressed.connect(self.savemask)

    def change_defs(self):
        """
        Change definitions.

        Returns
        -------
        None.

        """
        i = self.lw_prof_defs.currentRow()
        itxt = str(self.lw_prof_defs.item(i).text())

        if itxt not in self.mmc.classes:
            return

        self.mmc.curmodel = self.mmc.classes[itxt]

    def savemask(self):
        """
        Save mask to a file.

        Returns
        -------
        None.

        """
        if self.pb_classify.text() == 'Revert to classes':
            self.mask_index = self.mmc.class_index.copy()

        odir = os.path.dirname(self.indata['Raster'][0].filename)
        hfile = os.path.basename(self.indata['Raster'][0].filename)

        ofile = os.path.join(odir, 'mask_'+hfile[:-4]+'.hdr')

        datfin = copy.copy(self.indata['Raster'][0])
        datfin.data = self.mask_index[::-1].T
        datfin.data = (datfin.data == 2) | (datfin.data == 0)
        datfin = [datfin]

        export_gdal(ofile, datfin, 'ENVI')

    def setwidth(self, width):
        """
        Set the width of the edits on the profile view.

        Parameters
        ----------
        width : int
            Edit width.

        Returns
        -------
        None.

        """
        self.mmc.mywidth = width

    def pic_overview(self):
        """
        Horizontal slider to change picture opacity.

        Returns
        -------
        None.

        """
        self.mmc.init_grid_top(self.combo_overview.currentText(),
                               self.hs_overview.value())

    def pic_overview2(self):
        """
        Horizontal slider to change picture opacity.

        Returns
        -------
        None.

        """
        self.mmc.slide_grid_top(self.hs_overview.value())
        self.mmc.figure.canvas.draw()

    def slayer(self):
        """
        Change model layer.

        Returns
        -------
        None.

        """
        self.hs_layer.valueChanged.disconnect()
        self.hs_layer.setValue(self.sb_layer.value())
        self.hs_layer.valueChanged.connect(self.hlayer)

        self.mmc.slide_grid_top()
        self.mmc.update_line()
        self.mmc.figure.canvas.draw()

    def classify(self):
        """
        Test classification.

        Returns
        -------
        None.

        """

        if self.pb_classify.text() == 'Revert to classes':
            self.pb_classify.setText('Classify Data')
            self.mmc.class_index = self.class_index
            self.pic_overview2()
            return

        self.class_index = self.mmc.class_index.copy()

        self.pb_classify.setEnabled(False)
        self.pb_classify.setStyleSheet("background-color: red; color: white")
        self.pb_classify.setText('Busy...')
        QtWidgets.QApplication.processEvents()

        ker = self.SVCkernel.currentText()
        classifier = SVC(gamma='scale', kernel=ker)

        masks = {}
        for cname in self.mmc.classes:
            if cname == 'Erase':
                continue
            masks[cname] = (self.class_index.T ==
                            self.mmc.classes[cname])

        datall = []
        for i in self.indata['Raster']:
            datall.append(i.data)
        datall = np.array(datall)
        datall = np.moveaxis(datall, 0, -1)

        y = []
        x = []
        tlbls = []
        for i, lbl in enumerate(masks):
            y += [i]*masks[lbl].sum()
            x.append(datall[masks[lbl]])
            tlbls.append(lbl)

        y = np.array(y)
        x = np.vstack(x)
        lbls = np.unique(y)

        if len(lbls) < 2:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'You need at least two classes',
                                          QtWidgets.QMessageBox.Ok)
            self.pb_classify.setStyleSheet("")
            self.pb_classify.setText('Classify Data')
            self.pb_classify.setEnabled(True)
            return

        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y)

        classifier.fit(X_train, y_train)

        mask = ~self.indata['Raster'][0].data.mask
        datall = datall[mask]

        yout1 = classifier.predict(datall)
        yout = np.zeros_like(mask, dtype=int)
        yout[mask] = yout1
        self.mask_index = (yout.T+1)[::-1]

        self.mmc.class_index = self.mask_index

        self.pic_overview2()

        self.pb_classify.setStyleSheet("")
        self.pb_classify.setText('Revert to classes')
        self.pb_classify.setEnabled(True)
        self.pb_savemask.setEnabled(True)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Raster' not in self.indata:
            self.showprocesslog('No Data')
            return False

        self.class_index = np.zeros_like(self.indata['Raster'][0].data.T,
                                         dtype=int)
        self.mmc.class_index = self.class_index

        data = {}
        for i in self.indata['Raster']:
            data[i.dataid] = i.data.T

        self.mmc.data = data
        self.mmc.classes = {'Erase': 0, 'Core': 1, 'Tray': 2}

        self.combo_overview.currentIndexChanged.disconnect()
        self.combo_overview.clear()
        bands = [i.dataid for i in self.indata['Raster']]
        self.combo_overview.addItems(bands)
        self.combo_overview.setCurrentIndex(0)

        self.lw_prof_defs.addItems(self.mmc.classes.keys())
        update_lith_lw(self.mmc.classes, self.mmc.mlut, self.lw_prof_defs)

        self.combo_overview.currentIndexChanged.connect(self.pic_overview)

        self.mmc.init_grid_top(self.combo_overview.currentText(),
                               self.hs_overview.value())
        self.mmc.figure.canvas.draw()

        if not nodialog:
            tmp = self.exec_()
        else:
            tmp = 1

        if tmp != 1:
            return False

        return True


class MyMplCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas"""

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

        # self.lmod1 = parent.lmod1
        self.cbar = cm.get_cmap('jet')
        self.curmodel = 1
        self.mywidth = 10
        self.xold = None
        self.yold = None
        self.press = False
        self.newline = False

        self.classes = {}
        self.class_index = np.zeros([100, 100])
        self.mlut = {0: [255, 255, 255], 1: [255, 255, 0], 2: [0, 0, 255]}
        self.data = {}

        self.plotisinit = False
        self.lopac = 1.0
        self.xlims = None
        self.ylims = None
        self.crd = None
        self.myparent = parent

# Events
        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release)
        self.figure.canvas.mpl_connect('resize_event', self.on_resize)

# Initial Images
        self.laxes = fig.add_subplot(111)
        self.lims2 = self.laxes.imshow(self.class_index, aspect='equal', interpolation='none')

        self.lims = self.laxes.imshow(self.cbar(self.class_index),
                                      aspect='equal',
                                      interpolation='none')
        self.lims.format_cursor_data = lambda x: ''
        self.lims2.format_cursor_data = lambda x: ''

    def button_press(self, event):
        """
        Button press event.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return

        nmode = event.inaxes.get_navigate_mode()
        if event.button == 1 and nmode is None:
            self.press = True
            self.newline = True
            self.move(event)

    def button_release(self, event):
        """
        Button release event.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.press = False

    def move(self, event):
        """
        Mouse move event.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        curaxes = event.inaxes
        if curaxes != self.laxes:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            return

        mdata = self.class_index
        yptp, xptp = self.class_index.shape

        if self.figure.canvas.toolbar.mode == '':
            vlim = curaxes.viewLim
            tmp0 = curaxes.transData.transform((vlim.x0, vlim.y0))
            tmp1 = curaxes.transData.transform((vlim.x1, vlim.y1))
            width, height = tmp1-tmp0
            width /= mdata.shape[1]
            height /= mdata.shape[0]
            width *= xptp/vlim.width
            height *= yptp/vlim.height
            width *= self.mywidth
            height *= self.mywidth
            width = np.ceil(width)
            height = np.ceil(height)

            cbit = QtGui.QBitmap(int(width), int(height))
            cbit.fill(QtCore.Qt.color1)
            self.setCursor(QtGui.QCursor(cbit))

        if self.press is True:
            xdata = event.xdata
            ydata = event.ydata

            if self.newline is True:
                self.newline = False
                self.set_mdata(xdata, ydata, mdata)
            else:

                rrr = np.sqrt((self.xold-xdata)**2+(self.yold-ydata)**2)
                steps = int(rrr)+1
                xxx = np.linspace(self.xold, xdata, steps)
                yyy = np.linspace(self.yold, ydata, steps)

                for i, _ in enumerate(xxx):
                    self.set_mdata(xxx[i], yyy[i], mdata)

            self.xold = xdata
            self.yold = ydata

            self.class_index = mdata
            self.slide_grid_top()
            self.figure.canvas.draw()

    def set_mdata(self, xdata, ydata, mdata):
        """
        Routine to 'draw' the line on mdata. xdata and ydata are the cursor
        centre coordinates.

        Parameters
        ----------
        xdata : float
            X data.
        ydata : float
            Y data.
        mdata : numpy array
            Model array.

        Returns
        -------
        None.

        """
        if xdata < 0:
            xdata = 0
        if ydata < 0:
            ydata = 0

        hwidth = self.mywidth/2
        xstart = max(0, xdata-hwidth)
        xend = min(mdata.shape[1], xdata+hwidth)
        ystart = max(0, ydata-hwidth)
        yend = min(mdata.shape[0], ydata+hwidth)

        xstart = int(round(xstart))
        xend = int(round(xend))
        ystart = int(round(ystart))
        yend = int(round(yend))

        if xstart < xend and ystart < yend:
            mtmp = mdata[ystart:yend, xstart:xend]
            mtmp[np.logical_and(mtmp != -1, mtmp < 900)] = self.curmodel

    def luttodat(self, dat):
        """
        LUT to dat grid.

        Parameters
        ----------
        dat : numpy array
            DESCRIPTION.

        Returns
        -------
        tmp : numpy array
            DESCRIPTION.

        """
        tmp = np.zeros([dat.shape[0], dat.shape[1], 4])

        for i in np.unique(dat):
            if i == 0:
                ctmp = [0, 0, 0, 0]
            else:
                ctmp = np.array(self.mlut[int(i)]+[255])/255.

            tmp[dat[::-1] == i] = ctmp

        return tmp

    def on_resize(self, event):
        """
        Resize event.

        Used to make sure tight_layout happens on startup.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def init_grid_top(self, dat2=None, opac=100.0):
        """
        Initialise top grid.

        Parameters
        ----------
        dat2 : str, optional
            Comobox text. The default is None.
        opac : float, optional
            Opacity between 0 and 100. The default is 100.0.

        Returns
        -------
        None.

        """
        dat2 = self.myparent.combo_overview.currentText()

        ymax, xmax = self.class_index.shape

        extent = [0, xmax, 0, ymax]

        self.lopac = 1.0 - float(opac) / 100.
        dat = self.class_index

        tmp = self.luttodat(dat)

        self.lims.set_visible(False)
        self.lims2.set_visible(False)
        self.lims.set_data(tmp)
        self.lims.set_extent(extent)
        self.lims.set_alpha(self.lopac)

        if dat2 in self.data:
            self.lims2.set_visible(True)
            dat2 = self.data[dat2]
            self.lims2.set_data(dat2)
            self.lims2.set_extent(extent)
            ymin = dat2.mean()-2*dat2.std()
            ymax = dat2.mean()+2*dat2.std()
            self.lims2.set_clim(ymin, ymax)

        self.laxes.xaxis.set_major_formatter(frm)
        self.laxes.yaxis.set_major_formatter(frm)

        self.lims.set_visible(True)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def slide_grid_top(self, opac=None):
        """
        Slide top grid.

        Parameters
        ----------
        opac : float, optional
            Opacity between 0 and 100. The default is None.

        Returns
        -------
        None.

        """
        if opac is not None:
            self.lopac = 1.0 - float(opac) / 100.

        tmp = self.luttodat(self.class_index)
        self.lims.set_data(tmp)
        self.lims.set_alpha(self.lopac)

        self.laxes.draw_artist(self.lims2)
        self.laxes.draw_artist(self.lims)


class MySlider(QtWidgets.QSlider):
    """
    My Slider.

    Custom class which allows clicking on a horizontal slider bar with slider
    moving to click in a single step.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """
        Mouse press event.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))

    def mouseMoveEvent(self, event):
        """
        Mouse move event.

        Jump to pointer position while moving.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))


def update_lith_lw(lith_list, mlut, lwidget):
    """
    Update the lithology list widget.

    Parameters
    ----------
    lmod : LithModel
        3D model.
    lwidget : QListWidget
        List widget.

    Returns
    -------
    None.

    """
    lwidget.clear()
    for i in lith_list:
        lwidget.addItem(i)

    for i in range(lwidget.count()):
        tmp = lwidget.item(i)
        tindex = lith_list[str(tmp.text())]
        tcol = mlut[tindex]
        tmp.setBackground(QtGui.QColor(tcol[0], tcol[1], tcol[2], 255))

        L = (tcol[0]*299 + tcol[1]*587 + tcol[2]*114)/1000.
        if L > 128.:
            tmp.setForeground(QtGui.QColor('black'))
        else:
            tmp.setForeground(QtGui.QColor('white'))


def testfn():
    """Main testing routine."""
    ifile = r'c:\work\Workdata\HyperspectralScanner\PTest\smile\FENIX\clip_BV1_17_118m16_125m79_2020-06-30_12-43-14.dat'

    pbar = ProgressBarText()
    data = get_raster(ifile, piter=pbar.iter)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = CoreMask()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    testfn()
