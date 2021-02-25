# -----------------------------------------------------------------------------
# Name:        hypercore.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2021 Council for Geoscience
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
Core Metadata Routine.
terracore uses from to depth assignments.
"""

import json
import sys

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from PyQt5 import QtWidgets, QtCore

from pygmi.misc import frm
from pygmi.raster.iodefs import get_raster
from pygmi.misc import ProgressBarText


class GraphMap(FigureCanvasQTAgg):
    """
    Graph Map.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        self.figure = Figure()

        super().__init__(self.figure)
        self.setParent(parent)

        self.parent = parent
        self.data = []
        self.mindx = 0
        self.csp = None
        self.subplot = None
        self.numcores = 5
        self.depthmarkers = {}
        self.currentmark = ''
        self.format_coord = None

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.data[self.mindx]
        rows, cols = dat.data.T.shape

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)

        ymin = dat.data.mean()-2*dat.data.std()
        ymax = dat.data.mean()+2*dat.data.std()

        self.csp = self.subplot.imshow(dat.data.T, vmin=ymin, vmax=ymax)

        for yline in np.linspace(0, rows, self.numcores+1):
            self.subplot.plot([0, cols], [yline, yline], 'w')

        axes = self.figure.gca()
        axes.set_xlim((0, cols))
        axes.set_ylim((0, rows))
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        axes.format_coord = self.format_coord

        axes.xaxis.set_major_formatter(frm)
        axes.yaxis.set_major_formatter(frm)

        for key in self.depthmarkers:
            x, y, _ = self.depthmarkers[key]

            hrow = rows / self.numcores
            y0 = (y // hrow) * hrow
            y1 = y0 + hrow
            if key == self.currentmark:
                self.subplot.plot([x, x], [y0, y1], 'k')
            else:
                self.subplot.plot([x, x], [y0, y1], 'w')

        self.figure.canvas.draw()


class CoreMeta(QtWidgets.QDialog):
    """
    Core metadata and depth assignment routine.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.depthmarkers = {'Start': [0., 0., 0.]}
        self.nummarkers = 1
        self.depthfunc = None
        self.dx = 1.

        self.map = GraphMap(self)
        self.map.format_coord = self.format_coord
        self.combo = QtWidgets.QComboBox()
        self.combo_dmark = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)
        self.dsb_mdepth = QtWidgets.QDoubleSpinBox()
        self.dsb_traylen = QtWidgets.QDoubleSpinBox()
        self.sb_numcore = QtWidgets.QSpinBox()
        self.pb_deldmark = QtWidgets.QPushButton('Delete Current Depth Mark')
        self.pb_savedmark = QtWidgets.QPushButton('Save Depth Marks')
        self.lbl_info = QtWidgets.QLabel('')
        self.group_info = QtWidgets.QGroupBox('Information:')

        self.setupui()

        self.canvas = self.map.figure.canvas

        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.resize(800, 400)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        grid_main = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        infolayout = QtWidgets.QVBoxLayout(self.group_info)

        self.lbl_info.setWordWrap(True)

        self.setWindowTitle('Core Metadata and Depth Assignment')
        self.sb_numcore.setMinimum(1)
        self.sb_numcore.setMaximum(10)
        self.sb_numcore.setValue(5)

        self.dsb_mdepth.setMaximum(10000.0)
        self.dsb_mdepth.setValue(0.0)
        self.dsb_mdepth.setSuffix(' mm')

        self.dsb_traylen.setMaximum(10000.0)
        self.dsb_traylen.setValue(1500.0)
        self.dsb_traylen.setSuffix(' mm')

        self.pb_deldmark.setEnabled(False)

        lbl_combo = QtWidgets.QLabel('Display Band:')
        lbl_numcore = QtWidgets.QLabel('Number of core per tray:')
        lbl_traylen = QtWidgets.QLabel('Tray Length:')
        lbl_depthm = QtWidgets.QLabel('Current Depth Marker:')
        lbl_mdepth = QtWidgets.QLabel('Marker Depth:')

        infolayout.addWidget(self.lbl_info)

        grid_main.addWidget(lbl_combo, 0, 1)
        grid_main.addWidget(self.combo, 0, 2)
        grid_main.addWidget(lbl_numcore, 1, 1)
        grid_main.addWidget(self.sb_numcore, 1, 2)
        grid_main.addWidget(lbl_traylen, 2, 1)
        grid_main.addWidget(self.dsb_traylen, 2, 2)
        grid_main.addWidget(lbl_depthm, 3, 1)
        grid_main.addWidget(self.combo_dmark, 3, 2)
        grid_main.addWidget(lbl_mdepth, 4, 1)
        grid_main.addWidget(self.dsb_mdepth, 4, 2)
        grid_main.addWidget(self.pb_deldmark, 5, 1, 1, 2)
        grid_main.addWidget(self.group_info, 6, 1, 3, 2)
        grid_main.addWidget(self.pb_savedmark, 9, 1, 1, 2)

        grid_main.addWidget(self.map, 0, 0, 10, 1)
        grid_main.addWidget(self.mpl_toolbar, 11, 0)

        grid_main.addWidget(buttonbox, 12, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.sb_numcore.valueChanged.connect(self.ncore_change)
        self.dsb_traylen.valueChanged.connect(self.traylen_change)
        self.combo_dmark.currentIndexChanged.connect(self.dmark_change)
        self.dsb_mdepth.valueChanged.connect(self.dmarkval_change)
        self.pb_deldmark.clicked.connect(self.dmark_del)
        self.pb_savedmark.clicked.connect(self.savedmark)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def button_press_callback(self, event):
        """
        Button press callback.

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
        if event.button != 1:
            return

        ax = self.map.figure.gca()
        if ax.get_navigate_mode() is not None:
            return

        chk = QtWidgets.QMessageBox.question(self, 'New Depth Marker',
                                             'Add depth marker?',
                                             QtWidgets.QMessageBox.Yes |
                                             QtWidgets.QMessageBox.No)

        if chk == QtWidgets.QMessageBox.Yes:
            self.nummarkers += 1
            txt = 'Marker '+str(self.nummarkers)

            rows, cols = self.indata['Raster'][0].data.T.shape
            hrow = rows / self.sb_numcore.value()
            y1 = int((event.ydata // hrow) + 1)
            x1 = (y1-1)*cols + event.xdata
            depth = self.depthfunc(x1)
            self.depthmarkers[txt] = [event.xdata, event.ydata, depth]

            self.combo_dmark.disconnect()
            self.dsb_mdepth.disconnect()

            self.combo_dmark.clear()
            self.combo_dmark.addItems(self.depthmarkers.keys())

            self.combo_dmark.currentIndexChanged.connect(self.dmark_change)
            self.dsb_mdepth.valueChanged.connect(self.dmarkval_change)

            self.map.depthmarkers = self.depthmarkers
            self.map.init_graph()
            self.update_depthfunc()
            self.combo_dmark.setCurrentText(txt)

            if self.dsb_traylen.isEnabled():
                self.dsb_traylen.setEnabled(False)
                self.dsb_traylen.disconnect()
                self.lbl_info.setText('First depth added. Tray length now '
                                      'fixed.')

    def dmark_change(self):
        """
        Change depth marker combo.

        Returns
        -------
        None.

        """
        txt = self.combo_dmark.currentText()
        dval = self.depthmarkers[txt][2]

        self.dsb_mdepth.setValue(dval)

        if txt not in ['Start', 'End']:
            self.pb_deldmark.setEnabled(True)
        else:
            self.pb_deldmark.setEnabled(False)

        if txt == 'End':
            self.dsb_mdepth.setEnabled(False)
        else:
            self.dsb_mdepth.setEnabled(True)

        self.map.currentmark = txt
        self.map.init_graph()

    def dmark_del(self):
        """
        Delete depth marker.

        Returns
        -------
        None.

        """
        txt = self.combo_dmark.currentText()

        self.combo_dmark.setCurrentText('Start')

        if txt not in ['Start', 'End']:
            del self.depthmarkers[txt]

            self.combo_dmark.disconnect()
            self.dsb_mdepth.disconnect()

            self.combo_dmark.clear()
            self.combo_dmark.addItems(self.depthmarkers.keys())

            self.combo_dmark.currentIndexChanged.connect(self.dmark_change)
            self.dsb_mdepth.valueChanged.connect(self.dmarkval_change)

            self.map.depthmarkers = self.depthmarkers
            self.map.currentmark = 'Start'
            self.map.init_graph()
            self.update_depthfunc()

    def dmarkval_change(self):
        """
        Change depth marker value in spinbox.

        Returns
        -------
        None.

        """
        txt = self.combo_dmark.currentText()
        dval = self.dsb_mdepth.value()
        self.depthmarkers[txt][2] = dval
        self.map.currentmark = txt
        self.map.init_graph()
        self.update_depthfunc()

    def format_coord(self, x, y):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        rows, cols = self.indata['Raster'][0].data.T.shape
        hrow = rows / self.sb_numcore.value()

        y1 = int((y // hrow) + 1)
        x1 = (y1-1)*cols + x

        x2 = self.depthfunc(x1)

        return f'depth: {x2:1.2f}, core: {y1}'

    def ncore_change(self):
        """
        Number of cores has changed.

        Returns
        -------
        None.

        """
        self.map.numcores = self.sb_numcore.value()
        self.map.currentmark = self.combo_dmark.currentText()

        self.map.init_graph()
        self.update_depthfunc()

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.map.mindx = self.combo.currentIndex()
        self.map.currentmark = self.combo_dmark.currentText()
        self.map.init_graph()

    def savedmark(self):
        """
        Save depth marks to a json file, with same name as the raster image.

        Returns
        -------
        None.

        """

        sdata = {}

        ofile = self.indata['Raster'][0].filename[:-4]+'.json'

        sdata['numcores'] = self.sb_numcore.value()
        sdata['traylen'] = self.dsb_traylen.value()
        sdata['depthmarkers'] = self.depthmarkers

        with open(ofile, 'w') as todisk:
            json.dump(sdata, todisk, indent=4)

        self.lbl_info.setText('Save complete.')

    def traylen_change(self):
        """
        Called whenever the tray length parameter is changed.

        Returns
        -------
        None.

        """

        rows, cols = self.indata['Raster'][0].data.T.shape
        cores = self.sb_numcore.value()
        endlen = cores * self.dsb_traylen.value()
        self.dx = endlen/(cols*cores)

        self.depthmarkers['End'] = [cols, rows-1, endlen]

        if self.combo_dmark.currentText() == 'End':
            self.dsb_mdepth.setValue(endlen)

    def update_depthfunc(self):
        """
        Updates the depth function

        Returns
        -------
        None.

        """
        tmp = []
        for i in self.depthmarkers.values():
            tmp.append(i)

        tmp = np.array(tmp)

        x0 = tmp[:, 0]
        y0 = tmp[:, 1]
        z0 = tmp[:, 2]

        rows, cols = self.indata['Raster'][0].data.T.shape
        hrow = rows / self.sb_numcore.value()

        y1 = (y0 // hrow) + 1
        x1 = (y1-1)*cols + x0

        xy1 = np.transpose([x1, y1, z0])
        xy = xy1[np.argsort(xy1[:, 0])]

        x2 = xy[:, 0]
        z2 = xy[:, 2]
        z2[-1] = (x2[-1]-x2[-2])*self.dx + z2[-2]

        xend, yend, _ = self.depthmarkers['End']
        self.depthmarkers['End'] = [xend, yend, z2[-1]]

        self.depthfunc = interp1d(x2, z2)

        if not (z2 == np.sort(z2)).all():
            self.group_info.setStyleSheet('background-color: red; color: '
                                          'white')
            self.lbl_info.setText('Your depths are out of order! '
                                  'Please check.')
        else:
            self.group_info.setStyleSheet('')
            self.lbl_info.setText('')

    def settings(self, nodialog=False):
        """
        Settings.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Raster' not in self.indata:
            self.showprocesslog('Error: You must have a multi-band raster '
                                'dataset in addition to your cluster '
                                'analysis results')
            return False

        self.map.data = self.indata['Raster']

        bands = [i.dataid for i in self.indata['Raster']]

        self.combo.clear()
        self.combo.addItems(bands)
        self.combo.currentIndexChanged.connect(self.on_combo)

        self.traylen_change()

        self.combo_dmark.clear()
        self.combo_dmark.addItems(self.depthmarkers.keys())

        self.ncore_change()

        if self.combo_dmark.currentText() not in ['Start', 'End']:
            self.pb_deldmark.setEnabled(True)

        self.update_depthfunc()

        tmp = self.exec_()

        if tmp == 0:
            return False

        self.outdata['Raster'] = self.indata['Raster']

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

        # self.combo_class.setCurrentText(projdata['combo_class'])

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

        # projdata['combo_class'] = self.combo_class.currentText()

        return projdata


def testfn():
    """Main testing routine."""
    import matplotlib.pyplot as plt

    ifile = (r'c:\work\Workdata\HyperspectralScanner\PTest\smile\FENIX\\'
             r'clip_BV1_17_118m16_125m79_2020-06-30_12-43-14.dat')

    pbar = ProgressBarText()
    data = get_raster(ifile, piter=pbar.iter)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = CoreMeta()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    testfn()
