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
Hyperspectral Core Routines.

1) Initial import and corrections, includes smile correction, white balance
   and filtering
2) Clipping of tray
3) Masking of Boreholes
4) Assigning depths. Click a section and assign depth to that point. Depths are
   auto interpolated between assigned depths

Each tray gets a text/xlsx file, with relevant metadata such as:
    tray number, num cores, core number, depth from, depth to,
    tray x (auto get from core number?), tray y from, tray y to

section start depth
section end depth
QC on depth range to see it makes sense

Mask can be raster (separate or in main file with band called 'mask')
Mask can be a shapefile, but I think this is not great unless it is initially
created as a shapefile

"""

import copy
import os
import sys
import glob

import numpy as np
import scipy.interpolate as si
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.patches import Polygon as mPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from numba import jit
from PyQt5 import QtWidgets, QtCore
import geopandas as gpd

from pygmi.misc import frm
import pygmi.menu_default as menu_default
from pygmi.raster.iodefs import get_raster, export_gdal
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
        self.polyi = None
        self.data = []
        self.cdata = []
        self.mindx = [0, 0]
        self.csp = None
        self.subplot = None

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        # self.subplot.get_xaxis().set_visible(False)
        # self.subplot.get_yaxis().set_visible(False)

        self.csp = self.subplot.imshow(dat.data, extent=dat.extent,
                                       cmap=cm.get_cmap('jet'))
        axes = self.figure.gca()

        # axes.set_xlabel('ColumnsEastings')
        # axes.set_ylabel('Northings')

        axes.xaxis.set_major_formatter(frm)
        axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()

        self.polyi = PolygonInteractor(self.subplot)

    def update_graph(self):
        """
        Update graph.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        if mtmp[1] > 0:
            cdat = self.cdata[mtmp[1] - 1].data
            self.csp.set_data(cdat)
            self.csp.set_clim(cdat.min(), cdat.max())
        else:
            self.csp.set_data(dat.data)
            self.csp.set_clim(dat.data.min(), dat.data.max())

        self.csp.changed()
        self.figure.canvas.draw()


class PolygonInteractor(QtCore.QObject):
    """Polygon Interactor."""

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit
    polyi_changed = QtCore.pyqtSignal(list)

    def __init__(self, axtmp):
        super().__init__()
        self.ax = axtmp
        self.poly = mPolygon([(1, 1)], animated=True, ec='k',
                             joinstyle='miter')
        self.ax.add_patch(self.poly)
        self.canvas = self.poly.figure.canvas
        self.poly.set_alpha(0.5)
        self.background = None

        xtmp, ytmp = zip(*self.poly.xy)

        self.ind = None  # the active vert

        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self.motion_notify_callback)

    def draw_callback(self, event=None):
        """
        Draw callback.

        Parameters
        ----------
        event : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.ax.draw_artist(self.poly)

    def new_poly(self, npoly):
        """
        New polygon.

        Parameters
        ----------
        npoly : list or None, optional
            New polygon coordinates.

        Returns
        -------
        None.

        """
        self.poly.set_xy(npoly)

        self.update_plots()
        self.canvas.draw()

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

        if self.ax.get_navigate_mode() is not None:
            return

        self.set_line(event.xdata, event.ydata)

    def button_release_callback(self, event):
        """
        Button release callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.button != 1:
            return

        self.set_line(event.xdata, event.ydata)

        self.ind = None

        self.update_plots()

    def motion_notify_callback(self, event):
        """
        Motion notify on mouse movement.

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

        self.set_line(event.xdata, event.ydata)

    def get_ind_under_point(self, xdata, ydata):
        """
        Get the index of vertex under point if within epsilon tolerance.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        ind : int or None
            Index of vertex under point.

        """
        # display coords

        ptmp = self.poly.get_transform().transform([xdata, ydata])
        xytmp = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xytmp)

        xtt, ytt = xyt[:, 0], xyt[:, 1]
        dtt = np.sqrt((xtt - ptmp[0]) ** 2 + (ytt - ptmp[1]) ** 2)
        indseq = np.nonzero(np.equal(dtt, np.amin(dtt)))[0]
        ind = indseq[0]

        if dtt[ind] >= self.epsilon:
            ind = None

        return ind

    def update_plots(self):
        """
        Update plots.

        Returns
        -------
        None.

        """
        if self.poly.xy.size < 8:
            return
        self.polyi_changed.emit(self.poly.xy.tolist())

    def set_line(self, xdata, ydata):
        """


        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # We transform coords to screen coords so that picking tolerance is
        # always valid.
        xys = self.poly.get_transform().transform(self.poly.xy)
        ptmp = self.poly.get_transform().transform([xdata, ydata])

        self.ind = self.get_ind_under_point(xdata, ydata)

        if self.ind is None:
            dtmp = []
            for i in range(len(xys) - 1):
                dtmp.append(dist_point_to_segment(ptmp, xys[i], xys[i + 1]))

            dtmp = np.array(dtmp)
            imin = np.nonzero(dtmp < self.epsilon)[0]

            if imin.size == 0:
                return
            # self.ind = True
        else:
            imin = [self.ind]

        for i in imin:
            if i == self.ind:
                xdiff = xdata - self.poly.xy[i, 0]
                self.poly.xy[i, 0] = xdata
                self.poly.xy[i, 1] = ydata
                if i == 2:
                    self.poly.xy[1, 0] += xdiff
                    self.poly.xy[1, 1] = ydata
                if i == 1:
                    self.poly.xy[2, 0] += xdiff
                    self.poly.xy[2, 1] = ydata
                if i == 0:
                    self.poly.xy[3, 0] += xdiff
                    self.poly.xy[3, 1] = ydata
                if i == 3:
                    self.poly.xy[0, 0] += xdiff
                    self.poly.xy[0, 1] = ydata

                dx = self.poly.xy[1, 0] - self.poly.xy[0, 0]
                dy = self.poly.xy[1, 1] - self.poly.xy[0, 1]
                rad = np.arctan(dy/dx)
                print(rad)

            elif i in [0, 2]:
                self.poly.xy[i, 0] = xdata
                self.poly.xy[i+1, 0] = xdata
            else:
                self.poly.xy[i, 1] = ydata
                self.poly.xy[i+1, 1] = ydata

            if i == 0 and self.ind is None:
                self.poly.xy[4, 0] = xdata
            if i == 3 and self.ind is None:
                self.poly.xy[0, 1] = ydata

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.canvas.blit(self.ax.bbox)


class CorePrep(QtWidgets.QDialog):
    """
    Main Supervised Classification Tool Routine.

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
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.df = None

        self.map = GraphMap(self)
        self.combo = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        grid_main = QtWidgets.QGridLayout(self)
        group_map = QtWidgets.QGroupBox('Options')
        grid_right = QtWidgets.QGridLayout(group_map)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        loadshape = QtWidgets.QPushButton('Load Class Shapefile')
        saveshape = QtWidgets.QPushButton('Save Class Shapefile')

        self.setWindowTitle('Borehole Preparation')

        lbl_combo = QtWidgets.QLabel('Data Band:')

        grid_right.addWidget(lbl_combo, 0, 0, 1, 1)
        grid_right.addWidget(self.combo, 0, 1, 1, 2)

        grid_main.addWidget(self.map, 0, 0, 2, 1)
        grid_main.addWidget(self.mpl_toolbar, 2, 0, 1, 1)

        grid_main.addWidget(group_map, 0, 1, 1, 1)
        grid_main.addWidget(buttonbox, 2, 1, 1, 1)

        loadshape.clicked.connect(self.load_shape)
        saveshape.clicked.connect(self.save_shape)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def updatepoly(self, xycoords=None):
        """
        Update polygon.

        Parameters
        ----------
        xycoords : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
#         row = self.tablewidget.currentRow()
#         if row == -1:
#             return

#         self.df.loc[row] = None
#         self.df.loc[row, 'class'] = self.tablewidget.item(row, 0).text()
# #        self.df.loc[row, 'kappa'] = self.tablewidget.item(row, 1).text()

#         xycoords = self.map.polyi.poly.xy
#         if xycoords.size < 8:
#             self.df.loc[row, 'geometry'] = Polygon([])
#         else:
#             self.df.loc[row, 'geometry'] = Polygon(xycoords)

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.m[0] = self.combo.currentIndex()
        self.map.update_graph()

    def load_shape(self):
        """
        Load shapefile.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'Shapefile (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False

        df = gpd.read_file(filename)
        if 'class' not in df or 'geometry' not in df:
            return False

        self.df = df
        self.tablewidget.setRowCount(0)
        for index, _ in self.df.iterrows():
            self.tablewidget.insertRow(index)
            item = QtWidgets.QTableWidgetItem('Class '+str(index+1))
            self.tablewidget.setItem(index, 0, item)

        self.tablewidget.selectRow(0)
        coords = list(self.df.loc[0, 'geometry'].exterior.coords)
        self.map.polyi.new_poly(coords)

        return True

    def save_shape(self):
        """
        Save shapefile.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Shapefile (*.shp)')

        if filename == '':
            return False

        self.df.to_file(filename)
        return True

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

        self.map.init_graph()
        self.map.polyi.polyi_changed.connect(self.updatepoly)
        self.map.update_graph()

        axes = self.map.figure.gca()

        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()

        x1 = xmin+(xmax-xmin)*0.1
        x2 = xmax-(xmax-xmin)*0.1

        y1 = ymin+(ymax-ymin)*0.1
        y2 = ymax-(ymax-ymin)*0.1

        poly = [[x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1]]

        self.map.polyi.new_poly(poly)
        self.map.update_graph()

        # self.map.polyi.line.set_visible(True)

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
        self.combo_class.setCurrentText(projdata['combo_class'])
        self.KNalgorithm.setCurrentText(projdata['KNalgorithm'])
        self.DTcriterion.setCurrentText(projdata['DTcriterion'])
        self.RFcriterion.setCurrentText(projdata['RFcriterion'])
        self.SVCkernel.setCurrentText(projdata['SVCkernel'])
#        self.df.read_json(projdata['classes'])

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

        projdata['combo_class'] = self.combo_class.currentText()
        projdata['KNalgorithm'] = self.KNalgorithm.currentText()
        projdata['DTcriterion'] = self.DTcriterion.currentText()
        projdata['RFcriterion'] = self.RFcriterion.currentText()
        projdata['SVCkernel'] = self.SVCkernel.currentText()
#        projdata['classes'] = self.df.to_json()

        return projdata

    def update_map(self, polymask):
        """
        Update map.

        Parameters
        ----------
        polymask : numpy array
            Polygon mask.

        Returns
        -------
        None.

        """
        if max(polymask) is False:
            return

        mtmp = self.combo.currentIndex()
        mask = self.indata['Raster'][mtmp].data.mask

        polymask = np.array(polymask)
        polymask.shape = mask.shape
        polymask = np.logical_or(polymask, mask)

        dattmp = self.map.csp.get_array()
        dattmp.mask = polymask
        self.map.csp.changed()
        self.map.figure.canvas.draw()


class CoreInt(QtWidgets.QDialog):
    """
    Core Interpretation.

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
            pbar = ProgressBarText()
            self.piter = pbar.iter

        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.product = {}
        self.ratio = {}

        # self.combo_sensor = QtWidgets.QComboBox()
        self.lw_ratios = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.ratios')
        # label_sensor = QtWidgets.QLabel('Sensor:')
        label_ratios = QtWidgets.QLabel('Ratios:')

        # self.lw_ratios.setSelectionMode(self.lw_ratios.MultiSelection)

        # self.combo_sensor.addItems(['ASTER',
        #                             'Landsat 8 (OLI)',
        #                             'Landsat 7 (ETM+)',
        #                             'Landsat 4 and 5 (TM)',
        #                             'Sentinel-2'])
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Process Hyperspectral Features')

        # gridlayout_main.addWidget(label_sensor, 0, 0, 1, 1)
        # gridlayout_main.addWidget(self.combo_sensor, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_ratios, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.lw_ratios, 1, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

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

        if not nodialog:
            tmp = self.exec_()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

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

        # self.combo_sensor.setCurrentText(projdata['sensor'])
        # self.setratios()

        # for i in self.lw_ratios.selectedItems():
        #     if i.text()[2:] not in projdata['ratios']:
        #         i.setSelected(False)
        # self.set_selected_ratios()

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
        # projdata['sensor'] = self.combo_sensor.currentText()

        # rlist = []
        # for i in self.lw_ratios.selectedItems():
        #     rlist.append(i.text()[2:])

        # projdata['ratios'] = rlist

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        datfin = []




        self.outdata['Raster'] = datfin
        return True


class ImageCor(QtWidgets.QDialog):
    """
    Calculate Satellite Ratios.

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
            self.pbar = ProgressBarText()

        else:
            self.showprocesslog = parent.showprocesslog
            self.pbar = parent.pbar

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.product = {}
        self.ratio = {}

        self.idir = QtWidgets.QLineEdit('')
        self.odir = QtWidgets.QLineEdit('')
        self.dccor = QtWidgets.QCheckBox('DC Correction')
        self.smilecor = QtWidgets.QCheckBox('Geometric Smile Correction (FENIX only)')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.ratios')
        pb_idir = QtWidgets.QPushButton('Input Raw Directory')
        pb_odir = QtWidgets.QPushButton('Output Processed Directory')
        self.dccor.setChecked(True)
        self.smilecor.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Process Hyperspectral Features')

        gridlayout_main.addWidget(pb_idir, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.idir, 0, 1, 1, 1)
        gridlayout_main.addWidget(pb_odir, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.odir, 1, 1, 1, 1)
        gridlayout_main.addWidget(self.dccor, 2, 0, 1, 2)
        gridlayout_main.addWidget(self.smilecor, 3, 0, 1, 2)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_idir.pressed.connect(self.get_idir)
        pb_odir.pressed.connect(self.get_odir)


    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        # if 'Raster' not in self.indata:
        #     self.showprocesslog('No Satellite Data')
        #     return False

        if not nodialog:
            tmp = self.exec_()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

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

        # self.combo_sensor.setCurrentText(projdata['sensor'])
        # self.setratios()

        # for i in self.lw_ratios.selectedItems():
        #     if i.text()[2:] not in projdata['ratios']:
        #         i.setSelected(False)
        # self.set_selected_ratios()

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
        # projdata['sensor'] = self.combo_sensor.currentText()

        # rlist = []
        # for i in self.lw_ratios.selectedItems():
        #     rlist.append(i.text()[2:])

        # projdata['ratios'] = rlist

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        idir = self.idir.text()
        ifiles = glob.glob(os.path.join(idir, '**/*.raw'), recursive=True)
        ifiles = [i for i in ifiles if 'ref_' not in i.lower()]

        for ifile in ifiles:
            idir2 = os.path.dirname(ifile)
            hfile = os.path.basename(ifile)
            if 'OWL' in idir2:
                odir = os.path.join(self.odir.text(), r'OWL')
            elif 'RGB' in idir2:
                odir = os.path.join(self.odir.text(), r'RGB')
            elif 'FENIX' in idir2:
                odir = os.path.join(self.odir.text(), r'FENIX')
            else:
                continue

            if not os.path.exists(odir):
                os.makedirs(odir)

            self.showprocesslog('Processing '+hfile+'...')

            datah = get_raster(ifile, piter=self.pbar.iter)

            if datah is None:
                self.showprocesslog('Could not open file.')
                continue

            if self.dccor.isChecked():
                datah = dc_correct(idir2, hfile, datah, piter=self.pbar.iter,
                                   showprocesslog=self.showprocesslog)
            if self.smilecor.isChecked() and 'FENIX' in idir2:
                datah = smile(datah, piter=self.pbar.iter)

            if datah[0] is None:
                breakpoint()

            meta = 'reflectance scale factor = 10000\n'
            meta += 'wavelength = {\n'
            for i in datah:
                meta += i.dataid + ',\n'
            meta = meta[:-2]+'}\n'

            ofile = os.path.join(odir, hfile[:-4]+'.hdr')
            export_gdal(ofile, datah, 'ENVI', envimeta=meta)

        return True

    def get_idir(self, dirname=''):
        """
        Get input directory.

        Parameters
        ----------
        filename : str, optional
            Directory name submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        if dirname == '':
            dirname = QtWidgets.QFileDialog.getExistingDirectory(
                    self.parent, 'Open File')
            if dirname == '':
                return

        os.chdir(dirname)

        self.idir.setText(dirname)

    def get_odir(self, dirname=''):
        """
        Get output directory.

        Parameters
        ----------
        filename : str, optional
            Directory name submitted for testing. The default is ''.

        Returns
        -------
        None.

        """

        if dirname == '':
            dirname = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Open File')
            if dirname == '':
                return

        os.chdir(dirname)
        self.odir.setText(dirname)


def data_to_dict(dat):
    """Data to dictionary."""
    dat2 = {}
    for j in dat:
        dat2[j.dataid] = j.data

    return dat2


def dict_to_data(arr, data):
    """Dictionary to data"""
    dat = []
    for key in arr:
        tmp = copy.copy(data[0])
        tmp.data = arr[key]
        tmp.dataid = key
        dat.append(tmp)

    return dat


def dist_point_to_segment(p, s0, s1):
    """
    Dist point to segment.

    Reimplementation of Matplotlib's dist_point_to_segment, after it was
    depreciated. Follows http://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    p : numpy array
        Point.
    s0 : numpy array
        Start of segment.
    s1 : numpy array
        End of segment.

    Returns
    -------
    numpy array
        Distance of point to segment.

    """
    p = np.array(p)
    s0 = np.array(s0)
    s1 = np.array(s1)

    v = s1 - s0
    w = p - s0

    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s0)

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s1)

    b = c1/c2
    pb = s0 + b*v

    return np.linalg.norm(p - pb)


def dc_correct(idir, hfile, datah, piter=iter, showprocesslog=print):
    """main."""

    # ofile = os.path.join(odir, hfile[:-4]+'.hdr')
    dfile = 'darkref_'+hfile
    wfile = 'whiteref_'+hfile

    dfile = os.path.join(idir, dfile)
    wfile = os.path.join(idir, wfile)

    datad = get_raster(dfile, piter=piter)
    dataw = get_raster(wfile, piter=piter)

    if datad is None or dataw is None:
        showprocesslog('Error! Could not import white or dark file')
        return datah

    dath = data_to_dict(datah)
    datd = data_to_dict(datad)
    datw = data_to_dict(dataw)

    dath2 = {}
    i = 0
    for key in piter(dath):

        if 'OWL' in hfile and float(key) > 12300:
            continue

        i += 1
        datdm = datd[key].mean(0)
        datwm = datw[key].mean(0)

        if 'RGB' in hfile:
            datdm = datdm * 0.

        tmp = (dath[key]-datdm)/(datwm-datdm)
        tmp = tmp.astype(np.float32)

        dath2[key] = tmp

    for i in piter(dath2):
        dath2[i] *= 10000
        dath2[i][dath2[i] < 1] = 1
        dath2[i][dath2[i] > 11000] = 11000
        # dath2[i][dath2[i] > 65535] = 65535
        dath2[i] = dath2[i].astype(np.uint16)

    datfin = dict_to_data(dath2, datah)

    # meta = 'reflectance scale factor = 10000\n'
    # meta += 'wavelength = {\n'
    # for i in datfin:
    #     meta += i.dataid + ',\n'
    # meta = meta[:-2]+'}\n'

    # iodefs.export_gdal(ofile, datfin, 'ENVI', envimeta=meta)
    return datfin


def smile(datah, piter=iter):
    """:)"""
    dath = data_to_dict(datah)

    x = np.array([20, 50, 100, 150, 200, 250, 300, 350])
    y = np.array([3, 7, 12, 14, 15, 14, 11, 5])

    func = si.interp1d(x, y, kind='cubic', fill_value="extrapolate")

    xnew = np.arange(384)
    ynew = func(xnew)

    ynew2 = ynew.astype(int)
    maxy = ynew2.max()

    for i in piter(dath):

        datcor = np.zeros_like(dath[i])
        for j, _ in enumerate(datcor[:-maxy]):
            datcor[j] = dath[i][j+ynew2, xnew]
        dath[i] = datcor

    datfin = dict_to_data(dath, datah)

    return datfin


@jit(nopython=True)
def hampel_filter(input_series, window_size, n_sigmas=3):
    """From https://towardsdatascience.com/outlier-detection-with-hampel-
       filter-85ddf523c73d."""
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []

    for i in range((window_size), (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):
                                                  (i + window_size)] - x0))
        if np.abs(input_series[i] - x0) > n_sigmas * S0:
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


def testfn():
    """Main testing routine."""
    # app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    # tmp = ImageCor()
    # tmp.get_idir(r'D:\Workdata\HyperspectralScanner\Raw Data')
    # tmp.get_odir(r'D:\Workdata\HyperspectralScanner\PTest')
    # tmp.settings()


    # ifile = r'D:\Workdata\HyperspectralScanner\Raw Data\LWIR(OWL)\bv1_17_118m16_125m79_2020-06-30_12-43-14\capture\BV1_17_118m16_125m79_2020-06-30_12-43-14.raw'
    # ifile = r'D:\Workdata\HyperspectralScanner\Raw Data\VNIR-SWIR (FENIX)\bv1_17_118m16_125m79_2020-06-30_12-43-14\capture\BV1_17_118m16_125m79_2020-06-30_12-43-14.raw'

    # data = get_raster(ifile, piter=pbar.iter)

    import matplotlib.pyplot as plt
    from pygmi.raster.datatypes import pygmi_to_numpy
    pbar = ProgressBarText()

    ifile = r'C:\Work\Workdata\HyperspectralScanner\PTest\FENIX\BV1_17_118m16_125m79_2020-06-30_12-43-14.dat'
    data = get_raster(ifile, piter=pbar.iter)

    pdat = []
    for i in data:
        if float(i.dataid) < 479.54:
            continue
        if float(i.dataid) > 2482.69:
            continue
        pdat.append(i.data)

    pdat = np.array(pdat)
    pdat = np.moveaxis(pdat, 0, -1)

    ifile = r'C:\Work\Workdata\HyperspectralScanner\Processed Data\FENIX L201 Data Preparation v0810\BV1_17_extracted_image.img'
    data = get_raster(ifile, piter=pbar.iter)

    tdat = []
    for i in data:
        tdat.append(i.data)

    tdat = np.array(tdat)
    tdat = np.moveaxis(tdat, 0, -1)

    xcrd = 130
    ycrd = 500


    plt.figure(dpi = 200)
    pdat = pdat[7:961, 72:337]
    plt.imshow(pdat[:, :, 60])
    plt.plot(xcrd, ycrd, 'k.')
    plt.show()

    plt.imshow(tdat[:, :, 60])
    plt.plot(xcrd, ycrd, 'k.')
    plt.show()

    plt.plot(tdat[ycrd, xcrd], label='Terra')
    plt.legend()
    plt.show()

    plt.plot(pdat[ycrd, xcrd], label='Raw')
    plt.legend()
    plt.show()


    from scipy.signal import savgol_filter
    from statsmodels.nonparametric.smoothers_lowess import lowess

    res = savgol_filter(pdat[ycrd, xcrd], 11, 3)
    # res, outliers = hampel_filter(pdat[ycrd, xcrd], 10, 2)
    plt.plot(res, label='SavGol')
    plt.legend()
    plt.show()


    xxx = np.arange(tdat[ycrd, xcrd].size)

    res = lowess(pdat[ycrd, xcrd], xxx, is_sorted=True, frac=0.015, it=0)
    res = res[:, 1]

    plt.plot(res, label='Lowess')
    plt.legend()
    plt.show()


    breakpoint()



if __name__ == "__main__":
    testfn()
