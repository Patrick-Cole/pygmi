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

import numpy as np
import scipy.interpolate as si
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.patches import Polygon as mPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from PyQt5 import QtWidgets, QtCore
import geopandas as gpd

from pygmi.misc import frm
import pygmi.menu_default as menu_default
from pygmi.raster.iodefs import get_raster


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


class BorePrep(QtWidgets.QDialog):
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


def dc_correct(idir, hfile, odir, pbar=None):
    """main."""

    # ofile = os.path.join(odir, hfile[:-4]+'.hdr')
    dfile = 'darkref_'+hfile
    wfile = 'whiteref_'+hfile

    hfile = os.path.join(idir, hfile)
    dfile = os.path.join(idir, dfile)
    wfile = os.path.join(idir, wfile)

    datah = get_raster(hfile, piter=pbar.iter)
    datad = get_raster(dfile, piter=pbar.iter)
    dataw = get_raster(wfile, piter=pbar.iter)

    if datah is None or datad is None or dataw is None:
        print('Error! Could not import')
        print(hfile)
        return

    dath = data_to_dict(datah)
    datd = data_to_dict(datad)
    datw = data_to_dict(dataw)

    dath2 = {}
    i = 0
    for key in pbar.iter(dath):

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

    for i in pbar.iter(dath2):
        if 'FENIX' in idir:
            dath2[i] = smile(dath2[i])
        dath2[i] *= 10000
        dath2[i][dath2[i] < 1] = 1
        dath2[i][dath2[i] > 11000] = 11000
        # dath2[i][dath2[i] > 65535] = 65535
        dath2[i] = dath2[i].astype(np.uint16)

    datfin = dict_to_data(dath2, datah)

    meta = 'reflectance scale factor = 10000\n'
    meta += 'wavelength = {\n'
    for i in datfin:
        meta += i.dataid + ',\n'
    meta = meta[:-2]+'}\n'

    # iodefs.export_gdal(ofile, datfin, 'ENVI', envimeta=meta)


def smile(dat2):
    """:)"""

    x = np.array([20, 50, 100, 150, 200, 250, 300, 350])
    y = np.array([3, 7, 12, 14, 15, 14, 11, 5])

    func = si.interp1d(x, y, kind='cubic', fill_value="extrapolate")

    xnew = np.arange(384)
    ynew = func(xnew)

    ynew2 = ynew.astype(int)
    maxy = ynew2.max()

    datcor = np.zeros_like(dat2)
    for i, _ in enumerate(datcor[:-maxy]):
        datcor[i] = dat2[i+ynew2, xnew]

    return datcor


def testfn():
    """Main testing routine."""
    from pygmi.misc import ProgressBarText
    pbar = ProgressBarText()

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    ifile = r'C:\Work\Workdata\HyperspectralScanner\CCUS\Processed\RGB\UC850_D1_44_843m20_858m40_2020-08-07_10-10-21.hdr'

    data = get_raster(ifile, piter=pbar.iter)

    tmp = BorePrep(None)
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    testfn()
