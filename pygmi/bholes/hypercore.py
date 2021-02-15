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
from scipy.signal import savgol_filter
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.patches import Polygon as mPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from numba import jit
from PyQt5 import QtWidgets, QtCore
import gdal

from pygmi.raster.dataprep import data_to_gdal_mem, gdal_to_dat
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
        self.mindx = 0
        self.csp = None
        self.subplot = None

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.data[self.mindx]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)
        # self.subplot.get_xaxis().set_visible(False)
        # self.subplot.get_yaxis().set_visible(False)

        self.csp = self.subplot.imshow(dat.data.T, cmap=cm.get_cmap('jet'))
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
        dat = self.data[self.mindx]

        self.csp.set_data(dat.data.T)
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

        # xtmp, ytmp = zip(*self.poly.xy)

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

        if self.ax.get_navigate_mode() is not None:
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
        if self.ax.get_navigate_mode() is not None:
            return

        self.set_line(event.xdata, event.ydata)

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

        dtmp = []
        for i in range(len(xys) - 1):
            dtmp.append(dist_point_to_segment(ptmp, xys[i], xys[i + 1]))

        dtmp = np.array(dtmp)
        imin = np.nonzero(dtmp < self.epsilon)[0]

        if imin.size == 0:
            return

        for i in imin:
            if i in [0, 2]:
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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.map = GraphMap(self)
        self.combo = QtWidgets.QComboBox()
        self.combostart = QtWidgets.QComboBox()
        self.comboend = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)
        self.asave = QtWidgets.QCheckBox('Auto Save (adds clip_ prefix)')

        self.setupui()

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
        pb_fenix = QtWidgets.QPushButton('Default FENIX settings')
        pb_owl = QtWidgets.QPushButton('Default OWL settings')

        self.setWindowTitle('Tray Clipping and Band Selection')

        lbl_combo = QtWidgets.QLabel('Display Band:')
        lbl_combostart = QtWidgets.QLabel('Start Band:')
        lbl_comboend = QtWidgets.QLabel('End Band:')

        grid_main.addWidget(lbl_combo, 0, 1, 1, 1)
        grid_main.addWidget(self.combo, 0, 2, 1, 1)

        grid_main.addWidget(lbl_combostart, 1, 1, 1, 1)
        grid_main.addWidget(self.combostart, 1, 2, 1, 1)

        grid_main.addWidget(lbl_comboend, 2, 1, 1, 1)
        grid_main.addWidget(self.comboend, 2, 2, 1, 1)

        grid_main.addWidget(pb_fenix, 3, 1, 1, 2)
        grid_main.addWidget(pb_owl, 4, 1, 1, 2)
        grid_main.addWidget(self.asave, 9, 1, 1, 2)

        grid_main.addWidget(self.map, 0, 0, 10, 1)
        grid_main.addWidget(self.mpl_toolbar, 11, 0, 1, 1)

        grid_main.addWidget(buttonbox, 12, 0, 1, 1, QtCore.Qt.AlignLeft)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_fenix.pressed.connect(self.default_fenix)
        pb_owl.pressed.connect(self.default_owl)

    def default_fenix(self):
        """
        Default settings for FENIX.

        Returns
        -------
        None.

        """

        self.combostart.setCurrentText('479.54')
        self.comboend.setCurrentText('2482.69')

        ymax, xmax = self.indata['Raster'][0].data.shape

        x1 = xmax*0.1
        x2 = xmax-xmax*0.1
        y1 = ymax*0.1
        y2 = ymax-ymax*0.1

        if xmax > 330:
            x1 = 75
            x2 = 330

        if ymax > 975:
            y1 = 20
            y2 = 975

        poly = [[y1, x1],
                [y1, x2],
                [y2, x2],
                [y2, x1]]

        self.map.polyi.new_poly(poly)
        self.map.update_graph()

    def default_owl(self):
        """
        Default settings for OWL.

        Returns
        -------
        None.

        """
        self.combostart.setCurrentText('7603.31')
        self.comboend.setCurrentText('11992.60')

        ymax, xmax = self.indata['Raster'][0].data.shape

        x1 = xmax*0.1
        x2 = xmax-xmax*0.1
        y1 = ymax*0.1
        y2 = ymax-ymax*0.1

        # if xmax > 330:
        #     x1 = 75
        #     x2 = 330

        # if ymax > 975:
        #     y1 = 20
        #     y2 = 975

        poly = [[y1, x1],
                [y1, x2],
                [y2, x2],
                [y2, x1]]

        self.map.polyi.new_poly(poly)
        self.map.update_graph()

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.map.mindx = self.combo.currentIndex()
        self.map.update_graph()

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
        idir = os.path.dirname(self.indata['Raster'][0].filename)

        bands = [i.dataid for i in self.indata['Raster']]

        self.combo.clear()
        self.combo.addItems(bands)
        self.combo.currentIndexChanged.connect(self.on_combo)

        self.combostart.clear()
        self.combostart.addItems(bands)

        self.comboend.clear()
        self.comboend.addItems(bands)
        self.comboend.setCurrentIndex(len(bands)-1)

        self.map.init_graph()

        if 'FENIX' in idir.upper():
            self.default_fenix()
        elif 'OWL' in idir.upper():
            self.default_owl()
        else:
            ymax, xmax = self.indata['Raster'][0].data.shape

            x1 = xmax*0.1
            x2 = xmax-xmax*0.1
            y1 = ymax*0.1
            y2 = ymax-ymax*0.1

            poly = [[y1, x1],
                    [y1, x2],
                    [y2, x2],
                    [y2, x1]]

            self.map.polyi.new_poly(poly)
            self.map.update_graph()

        tmp = self.exec_()

        if tmp == 0:
            return False

        self.acceptall()

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

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """

        data = copy.copy(self.indata['Raster'])
        xy = self.map.polyi.poly.xy

        xy = xy.astype(int)
        xy = np.abs(xy)

        rows = np.unique(xy[:, 0])
        cols = np.unique(xy[:, 1])

        for dat in data:
            dat.data = dat.data[rows.min():rows.max(),
                                cols.min():cols.max()]
        datfin = []
        start = False
        for i in data:
            if i.dataid == self.combostart.currentText():
                start = True
            if not start:
                continue
            if i.dataid == self.comboend.currentText():
                break
            datfin.append(i)

        if self.asave.isChecked():
            meta = 'reflectance scale factor = 10000\n'
            meta += 'wavelength = {\n'
            for i in datfin:
                meta += i.dataid + ',\n'
            meta = meta[:-2]+'}\n'

            odir = os.path.dirname(self.indata['Raster'][0].filename)
            hfile = os.path.basename(self.indata['Raster'][0].filename)
            ofile = os.path.join(odir, 'clip_'+hfile[:-4]+'.hdr')
            export_gdal(ofile, datfin, 'ENVI', envimeta=meta, piter=self.piter)

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
            self.piter = ProgressBarText().iter

        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.product = {}
        self.ratio = {}

        self.idir = QtWidgets.QLineEdit('')
        self.odir = QtWidgets.QLineEdit('')
        self.dccor = QtWidgets.QCheckBox('DC Correction')
        self.smilecor = QtWidgets.QCheckBox('Geometric Smile Correction '
                                            '(FENIX only)')
        self.hampel = QtWidgets.QCheckBox('Spike removal (Hampel Filter)')
        self.savgol = QtWidgets.QCheckBox('Smoothing (Savitzky-Golay filter)')

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
        gridlayout_main.addWidget(self.hampel, 4, 0, 1, 2)
        gridlayout_main.addWidget(self.savgol, 5, 0, 1, 2)

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

            datah = get_raster(ifile, piter=self.piter)

            if datah is None:
                self.showprocesslog('Could not open file.')
                continue

            if self.dccor.isChecked():
                datah = dc_correct(idir2, hfile, datah, piter=self.piter,
                                   showprocesslog=self.showprocesslog)
            if self.smilecor.isChecked() and 'FENIX' in idir2:
                datah = smile(datah, piter=self.piter)
            if self.hampel.isChecked():
                datah = filter_data(datah, 'hampel', piter=self.piter)
            if self.savgol.isChecked():
                datah = filter_data(datah, 'savgol', piter=self.piter)

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


def smile(dat, piter=iter):
    """:)"""
    x = np.array([20, 50, 100, 150, 200, 250, 300, 350])
    y = np.array([3, 7, 12, 14, 15, 14, 11, 5])

    func = si.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    col = np.arange(384, dtype=float)
    row = func(col)

    arows, _ = dat[0].data.shape

    gcps = []
    for j in np.linspace(0., arows-1, 3):
        for i, _ in enumerate(col):
            gcps.append(gdal.GCP(col[i], -j, 0., col[i], j+row[i]))

    dat2 = []
    for data in piter(dat):
        doffset = 0.0
        data.data.set_fill_value(data.nullvalue)
        data.data = np.ma.array(data.data.filled(), mask=data.data.mask)
        if data.data.min() <= 0:
            doffset = data.data.min()-1.
            data.data = data.data - doffset
        gtr0 = data.get_gtr()
        orig_wkt = data.wkt

        drows, dcols = data.data.shape
        src = data_to_gdal_mem(data, gtr0, orig_wkt, dcols, drows)

        src.SetGCPs(gcps, orig_wkt)

        dest = gdal.AutoCreateWarpedVRT(src)

        gdal.ReprojectImage(src, dest, None, None, gdal.GRA_Bilinear)

        dat2.append(gdal_to_dat(dest, data.dataid))
        dat2[-1].data = dat2[-1].data + doffset
        data.data = data.data + doffset

    return dat2


def filter_data(datah, ftype, piter=iter):
    """
    Filter data

    Parameters
    ----------
    datah : TYPE
        DESCRIPTION.
    ftype : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    print('Filtering using '+ftype+'...')

    pdat = []
    for i in datah:
        pdat.append(i.data)

    pdat = np.array(pdat)
    pdat = np.moveaxis(pdat, 0, -1)

    rows, cols, _ = pdat.shape

    for i in piter(range(rows)):
        for j in range(cols):
            if ftype == 'hampel':
                pdat[i, j], _ = hampel_filter(pdat[i, j], 5, 3)
            if ftype == 'savgol':
                pdat[i, j] = savgol_filter(pdat[i, j], 7, 2)

    for i, val in enumerate(datah):
        val.data = pdat[:, :, i]

    # res, outliers = hampel_filter(pdat[ycrd, xcrd], 10, 2)

    return datah


@jit(nopython=True)
def hampel_filter(input_series, window_size, n_sigmas=3):
    """From https://towardsdatascience.com/outlier-detection-with-hampel-
       filter-85ddf523c73d"""
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
    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = ImageCor()
    tmp.get_idir(r'c:\work\Workdata\HyperspectralScanner\Raw Data\VNIR-SWIR (FENIX)')
    tmp.get_odir(r'c:\work\Workdata\HyperspectralScanner\PTest\smile')
    tmp.settings()


def testfn2():
    """Main testing routine."""
    import matplotlib.pyplot as plt
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

    # ifile = r'C:\Work\Workdata\HyperspectralScanner\Processed Data\FENIX L201 Data Preparation v0810\BV1_17_extracted_image.img'
    # data = get_raster(ifile, piter=pbar.iter)

    # tdat = []
    # for i in data:
    #     tdat.append(i.data)

    # tdat = np.array(tdat)
    # tdat = np.moveaxis(tdat, 0, -1)

    xcrd = 130
    ycrd = 500

    plt.figure(dpi=200)
    pdat = pdat[7:961, 72:337]
    plt.imshow(pdat[:, :, 60])
    plt.plot(xcrd, ycrd, 'k.')
    plt.show()

    # plt.imshow(tdat[:, :, 60])
    # plt.plot(xcrd, ycrd, 'k.')
    # plt.show()

    # plt.plot(tdat[ycrd, xcrd], label='Terra')
    # plt.legend()
    # plt.show()

    plt.plot(pdat[ycrd, xcrd], label='Raw')
    plt.legend()
    plt.show()

    # res = savgol_filter(pdat[ycrd, xcrd], 7, 2)
    res, outliers = hampel_filter(pdat[ycrd, xcrd], 7, 3)
    plt.plot(res, 'k', label='SavGol')
    plt.vlines(outliers, 1250, 2750)
    plt.legend()
    plt.show()


def testfn3():
    """Main testing routine."""
    import matplotlib.pyplot as plt
    pbar = ProgressBarText()

    ifile = r'C:\Work\Workdata\HyperspectralScanner\PTest\smile\FENIX\BV1_17_118m16_125m79_2020-06-30_12-43-14.dat'
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

    ifile = r'D:\Workdata\HyperspectralScanner\PTest\nosmile\FENIX\BV1_17_118m16_125m79_2020-06-30_12-43-14.dat'
    data = get_raster(ifile, piter=pbar.iter)

    tdat = []
    for i in data:
        if float(i.dataid) < 479.54:
            continue
        if float(i.dataid) > 2482.69:
            continue
        tdat.append(i.data)

    tdat = np.array(tdat)
    tdat = np.moveaxis(tdat, 0, -1)

    pdat = pdat[7:100, 72:337]
    tdat = tdat[7:100, 72:337]

    plt.figure(dpi=200)
    plt.imshow(pdat[:, :, 60])
    plt.show()

    plt.figure(dpi=200)
    plt.imshow(tdat[:, :, 60])
    plt.show()


def testfn4():
    """Main testing routine."""
    import matplotlib.pyplot as plt

    ifile = r'c:\work\Workdata\HyperspectralScanner\PTest\smile\FENIX\BV1_17_118m16_125m79_2020-06-30_12-43-14.dat'

    pbar = ProgressBarText()
    data = get_raster(ifile, piter=pbar.iter)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = CorePrep()
    tmp.indata['Raster'] = data
    tmp.settings()

    # datfin = tmp.outdata['Raster']

    # plt.imshow(datfin[0].data)
    # plt.show()


if __name__ == "__main__":
    testfn4()

"""
SOM - borehole SOM (box only) and SOM of SOMs (whole borehole, no library
      matching, pure pixel spectra, unsupervised? 100 classes, some are
      combined).
pearson correlation.
Compare directly mean spectra? to library. Dominant mineral map.

"""
