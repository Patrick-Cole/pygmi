# -----------------------------------------------------------------------------
# Name:        hyperspec.py (part of PyGMI)
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
"""Hyperspectral Interpretation Routines."""
import sys
import re
import os

import numpy as np
import numexpr as ne
from numba import jit
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from pygmi.misc import frm, BasicModule
from pygmi import menu_default
from pygmi.raster.iodefs import get_raster
from pygmi.raster.datatypes import numpy_to_pygmi
from pygmi.raster.iodefs import export_raster
from pygmi.rsense import features
from pygmi.raster.modest_image import imshow


class GraphMap(FigureCanvasQTAgg):
    """
    Graph Map.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        self.figure = Figure(layout='constrained')

        super().__init__(self.figure)
        self.setParent(parent)
        self.rgb = True

        self.parent = parent
        self.datarr = []
        self.wvl = []
        self.mindx = 0
        self.csp = None
        self.format_coord = None
        self.feature = None
        self.row = 20
        self.col = 20
        self.remhull = False
        self.currentspectra = 'None'
        self.spectra = None
        self.refl = 1.
        self.rotate = False
        self.nodata = 0.
        self.ax1 = None

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.datarr[self.mindx].data

        if self.refl != 1.:
            dat = dat/self.refl

        rows, cols = dat.shape

        self.figure.clf()
        ax1 = self.figure.add_subplot(211)
        self.ax1 = ax1

        self.compute_initial_figure()

        # ymin = dat.mean()-2*dat.std()
        # ymax = dat.mean()+2*dat.std()

        # if self.rotate is True:
        #     self.csp = ax1.imshow(dat.T, vmin=ymin, vmax=ymax,
        #                           interpolation='none')
        #     rows, cols = cols, rows
        # else:
        #     self.csp = ax1.imshow(dat, vmin=ymin, vmax=ymax,
        #                           interpolation='none')

        # ax1.set_xlim((0, cols))
        # ax1.set_ylim((0, rows))
        # ax1.xaxis.set_visible(False)
        # ax1.yaxis.set_visible(False)

        # ax1.xaxis.set_major_formatter(frm)
        # ax1.yaxis.set_major_formatter(frm)

        # if self.rotate is True:
        #     ax1.plot(self.row, self.col, '+w')
        # else:
        #     ax1.plot(self.col, self.row, '+w')

        ax2 = self.figure.add_subplot(212)

        prof = [i.data[self.row, self.col] for i in self.datarr]

        prof = np.ma.stack(prof).filled(0)/self.refl

        ax2.format_coord = lambda x, y: f'Wavelength: {x:1.2f}, Y: {y:1.2f}'
        ax2.grid(True)
        ax2.set_xlabel('Wavelength')

        if self.remhull is True:
            hull = phull(prof)
            ax2.plot(self.wvl, prof/hull)
        else:
            ax2.plot(self.wvl, prof)

        ax2.axvline(self.feature[0], ls='--', c='r')

        ax2.xaxis.set_major_formatter(frm)
        ax2.yaxis.set_major_formatter(frm)

        if self.currentspectra != 'None':
            spec = self.spectra[self.currentspectra]
            prof2 = spec['refl']

            if self.remhull is True:
                hull = phull(prof2)
                ax2.plot(spec['wvl'], prof2/hull)
                ax2.set_ylim(top=1.01)
            else:
                ax2.plot(spec['wvl'], prof2)

        zmin, zmax = ax2.get_ylim()

        bmin = self.feature[1]
        bmax = self.feature[2]

        rect = mpatches.Rectangle((bmin, zmin), bmax-bmin, zmax-zmin)
        rect.set_facecolor([0, 1, 0])
        rect.set_alpha(0.5)
        ax2.add_patch(rect)

        self.figure.canvas.draw()

    def compute_initial_figure(self):
        """Compute initial figure."""
        clippercu = 1
        clippercl = 1
        dat = self.datarr

        redidx = (np.abs(self.wvl - 630)).argmin()
        greenidx = (np.abs(self.wvl - 532)).argmin()
        blueidx = (np.abs(self.wvl - 465)).argmin()

        if self.rgb is True:
            red = dat[redidx].data/self.refl
            green = dat[greenidx].data/self.refl
            blue = dat[blueidx].data/self.refl

            data = [red, green, blue]
            data = np.ma.array(data)
            data = np.moveaxis(data, 0, -1)
            lclip = [0, 0, 0]
            uclip = [0, 0, 0]

            lclip[0], uclip[0] = np.percentile(red.compressed(),
                                               [clippercl, 100-clippercu])
            lclip[1], uclip[1] = np.percentile(green.compressed(),
                                               [clippercl, 100-clippercu])
            lclip[2], uclip[2] = np.percentile(blue.compressed(),
                                               [clippercl, 100-clippercu])
        else:
            data = dat[self.mindx].data/self.refl
            lclip, uclip = np.percentile(data.compressed(),
                                         [clippercl, 100-clippercu])

        extent = dat[self.mindx].extent
        # breakpoint()
        self.im1 = imshow(self.ax1, data, extent=extent)
        self.im1.rgbmode = 'RGB Ternary'

        if self.rgb is True:
            self.im1.rgbclip = [[lclip[0], uclip[0]],
                                [lclip[1], uclip[1]],
                                [lclip[2], uclip[2]]]
        else:
            self.im1.set_clim(lclip, uclip)

        if dat[self.mindx].crs.is_geographic:
            self.ax1.set_xlabel('Longitude')
            self.ax1.set_ylabel('Latitude')
        else:
            self.ax1.set_xlabel('Eastings')
            self.ax1.set_ylabel('Northings')

        self.ax1.xaxis.set_major_formatter(frm)
        self.ax1.yaxis.set_major_formatter(frm)


class AnalSpec(BasicModule):
    """Analyse spectra."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.spectra = None
        self.feature = {}
        self.feature[900] = [776, 1050, 850, 910]
        self.feature[1300] = [1260, 1420]
        self.feature[1800] = [1740, 1820]
        self.feature[2080] = [2000, 2150]
        self.feature[2200] = [2120, 2245]
        self.feature[2290] = [2270, 2330]
        self.feature[2330] = [2120, 2370]

        self.map = GraphMap(self)
        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_feature = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)
        self.lbl_info = QtWidgets.QLabel('')
        self.gbox_info = QtWidgets.QGroupBox('Information:')
        self.cb_hull = QtWidgets.QCheckBox('Remove Hull')
        self.cb_rgb = QtWidgets.QCheckBox('True Colour Ternary')
        self.lw_speclib = QtWidgets.QListWidget()

        self.setupui()

        self.canvas = self.map.figure.canvas

        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        vbl_info = QtWidgets.QVBoxLayout(self.gbox_info)
        pb_speclib = QtWidgets.QPushButton('Load ENVI Spectral Library')
        self.cb_rgb.setChecked(True)

        self.lbl_info.setWordWrap(True)
        self.lw_speclib.addItem('None')
        self.setWindowTitle('Analyse Features')
        lbl_combo = QtWidgets.QLabel('Display Band:')
        lbl_feature = QtWidgets.QLabel('Feature:')

        vbl_info.addWidget(self.lbl_info)

        gl_main.addWidget(lbl_combo, 0, 1)
        gl_main.addWidget(self.cmb_1, 0, 2)
        gl_main.addWidget(lbl_feature, 1, 1)
        gl_main.addWidget(self.cmb_feature, 1, 2)
        gl_main.addWidget(self.cb_rgb, 2, 1)
        gl_main.addWidget(self.cb_hull, 2, 2)
        gl_main.addWidget(pb_speclib, 3, 1, 1, 2)
        gl_main.addWidget(self.lw_speclib, 4, 1, 1, 2)

        gl_main.addWidget(self.gbox_info, 5, 1, 8, 2)

        gl_main.addWidget(self.map, 0, 0, 10, 1)
        gl_main.addWidget(self.mpl_toolbar, 11, 0)

        gl_main.addWidget(buttonbox, 12, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.cmb_feature.currentIndexChanged.connect(self.feature_change)
        self.cb_hull.clicked.connect(self.hull)
        self.cb_rgb.clicked.connect(self.rotate_view)
        pb_speclib.clicked.connect(self.load_splib)
        self.lw_speclib.currentRowChanged.connect(self.disp_splib)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def button_press_callback(self, event):
        """
        Button press callback.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse Event.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        if event.inaxes != self.map.ax1:
            return

        ax = event.inaxes
        if ax.get_navigate_mode() is not None:
            return

        if ax != self.map.ax1:
            return

        self.map.row = int(event.ydata)
        self.map.col = int(event.xdata)

        dat = self.map.datarr[self.map.mindx]

        self.map.row = int((dat.extent[-1]-self.map.row)//dat.ydim)
        self.map.col = int((self.map.col-dat.extent[0])//dat.xdim)

        # if self.cb_rgb.isChecked():
        #     self.map.row, self.map.col = self.map.col, self.map.row

        self.map.init_graph()

    def disp_splib(self, row):
        """
        Change library spectra for display.

        Parameters
        ----------
        row : int
            row of table, unused.

        Returns
        -------
        None.

        """
        self.map.currentspectra = self.lw_speclib.currentItem().text()

        self.map.init_graph()

    def feature_change(self):
        """
        Change depth marker combo.

        Returns
        -------
        None.

        """
        txt = self.cmb_feature.currentText()

        self.map.feature = [int(txt)] + self.feature[int(txt)]

        self.map.init_graph()

    def hull(self):
        """
        Change whether hull is removed or not.

        Returns
        -------
        None.

        """
        self.map.remhull = self.cb_hull.isChecked()
        self.map.init_graph()

    def load_splib(self):
        """
        Load ENVI spectral library data.

        Returns
        -------
        None.

        """
        ext = 'ENVI Spectral Library (*.sli)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        self.spectra = readsli(filename)

        self.lw_speclib.disconnect()
        self.lw_speclib.clear()
        tmp = ['None'] + list(self.spectra.keys())
        self.lw_speclib.addItems(tmp)
        self.lw_speclib.currentRowChanged.connect(self.disp_splib)

        self.map.spectra = self.spectra

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.map.mindx = self.cmb_1.currentIndex()
        self.map.init_graph()

    def rotate_view(self):
        """
        Rotates view.

        Returns
        -------
        None.

        """
        self.map.rgb = self.cb_rgb.isChecked()
        self.map.init_graph()

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
        if 'Raster' not in self.indata:
            self.showlog('Error: You must have a multi-band raster '
                         'dataset in addition to your cluster '
                         'analysis results')
            return False

        if 'wavelength' not in self.indata['Raster'][0].metadata['Raster']:
            self.showlog('Error: Your data should have wavelengths in'
                         ' the metadata')
            return False

        dat = self.indata['Raster']

        needsmerge = False
        rows, cols = dat[0].data.shape

        for i in dat:
            irows, icols = i.data.shape
            if irows != rows or icols != cols:
                needsmerge = True

        if needsmerge is True:
            self.showlog('Error: Your data bands have different sizes. '
                         'Use Layer Stack to fix this first')
            return False

        wavelengths = []
        dat2 = []
        for i in dat:
            if 'wavelength' in i.metadata['Raster']:
                wavelengths.append(i.metadata['Raster']['wavelength'])
                dat2.append(i)

        dat = [i for _, i in sorted(zip(wavelengths, dat2))]

        if 'reflectance_scale_factor' in dat[0].metadata['Raster']:
            self.map.refl = float(
                dat[0].metadata['Raster']['reflectance_scale_factor'])

        # dat2 = []
        wvl = []
        for j in dat:
            # dat2.append(j.data)
            wvl.append(float(j.metadata['Raster']['wavelength']))

        dat2 = np.ma.array(dat2)

        # self.map.datarr = dat2
        self.map.datarr = dat
        self.map.nodata = dat[0].nodata
        self.map.wvl = np.array(wvl)
        if self.map.wvl.max() < 20:
            self.map.wvl = self.map.wvl*1000.
            self.showlog('Wavelengths appear to be in nanometers. '
                         'Converting to micrometers.')

        bands = [i.dataid for i in self.indata['Raster']]

        self.cmb_1.clear()
        self.cmb_1.addItems(bands)
        self.cmb_1.currentIndexChanged.connect(self.on_combo)

        ftxt = [str(i) for i in self.feature]
        self.cmb_feature.disconnect()
        self.cmb_feature.clear()
        self.cmb_feature.addItems(ftxt)
        self.feature_change()
        self.cmb_feature.currentIndexChanged.connect(self.feature_change)

        tmp = self.exec()

        if tmp == 0:
            return False

        self.outdata['Raster'] = self.indata['Raster']

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """


class ProcFeatures(BasicModule):
    """Process Hyperspectral Features."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.product = {}
        self.ratio = {}
        self.feature = None

        self.cmb_ratios = QtWidgets.QComboBox()
        self.cb_rfiltcheck = QtWidgets.QCheckBox('If the final product is a '
                                                 'ratio, filter out values '
                                                 'less than 1.')
        self.cb_filtercheck = QtWidgets.QCheckBox('Filter Albedo and '
                                                  'Vegetation')
        self.tablewidget = QtWidgets.QTableWidget()

        self.setupui()

        self.resize(500, 350)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.pfeat')
        lbl_ratios = QtWidgets.QLabel('Product:')
        lbl_details = QtWidgets.QLabel('Details:')

        self.tablewidget.setRowCount(2)
        self.tablewidget.setColumnCount(3)
        self.tablewidget.setHorizontalHeaderLabels(['Feature', 'Filter',
                                                    'Threshold'])
        self.tablewidget.resizeColumnsToContents()
        self.cb_filtercheck.setChecked(True)
        self.cb_rfiltcheck.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Process Hyperspectral Features')

        gl_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_ratios, 1, 1, 1, 1)
        gl_main.addWidget(lbl_details, 2, 0, 1, 1)
        gl_main.addWidget(self.tablewidget, 2, 1, 1, 1)
        gl_main.addWidget(self.cb_filtercheck, 3, 0, 1, 2)
        gl_main.addWidget(self.cb_rfiltcheck, 4, 0, 1, 2)

        gl_main.addWidget(helpdocs, 6, 0, 1, 1)
        gl_main.addWidget(buttonbox, 6, 1, 1, 3)

        self.cmb_ratios.currentIndexChanged.connect(self.product_change)
        self.cb_filtercheck.stateChanged.connect(self.product_change)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def product_change(self):
        """
        Change product combo.

        Returns
        -------
        None.

        """
        txt = self.cmb_ratios.currentText()
        self.tablewidget.clear()

        product = self.product[txt]

        if self.cb_filtercheck.isChecked():
            product = product + self.product['filter']

        numrows = len(product)

        self.tablewidget.setRowCount(numrows)
        self.tablewidget.setColumnCount(4)
        self.tablewidget.setHorizontalHeaderLabels(['Feature', 'Filter',
                                                    'Threshold',
                                                    'Description'])

        item = QtWidgets.QTableWidgetItem(str(product[0]))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.tablewidget.setItem(0, 0, item)

        item = QtWidgets.QTableWidgetItem('None')
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.tablewidget.setItem(0, 1, item)

        item = QtWidgets.QTableWidgetItem('None')
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.tablewidget.setItem(0, 2, item)

        if product[0] in self.feature:
            desc = 'Feature Depth'
        elif product[0] in self.ratio:
            desc = self.ratio[product[0]]
        else:
            desc = 'None'

        item = QtWidgets.QTableWidgetItem(desc)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.tablewidget.setItem(0, 3, item)

        for i in range(1, numrows):
            cmb_1 = QtWidgets.QComboBox()
            cmb_1.addItems(['<', '>'])
            self.tablewidget.setCellWidget(i, 1, cmb_1)

            txt2 = str(product[i])
            txt2 = txt2.split()

            cmb_1.setCurrentText(txt2[1])
            item = QtWidgets.QTableWidgetItem(txt2[0])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tablewidget.setItem(i, 0, item)
            item = QtWidgets.QTableWidgetItem(txt2[2])
            self.tablewidget.setItem(i, 2, item)

            if txt2[0] in self.ratio:
                desc = self.ratio[txt2[0]]
            else:
                desc = 'Feature between ' + str(self.feature[txt2[0]])
            item = QtWidgets.QTableWidgetItem(desc)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tablewidget.setItem(i, 3, item)

        self.tablewidget.resizeColumnsToContents()

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
        if 'Raster' not in self.indata and 'RasterFileList' not in self.indata:
            self.showlog('No Satellite Data')
            return False

        self.feature = features.feature
        self.ratio = features.ratio

        self.cmb_ratios.disconnect()
        self.product = features.product.copy()

        self.product = dict(sorted(self.product.items()))

        del self.product['filter']
        self.cmb_ratios.clear()
        self.cmb_ratios.addItems(self.product)

        # The filter line is added after the other products so that it does
        # not make it into the list widget
        self.product['filter'] = features.product['filter']
        self.cmb_ratios.currentIndexChanged.connect(self.product_change)
        self.product_change()

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
        self.saveobj(self.cmb_ratios)
        self.saveobj(self.cb_rfiltcheck)
        self.saveobj(self.cb_filtercheck)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        datfin = []

        mineral = self.cmb_ratios.currentText()
        rfilt = self.cb_rfiltcheck.isChecked()

        product = self.product

        try:
            product = [int(self.tablewidget.item(0, 0).text())]
        except ValueError:
            product = [self.tablewidget.item(0, 0).text()]
        for i in range(1, self.tablewidget.rowCount()):
            product.append(self.tablewidget.item(i, 0).text())
            product[-1] += (' ' +
                            self.tablewidget.cellWidget(i, 1).currentText())
            product[-1] += ' ' + self.tablewidget.item(i, 2).text()

        if 'RasterFileList' in self.indata:
            flist = self.indata['RasterFileList']
            odir = os.path.join(os.path.dirname(flist[0]), 'feature')

            os.makedirs(odir, exist_ok=True)
            for ifile in flist:
                self.showlog('Processing '+os.path.basename(ifile))

                dat = get_raster(ifile)
                datfin = calcfeatures(dat, mineral, self.feature, self.ratio,
                                      product, rfilt, piter=self.piter)

                ofile = (os.path.basename(ifile).split('.')[0] + '_' +
                         mineral.replace(' ', '_') + '.tif')
                ofile = os.path.join(odir, ofile)
                if datfin[0].data.mask.min() == True:
                    self.showlog(' Could not find any ' + mineral +
                                 '. No data to export.')
                else:
                    self.showlog('Exporting '+os.path.basename(ofile))
                    export_raster(ofile, datfin, 'GTiff', piter=self.piter,
                                  showlog=self.showlog)

        elif 'Raster' in self.indata:
            dat = self.indata['Raster']
            datfin = calcfeatures(dat, mineral, self.feature, self.ratio,
                                  product, rfilt, piter=self.piter)

        if datfin[0].data.mask.min() == True:
            QtWidgets.QMessageBox.warning(self.parent, 'Warning',
                                          ' Could not find any ' + mineral +
                                          '. No data to export.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        self.outdata['Raster'] = datfin
        return True


def calcfeatures(dat, mineral, feature, ratio, product, rfilt=True,
                 piter=iter):
    """
    Calculate feature dataset.

    Parameters
    ----------
    dat : list of PyGMI Data
        Input PyGMI data.
    mineral : str
        Mineral description.
    feature : dictionary
        Dictionary containing the hyperspectral features.
    ratio : dictionary
        Dictionary containing string definitions of ratios.
    product : dictionary
        Final hyperspectral products. Each dictionary value, is a list of
        features or ratios with thresholds to be combined.
    rfilt : bool
        Flag to decide whether to filter final ratio products less than 1.0
    piter : function, optional
        Progress bar iterable. The default is iter.

    Returns
    -------
    datfin : list of PyGMI Data.
        Output datasets.

    """
    # allfeatures = [i.split()[0] for i in product if i[0] == 'f']
    # allratios = [i.split()[0] for i in product if i[0] != 'f']

    allfeatures = []
    for f in feature:
        for p in product:
            if f in p and f not in allfeatures:
                allfeatures.append(f)

    allratios = []
    for r in ratio:
        for p in product:
            if r in p and r not in allratios:
                allratios.append(r)

    # Get list of wavelengths and data
    dat2 = []
    xval = []
    for j in piter(dat):
        dat2.append(j.data)
        refl = float(re.findall(r'[\d\.\d]+', j.dataid)[-1])
        if refl < 100.:
            refl = refl * 1000
        refl = round(refl, 2)
        xval.append(refl)

    xval = np.array(xval)
    dat2 = np.ma.array(dat2)  # This line is very slow.

    # This gets nearest wavelength and assigns to R number.
    # It does not interpolate.
    RBands = {}
    for j in range(1, 2501):
        i = abs(xval-j).argmin()
        RBands['R'+str(j)] = dat2[i]

    datfin = []
    # Calclate ratios
    datcalc = {}
    for j in piter(allratios):
        if j in datcalc:
            continue
        tmp = indexcalc(ratio[j], RBands)
        datcalc[j] = tmp

    # Start processing
    depths = {}
    wvl = {}
    dmax = {}
    for fname in allfeatures:
        if len(feature[fname]) == 4:
            fmin, fmax, lmin, lmax = feature[fname]
        else:
            fmin, fmax = feature[fname]
            lmin, lmax = fmin, fmax

        # get index of closest wavelength
        i1 = abs(xval-fmin).argmin()
        i2 = abs(xval-fmax).argmin()

        fdat = dat2[i1:i2+1]
        xdat = xval[i1:i2+1]

        # Raster calculation
        _, rows, cols = dat2.shape
        dtmp = np.zeros((rows, cols))
        ptmp = np.zeros((rows, cols))
        mtmp = np.zeros((rows, cols))

        tmp = np.nonzero((xdat > lmin) & (xdat < lmax))[0]
        i1a = tmp[0]
        i2a = tmp[-1]

        fdat = np.moveaxis(fdat, 0, -1)

        for i in piter(range(rows)):
            ptmp[i], dtmp[i], mtmp[i] = fproc(fdat[i].data, ptmp[i], dtmp[i], i1a, i2a,
                                              xdat, mtmp[i])
        depths[fname] = dtmp
        wvl[fname] = ptmp
        datcalc[fname] = dtmp
        dmax[fname] = mtmp

    datout = None
    datout2 = None
    tmpw = None

    for i in product:
        if ('>' in i or '<' in i or '=' in i) and i.count('f') > 1:
            dattmp = {}
            for j in datcalc:
                if j in i:
                    dattmp[j] = datcalc[j]*dmax[j]
            tmp = ne.evaluate(i, dattmp)
            # breakpoint()
        elif '>' in i or '<' in i or '=' in i or i[0] == 'r':
            tmp = ne.evaluate(i, datcalc)
        else:
            tmp = depths[i]
            tmpw = wvl[i]

        if datout is None:
            datout = np.nan_to_num(tmp)
            datout2 = np.nan_to_num(tmpw)
        else:
            if tmp.max() > 1:
                print('Problem with filter. Max value greater that 1')
                return datfin
            datout = datout * np.nan_to_num(tmp)
            if datout2 is not None:
                datout2 = datout2 * np.nan_to_num(tmp)

    if product[0][0] == 'f':
        label = f'{mineral} depth'
    else:
        label = f'{mineral} ratio'
        if rfilt is True:
            datout[datout < 1] = 0

    datout = np.ma.masked_equal(datout, 0)
    datfin.append(numpy_to_pygmi(datout, dat[0], label))

    if datout2 is not None:
        datout2 = np.ma.masked_equal(datout2, 0)
        datfin.append(numpy_to_pygmi(datout2, dat[0], f'{mineral} wvl'))

    return datfin


def indexcalc(formula, dat):
    """
    Calculate an index using numexpr.

    Parameters
    ----------
    formula : str
        string expression containing index formula.
    dat : dict
        Dictionary of variables to be used in calculation.

    Returns
    -------
    out : numpy array
        This can be a masked array.

    """
    out = ne.evaluate(formula, dat)

    key = list(dat.keys())[0]

    if np.ma.isMaskedArray(dat[key]):
        mask = dat[key].mask
        out = np.ma.array(out, mask=mask)

    return out


@jit(nopython=True)
def fproc(fdat, ptmp, dtmp, i1a, i2a, xdat, mtmp):
    """
    Feature process.

    This function finds the minimum value of a feature.

    Parameters
    ----------
    fdat : numpy array
        Feature data
    ptmp : numpy array
        Feature wavelengths.
    dtmp : numpy array
        Feature depths.
    i1a : int
        Start index of feature definition.
    i2a : int
        End Index of feature definition.
    xdat : numpy array
        Wavelengths of feature definition.

    Returns
    -------
    ptmp : numpy array
        Feature wavelengths.
    dtmp : numpy array
        Feature depths.

    """
    cols, _ = fdat.shape

    for j in range(cols):
        yval = fdat[j]
        if yval.mean() == 0:
            continue

        yhull = phull(yval)
        crem = yval/yhull
        mtmp[j] = -(yval - yhull).min()

        imin = crem[i1a:i2a].argmin()

        if imin == 0 or imin == (i2a-i1a-1):
            dtmp[j] = 1. - crem[i1a:i2a][imin]
            ptmp[j] = xdat[i1a:i2a][imin]
            continue

        x, y = cubic_calc(xdat[i1a:i2a], crem[i1a:i2a], imin)

        ptmp[j] = x
        dtmp[j] = 1. - y

    return ptmp, dtmp, mtmp


@jit(nopython=True)
def cubic_calc(xdat, crem, imin):
    """
    Find minimum of function using an analytic cubic calculation for speed.

    Parameters
    ----------
    xdat : numpy array
        wavelengths - x data.
    crem : numpy array
        continuum removed data - y data.
    imin : int
        Index for estimated minimum.

    Returns
    -------
    x : float
        wavelength at minimum.
    y : float
        y value at minimum.

    """
    x1 = xdat[imin-1]
    x2 = xdat[imin]
    x3 = xdat[imin+1]

    y1 = crem[imin-1]
    y2 = crem[imin]
    y3 = crem[imin+1]

    x = x2
    y = y2

    a1 = (2*x1**3*x2*y3 - 2*x1**3*x3*y2 - x1**2*x2**2*y2 - 3*x1**2*x2**2*y3 +
          2*x1**2*x2*x3*y2 + 2*x1**2*x3**2*y2 + x1*x2**3*y1 + x1*x2**3*y3 +
          x1*x2**2*x3*y1 + x1*x2**2*x3*y2 - 2*x1*x2*x3**2*y1 -
          2*x1*x2*x3**2*y2 - 2*x2**3*x3*y1 +
          2*x2**2*x3**2*y1)/(2*(x1 - x2)**2*(x1 - x3)*(x2 - x3))
    b1 = (2*x1**3*y2 - 2*x1**3*y3 - 4*x1*x2**2*y1 + x1*x2**2*y2 +
          3*x1*x2**2*y3 + 2*x1*x2*x3*y1 - 2*x1*x2*x3*y2 + 2*x1*x3**2*y1 -
          2*x1*x3**2*y2 + x2**3*y1 - x2**3*y3 + x2**2*x3*y1 - x2**2*x3*y2 -
          2*x2*x3**2*y1 + 2*x2*x3**2*y2)/(2*(x1 - x2)**2*(x1 - x3)*(x2 - x3))
    c1 = -3*x1*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 -
                x3*y2)/(2*(x1 - x2)**2*(x1 - x3)*(x2 - x3))
    d1 = (x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 -
          x3*y2)/(2*(x1 - x2)**2*(x1 - x3)*(x2 - x3))

    a2 = (2*x1**2*x2**2*y3 - 2*x1**2*x2*x3*y2 - 2*x1**2*x2*x3*y3 +
          2*x1**2*x3**2*y2 - 2*x1*x2**3*y3 + x1*x2**2*x3*y2 + x1*x2**2*x3*y3 +
          2*x1*x2*x3**2*y2 - 2*x1*x3**3*y2 + x2**3*x3*y1 + x2**3*x3*y3 -
          3*x2**2*x3**2*y1 - x2**2*x3**2*y2 +
          2*x2*x3**3*y1)/(2*(x1 - x2)*(x1 - x3)*(x2 - x3)**2)
    b2 = (2*x1**2*x2*y2 - 2*x1**2*x2*y3 - 2*x1**2*x3*y2 + 2*x1**2*x3*y3 -
          x1*x2**2*y2 + x1*x2**2*y3 - 2*x1*x2*x3*y2 + 2*x1*x2*x3*y3 -
          x2**3*y1 + x2**3*y3 + 3*x2**2*x3*y1 + x2**2*x3*y2 - 4*x2**2*x3*y3 -
          2*x3**3*y1 + 2*x3**3*y2)/(2*(x1 - x2)*(x1 - x3)*(x2 - x3)**2)
    c2 = 3*x3*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 -
               x3*y2)/(2*(x1 - x2)*(x1 - x3)*(x2 - x3)**2)
    d2 = -(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 -
           x3*y2)/(2*(x1 - x2)*(x1 - x3)*(x2 - x3)**2)

    if abs(d1) > 2.22e+16:
        min1 = [(-c1 + np.sqrt(-3*b1*d1 + c1**2))/(3*d1),
                -(c1 + np.sqrt(-3*b1*d1 + c1**2))/(3*d1)]
        for i in min1:
            if x1 < i < x2:
                x = i
                y = a1+b1*x+c1*x**2+d1*x**3

    if abs(d2) > 2.22e+16:
        min2 = [(-c2 + np.sqrt(-3*b2*d2 + c2**2))/(3*d2),
                -(c2 + np.sqrt(-3*b2*d2 + c2**2))/(3*d2)]

        for i in min2:
            if x2 < i < x3:
                x = i
                y = a2+b2*x+c2*x**2+d2*x**3

    return x, y


@jit(nopython=True)
def phull(sample1):
    """
    Hull Calculation.

    Parameters
    ----------
    sample1 : numpy array
        Sample to create a hull for.

    Returns
    -------
    out : numpy array
        Output hull.

    """
    xvals = np.arange(sample1.size, dtype=np.int64)
    sample = np.empty((sample1.size, 2))
    sample[:, 0] = xvals
    sample[:, 1] = sample1

    edge = sample[:1].copy()
    rest = sample[1:]

    hull = [0]
    while len(rest) > 0:
        grad = rest - edge
        grad = grad[:, 1]/grad[:, 0]
        pivot = np.argmax(grad)
        edge[0, 0] = rest[pivot, 0]
        edge[0, 1] = rest[pivot, 1]
        rest = rest[pivot+1:]
        hull.append(pivot)

    hull = np.array(hull) + 1
    hull = hull.cumsum()-1

    take = np.take(sample[:, 1], hull)

    out = np.interp(xvals, hull, take)

    return out


def readsli(ifile):
    """
    Read an ENVI sli file.

    Parameters
    ----------
    ifile : str
        Input sli spectra file.

    Returns
    -------
    spectra : dictionary
        Dictionary of spectra with wavelengths and reflectances.
    """
    with open(ifile[:-4]+'.hdr', encoding='utf-8') as file:
        hdr = file.read()

    hdr = hdr.split('\n')

    hdr2 = []
    i = -1
    brackets = False
    while hdr:
        tmp = hdr.pop(0)
        if not brackets:
            hdr2.append(tmp)
        else:
            hdr2[-1] += tmp
        if '{' in tmp:
            brackets = True
        if '}' in tmp:
            brackets = False

    hdr3 = {}
    for i in hdr2:
        tmp = i.split('=')
        if len(tmp) > 1:
            hdr3[tmp[0].strip()] = tmp[1].strip()

    for i in hdr3:
        if i in ['samples', 'lines', 'bands', 'header offset', 'data type']:
            hdr3[i] = int(hdr3[i])
            continue
        if i in ['reflectance scale factor']:
            hdr3[i] = float(hdr3[i])
            continue
        if '{' in hdr3[i]:
            hdr3[i] = hdr3[i].replace('{', '')
            hdr3[i] = hdr3[i].replace('}', '')
            hdr3[i] = hdr3[i].split(',')
            hdr3[i] = [j.strip() for j in hdr3[i]]
            if i in ['wavelength', 'z plot range']:
                hdr3[i] = [float(j) for j in hdr3[i]]

    if hdr3['bands'] > 1:
        print('More than one band in sli file. Cannot import')
        return None

    dtype = hdr3['data type']
    dt2np = {}
    dt2np[1] = np.uint8
    dt2np[2] = np.int16
    dt2np[3] = np.int32
    dt2np[4] = np.float32
    dt2np[5] = np.float64
    dt2np[6] = np.complex64
    dt2np[9] = np.complex128
    dt2np[12] = np.uint16
    dt2np[13] = np.uint32
    dt2np[14] = np.int64
    dt2np[15] = np.uint64

    data = np.fromfile(ifile, dtype=dt2np[dtype])
    data[data == -1.23e+34] = np.nan
    data = data / hdr3['reflectance scale factor']

    data.shape = (hdr3['lines'], hdr3['samples'])

    spectra = {}
    wmult = 1.

    if hdr3['wavelength units'].lower() == 'micrometers':
        wmult = 1000.

    hdr3['wavelength'] = np.array(hdr3['wavelength'])*wmult

    for i, val in enumerate(hdr3['spectra names']):
        spectra[val] = {'wvl': hdr3['wavelength'],
                        'refl': data[i]}

    # breakpoint()
    return spectra


def _testfn():
    """Test routine."""
    from pygmi.rsense.iodefs import get_data

    app = QtWidgets.QApplication(sys.argv)

    # ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\hyperspectral\071_0818-0932_ref_rect_BSQ.hdr"
    # ifile = r"D:\cut_048-055_ref_rect_DEFLATE.tif"
    ifile = r"D:\Cu-hyperspec-testarea.tif"

    data = get_data(ifile)

    tmp = ProcFeatures(None)
    tmp.indata['Raster'] = data
    tmp.settings()

    dat = tmp.outdata['Raster'][0]

    plt.figure(dpi=150)
    plt.imshow(dat.data, extent=dat.extent)
    plt.colorbar()
    plt.show()

    # print(dat.data.mean())

    # plt.figure(dpi=150)
    # plt.hist(dat.data.flatten(), bins=200)
    # plt.show()

    # tmp = np.histogram(dat.data[dat.data > 0])

    # breakpoint()


def _testfn2():
    """Test routine."""
    from pygmi.rsense.iodefs import get_data
    from pygmi.raster.dataprep import lstack

    # ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\hyperspectral\071_0818-0932_ref_rect_BSQ.hdr"
    ifile = r"D:\Cu-hyperspec-testarea.tif"

    data = get_data(ifile)

    # data = lstack(data)

    app = QtWidgets.QApplication(sys.argv)
    tmp = AnalSpec()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    _testfn2()
