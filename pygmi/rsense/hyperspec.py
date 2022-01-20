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
"""
Hyperspectral Interpretation Routines.

1) Spectral Feature Examination
2) Spectral Interpretation and Processing
3) Borehole display and interpretation

Spectral feature examination is a GUI which allows for the comparison of
spectra from the dataset with library spectra.

Must be able to:

1) Zoom into features
2) Select features and apply settings such as threshold if necessary
3) Combine multiple features into final filtered result.
4) Save spectra into a library for feature examination now or later.
5) Features can be depths, widths, or ratios
6) Output successful feature combinations into formulae for processing.

Note: feature or ratio x unit is wavelength and not band number.

Spectral Interpretation and processing allows for processing of a scene.
1) Must have a list of features or filters and thresholds
2) Must have a 'quick name' dropdown for premade combinations.
3) Must be able to save a combination
4) Must be able to import combinations

There should be a library file for features or filters
There should be a combo file for combinations.

Borehole display and interpretation shows a list of trays and can display core
as one borehole. It can also allow for manual interpretation as well as
comparison between existing logs and the core images.

"""
import json
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
# from scipy.optimize import minimize
# from scipy.interpolate import CubicSpline

from pygmi.misc import frm
import pygmi.menu_default as menu_default
from pygmi.raster.iodefs import get_raster
from pygmi.misc import ProgressBarText
from pygmi.raster.datatypes import numpy_to_pygmi
from pygmi.raster.iodefs import export_raster
# from pygmi.raster.modest_image import imshow
import pygmi.rsense.features as features


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

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.datarr[self.mindx]/self.refl
        # dat = np.ma.masked_equal(dat, self.nodata)

        rows, cols = dat.shape

        self.figure.clf()
        ax1 = self.figure.add_subplot(211)

        ymin = dat.mean()-2*dat.std()
        ymax = dat.mean()+2*dat.std()

        # if self.rotate is True:
        #     self.csp = imshow(ax1, dat.T, vmin=ymin, vmax=ymax)
        #     rows, cols = cols, rows
        # else:
        #     self.csp = imshow(ax1, dat, vmin=ymin, vmax=ymax)

        if self.rotate is True:
            self.csp = ax1.imshow(dat.T, vmin=ymin, vmax=ymax)
            rows, cols = cols, rows
        else:
            self.csp = ax1.imshow(dat, vmin=ymin, vmax=ymax)

        ax1.set_xlim((0, cols))
        ax1.set_ylim((0, rows))
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)

        ax1.xaxis.set_major_formatter(frm)
        ax1.yaxis.set_major_formatter(frm)

        ax1.plot(self.col, self.row, '+w')

        ax2 = self.figure.add_subplot(212)
        prof = self.datarr[:, self.row, self.col]/self.refl

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
        rect.set_fc([0, 1, 0])
        rect.set_alpha(0.5)
        ax2.add_patch(rect)

        self.figure.tight_layout()
        self.figure.canvas.draw()


class AnalSpec(QtWidgets.QDialog):
    """
    Analyse spectra.

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
        self.combo = QtWidgets.QComboBox()
        self.combo_feature = QtWidgets.QComboBox()
        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)
        self.dsb_mdepth = QtWidgets.QDoubleSpinBox()
        self.pb_save = QtWidgets.QPushButton('Save')
        self.lbl_info = QtWidgets.QLabel('')
        self.group_info = QtWidgets.QGroupBox('Information:')
        self.chk_hull = QtWidgets.QCheckBox('Remove Hull')
        self.chk_rot = QtWidgets.QCheckBox('Rotate View')
        self.lw_speclib = QtWidgets.QListWidget()

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
        pb_speclib = QtWidgets.QPushButton('Load ENVI Spectral Library')

        self.lbl_info.setWordWrap(True)
        self.lw_speclib.addItem('None')
        self.setWindowTitle('Analyse Features')
        lbl_combo = QtWidgets.QLabel('Display Band:')
        lbl_feature = QtWidgets.QLabel('Feature:')

        infolayout.addWidget(self.lbl_info)

        grid_main.addWidget(lbl_combo, 0, 1)
        grid_main.addWidget(self.combo, 0, 2)
        grid_main.addWidget(lbl_feature, 1, 1)
        grid_main.addWidget(self.combo_feature, 1, 2)
        grid_main.addWidget(self.chk_rot, 2, 1)
        grid_main.addWidget(self.chk_hull, 2, 2)
        grid_main.addWidget(pb_speclib, 3, 1, 1, 2)
        grid_main.addWidget(self.lw_speclib, 4, 1, 1, 2)

        grid_main.addWidget(self.group_info, 5, 1, 8, 2)
        # grid_main.addWidget(self.pb_save, 9, 1, 1, 2)

        grid_main.addWidget(self.map, 0, 0, 10, 1)
        grid_main.addWidget(self.mpl_toolbar, 11, 0)

        grid_main.addWidget(buttonbox, 12, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.combo_feature.currentIndexChanged.connect(self.feature_change)
        self.chk_hull.clicked.connect(self.hull)
        self.chk_rot.clicked.connect(self.rotate_view)
        pb_speclib.clicked.connect(self.load_splib)
        self.lw_speclib.currentRowChanged.connect(self.disp_splib)

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

        self.map.row = int(event.ydata)
        self.map.col = int(event.xdata)

        if self.chk_rot.isChecked():
            self.map.row, self.map.col = self.map.col, self.map.row

        self.map.init_graph()

    def disp_splib(self, row):
        """
        Change library spectra for display.

        Parameters
        ----------
        row : TYPE
            Unused.

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
        txt = self.combo_feature.currentText()

        self.map.feature = [int(txt)] + self.feature[int(txt)]

        self.map.init_graph()

    def hull(self):
        """
        Change whether hull is removed or not.

        Returns
        -------
        None.

        """
        self.map.remhull = self.chk_hull.isChecked()
        self.map.init_graph()

    def load_splib(self):
        """
        Load ENVI spectral library data.

        Returns
        -------
        None.

        """
        ext = ('ENVI Spectral Library (*.sli)')

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
        self.map.mindx = self.combo.currentIndex()
        # self.map.currentmark = self.combo_dmark.currentText()
        self.map.init_graph()

    def rotate_view(self):
        """
        Rotates view.

        Returns
        -------
        None.

        """
        self.map.rotate = self.chk_rot.isChecked()
        self.map.init_graph()

    def save(self):
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
            self.showprocesslog('Error: You must have a multi-band raster '
                                'dataset in addition to your cluster '
                                'analysis results')
            return False

        if 'wavelength' not in self.indata['Raster'][0].metadata['Raster']:
            self.showprocesslog('Error: Your data should have wavelengths in'
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
            self.showprocesslog('Error: Your data bands have different sizes. '
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
            self.map.refl = float(dat[0].metadata['Raster']['reflectance_scale_factor'])

        dat2 = []
        wvl = []
        for j in dat:
            if self.chk_rot.isChecked():
                dat2.append(j.data.T)
            else:
                dat2.append(j.data)
            wvl.append(float(j.metadata['Raster']['wavelength']))

        self.map.datarr = np.array(dat2)
        self.map.nodata = dat[0].nodata
        self.map.wvl = np.array(wvl)
        if self.map.wvl.max() < 20:
            self.map.wvl = self.map.wvl*1000.
            self.showprocesslog('Wavelengths appear to be in nanometers. '
                                'Converting to micrometers.')

        bands = [i.dataid for i in self.indata['Raster']]

        self.combo.clear()
        self.combo.addItems(bands)
        self.combo.currentIndexChanged.connect(self.on_combo)

        ftxt = [str(i) for i in self.feature.keys()]
        self.combo_feature.disconnect()
        self.combo_feature.clear()
        self.combo_feature.addItems(ftxt)
        self.feature_change()
        self.combo_feature.currentIndexChanged.connect(self.feature_change)

        tmp = self.exec_()

        if tmp == 0:
            return False

        self.outdata['Raster'] = self.indata['Raster']

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


class ProcFeatures(QtWidgets.QDialog):
    """
    Process Hyperspectral Features.

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
        self.feature = None

        # self.combo_sensor = QtWidgets.QComboBox()
        self.cb_ratios = QtWidgets.QComboBox()
        self.rfiltcheck = QtWidgets.QCheckBox('If the final product is a '
                                              'ratio, filter out values less '
                                              'than 1.')
        self.filtercheck = QtWidgets.QCheckBox('Filter Albedo and Vegetation')
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
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.pfeat')
        # label_sensor = QtWidgets.QLabel('Sensor:')
        lbl_ratios = QtWidgets.QLabel('Product:')
        lbl_details = QtWidgets.QLabel('Details:')

        self.tablewidget.setRowCount(2)
        self.tablewidget.setColumnCount(3)
        self.tablewidget.setHorizontalHeaderLabels(['Feature', 'Filter',
                                                    'Threshold'])
        self.tablewidget.resizeColumnsToContents()
        self.filtercheck.setChecked(True)
        self.rfiltcheck.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Process Hyperspectral Features')

        gridlayout_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.cb_ratios, 1, 1, 1, 1)
        gridlayout_main.addWidget(lbl_details, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.tablewidget, 2, 1, 1, 1)
        gridlayout_main.addWidget(self.filtercheck, 3, 0, 1, 2)
        gridlayout_main.addWidget(self.rfiltcheck, 4, 0, 1, 2)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        self.cb_ratios.currentIndexChanged.connect(self.product_change)
        self.filtercheck.stateChanged.connect(self.product_change)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def product_change(self):
        """
        Change product combo.

        Returns
        -------
        None.

        """
        txt = self.cb_ratios.currentText()
        self.tablewidget.clear()

        product = self.product[txt]

        if self.filtercheck.isChecked():
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
            cb = QtWidgets.QComboBox()
            cb.addItems(['<', '>'])
            self.tablewidget.setCellWidget(i, 1, cb)

            txt2 = str(product[i])
            txt2 = txt2.split()

            cb.setCurrentText(txt2[1])
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
            self.showprocesslog('No Satellite Data')
            return False

        self.feature = features.feature
        self.ratio = features.ratio

        self.cb_ratios.disconnect()
        self.product = features.product.copy()
        del self.product['filter']
        self.cb_ratios.clear()
        self.cb_ratios.addItems(self.product)

        # The filter line is added after the other products so that it does
        # not make it into the list widget
        self.product['filter'] = features.product['filter']
        self.cb_ratios.currentIndexChanged.connect(self.product_change)
        self.product_change()

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

        mineral = self.cb_ratios.currentText()
        rfilt = self.rfiltcheck.isChecked()

        # feature = self.feature
        # ratio = self.ratio
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
                self.showprocesslog('Processing '+os.path.basename(ifile))

                dat = get_raster(ifile)
                datfin = calcfeatures(dat, mineral, self.feature, self.ratio,
                                      product, rfilt, piter=self.piter)

                ofile = (os.path.basename(ifile).split('.')[0] + '_' +
                         mineral.replace(' ', '_') + '.tif')
                ofile = os.path.join(odir, ofile)
                if datfin[0].data.mask.min() == True:
                    self.showprocesslog(' Could not find any ' + mineral +
                                        '. No data to export.')
                else:
                    self.showprocesslog('Exporting '+os.path.basename(ofile))
                    export_raster(ofile, datfin, 'GTiff', piter=self.piter)

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
    dat : PyGMI Data
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
    piter : iter, optional
        Progress bar iterable. The default is iter.

    Returns
    -------
    datfin : list
        Output datasets.

    """
    allfeatures = [i.split()[0] for i in product if i[0] == 'f']
    allratios = [i.split()[0] for i in product if i[0] != 'f']

    # Get list of wavelengths and data
    dat2 = []
    xval = []
    for j in dat:
        dat2.append(j.data)
        # refl = round(float(re.findall(r'[\d\.\d]+', j.dataid)[-1])*1000, 2)
        refl = float(re.findall(r'[\d\.\d]+', j.dataid)[-1])
        if refl < 100.:
            refl = refl * 1000
        refl = round(refl, 2)
        xval.append(refl)

    xval = np.array(xval)
    dat2 = np.ma.array(dat2)

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

        tmp = np.nonzero((xdat > lmin) & (xdat < lmax))[0]
        i1a = tmp[0]
        i2a = tmp[-1]

        fdat = np.moveaxis(fdat, 0, -1)

        for i in piter(range(rows)):
            ptmp[i], dtmp[i] = fproc(fdat[i].data, ptmp[i], dtmp[i], i1a, i2a,
                                     xdat)
        depths[fname] = dtmp
        wvl[fname] = ptmp
        datcalc[fname] = dtmp

    datout = None
    datout2 = None
    tmpw = None

    for i in product:
        if '>' in i or '<' in i or '=' in i or i[0] == 'r':
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
                breakpoint()
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
def fproc(fdat, ptmp, dtmp, i1a, i2a, xdat):
    """
    Feature process.

    This function finds the minimum value of a feature.

    Parameters
    ----------
    fdat : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    cols, _ = fdat.shape

    for j in range(cols):
        yval = fdat[j]
        if yval.mean() == 0:
            continue
        # if True in yval.mask:
        #     continue

        yhull = phull(yval)
        crem = yval/yhull
        imin = crem[i1a:i2a].argmin()
        # dtmp[j] = 1. - crem[i1a:i2a][imin]
        # ptmp[j] = xdat[i1a:i2a][imin]

        if imin == 0 or imin == (i2a-i1a-1):
            dtmp[j] = 1. - crem[i1a:i2a][imin]
            ptmp[j] = xdat[i1a:i2a][imin]
            continue
        x, y = cubic_calc(xdat[i1a:i2a], crem[i1a:i2a], imin)

        ptmp[j] = x
        dtmp[j] = 1. - y

        # if ptmp[j] > 880. and ptmp[j] < 884:
        #     # fun = CubicSpline(xdat[i1a:i2a], crem[i1a: i2a], bc_type='natural')
        #     fun = CubicSpline(xdat[i1a:i2a][imin-2:imin+3], crem[i1a:i2a][imin-2: imin+3])
        #     xxx = np.arange(xdat[i1a:i2a][imin-2], xdat[i1a:i2a][imin+2], 0.1)
        #     tmp = minimize(fun, xdat[i1a:i2a][imin])

        #     x, y = cubic_calc(xdat[i1a:i2a], crem[i1a:i2a], imin)

        #     plt.figure(dpi=150)
        #     plt.plot(xdat[i1a:i2a], crem[i1a:i2a], '-+')
        #     plt.plot(xxx, fun(xxx))
        #     plt.plot(tmp.x, tmp.fun, 'o')
        #     plt.plot(x, y, 'k+')
        #     print(tmp.x, x)
        #     plt.show()

        #     yval = fdat[j-1]
        #     yhull = phull(yval)
        #     crem = yval/yhull
        #     imin = crem[i1a:i2a].argmin()

        #     x, y = cubic_calc(xdat[i1a:i2a], crem[i1a:i2a], imin)

        #     fun = CubicSpline(xdat[i1a:i2a][imin-2:imin+3], crem[i1a: i2a][imin-2:imin+3])
        #     xxx = np.arange(xdat[i1a:i2a][imin-2], xdat[i1a:i2a][imin+2], 0.1)
        #     tmp = minimize(fun, xdat[i1a:i2a][imin])

        #     plt.figure(dpi=150)
        #     plt.plot(xxx, fun(xxx))
        #     plt.plot(xdat[i1a:i2a], crem[i1a:i2a], '-+')
        #     plt.plot(tmp.x, tmp.fun, 'o')
        #     plt.plot(x, y, 'k+')

        #     plt.show()
        #     print(tmp.x, x)

        #     breakpoint()

    return ptmp, dtmp


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
    # if imin == 0 or imin == (i2a-i1a-1):
    #     dtmp[j] = 1. - crem[i1a:i2a][imin]
    #     ptmp[j] = xdat[i1a:i2a][imin]
    #     continue

    x1 = xdat[imin-1]
    x2 = xdat[imin]
    x3 = xdat[imin+1]

    y1 = crem[imin-1]
    y2 = crem[imin]
    y3 = crem[imin+1]

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

    min1 = [(-c1 + np.sqrt(-3*b1*d1 + c1**2))/(3*d1),
            -(c1 + np.sqrt(-3*b1*d1 + c1**2))/(3*d1)]

    min2 = [(-c2 + np.sqrt(-3*b2*d2 + c2**2))/(3*d2),
            -(c2 + np.sqrt(-3*b2*d2 + c2**2))/(3*d2)]

    for i in min1:
        if x1 < i and i < x2:
            x = i
            y = a1+b1*x+c1*x**2+d1*x**3

    for i in min2:
        if x2 < i and i < x3:
            x = i
            y = a2+b2*x+c2*x**2+d2*x**3

    return x, y


@jit(nopython=True)
def phull(sample1):
    """
    Hull Calculation.

    Parameters
    ----------
    sample : numpy array
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
    with open(ifile[:-4]+'.hdr') as file:
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
    data = data / hdr3['reflectance scale factor']

    data.shape = (hdr3['lines'], hdr3['samples'])

    spectra = {}
    for i, val in enumerate(hdr3['spectra names']):
        spectra[val] = {'wvl': hdr3['wavelength'],
                        'refl': data[i]}

    return spectra


def _testfn():
    """Test routine."""
    from pygmi.rsense.iodefs import get_data

    pbar = ProgressBarText()

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    # ifile = r"C:\Workdata\Lithosphere\merge\cut-087-0824_iMNF15.hdr"
    ifile = r"E:\Workdata\Hyperspectral\080_0824-0920_ref_rect_clip.hdr"
    ifile = r"E:\Workdata\Remote Sensing\hyperion\EO1H1760802013198110KF_1T.ZIP"

    xoff = 0
    yoff = 2000
    xsize = None
    ysize = 1000
    nodata = 15000
    nodata = 0

    iraster = (xoff, yoff, xsize, ysize)
    iraster = None

    # data = get_raster(ifile, nval=nodata, iraster=iraster, piter=pbar.iter)
    data = get_data(ifile, extscene='Hyperion')

    # data = get_raster(ifile, piter=pbar.iter)

    tmp = ProcFeatures(None)
    tmp.indata['Raster'] = data
    # tmp.cb_ratios.setCurrentText('ferric iron')
    tmp.settings()

    dat = tmp.outdata['Raster'][0]

    plt.figure(dpi=150)
    plt.imshow(dat.data)
    plt.colorbar()
    plt.show()

    print(dat.data.mean())

    plt.figure(dpi=150)
    plt.hist(dat.data.flatten(), bins=200)
    plt.show()

    tmp = np.histogram(dat.data[dat.data > 0])


def _testfn2():
    """Test routine."""
    from pygmi.rsense.iodefs import get_data
    from pygmi.raster.dataprep import lstack

    ifile = r'C:\Workdata\Hyperspectral\071_0818-0932_ref_rect_BSQ.hdr'
    ifile = r"E:\Workdata\Remote Sensing\hyperion\EO1H1760802013198110KF_1T.ZIP"
    ifile = r"E:\Workdata\Remote Sensing\Landsat\LC08_L1TP_176080_20190820_20190903_01_T1.tar.gz"
    # ifile = r"E:\Workdata\Remote Sensing\Sentinel-2\S2A_MSIL2A_20210305T075811_N0214_R035_T35JML_20210305T103519.zip"
    ifile = r"E:\Workdata\Remote Sensing\AST_07XT_00307292005085059_20210608060928_376.hdf"
    ifile = r"E:\Workdata\Remote Sensing\ASTER\old\AST_07XT_00309042002082052_20200518021740_29313.zip"


    # xoff = 0
    # yoff = 2000
    # xsize = None
    # ysize = 1000
    # nodata = 15000
    # nodata = 0

    # iraster = (xoff, yoff, xsize, ysize)
    # iraster = None

    # data = get_raster(ifile, nval=nodata, iraster=iraster)
    data = get_data(ifile, extscene='Sentinel-2')

    data = lstack(data)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = AnalSpec()
    tmp.indata['Raster'] = data
    tmp.settings()


def _testfn3():
    """Test."""
    ifile1 = r'C:\Workdata\Lithosphere\merge\cut-087-0824_iMNF15.hdr'
    ifile2 = r'C:\Workdata\Lithosphere\merge\cut-088-0824_iMNF15.hdr'
    ifile3 = r'C:\Workdata\Lithosphere\merge\cut-089-0824_iMNF15.hdr'

    feat = 18

    yoff = 2425
    ysize = 400
    nodata = 0
    iraster = (0, yoff, None, ysize)

    data1 = get_raster(ifile1, nval=nodata, iraster=iraster)
    data2 = get_raster(ifile2, nval=nodata, iraster=iraster)
    data3 = get_raster(ifile3, nval=nodata, iraster=iraster)

    plt.figure(dpi=150)
    plt.imshow(data1[0].data, extent=data1[0].extent)
    plt.plot(277545, 6774900, '+k')
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(data2[0].data, extent=data2[0].extent)
    plt.plot(277545, 6774900, '+k')
    plt.plot(279900, 6774900, '+k')
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(data3[0].data, extent=data3[0].extent)
    plt.plot(279900, 6774900, '+k')
    plt.show()

    yoff = 2625
    ysize = 1
    nodata = 0
    iraster = (0, yoff, None, ysize)
    # iraster = None

    data1 = get_raster(ifile1, nval=nodata, iraster=iraster)
    data2 = get_raster(ifile2, nval=nodata, iraster=iraster)
    data3 = get_raster(ifile3, nval=nodata, iraster=iraster)

    for i in range(2):
        if i == 0:
            xxx = 277545
            yyy = 6774900
        else:
            xxx = 279900
            yyy = 6774900
            data1 = data3

        i1 = int((xxx-data1[0].extent[0])/data1[0].xdim)
        i2 = int((xxx-data2[0].extent[0])/data2[0].xdim)

        # Get list of wavelengths and data
        dat1 = []
        xval1 = []
        for j in data1:
            dat1.append(j.data)
            refl = round(float(re.findall(r'[\d\.\d]+', j.dataid)[-1])*1000, 2)
            xval1.append(refl)

        xval1 = np.array(xval1)
        dat1 = np.array(dat1)

        dat2 = []
        xval2 = []
        for j in data2:
            dat2.append(j.data)
            refl = round(float(re.findall(r'[\d\.\d]+', j.dataid)[-1])*1000, 2)
            xval2.append(refl)

        xval2 = np.array(xval2)
        dat2 = np.array(dat2)

        dat1 = dat1[55: 98]
        dat2 = dat2[55: 98]
        xval1 = xval1[55: 98]
        xval2 = xval2[55: 98]

        fdat = []
        xdat = []
        fdat.append(dat1[:, 0, i1])
        fdat.append(dat2[:, 0, i2])
        # xdat.append(xval1)
        # xdat.append(xval2)
        xdat = xval1

        fdat = np.array(fdat)

        ptmp = np.array([0, 0])
        dtmp = np.array([0, 0])

        ptmp, dtmp = fproc(fdat, ptmp, dtmp, 11, 19, xdat)

        spec1 = dat1[:, 0, i1]
        spec2 = dat2[:, 0, i2]

        plt.figure(dpi=150)
        plt.title(str(i)+': '+str(i1))
        plt.plot(xval1, spec1)
        plt.plot(xval2, spec2, '-.')
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(ptmp[0], ymin, ymax, 'k')
        plt.vlines(ptmp[1], ymin, ymax, 'k')

        spec1 = spec1/phull(spec1)
        spec2 = spec2/phull(spec2)

        plt.figure(dpi=150)
        plt.title(str(i)+': '+str(i1))
        plt.plot(xval1, spec1)
        plt.plot(xval2, spec2, 'r-.')

        plt.plot(xval1[11:19], spec1[11:19], '.')
        plt.plot(xval2[11:19], spec2[11:19], '.')

        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(ptmp[0], ymin, ymax, label=str(ptmp[0]))
        plt.vlines(ptmp[1], ymin, ymax, colors='r', linestyles='-.',
                   label=str(ptmp[1]))
        plt.legend()
        plt.show()

        plt.figure(dpi=150)

        dat1 = np.ma.masked_equal(dat1, 0)
        dat2 = np.ma.masked_equal(dat2, 0)

        x1 = np.linspace(data1[0].extent[0], data1[0].extent[1],
                         dat1[feat, 0].size)
        x2 = np.linspace(data2[0].extent[0], data2[0].extent[1],
                         dat2[feat, 0].size)
        plt.plot(x1, dat1[feat, 0])
        plt.plot(x2, dat2[feat, 0], '-.')

        ymin, ymax = plt.gca().get_ylim()

        plt.vlines(xxx, ymin, ymax, 'k')
        plt.show()

    # breakpoint()


if __name__ == "__main__":
    # _testfn3()
    _testfn2()
