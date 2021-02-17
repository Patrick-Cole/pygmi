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

import os
import sys
import re

import numpy as np
import numexpr as ne
from PyQt5 import QtWidgets, QtCore

import pygmi.menu_default as menu_default
from pygmi.raster.iodefs import get_raster
from pygmi.misc import ProgressBarText
from pygmi.raster.datatypes import numpy_to_pygmi


import os
import shutil
import json
import sys

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.patches as mpatches
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

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        dat = self.datarr[self.mindx]

        rows, cols = dat.shape

        self.figure.clf()
        ax1 = self.figure.add_subplot(211)

        ymin = dat.mean()-2*dat.std()
        ymax = dat.mean()+2*dat.std()

        self.csp = ax1.imshow(dat, vmin=ymin, vmax=ymax)

        ax1.set_xlim((0, cols))
        ax1.set_ylim((0, rows))
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)

        ax1.xaxis.set_major_formatter(frm)
        ax1.yaxis.set_major_formatter(frm)

        ax1.plot(self.col, self.row, '+w')

        ax2 = self.figure.add_subplot(212)
        prof = self.datarr[:, self.row, self.col]

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
        self.setWindowTitle('Core Metadata and Depth Assignment')
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
        self.map.init_graph()

    def disp_splib(self, row):
        """
        Change library spectra for display

        Parameters
        ----------
        row : TYPE
            DESCRIPTION.

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
        dat = self.map.datarr
        self.map.datarr = np.moveaxis(dat, 1, 2)
        self.map.row, self.map.col = self.map.col, self.map.row

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

        if 'wavelength' not in self.indata['Raster'][0].metadata['Raster']:
            self.showprocesslog('Error: Your data should have wavelengths in'
                                'the metadata')
            return False

        dat = self.indata['Raster']
        refl = float(dat[0].metadata['Raster']['reflectance_scale_factor'])

        dat2 = []
        wvl = []
        for j in dat:
            if self.chk_rot.isChecked():
                dat2.append(j.data.T)
            else:
                dat2.append(j.data)
            wvl.append(float(j.metadata['Raster']['wavelength']))

        self.map.datarr = np.array(dat2)/refl
        self.map.wvl = np.array(wvl)

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

        # self.map.init_graph()

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


def phull(sample):
    """
    Hull Calculation

    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    xvals = np.arange(sample.size)
    sample = np.transpose([xvals, sample])

    edge = sample[:1]
    rest = sample[1:]

    hull = [0]
    while len(rest) > 0:
        grad = rest - edge
        grad = grad[:, 1]/grad[:, 0]
        pivot = np.argmax(grad)
        edge = rest[pivot]
        rest = rest[pivot+1:]
        hull.append(pivot)

    hull = np.array(hull) + 1
    hull = hull.cumsum()-1
    out = np.transpose([hull, np.take(sample[:, 1], hull)])
    out = np.interp(xvals, out[:, 0], out[:, 1])

    return out


def readsli(ifile):
    """
    Reads an ENVI sli file.

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
        if 'Raster' not in self.indata and 'RasterFileList' not in self.indata:
            self.showprocesslog('No Satellite Data')
            return False

        self.feature = {}
        self.feature[900] = [776, 1050, 850, 910]
        self.feature[1300] = [1260, 1420]
        self.feature[1800] = [1740, 1820]
        self.feature[2080] = [2000, 2150]
        self.feature[2500] = [2500, 2500]
        self.feature[2200] = [2120, 2245]
        self.feature[2290] = [2270, 2330]
        self.feature[2330] = [2120, 2370]

        self.ratio = {}
        self.ratio['NDVI'] = '(R860-R687)/(R860+R687)'
        self.ratio['dryveg'] = '(R2006+R2153)/(R2081+R2100)'
        self.ratio['albedo'] = 'R1650'

        self.ratio['r2350De'] = '(R2326+R2376)/(R2343+R2359)'
        self.ratio['r2160D2190'] = '(R2136+R2188)/(R2153+R2171)' # Kaolin from non kaolin
        self.ratio['r2250D'] = '(R2227+R2275)/(R2241+R2259)'  # Chlorite epidote biotite
        self.ratio['r2380D'] = '(R2365+R2415)/(R2381+R2390)'  # Amphibole, talc
        self.ratio['r2330D'] = '(R2265+R2349)/(R2316+R2333)'  # MgOH and CO3


        self.product['mica'] = [2200, 'r2350De > 1.02', 'r2160D2190 < 1.005']
        self.product['smectite'] = [2200, 'r2350De < 1.02', 'r2160D2190 < 1.005']
        self.product['kaolin'] = [2200, 'r2160D2190 > 1.005']
        self.product['chlorite, epidote'] = ['r2250D', 'r2330D > 1.06']

        self.lw_ratios.clear()
        self.lw_ratios.addItems(self.product)

        self.product['filter'] = ['NDVI < .25', 'dryveg < 1.015', 'albedo > 1000']


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

        mineral = self.lw_ratios.currentItem().text()

        feature = self.feature
        ratio = self.ratio
        product = self.product

        allfeatures = [i for i in product[mineral] if isinstance(i, int)]
        allratios = [i.split()[0] for i in product[mineral]
                     if not isinstance(i, int)]
        allratios += [i.split()[0] for i in product['filter']
                      if not isinstance(i, int)]

        # Get list of wavelengths and data
        dat2 = []
        xval = []
        for j in self.indata['Raster']:
            dat2.append(j.data)
            refl = round(float(re.findall(r'[\d\.\d]+', j.dataid)[-1])*1000, 2)
            xval.append(refl)

        xval = np.array(xval)
        dat2 = np.array(dat2)

        # This gets nearest wavelength adn assigns to R number.
        # It does not interpolate.
        RBands = {}
        for j in range(1, 2501):
            i = abs(xval-j).argmin()
            RBands['R'+str(j)] = dat2[i]

        # Calclate ratios
        datcalc = {}
        for j in allratios:
            if j in datcalc:
                continue
            tmp = indexcalc(ratio[j], RBands)
            datcalc[j] = tmp

        # Start processing
        depths = {}
        # wvl = {}
        for fname in allfeatures:
            if len(feature[fname]) == 4:
                fmin, fmax, lmin, lmax = feature[fname]
            else:
                fmin, fmax = feature[fname]
                # lmin, lmax = fmin, fmax

            # get index of closest wavelength
            i1 = abs(xval-fmin).argmin()
            i2 = abs(xval-fmax).argmin()

            fdat = dat2[i1:i2+1]
            xdat = xval[i1:i2+1]

            # Raster calculation
            _, rows, cols = dat2.shape
            dtmp = np.zeros((rows, cols))
            ptmp = np.zeros((rows, cols))

            # tmp = np.nonzero((xdat > lmin) & (xdat < lmax))[0]
            # i1a = tmp[0]
            # i2a = tmp[-1]

            for i in self.piter(range(rows)):
                for j in range(cols):
                    yval = fdat[:, i, j]
                    if yval.max() == 0:
                        continue

                    yhull = phull(yval)
                    crem = yval/yhull

                    imin = crem.argmin()
                    dtmp[i, j] = crem[imin]
                    ptmp[i, j] = xdat[imin]


            depths[fname] = 1. - dtmp
            # wvl[fname] = ptmp

        datout = None
        for i in product[mineral]:
            if isinstance(i, int):
                tmp = depths[i]
            else:
                tmp = ne.evaluate(i, datcalc)

            if datout is None:
                datout = tmp
            else:
                datout = datout * tmp

        datout = np.ma.masked_equal(datout, 0)
        datfin.append(numpy_to_pygmi(datout, self.indata['Raster'][0],
                                     f'{mineral} depth'))
        # datfin.append(numpy_to_pygmi(pos1, dat[0], f'{product} {fname} wvl'))


        self.outdata['Raster'] = datfin
        return True


def indexcalc(formula, dat):
    """
    Calculates an index using numexpr.

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


def phull(sample):
    """
    Hull Calculation

    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    xvals = np.arange(sample.size)
    sample = np.transpose([xvals, sample])

    edge = sample[:1]
    rest = sample[1:]

    hull = [0]
    while len(rest) > 0:
        grad = rest - edge
        grad = grad[:, 1]/grad[:, 0]
        pivot = np.argmax(grad)
        edge = rest[pivot]
        rest = rest[pivot+1:]
        hull.append(pivot)

    hull = np.array(hull) + 1
    hull = hull.cumsum()-1
    out = np.transpose([hull, np.take(sample[:, 1], hull)])
    out = np.interp(xvals, out[:, 0], out[:, 1])

    return out


def testfn():
    """Main testing routine."""
    pbar = ProgressBarText()

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    ifile = r'c:\work\Workdata\Richtersveld\Reprocessed\057_0818-1117_ref_rect_BSQ.hdr'

    xoff = 0
    yoff = 2000
    xsize = None
    ysize = 1000
    nodata = 15000
    nodata = 0

    iraster = (xoff, yoff, xsize, ysize)
    # iraster = None

    data = get_raster(ifile, nval=nodata, iraster=iraster, piter = pbar.iter)

    # data = get_raster(ifile, piter=pbar.iter)

    tmp = ProcFeatures(None)
    tmp.indata['Raster'] = data
    tmp.settings()


def testfn2():
    """Main testing routine."""
    import matplotlib.pyplot as plt

    pbar = ProgressBarText()

    # ifile = (r'c:\work\Workdata\HyperspectralScanner\PTest\smile\FENIX\\'
    #          r'clip_BV1_17_118m16_125m79_2020-06-30_12-43-14.dat')

    # ifile = (r'C:\Work\Workdata\HyperspectralScanner\Processed Data\\'
    #           r'FENIX L201 Data Preparation v0810\BV1_17_extracted_image.img')

    # data = get_raster(ifile, piter=pbar.iter)

    ifile = r'c:\work\Workdata\Richtersveld\Reprocessed\057_0818-1117_ref_rect_BSQ.hdr'

    xoff = 0
    yoff = 2000
    xsize = None
    ysize = 1000
    nodata = 15000
    nodata = 0

    iraster = (xoff, yoff, xsize, ysize)
    # iraster = None

    data = get_raster(ifile, nval=nodata, iraster=iraster, piter = pbar.iter)

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = AnalSpec()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    testfn2()
