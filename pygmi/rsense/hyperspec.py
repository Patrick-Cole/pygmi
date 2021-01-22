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


Spectral feature examination is a GUI which allows for teh comparison of
spectra from the dataset with library spectra.

Must be able to:

1) Zoom into features
2) Select features and apply settings such as threshold if necessary
3) Combine multiple features into final filtered result.
4) Output successful feature combinations into formulae for processing.
5) Save spectra into a library for feature examination now or later.


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


class AnalSpec(QtWidgets.QDialog):
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
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.combo_sensor = QtWidgets.QComboBox()
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
        label_sensor = QtWidgets.QLabel('Sensor:')
        label_ratios = QtWidgets.QLabel('Ratios:')

        self.lw_ratios.setSelectionMode(self.lw_ratios.MultiSelection)

        self.combo_sensor.addItems(['ASTER',
                                    'Landsat 8 (OLI)',
                                    'Landsat 7 (ETM+)',
                                    'Landsat 4 and 5 (TM)',
                                    'Sentinel-2'])
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Band Ratio Calculations')

        gridlayout_main.addWidget(label_sensor, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.combo_sensor, 0, 1, 1, 1)
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

        self.combo_sensor.setCurrentText(projdata['sensor'])
        self.setratios()

        for i in self.lw_ratios.selectedItems():
            if i.text()[2:] not in projdata['ratios']:
                i.setSelected(False)
        self.set_selected_ratios()

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
        projdata['sensor'] = self.combo_sensor.currentText()

        rlist = []
        for i in self.lw_ratios.selectedItems():
            rlist.append(i.text()[2:])

        projdata['ratios'] = rlist

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


class ProcFeatures(QtWidgets.QDialog):
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


if __name__ == "__main__":
    testfn()
