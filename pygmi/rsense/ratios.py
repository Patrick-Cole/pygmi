# -----------------------------------------------------------------------------
# Name:        ratios.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2020 Council for Geoscience
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
"""Import Data."""

import copy
import os
import sys
import re
import numexpr as ne
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pygmi.menu_default as menu_default
import pygmi.rsense.iodefs as iodefs
from pygmi.raster.iodefs import export_gdal
from pygmi.raster.dataprep import merge


class SatRatios(QtWidgets.QDialog):
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
        self.pbar = parent.pbar

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
        self.setratios()

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
        self.lw_ratios.clicked.connect(self.set_selected_ratios)
        self.combo_sensor.currentIndexChanged.connect(self.setratios)

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
        sensor = self.combo_sensor.currentText()

        if 'RasterFileList' in self.indata:
            flist = self.indata['RasterFileList']
            if sensor == 'ASTER':
                flist = get_aster_list(flist)
            elif 'Landsat' in sensor:
                flist = get_landsat_list(flist, sensor)
            elif 'Sentinel-2' in sensor:
                flist = get_sentinel_list(flist)
            if not flist:
                self.showprocesslog('Could not find '+sensor+' data')
                return False
        else:
            flist = [self.indata['Raster']]

        rlist = []
        for i in self.lw_ratios.selectedItems():
            rlist.append(i.text()[2:])

        for ifile in flist:
            if isinstance(ifile, str):
                dat = iodefs.get_data(ifile, showprocesslog=self.showprocesslog)
                ofile = ifile
            elif isinstance(ifile, list) and 'RasterFileList' in self.indata:
                dat = []
                for jfile in ifile:
                    dat += iodefs.get_data(jfile,
                                           showprocesslog=self.showprocesslog)
                ofile = ifile[-1]
            else:
                dat = ifile
                ofile = dat[0].filename

            if dat is None:
                continue

            dat = merge(dat)

            datd = {}
            for i in dat:
                tmp = i.dataid.split()
                txt = tmp[0]
                if txt == 'Band':
                    txt = tmp[0]+tmp[1]

                if 'Band' not in txt and 'B' in txt and ',' in txt:
                    txt = txt.replace('B', 'Band')
                    txt = txt.replace(',', '')

                if txt == 'Band3N':
                    txt = 'Band3'

                if txt == 'Band8A':
                    txt = 'Band8'

                datd[txt] = i.data

            datfin = []
            for i in self.pbar.iter(rlist):
                self.showprocesslog('Calculating '+i)
                formula = i.split(' ')[0]
                formula = re.sub(r'(\d+)', r'Band\1', formula)
                blist = formula
                for j in ['/', '*', '+', '-', '(', ')']:
                    blist = blist.replace(j, ' ')
                blist = blist.split()
                blist = list(set(blist))

                abort = []
                for j in blist:
                    if j not in datd:
                        abort.append(j)
                if abort:
                    self.showprocesslog('Error:'+', '.join(abort)+'missing.')
                    continue

                ratio = ne.evaluate(formula, datd)

                ratio = ratio.astype(np.float32)
                ratio = np.ma.masked_invalid(ratio)
                ratio.set_fill_value(dat[0].nullvalue)
                ratio = np.ma.fix_invalid(ratio)
                rband = copy.copy(dat[0])
                rband.data = ratio
                rband.dataid = i.replace(r'/', 'div')
                datfin.append(rband)
            ofile = ofile.split('.')[0] + '_ratio.tif'
            self.pbar.setValue(0)
            if datfin:
                self.showprocesslog('Exporting to '+ofile)
                export_gdal(ofile, datfin, 'GTiff', piter=self.pbar.iter)

        self.outdata['Raster'] = datfin
        return True

    def setratios(self):
        """
        Set the available ratios.

        Returns
        -------
        None.

        """
        sensor = self.combo_sensor.currentText()

        sdict = {}

        sdict['ASTER'] = {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                          '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
                          '11': '11', '12': '12', '13': '13', '14': '14'}
        sdict['Landsat 8 (OLI)'] = {'0': '2', '1': '3', '2': '4', '3': '5',
                                    '4': '6', '5': '7'}
        sdict['Landsat 7 (ETM+)'] = {'0': '1', '1': '2', '2': '3', '3': '4',
                                     '4': '5', '5': '7'}
        sdict['Landsat 4 and 5 (TM)'] = sdict['Landsat 7 (ETM+)']
        sdict['Sentinel-2'] = {'0': '2', '1': '3', '2': '4', '3': '8',
                               '4': '11', '5': '12'}

        rlist = []

        # carbonates/mafic minerals bands
        rlist += [r'(7+9)/8 carbonate chlorite epidote',
                  r'(6+9)/(7+8) epidote chlorite amphibole',
                  r'(6+9)/8 amphibole MgOH',
                  r'6/8 amphibole',
                  r'(6+8)/7 dolomite',
                  r'13/14 carbonate']

        # iron bands (All, but possibly only swir and vnir)
        rlist += [r'2/1 Ferric Iron Fe3+',
                  r'2/0 Iron Oxide',
                  r'5/3+1/2 Ferrous Iron Fe2+',
                  r'4/5 Laterite or Alteration',
                  r'4/2 Gossan',
                  r'5/4 Ferrous Silicates (biotite, chloride, amphibole)',
                  r'4/3 Ferric Oxides (can be ambiguous)']  # lsat ferrous?

        # silicates bands
        rlist += [r'(5+7)/6 sericite muscovite illite smectite',
                  r'(4+6)/5 alunite kaolinite pyrophyllite',
                  r'5/6 phengitic or host rock',
                  r'7/6 muscovite',
                  r'7/5 kaolinite',
                  r'(5*7)/(6*6) clay']

        # silica
        rlist += [r'14/12 quartz',
                  r'12/13 basic degree index (gnt cpx epi chl) or SiO2',
                  r'13/12 SiO2 same as 14/12',
                  r'(11*11)/(10*12) siliceous rocks',
                  r'11/10 silica',
                  r'11/12 silica',
                  r'13/10 silica']

        # Other
        rlist += [r'3/2 Vegetation',
                  r'(3-2)/(3+2) NDVI',
                  r'(3-4)/(3+4) NDWI/NDMI water in leaves',
                  r'(1-3)/(1+3) NDWI water bodies ']

        bandmap = sdict[sensor]
        svalues = set(bandmap.keys())
        rlist2 = []
        for i in rlist:
            formula = i.split(' ')[0]
            lbl = i[i.index(' '):]
            bands = set(re.findall(r'\d+', formula))
            if bands.issubset(svalues):
                tmp = re.sub(r'(\d+)', r'B\1', formula)
                for j in svalues:
                    tmp = tmp.replace('B'+j, bandmap[j])
                rlist2.append(tmp+lbl)

        self.lw_ratios.clear()
        self.lw_ratios.addItems(rlist2)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(True)
            item.setText('\u2713 ' + item.text())

    def set_selected_ratios(self):
        """
        Set the selected ratios.

        Returns
        -------
        None.

        """
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            if item.isSelected():
                item.setText('\u2713' + item.text()[1:])
            else:
                item.setText(' ' + item.text()[1:])


def get_aster_list(flist):
    """
    Get ASTER files from a file list.

    Parameters
    ----------
    flist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(flist[0], list):
        if 'AST_' in flist[0][0].filename:
            return flist
        return []

    names = {}
    for i in flist:
        if os.path.basename(i)[:3] != 'AST':
            continue

        adate = os.path.basename(i).split('_')[2]
        if adate not in names:
            names[adate] = []
        names[adate].append(i)

    for adate in names:
        has_07xt = [True for i in names[adate] if '_07XT_' in i]
        has_07 = [True for i in names[adate] if '_07_' in i]
        if len(has_07xt) > 0 and len(has_07) > 0:
            names[adate] = [i for i in names[adate] if '_07_' not in i]

    flist = []
    for adate in names:
        flist.append(names[adate])

    return flist


def get_landsat_list(flist, sensor):
    """
    Get Landsat files from a file list.

    Parameters
    ----------
    flist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(flist[0], list):
        bfile = os.path.basename(flist[0][0].filename)
        if bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08', 'LM05']:
            return flist
        return []

    if sensor == 'Landsat 8 (OLI)':
        fid = ['LC08']
    elif sensor == 'Landsat 7 (ETM+)':
        fid = ['LE07']
    elif sensor == 'Landsat 4 and 5 (TM)':
        fid = ['LT04', 'LT05']
    else:
        return None

    flist2 = []
    for i in flist:
        if os.path.basename(i)[:4] not in fid:
            continue
        if '.tif' in i:
            continue
        flist2.append(i)

    return flist2


def get_sentinel_list(flist):
    """
    Get Sentinel-2 files from a file list.

    Parameters
    ----------
    flist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(flist[0], list):
        if '.SAFE' in flist[0][0].filename:
            return flist
        return []

    flist2 = []
    for i in flist:
        if '.SAFE' not in i:
            continue
        flist2.append(i)

    return flist2


def testfn():
    """Main testing routine."""
    ifile = r'C:\Work\Workdata\ASTER\AST_05_00302282018211606_20180814024609_27608.hdf'
    dat = iodefs.get_data(ifile)

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    idir = r'C:\Work\Workdata\Sentinel-2'

    SR = SatRatios()
    SR.indata['Raster'] = dat  #single file only

    # IO = iodefs.ImportBatch()
    # IO.idir = idir
    # IO.settings(True)
    # SR.indata = IO.outdata

    SR.settings()


if __name__ == "__main__":
    testfn()
