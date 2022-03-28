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
"""Calculate remote sensing ratios."""

import copy
import os
import sys
import re
import numexpr as ne
import numpy as np
from PyQt5 import QtWidgets, QtCore

from pygmi import menu_default
from pygmi.rsense import iodefs
from pygmi.raster.iodefs import export_raster
from pygmi.raster.dataprep import lstack
from pygmi.misc import ProgressBarText


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
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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
        btn_invert = QtWidgets.QPushButton('Invert Selection')
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.ratios')
        label_sensor = QtWidgets.QLabel('Sensor:')
        label_ratios = QtWidgets.QLabel('Ratios:')

        self.lw_ratios.setSelectionMode(self.lw_ratios.MultiSelection)

        self.combo_sensor.addItems(['ASTER',
                                    'Landsat 8 (OLI)',
                                    'Landsat 7 (ETM+)',
                                    'Landsat 4 and 5 (TM)',
                                    'Sentinel-2', 'WorldView'])
        # self.setratios()

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Band Ratio Calculations')

        gridlayout_main.addWidget(label_sensor, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.combo_sensor, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_ratios, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.lw_ratios, 1, 1, 1, 1)
        gridlayout_main.addWidget(btn_invert, 2, 0, 1, 2)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.lw_ratios.clicked.connect(self.set_selected_ratios)
        self.combo_sensor.currentIndexChanged.connect(self.setratios)
        btn_invert.clicked.connect(self.invert_selection)

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

        if 'RasterFileList' in self.indata:
            bfile = os.path.basename(self.indata['RasterFileList'][0])
        else:
            bfile = os.path.basename(self.indata['Raster'][0].filename)

        if 'AST_' in bfile and 'hdf' in bfile.lower():
            self.combo_sensor.setCurrentText('ASTER')
        elif bfile[:4] in ['LC08']:
            self.combo_sensor.setCurrentText('Landsat 8 (OLI)')
        elif bfile[:4] in ['LE07']:
            self.combo_sensor.setCurrentText('Landsat 7 (ETM+)')
        elif bfile[:4] in ['LT04', 'LT05']:
            self.combo_sensor.setCurrentText('Landsat 4 and 5 (TM)')
        else:
            self.combo_sensor.setCurrentText('Sentinel-2')

        self.setratios()

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
        # evi = None
        # tci = None
        # vci = None

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

        if not rlist:
            self.showprocesslog('You need to select a ratio to calculate.')
            return False

        for ifile in flist:
            if isinstance(ifile, str):
                dat = iodefs.get_data(ifile,
                                      showprocesslog=self.showprocesslog,
                                      piter=self.piter)
                ofile = ifile
            elif isinstance(ifile, list) and 'RasterFileList' in self.indata:
                dat = []
                for jfile in ifile:
                    dat += iodefs.get_data(jfile,
                                           showprocesslog=self.showprocesslog,
                                           piter=self.piter)
                ofile = ifile[-1]
            else:
                dat = ifile
                ofile = dat[0].filename

            if dat is None:
                continue

            datsml = []
            for i in dat:
                tmp = i.dataid.split()
                txt = tmp[0]

                if 'Band' not in txt and 'B' in txt:
                    txt = txt.replace('B', 'Band')

                if 'Band' not in txt and 'LST' not in txt:
                    continue
                datsml.append(i)

            dat = lstack(datsml, self.piter, pprint=self.showprocesslog)
            # commonmask=True)

            datd = {}
            newmask = None
            for i in dat:
                tmp = i.dataid.split()
                txt = tmp[0]
                if txt == 'Band':
                    txt = tmp[0]+tmp[1]

                if 'Band' not in txt and 'B' in txt and ',' in txt:
                    txt = txt.replace('B', 'Band')
                    txt = txt.replace(',', '')

                if 'Band' not in txt and 'B' in txt:
                    txt = txt.replace('B', 'Band')

                if txt == 'Band3N':
                    txt = 'Band3'

                if txt == 'Band8A':
                    txt = 'Band8'

                datd[txt] = i.data
                # if newmask is None:
                #     newmask = i.data.mask
                # else:
                #     newmask = (newmask | i.data.mask)

            datfin = []
            for i in self.piter(rlist):
                self.showprocesslog('Calculating '+i)
                formula = i.split(' ')[0]
                formula = re.sub(r'B(\d+)', r'Band\1', formula)
                blist = formula
                for j in ['/', '*', '+', '-', '(', ')']:
                    blist = blist.replace(j, ' ')
                blist = blist.split()
                blist = list(set(blist))

                abort = []
                for j in blist:
                    if 'B' not in j:
                        continue
                    if j not in datd:
                        abort.append(j)
                if abort:
                    self.showprocesslog('Error:'+' '.join(abort)+'missing.')
                    continue

                newmask = datd[blist[0]].mask
                for j in blist:
                    newmask = (newmask | datd[blist[j]].mask)

                ratio = ne.evaluate(formula, datd)

                ratio = ratio.astype(np.float32)
                ratio[newmask] = dat[0].nodata
                ratio = np.ma.array(ratio, mask=newmask,
                                    fill_value=dat[0].nodata)

                ratio = np.ma.fix_invalid(ratio)

                rband = copy.deepcopy(dat[0])
                rband.data = ratio
                rband.dataid = i.replace(r'/', 'div')
                datfin.append(rband)

            ofile = ofile.split('.')[0] + '_ratio.tif'
            if datfin:
                self.showprocesslog('Exporting to '+ofile)
                export_raster(ofile, datfin, 'GTiff', piter=self.piter)
                self.outdata['Raster'] = datfin

        return True

    def setratios(self):
        """
        Set the available ratios.

        The ratio definitions are for the ASTER satellite. Band 0 refers to
        an imaginary blue band.

        Returns
        -------
        None.

        """
        sensor = self.combo_sensor.currentText()

        sdict = {}

        sdict['ASTER'] = {'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'B4': 'B4',
                          'B5': 'B5', 'B6': 'B6', 'B7': 'B7', 'B8': 'B8',
                          'B9': 'B9', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12',
                          'B13': 'B13', 'B14': 'B14'}
        sdict['Landsat 8 (OLI)'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4',
                                    'B3': 'B5', 'B4': 'B6', 'B5': 'B7'}
        sdict['Landsat 7 (ETM+)'] = {'B0': 'B1', 'B1': 'B2', 'B2': 'B3',
                                     'B3': 'B4', 'B4': 'B5', 'B5': 'B7'}
        sdict['Landsat 4 and 5 (TM)'] = sdict['Landsat 7 (ETM+)']
        sdict['Sentinel-2'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4', 'B3': 'B8',
                               'B4': 'B11', 'B5': 'B12'}
        sdict['WorldView'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B5', 'B3': 'B7',}
                              # 'B4': 'B11', 'B5': 'B13', 'B6': 'B14',
                              # 'B7': 'B15', 'B8': 'B16'}

        rlist = []

        # carbonates/mafic minerals bands
        rlist += [r'(B7+B9)/B8 carbonate chlorite epidote',
                  r'(B6+B9)/(B7+B8) epidote chlorite amphibole',
                  r'(B6+B9)/B8 amphibole MgOH',
                  r'B6/B8 amphibole',
                  r'(B6+B8)/B7 dolomite',
                  r'B13/B14 carbonate']

        # iron bands (All, but possibly only swir and vnir)
        rlist += [r'B2/B1 Ferric Iron Fe3+',
                  r'B2/B0 Iron Oxide',
                  r'B5/B3+B1/B2 Ferrous Iron Fe2+',
                  r'B4/B5 Laterite or Alteration',
                  r'B4/B2 Gossan',
                  r'B5/B4 Ferrous Silicates (biotite, chloride, amphibole)',
                  r'B4/B3 Ferric Oxides (can be ambiguous)']  # lsat ferrous?

        # silicates bands
        rlist += [r'(B5+B7)/B6 sericite muscovite illite smectite',
                  r'(B4+B6)/B5 alunite kaolinite pyrophyllite',
                  r'B5/B6 phengitic or host rock',
                  r'B7/B6 muscovite',
                  r'B7/B5 kaolinite',
                  r'(B5*B7)/(B6*B6) clay']

        # silica
        rlist += [r'B14/B12 quartz',
                  r'B12/B13 basic degree index (gnt cpx epi chl) or SiO2',
                  r'B13/B12 SiO2 same as B14/B12',
                  r'(B11*B11)/(B10*B12) siliceous rocks',
                  r'B11/B10 silica',
                  r'B11/B12 silica',
                  r'B13/B10 silica']

        # Other
        rlist += [r'B3/B2 Vegetation',
                  r'(B3-B2)/(B3+B2) NDVI',
                  r'(B3-B4)/(B3+B4) NDWI/NDMI water in leaves',
                  r'(B1-B3)/(B1+B3) NDWI water bodies ',
                  r'2.5*((B3-B2)/(B3+6.0*B2-7.5*B0+1)) EVI',
                  r'0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2']

        bandmap = sdict[sensor]
        svalues = set(bandmap.keys())
        rlist2 = []
        for i in rlist:
            formula = i.split(' ')[0]
            lbl = i[i.index(' '):]
            bands = set(re.findall(r'B\d+', formula))
            if bands.issubset(svalues):
                tmp = re.sub(r'B(\d+)', r'tmpB\1', formula)
                for j in svalues:
                    tmp = tmp.replace('tmp'+j, bandmap[j])
                rlist2.append(tmp+lbl)

        # rlist2 += ['VCI']

        # if 'Landsat' in sensor:
        #     rlist2 += ['TCI', 'VHI']

        self.lw_ratios.clear()
        self.lw_ratios.addItems(rlist2)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(True)
            item.setText('\u2713 ' + item.text())

    def invert_selection(self):
        """
        Invert the selected ratios.

        Returns
        -------
        None.

        """
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(not(item.isSelected()))

        self.set_selected_ratios()

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


class ConditionIndices(QtWidgets.QDialog):
    """
    Calculate Satellite Condition Indices.

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
        self.bfile = None

        self.combo_index = QtWidgets.QComboBox()
        self.lw_ratios = QtWidgets.QListWidget()
        self.combo_sensor = QtWidgets.QComboBox()

        # self.label_sensor = QtWidgets.QLabel('Sensor:')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton('Invert Selection')
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.ratios')
        label_index = QtWidgets.QLabel('Index:')
        label_ratios = QtWidgets.QLabel('Condition Indices:')
        label_sensor = QtWidgets.QLabel('Sensor:')

        self.lw_ratios.setSelectionMode(self.lw_ratios.MultiSelection)

        self.combo_index.addItems(['EVI',
                                   'NDVI',
                                   'MSAVI2'])

        self.combo_sensor.addItems(['ASTER',
                                    'Landsat 8 (OLI)',
                                    'Landsat 7 (ETM+)',
                                    'Landsat 4 and 5 (TM)',
                                    'Sentinel-2', 'WorldView'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Condition Indices Calculations')

        gridlayout_main.addWidget(label_sensor, 0, 0, 1, 2)
        gridlayout_main.addWidget(self.combo_sensor, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_index, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.combo_index, 1, 1, 1, 1)
        gridlayout_main.addWidget(label_ratios, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.lw_ratios, 2, 1, 1, 1)
        gridlayout_main.addWidget(btn_invert, 3, 0, 1, 2)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.lw_ratios.clicked.connect(self.set_selected_ratios)
        # self.combo_index.currentIndexChanged.connect(self.setratios)
        btn_invert.clicked.connect(self.invert_selection)

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
        if 'RasterFileList' not in self.indata:
            self.showprocesslog('You need a raster file list as input.')
            return False

        bfile = os.path.basename(self.indata['RasterFileList'][0])
        self.bfile = bfile[:4]

        if 'AST_' in bfile and 'hdf' in bfile.lower():
            self.combo_sensor.setCurrentText('ASTER')
        elif bfile[:4] in ['LC08']:
            self.combo_sensor.setCurrentText('Landsat 8 (OLI)')
        elif bfile[:4] in ['LE07']:
            self.combo_sensor.setCurrentText('Landsat 7 (ETM+)')
        elif bfile[:4] in ['LT04', 'LT05']:
            self.combo_sensor.setCurrentText('Landsat 4 and 5 (TM)')
        else:
            self.combo_sensor.setCurrentText('Sentinel-2')

        self.setratios()

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
        index = self.combo_index.currentText()
        sensor = self.combo_sensor.currentText()

        rlist1 = []
        for i in self.lw_ratios.selectedItems():
            rlist1.append(i.text()[2:])

        rlist = []
        if 'VCI' in rlist1 and 'EVI' in index:
            rlist += [r'2.5*((B3-B2)/(B3+6.0*B2-7.5*B0+1)) EVI']
        elif 'VCI' in rlist1 and 'NDVI' in index:
            rlist += [r'(B3-B2)/(B3+B2) NDVI']
        elif 'VCI' in rlist1 and 'MSAVI2' in index:
            rlist += [r'0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2']

        rlist = correct_bands(rlist, sensor)

        evi = []
        tci = []
        vci = []
        vhi = []
        lst = []

        # datfin = []
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

        # flist = self.indata['RasterFileList']
        # if 'ASTER' in sensor:
        #     flist = get_aster_list(flist)
        # elif 'Landsat' in sensor:
        #     flist = get_landsat_list(flist, sensor, True)
        # elif 'Sentinel-2' in sensor:
        #     flist = get_sentinel_list(flist)
        # if not flist:
        #     self.showprocesslog('Could not find '+sensor+' data')
        #     return False

        if not rlist:
            self.showprocesslog('You need to select a ratio to calculate.')
            return False

        for ifile in flist:
            dat = iodefs.get_data(ifile, showprocesslog=self.showprocesslog)

            if dat is None:
                continue

            # Prepare for layer stacking
            datsml = []
            for i in dat:
                txt = i.dataid.split()[0]

                if 'Band' not in txt and 'B' in txt:
                    txt = txt.replace('B', 'Band')

                if 'Band' not in txt and 'LST' not in txt:
                    continue

                i.data = i.data.astype(float)
                i.data = i.data.filled(1e+20)
                i.data = np.ma.masked_equal(i.data, 1e+20)
                i.nodata = 1e+20
                datsml.append(i)

            dat = lstack(datsml, self.piter, pprint=self.showprocesslog)

            # Correct band names
            datd = {}
            newmask = None
            for i in dat:
                tmp = i.dataid.split()
                txt = tmp[0]
                if txt == 'Band':
                    txt = tmp[0]+tmp[1]

                if 'Band' not in txt and 'B' in txt and ',' in txt:
                    txt = txt.replace('B', 'Band')
                    txt = txt.replace(',', '')

                if 'Band' not in txt and 'B' in txt:
                    txt = txt.replace('B', 'Band')

                if txt == 'Band3N':
                    txt = 'Band3'

                if txt == 'Band8A':
                    txt = 'Band8'

                datd[txt] = i.data
                if newmask is None:
                    newmask = i.data.mask
                else:
                    newmask = (newmask | i.data.mask)

                if 'LST' in txt:
                    lst.append(i)

            # Calculate ratios
            for i in self.piter(rlist):
                self.showprocesslog('Calculating '+i)
                formula = i.split(' ')[0]
                formula = re.sub(r'B(\d+)', r'Band\1', formula)
                blist = formula
                for j in ['/', '*', '+', '-', '(', ')']:
                    blist = blist.replace(j, ' ')
                blist = blist.split()
                blist = list(set(blist))

                abort = []
                for j in blist:
                    if 'B' not in j:
                        continue
                    if j not in datd:
                        abort.append(j)
                if abort:
                    self.showprocesslog('Error:'+' '.join(abort)+'missing.')
                    continue

                ratio = ne.evaluate(formula, datd)

                ratio = ratio.astype(np.float32)
                ratio[newmask] = 1e+20
                # ratio[newmask] = dat[0].nodata
                ratio = np.ma.array(ratio, mask=newmask,
                                    fill_value=dat[0].nodata)

                ratio = np.ma.fix_invalid(ratio)

                rmask = ratio.mask | (ratio < -1) | (ratio > 1)
                ratio.mask = rmask.data
                tmp = copy.deepcopy(dat[0])
                tmp.data = ratio
                tmp.nodata = 1e+20
                evi.append(tmp)

                # rband = copy.deepcopy(dat[0])
                # rband.data = ratio
                # rband.dataid = i.replace(r'/', 'div')
                # datfin.append(rband)

                print(ifile)
                print('Index:', index, ratio.min(), ratio.max())

        if lst:
            lst = lstack(lst, commonmask=True)
        if evi:
            evi = lstack(evi, commonmask=True)

        ofile = ''
        if 'TCI' in rlist1 or 'VHI' in rlist1 and lst:
            tci = get_TCI(lst)
            ofile += '_TCI'
        if 'VCI' in rlist1 or 'VHI' in rlist1 and evi:
            vci = get_VCI(evi, index)
            ofile += '_VCI_'+index
        if 'VHI' in rlist1 and tci and vci:
            vhi = get_VHI(tci, vci)
            ofile += '_VHI'

        datfin = tci+vci+vhi

        for i in datfin:
            i.data = i.data.astype(np.float32)

        ofile = os.path.join(os.path.dirname(ifile),  'CI'+ofile+'.tif')

        if datfin:
            self.showprocesslog('Exporting to '+ofile)
            export_raster(ofile, datfin, 'GTiff', piter=self.piter)
            self.outdata['Raster'] = datfin

        return True

    def setratios(self):
        """
        Set the available ratios.

        The ratio definitions are for the ASTER satellite. Band 0 refers to
        an imaginary blue band.

        Returns
        -------
        None.

        """
        sensor = self.combo_sensor.currentText()

        rlist = ['VCI']

        if 'Landsat' in sensor:
            rlist += ['TCI', 'VHI']

        self.lw_ratios.clear()
        self.lw_ratios.addItems(rlist)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(True)
            item.setText('\u2713 ' + item.text())

    def invert_selection(self):
        """
        Invert the selected ratios.

        Returns
        -------
        None.

        """
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(not(item.isSelected()))

        self.set_selected_ratios()

    def set_selected_ratios(self):
        """
        Set the selected ratios.

        Returns
        -------
        None.

        """
        currentitem = self.lw_ratios.currentItem()

        idict = {}
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            idict[item.text()[2:]] = i

        if currentitem.text()[2:] == 'VHI':
            for i in range(self.lw_ratios.count()):
                self.lw_ratios.item(i).setSelected(currentitem.isSelected())
        elif not currentitem.isSelected():
            self.lw_ratios.item(idict['VHI']).setSelected(False)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            if item.isSelected():
                item.setText('\u2713' + item.text()[1:])
            else:
                item.setText(' ' + item.text()[1:])


def correct_bands(rlist, sensor):
    """
    Correct the band designations.

    Ratio formula are defined in terms of ASTER bands. This converts that to
    the target sensor.

    Parameters
    ----------
    rlist : list
        List of input ratios.
    sensor : str
        Target sensor.

    Returns
    -------
    rlist2 : list
        List of converted ratios.

    """
    # if 'AST_' in bfile and 'hdf' in bfile.lower():
    #     sensor = 'ASTER'
    # elif bfile[:4] in ['LC08']:
    #     sensor = 'Landsat 8 (OLI)'
    # elif bfile[:4] in ['LE07']:
    #     sensor = 'Landsat 7 (ETM+)'
    # elif bfile[:4] in ['LT04', 'LT05']:
    #     sensor = 'Landsat 4 and 5 (TM)'
    # else:
    #     sensor = 'Sentinel-2'

    sdict = {}

    sdict['ASTER'] = {'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'B4': 'B4',
                      'B5': 'B5', 'B6': 'B6', 'B7': 'B7', 'B8': 'B8',
                      'B9': 'B9', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12',
                      'B13': 'B13', 'B14': 'B14'}
    sdict['Landsat 8 (OLI)'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4',
                                'B3': 'B5', 'B4': 'B6', 'B5': 'B7'}
    sdict['Landsat 7 (ETM+)'] = {'B0': 'B1', 'B1': 'B2', 'B2': 'B3',
                                 'B3': 'B4', 'B4': 'B5', 'B5': 'B7'}
    sdict['Landsat 4 and 5 (TM)'] = sdict['Landsat 7 (ETM+)']
    sdict['Sentinel-2'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4', 'B3': 'B8',
                           'B4': 'B11', 'B5': 'B12'}

    sdict['WorldView'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B5', 'B3': 'B7',}
                          # 'B4': 'B11', 'B5': 'B13', 'B6': 'B14', 'B7': 'B15',
                          # 'B8': 'B16'}

    bandmap = sdict[sensor]
    svalues = set(bandmap.keys())
    rlist2 = []
    for i in rlist:
        formula = i.split(' ')[0]
        lbl = i[i.index(' '):]
        bands = set(re.findall(r'B\d+', formula))
        if bands.issubset(svalues):
            tmp = re.sub(r'B(\d+)', r'tmpB\1', formula)
            for j in svalues:
                tmp = tmp.replace('tmp'+j, bandmap[j])
            rlist2.append(tmp+lbl)

    return rlist2


def get_TCI(lst):
    """
    Calculate TCI.

    Parameters
    ----------
    lst : numpy array
        array of land surface temperatures.

    Returns
    -------
    tci : numpy array
        output TCI.

    """
    lst2 = []
    for j in lst:
        lst2.append(j.data)
    lst2 = np.ma.array(lst2)

    lstmax = lst2.max(0)
    lstmin = lst2.min(0)

    tci = []
    for dat in lst:
        tmp = copy.deepcopy(dat)

        tmp.data = (lstmax-dat.data)/(lstmax-lstmin)

        tmp.dataid = os.path.basename(dat.filename)[:-4]+'_TCI'
        tci.append(tmp)

    return tci


def get_VCI(evi, index):
    """
    Calculate VCI.

    Parameters
    ----------
    evi : numpy array
        array of land surface temperatures.

    Returns
    -------
    vci : numpy array
        output VCI.

    """
    evi2 = []
    for j in evi:
        evi2.append(j.data)

    evi2 = np.ma.array(evi2)

    evimax = evi2.max(0)
    evimin = evi2.min(0)

    vci = []
    for dat in evi:
        tmp = copy.deepcopy(dat)

        tmp.data = (dat.data-evimin)/(evimax-evimin)

        tmp.dataid = os.path.basename(dat.filename)[:-4]+'_VCI_'+index
        vci.append(tmp)

    return vci


def get_VHI(tci, vci, alpha=0.5):
    """
    Calculate VHI.

    Parameters
    ----------
    lst : numpy array
        array of land surface temperatures.

    Returns
    -------
    ratio : numpy array
        output VHI.

    """
    vhi = []
    for tci1 in tci:
        for vci1 in vci:
            if tci1.filename == vci1.filename:
                tmp = copy.deepcopy(tci1)
                tmp.data = vci1.data*alpha+tci1.data*(1-alpha)
                tmp.dataid = os.path.basename(tci1.filename)[:-4]+'_VHI'

                vhi.append(tmp)

    return vhi


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


def get_landsat_list(flist, sensor, allsats=False):
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
        if bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08']:
            return flist
        return []

    if allsats is True:
        fid = ['LT04', 'LT05', 'LE07', 'LC08']

    elif sensor == 'Landsat 8 (OLI)':
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


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt

    piter = ProgressBarText().iter
    # ifile = r'd:\Work\Workdata\ASTER\AST_05_00302282018211606_20180814024609_27608.hdf'
    ifile = r"d:\Workdata\Remote Sensing\Landsat\LM05_L1TP_171078_19840629_20180410_01_T2.tar.gz"
    ifile = r"d:\Workdata\Remote Sensing\Sentinel-2\S2A_MSIL2A_20210305T075811_N0214_R035_T35JML_20210305T103519.zip"
    extscene = 'Sentinel-2'


    ifile = r"d:\Workdata\Remote Sensing\ASTER\test\AST_07XT_00311172002085850_20220121015142_25162.hdf"
    extscene = None

    dat = iodefs.get_data(ifile, extscene=extscene, piter=piter)

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes


    SR = SatRatios()
    SR.indata['Raster'] = dat  # single file only
    SR.settings()

    dat2 = SR.outdata['Raster']
    for i in dat+dat2:
        plt.title(i.dataid)
        plt.imshow(i.data)
        plt.colorbar()
        plt.show()

    breakpoint()


def _testfn2():
    """Test routine."""
    import glob
    import matplotlib.pyplot as plt

    ifiles = glob.glob("d:/Workdata/NRF/172-079/*.tar")
    ifiles = glob.glob(r"d:\Workdata\Remote Sensing\Landsat\VHI\*.tar")

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    SR = ConditionIndices()
    SR.indata['RasterFileList'] = ifiles
    SR.settings()

    dat = SR.outdata["Raster"]

    for i in dat:
        plt.imshow(i.data, extent=i.extent)
        plt.colorbar()
        plt.title(i.dataid)
        plt.show()

    breakpoint()


def _testfn3():
    """Test Function"""
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt

    ifile = r"D:\Workdata\Remote Sensing\wv2\014568829030_01_P001_MUL.tif"
    ifile = r"C:\WorkProjects\Sentinel-2_Winter\clip_S2A_MSIL2A_20160828T080012_T35JML.tif"

    dat = get_raster(ifile)

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    SR = SatRatios()
    SR.indata['Raster'] = dat  # single file only

    SR.settings()

    dat = SR.outdata['Raster']

    plt.title(dat[0].dataid)
    plt.imshow(dat[0].data)
    plt.colorbar()
    plt.show()

    breakpoint()


if __name__ == "__main__":
    _testfn3()
