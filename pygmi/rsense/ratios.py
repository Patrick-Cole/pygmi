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
from pygmi.rsense.iodefs import get_from_rastermeta, set_export_filename
from pygmi.raster.iodefs import export_raster
from pygmi.raster.dataprep import lstack
from pygmi.misc import BasicModule


class SatRatios(BasicModule):
    """Calculate Satellite Ratios."""

    def __init__(self, parent=None):
        super().__init__(parent)

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
                                    'Landsat 8 and 9 (OLI)',
                                    'Landsat 7 (ETM+)',
                                    'Landsat 4 and 5 (TM)',
                                    'Sentinel-2', 'WorldView', 'Unknown'])
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
            self.showlog('No Satellite Data')
            return False

        if 'RasterFileList' in self.indata:
            dat = self.indata['RasterFileList']
            instr = dat[0].sensor
        else:
            dat = self.indata['Raster']
            instr = dat[0].metadata['Raster']['Sensor']

        if 'ASTER' in instr:
            self.combo_sensor.setCurrentText('ASTER')
        elif 'LC08' in instr or 'LC09' in instr:
            self.combo_sensor.setCurrentText('Landsat 8 and 9 (OLI)')
        elif 'LE07' in instr:
            self.combo_sensor.setCurrentText('Landsat 7 (ETM+)')
        elif 'LT04' in instr or 'LT05' in instr:
            self.combo_sensor.setCurrentText('Landsat 4 and 5 (TM)')
        elif 'WorldView' in instr and 'Multi' in instr:
            self.combo_sensor.setCurrentText('WorldView')
        elif 'Sentinel-2' in instr:
            self.combo_sensor.setCurrentText('Sentinel-2')
        else:
            self.combo_sensor.setCurrentText('Unknown')

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
        sensor = self.combo_sensor.currentText()

        rlist = []
        for i in self.lw_ratios.selectedItems():
            rlist.append(i.text()[2:])

        if 'RasterFileList' in self.indata:
            data = self.indata['RasterFileList']
        else:
            data = self.indata['Raster']

        if 'RasterFileList' in self.indata:
            flist = self.indata['RasterFileList']
            if sensor == 'ASTER':
                flist = get_aster_list(flist)
            elif 'Landsat' in sensor:
                flist = get_landsat_list(flist, sensor)
            elif 'Sentinel-2' in sensor:
                flist = get_sentinel_list(flist)
            if not flist:
                self.showlog('Warning: This might not be ' + sensor +
                             ' data. Will attempt to do calculation '
                             'anyway.')
                flist = self.indata['RasterFileList']
        else:
            flist = [self.indata['Raster']]

        rlist = []
        for i in self.lw_ratios.selectedItems():
            rlist.append(i.text()[2:])

        if not rlist:
            self.showlog('You need to select a ratio to calculate.')
            return False

        for ifile in flist:
            if 'RasterFileList' in self.indata:
                dat = get_from_rastermeta(ifile, piter=self.piter,
                                          showlog=self.showlog)
            else:
                dat = ifile

            if dat is None:
                continue

            odir = os.path.dirname(dat[0].filename)

            datfin = calc_ratios(dat, rlist, showlog=self.showlog,
                                 piter=self.piter)

            if datfin:
                odir = os.path.dirname(data[0].filename)
                odir = os.path.join(odir, 'ratios')

                os.makedirs(odir, exist_ok=True)

                ofile = set_export_filename(dat, odir, 'ratio')

                self.showlog('Exporting to '+ofile)
                export_raster(ofile, datfin, 'GTiff', piter=self.piter,
                              compression='DEFLATE', showlog=self.showlog)
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
                  r'(B1-B3)/(B1+B3) NDWI water bodies',
                  r'2.5*(B3-B2)/(B3+6.0*B2-7.5*B0+1) EVI',
                  r'0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2',
                  r'(B3A-B4+B5)/(B3A+B4-B5) NMDI',
                  r'((B4+B2)-(B3+B0))/((B4+B2)+(B3+B0)) BSI']

        # Colour composite

        rlist += [r'B5/B3 Used in colour composites',
                  r'B4/B0 Used in colour composites',
                  r'B5/B1 Used in colour composites',
                  r'B4/B7 Used in colour composites']

        # Landslides

        rlist += ['B0,B1,B2,B3,B4 Landslide Index']

        rlist2 = correct_bands(rlist, sensor)

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
            item.setSelected(not item.isSelected())

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


class ConditionIndices(BasicModule):
    """Calculate Satellite Condition Indices."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.bfile = None

        self.combo_index = QtWidgets.QComboBox()
        self.lw_ratios = QtWidgets.QListWidget()
        self.combo_sensor = QtWidgets.QComboBox()

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
        helpdocs = menu_default.HelpButton('pygmi.rsense.cind')
        label_index = QtWidgets.QLabel('Index:')
        label_ratios = QtWidgets.QLabel('Condition Indices:')
        label_sensor = QtWidgets.QLabel('Sensor:')

        self.lw_ratios.setSelectionMode(self.lw_ratios.MultiSelection)

        self.combo_index.addItems(['EVI',
                                   'NDVI',
                                   'MSAVI2'])

        self.combo_sensor.addItems(['ASTER',
                                    'Landsat 8 and 9 (OLI)',
                                    'Landsat 7 (ETM+)',
                                    'Landsat 4 and 5 (TM)',
                                    'Landsat (All)',
                                    'Sentinel-2', 'WorldView', 'Unknown'])

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
        if 'RasterFileList' not in self.indata:
            self.showlog('You need a raster file list as input.')
            return False

        bfile = os.path.basename(self.indata['RasterFileList'][0].filename)
        self.bfile = bfile[:4]

        dat = self.indata['RasterFileList'][0]

        instr = dat.sensor

        if 'ASTER' in instr:
            self.combo_sensor.setCurrentText('ASTER')
        elif 'LC08' in instr or 'LC09' in instr:
            self.combo_sensor.setCurrentText('Landsat 8 and 9 (OLI)')
        elif 'LE07' in instr:
            self.combo_sensor.setCurrentText('Landsat 7 (ETM+)')
        elif 'LT04' in instr or 'LT05' in instr:
            self.combo_sensor.setCurrentText('Landsat 4 and 5 (TM)')
        elif 'WorldView' in instr and 'Multi' in instr:
            self.combo_sensor.setCurrentText('WorldView')
        elif 'Sentinel-2' in instr:
            self.combo_sensor.setCurrentText('Sentinel-2')
        else:
            self.combo_sensor.setCurrentText('Unknown')

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

        if not rlist1:
            self.showlog('You need to select a condition index to '
                         'calculate.')
            return False

        rlist2 = []
        if 'VCI' in rlist1 and 'EVI' in index:
            rlist2 += [r'2.5*(B3-B2)/(B3+6.0*B2-7.5*B0+1) EVI']
        elif 'VCI' in rlist1 and 'NDVI' in index:
            rlist2 += [r'(B3-B2)/(B3+B2) NDVI']
        elif 'VCI' in rlist1 and 'MSAVI2' in index:
            rlist2 += [r'0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2']

        evi = []
        tci = []
        vci = []
        vhi = []
        lst = []

        if 'RasterFileList' in self.indata:
            flist = self.indata['RasterFileList']
            if sensor == 'ASTER':
                flist = get_aster_list(flist)
            elif 'Landsat' in sensor:
                flist = get_landsat_list(flist, sensor)
            elif 'Sentinel-2' in sensor:
                flist = get_sentinel_list(flist)
            if not flist:
                self.showlog('Warning: This might not be ' + sensor +
                             ' data. Will attempt to do calculation '
                             'anyway.')
                flist = self.indata['RasterFileList']
        else:
            flist = [self.indata['Raster']]

        for ifile in flist:
            dat = get_from_rastermeta(ifile, piter=self.piter,
                                      showlog=self.showlog)

            if dat is None:
                continue

            ofile = dat[0].filename

            # Prepare for layer stacking
            if sensor == 'WorldView':
                wvlabels = {'CoastalBlue': 'B1',
                            'Blue': 'B2',
                            'Green': 'B3',
                            'Yellow': 'B4',
                            'Red': 'B5',
                            'RedEdge': 'B6',
                            'NIR1': 'B7',
                            'NIR2': 'B8'}
                for i in dat:
                    if i.dataid.split()[0] in wvlabels:
                        i.dataid = wvlabels[i.dataid.split()[0]]

            bfile = os.path.basename(ifile)
            rlist = correct_bands(rlist2, sensor, bfile)

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

                formula = ','.join(rlist)
                formula = re.sub(r'B(\d+)', r'Band\1', formula)

                if txt in formula:
                    datsml.append(i)

            dat = lstack(datsml, piter=self.piter, showlog=self.showlog)

            del datsml

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

                datd[txt] = i.data

                if 'LST' in txt:
                    lst.append(i)

            # Calculate ratios
            for i in self.piter(rlist):
                self.showlog('Calculating '+i)
                formula = i.split(' ')[0]
                formula = re.sub(r'B(\d+)', r'Band\1', formula)
                blist = formula
                for j in ['/', '*', '+', '-', '(', ')']:
                    blist = blist.replace(j, ' ')
                blist = blist.split()
                blist = list(set(blist))
                blist = [i for i in blist if 'Band' in i]

                abort = []
                for j in blist:
                    if 'B' not in j:
                        continue
                    if j not in datd:
                        abort.append(j)
                if abort:
                    self.showlog('Error: '+' '.join(abort)+' missing.')
                    continue

                newmask = datd[blist[0]].mask
                for j in blist:
                    newmask = (newmask | datd[j].mask)

                if len(formula.split(r'/')) == 2:
                    f1, f2 = formula.split(r'/')
                    a1 = ne.evaluate(f1, datd)
                    a2 = ne.evaluate(f2, datd)

                    a2[np.isclose(a2, 0.)] = 0.
                    ratio = a1/a2
                else:
                    ratio = ne.evaluate(formula, datd)

                newmask = newmask | (ratio < -1) | (ratio > 1)
                ratio = ratio.astype(np.float32)
                ratio[newmask] = 1e+20
                ratio = np.ma.array(ratio, mask=newmask,
                                    fill_value=1e+20)

                ratio = np.ma.fix_invalid(ratio)

                tmp = copy.deepcopy(dat[0])
                tmp.data = ratio
                tmp.nodata = 1e+20
                evi.append(tmp)

        if lst:
            lst = lstack(lst, piter=self.piter, showlog=self.showlog,
                         commonmask=True)
        if evi:
            evi = lstack(evi, piter=self.piter, showlog=self.showlog,
                         commonmask=True)

        ofile = ''
        if ('TCI' in rlist1 or 'VHI' in rlist1) and lst:
            tci = get_TCI(lst)
            ofile += '_TCI'
        if ('VCI' in rlist1 or 'VHI' in rlist1) and evi:
            vci = get_VCI(evi, index)
            ofile += '_VCI_'+index
        if 'VHI' in rlist1 and tci and vci:
            vhi = get_VHI(tci, vci)
            ofile += '_VHI'

        datfin = tci+vci+vhi

        for i in datfin:
            i.data = i.data.astype(np.float32)

        if datfin:
            self.outdata['Raster'] = datfin

        return True

    def setratios(self):
        """
        Set the available indices.

        Returns
        -------
        None.

        """
        sensor = self.combo_sensor.currentText()
        rlist = []

        if 'Unknown' not in sensor:
            rlist += ['VCI']

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
            item.setSelected(not item.isSelected())

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

        if currentitem.text()[2:] == 'VHI' and currentitem.isSelected():
            for i in range(self.lw_ratios.count()):
                self.lw_ratios.item(i).setSelected(currentitem.isSelected())
        elif not currentitem.isSelected() and 'VHI' in idict:
            self.lw_ratios.item(idict['VHI']).setSelected(False)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            if item.isSelected():
                item.setText('\u2713' + item.text()[1:])
            else:
                item.setText(' ' + item.text()[1:])


def calc_ratios(dat, rlist, showlog=print, piter=iter):
    """
    Calculate Band ratios.

    Note that this routine assumes that the ratio you supply is correct for
    your data.

    Parameters
    ----------
    dat : list
        List of PyGMI Data.
    rlist : list
        List of strings, containing ratios to calculate..
    showlog : print, optional
        Display information. The default is print.
    piter : iter, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    datfin : list
        List of PyGMI Data.

    """
    datsml = []

    for i in dat:
        tmp = i.dataid.split()
        txt = tmp[0]

        if 'Band' not in txt and 'B' in txt:
            txt = txt.replace('B', 'Band')

        if 'Band' not in txt and 'LST' not in txt:
            continue

        formula = ','.join(rlist)
        formula = re.sub(r'B(\d+)', r'Band\1', formula)

        if txt == 'Band3N':
            txt = 'Band3'

        if txt in formula:
            datsml.append(i)

    dat = lstack(datsml, piter=piter, showlog=showlog)

    del datsml

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

        datd[txt] = i.data

    datfin = []
    for i in piter(rlist):
        showlog('Calculating '+i)
        if 'Landslide Index' in i:
            rband = landslide_index(dat, showlog, piter)
            datfin += rband
            continue

        formula = i.split(' ')[0]
        formula = re.sub(r'B(\d+)', r'Band\1', formula)
        blist = formula
        for j in ['/', '*', '+', '-', '(', ')']:
            blist = blist.replace(j, ' ')
        blist = blist.split()
        blist = list(set(blist))
        blist = [i for i in blist if 'Band' in i]

        abort = []
        for j in blist:
            if 'B' not in j:
                continue
            if j not in datd:
                abort.append(j)
        if abort:
            showlog('Error: '+' '.join(abort)+' missing.')
            continue

        newmask = datd[blist[0]].mask
        for j in blist:
            newmask = (newmask | datd[j].mask)

        if len(formula.split(r'/')) == 2:
            f1, f2 = formula.split(r'/')
            a1 = ne.evaluate(f1, datd)
            a1 = a1.astype(np.float32)
            a2 = ne.evaluate(f2, datd)
            a2 = a2.astype(np.float32)

            a2[np.isclose(a2, 0.)] = 0.
            ratio = a1/a2

            del a1
            del a2
        else:
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

    return datfin


def correct_bands(rlist, sensor, bfile=None):
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
    # custom_indices = ['Landslide Index']

    sdict = {}

    sdict['ASTER'] = {'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'B4': 'B4',
                      'B3A': 'B3',
                      'B5': 'B5', 'B6': 'B6', 'B7': 'B7', 'B8': 'B8',
                      'B9': 'B9', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12',
                      'B13': 'B13', 'B14': 'B14', 'B3A': 'B3'}
    sdict['Landsat 8 and 9 (OLI)'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4',
                                      'B3': 'B5', 'B4': 'B6', 'B5': 'B7',
                                      'B3A': 'B5'}
    sdict['Landsat 7 (ETM+)'] = {'B0': 'B1', 'B1': 'B2', 'B2': 'B3',
                                 'B3': 'B4', 'B4': 'B5', 'B5': 'B7',
                                 'B3A': 'B4'}
    sdict['Landsat 4 and 5 (TM)'] = sdict['Landsat 7 (ETM+)']
    sdict['Sentinel-2'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B4', 'B3': 'B8',
                           'B4': 'B11', 'B5': 'B12', 'B3A': 'B8A'}
    sdict['WorldView'] = {'B0': 'B2', 'B1': 'B3', 'B2': 'B5', 'B3': 'B7',
                          'B3A': 'B7'}
    sdict['Unknown'] = {}

    if sensor == 'Landsat (All)':
        if 'LC09' in bfile or 'LC08' in bfile:
            sensor = 'Landsat 8 and 9 (OLI)'
        elif 'LE07' in bfile:
            sensor = 'Landsat 7 (ETM+)'
        else:
            sensor = 'Landsat 4 and 5 (TM)'

    bandmap = sdict[sensor]
    # Sort the keys so we do long names like B3A first
    svalues = set(sorted(bandmap.keys(), key=lambda el: len(el))[::-1])
    rlist2 = []
    for i in rlist:
        formula = i.split(' ')[0]
        lbl = i[i.index(' '):]
        bands = set(re.findall(r'B\d+\w?', formula))
        if bands.issubset(svalues):
            tmp = re.sub(r'B(\d+\w?)', r'tmpB\1', formula)
            for j in svalues:
                tmp = tmp.replace('tmp'+j, bandmap[j])

            # lbl = lbl.strip()
            # if lbl in custom_indices:
            #     tmp = ''
            # else:
            #     tmp += ' '
            rlist2.append(tmp+lbl)

    return rlist2


def get_aster_list(flist):
    """
    Get ASTER files from a file list.

    Parameters
    ----------
    flist : list
        List of filenames.

    Returns
    -------
    flist : list
        List of filenames.

    """
    flist2 = []
    for i in flist:
        if 'ASTER' not in i.sensor:
            continue
        flist2.append(i)

    return flist2


def get_landsat_list(flist, sensor=None, allsats=False):
    """
    Get Landsat files from a file list.

    Parameters
    ----------
    flist : list
        List of filenames.

    Returns
    -------
    flist : list
        List of filenames.

    """
    if isinstance(flist[0], list):
        bfile = os.path.basename(flist[0][0].filename)
        if bfile[:4] in ['LT04', 'LT05', 'LE07', 'LC08', 'LC09']:
            return flist
        return []

    if allsats is True or sensor is None:
        fid = ['LT04', 'LT05', 'LE07', 'LC08', 'LC09']
    elif sensor == 'Landsat 8 and 9 (OLI)':
        fid = ['LC08', 'LC09']
    elif sensor == 'Landsat 7 (ETM+)':
        fid = ['LE07']
    elif sensor == 'Landsat 4 and 5 (TM)':
        fid = ['LT04', 'LT05']
    else:
        return None

    flist2 = []
    for i in flist:
        for j in fid:
            if j not in i.sensor:
                continue
            if '.tif' in i.filename:
                continue
            flist2.append(i)

    return flist2


def get_sentinel_list(flist):
    """
    Get Sentinel-2 files from a file list.

    Parameters
    ----------
    flist : list
        List of filenames.

    Returns
    -------
    flist : list
        List of filenames.

    """
    flist2 = []
    for i in flist:
        if 'Sentinel-2' not in i.sensor:
            continue
        flist2.append(i)

    return flist2


def get_TCI(lst):
    """
    Calculate TCI.

    Parameters
    ----------
    lst : list
        list of PyGMI datasets - land surface temperatures.

    Returns
    -------
    tci : list
        output TCI datasets.

    """
    tci = []
    lst2 = []

    for j in lst:
        lst2.append(j.data)
    lst2 = np.ma.array(lst2)

    lstmax = lst2.max(0)
    lstmin = lst2.min(0)

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
    evi : list
        list of EVI datasets.
    index : str
        index for dataid.

    Returns
    -------
    vci : list
        output VCI datasets.

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
    tci : list
        TCI dataset list.
    vci : list
        VCI dataset list.
    alpha : float, optional
        Weight for proportion of TCI and VCI. The default is 0.5.

    Returns
    -------
    vhi : list
        Output VHI datasets.

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


def landslide_index(dat, showlog=print, piter=iter):
    """
    Calculate Band ratios.

    Note that this routine assumes that the ratio you supply is correct for
    your data.

    Parameters
    ----------
    dat : list
        List of PyGMI Data.
    showlog : print, optional
        Display information. The default is print.
    piter : iter, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    datfin : list
        Red, green and blue PyGMI Data.

    """
    rlist = [r'(B3-B2)/(B3+B2) NDVI',
             r'(B1-B3)/(B1+B3) NDWI water bodies',
             r'B4 SWIR',
             r'((B4+B2)-(B3+B0))/((B4+B2)+(B3+B0)) BSI']

    sensor = dat[0].metadata['Raster']['Sensor']
    rlist = correct_bands(rlist, sensor)

    datfin = calc_ratios(dat, rlist, showlog=showlog, piter=piter)

    for i in datfin:
        if 'NDVI' in i.dataid:
            NDVI = i.data
        elif 'NDWI' in i.dataid:
            NDWI = i.data
        elif 'SWIR' in i.dataid:
            SWIR = i.data
        elif 'BSI' in i.dataid:
            BSI = i.data

    red = copy.deepcopy(dat[0])
    green = copy.deepcopy(dat[0])
    blue = copy.deepcopy(dat[0])

    red.data[:] = 3.5*BSI
    green.data[:] = 0.3
    blue.data[:] = 0.

    filt = ((SWIR > 0.8) | (NDVI < 0.15))
    red.data[filt] = 1.5
    green.data[filt] = 0.7
    blue.data[filt] = -1.

    filt = (NDVI > 0.25)
    red.data[filt] = 0.
    green.data[filt] = 0.2*NDVI[filt]
    blue.data[filt] = 0.

    filt = (NDWI > 0.15)
    red.data[filt] = 0.
    green.data[filt] = 0.2
    blue.data[filt] = NDWI[filt]

    red.dataid = 'Landslide Index Red'
    green.dataid = 'Landslide Index Green'
    blue.dataid = 'Landslide Index Blue'

    red.data = np.ma.masked_equal(red.data.filled(1e+20), 1e+20)
    red.nodata = 1e+20

    green.data = np.ma.masked_equal(green.data.filled(1e+20), 1e+20)
    green.nodata = 1e+20

    blue.data = np.ma.masked_equal(blue.data.filled(1e+20), 1e+20)
    blue.nodata = 1e+20

    return [red, green, blue]


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt
    import winsound
    from pygmi.rsense.iodefs import ImportBatch

    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Sentinel-2\S2A_MSIL2A_20210305T075811_N0214_R035_T35JML_20210305T103519.zip"
    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Landsat\LC081740432017101901T1-SC20180409064853.tar.gz"
    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\wv2\014568829030_01_P001_MUL\16MAY28083210-M3DS-014568829030_01_P001.XML"
    ifile = r"D:\Workdata\PyGMI Test Data\Remote Sensing\Import\ASTER\new\AST_07XT_00308302021082202_20230215122255_9222.zip"

    idir = r'd:\sentinel2'
    os.chdir(r'D:\\')

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportBatch()
    tmp1.idir = idir
    tmp1.get_sfile(True)
    tmp1.settings()

    tmp1.outdata['RasterFileList'] = [tmp1.outdata['RasterFileList'][0]]

    SR = SatRatios()
    SR.indata = tmp1.outdata
    SR.settings()

    dat2 = SR.outdata['Raster']
    for i in dat2:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        vmin = i.data.mean()-2*i.data.std()
        vmax = i.data.mean()+2*i.data.std()
        plt.imshow(i.data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

    dat = [i.data for i in dat2]
    dat = np.moveaxis(dat, 0, -1)

    plt.figure(dpi=200)
    plt.imshow(dat, vmin=0, vmax=1)
    plt.show()

    plt.figure(dpi=200)
    plt.imshow(dat)
    plt.show()

    winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)


def _testfn2():
    """Test routine."""
    import glob
    import matplotlib.pyplot as plt

    ifiles = glob.glob(r'D:\Workdata\PyGMI Test Data\Remote Sensing\ConditionIndex\*.tar')

    app = QtWidgets.QApplication(sys.argv)

    SR = ConditionIndices()
    SR.indata['RasterFileList'] = ifiles
    SR.settings()

    dat = SR.outdata["Raster"]

    for i in dat:
        plt.figure(dpi=200)
        plt.imshow(i.data, extent=i.extent)
        plt.colorbar()
        plt.title(i.dataid)
        plt.show()


def _testfn3():
    """Test Function."""
    import matplotlib.pyplot as plt

    ifile = r"C:/Workdata/Remote Sensing/Sentinel-2/S2A_MSIL2A_20210305T075811_N0214_R035_T35JML_20210305T103519.zip"

    dat = iodefs.get_data(ifile)

    app = QtWidgets.QApplication(sys.argv)

    SR = SatRatios()
    SR.indata['Raster'] = dat  # single file only

    SR.settings()

    plt.title(dat[2].dataid)
    plt.imshow(dat[2].data)
    plt.colorbar()
    plt.show()

    plt.title(dat[0].dataid)
    plt.imshow(dat[0].data)
    plt.colorbar()
    plt.show()

    plt.title(dat[3].dataid)
    plt.imshow(dat[3].data)
    plt.colorbar()
    plt.show()

    dat2 = SR.outdata['Raster']

    plt.title(dat2[0].dataid)
    plt.imshow(dat2[0].data, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


def _testfn4():
    """Test routine."""
    import glob
    import matplotlib.pyplot as plt
    from pygmi.raster.dataprep import DataMerge

    ifiles = glob.glob(r"C:\WorkProjects\ratios\*.zip")

    app = QtWidgets.QApplication(sys.argv)

    ifiles = glob.glob(r"C:\WorkProjects\ratios\*.tif")

    dat = []
    for ifile in ifiles:
        dat += iodefs.get_data(ifile)

    DM = DataMerge()
    DM.indata['Raster'] = dat
    DM.settings()

    dat += DM.outdata['Raster']

    vmin = dat[0].data.mean() - dat[0].data.std()
    vmax = dat[0].data.mean() + dat[0].data.std()

    for i in dat:
        plt.title(i.dataid)
        plt.imshow(i.data, extent=i.extent, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    _testfn()
