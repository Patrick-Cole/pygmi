# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
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

import os
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
from pygmi.vector.datatypes import LData
import pygmi.menu_default as menu_default


class ImportCG5(QtWidgets.QDialog):
    """
    Import Line Data.

    This class imports ASCII point data.

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.name = 'Import CG-5 Data: '
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

        self.df_cg5 = None
        self.df_gps = None

        self.line = QtWidgets.QComboBox()
        self.station = QtWidgets.QComboBox()
        self.xchan = QtWidgets.QComboBox()
        self.ychan = QtWidgets.QComboBox()
        self.zchan = QtWidgets.QComboBox()
        self.nodata = QtWidgets.QLineEdit('-99999')
        self.cg5file = QtWidgets.QLineEdit('')
        self.gpsfile = QtWidgets.QLineEdit('')
        self.basethres = QtWidgets.QLineEdit('10000')

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
        helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        label_line = QtWidgets.QLabel('Line:')
        label_station = QtWidgets.QLabel('Station:')
        label_xchan = QtWidgets.QLabel('Longitude:')
        label_ychan = QtWidgets.QLabel('Latitude:')
        label_zchan = QtWidgets.QLabel('Ellipsoid (GPS) Elevation:')
        label_bthres = QtWidgets.QLabel('Minimum Base Station Number:')
        pb_cg5 = QtWidgets.QPushButton('Load CG-5 File')
        pb_gps = QtWidgets.QPushButton('Load GPS File')

        self.line.setEnabled(False)
        self.station.setEnabled(False)
        self.xchan.setEnabled(False)
        self.ychan.setEnabled(False)
        self.zchan.setEnabled(False)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Import CG-5 Data')

        gridlayout_main.addWidget(self.cg5file, 0, 0, 1, 1)
        gridlayout_main.addWidget(pb_cg5, 0, 1, 1, 1)

        gridlayout_main.addWidget(self.gpsfile, 1, 0, 1, 1)
        gridlayout_main.addWidget(pb_gps, 1, 1, 1, 1)

        gridlayout_main.addWidget(label_line, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.line, 2, 1, 1, 1)

        gridlayout_main.addWidget(label_station, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.station, 3, 1, 1, 1)

        gridlayout_main.addWidget(label_xchan, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.xchan, 4, 1, 1, 1)

        gridlayout_main.addWidget(label_ychan, 5, 0, 1, 1)
        gridlayout_main.addWidget(self.ychan, 5, 1, 1, 1)

        gridlayout_main.addWidget(label_zchan, 6, 0, 1, 1)
        gridlayout_main.addWidget(self.zchan, 6, 1, 1, 1)

        gridlayout_main.addWidget(label_bthres, 7, 0, 1, 1)
        gridlayout_main.addWidget(self.basethres, 7, 1, 1, 1)

#        gridlayout_main.addWidget(label_nodata, 2, 0, 1, 1)
#        gridlayout_main.addWidget(self.nodata, 2, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 8, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 8, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_cg5.pressed.connect(self.get_cg5)
        pb_gps.pressed.connect(self.get_gps)

    def settings(self, test=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not test:
            tmp = self.exec_()

            if tmp != 1 or self.df_cg5 is None or self.df_gps is None:
                return tmp

        # Rename columns
        cren = {}
        cren[self.line.currentText()] = 'line'
        cren[self.station.currentText()] = 'station'
        self.df_gps.rename(columns=cren)

        # Get rid of text in line columns
        self.df_gps['line'] = self.df_gps['line'].str.replace(r'\D', '')

        # Convert line and station to numbers
        self.df_gps['station'] = pd.to_numeric(self.df_gps['station'],
                                               errors='coerce',
                                               downcast='float')

        self.df_gps['line'] = pd.to_numeric(self.df_gps['line'],
                                            errors='coerce',
                                            downcast='float')

        # Merge data
        dfmerge = pd.merge(self.df_cg5, self.df_gps,
                           left_on=['LINE', 'STATION'],
                           right_on=['line', 'station'], how='left')

        filt = dfmerge['STATION'] < float(self.basethres.text())
        filt = filt & dfmerge['longitude'].isna()

        dfmerge = dfmerge[~filt]

        dat = {}
        lines = dfmerge.LINE.unique()
        for line in lines:
            tmp = dfmerge.loc[dfmerge['LINE'] == line]
            dat[str(line)] = tmp.to_records(index=False)

        dat2 = LData()
        dat2.xchannel = self.xchan.currentText()
        dat2.ychannel = self.ychan.currentText()
        dat2.zchannel = self.zchan.currentText()
        dat2.data = dat
        dat2.dataid = 'Gravity'
#        dat2.nullvalue = nodata

        self.outdata['Line'] = dat2

        return True

    def get_cg5(self, filename=''):
        """
        Get CG-5 filename.

        Parameters
        ----------
        filename : str, optional
            CG-5 filename submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        ext = ('CG-5 ASCII (*.txt *.xyz)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        with open(filename) as fno:
            tmp = fno.readlines()

        data = []
        for i in tmp:
            if i[0] != r'/' and 'Line' not in i and ',' not in i:
                data.append(i)

        names = ['LINE', 'STATION', 'ALT', 'GRAV', 'SD', 'TILTX', 'TILTY',
                 'TEMP', 'TIDE', 'DUR', 'REJ', 'TIME', 'DECTIMEDATE',
                 'TERRAIN']

        dtype = {}
        dtype['names'] = names
        dtype['formats'] = ['f4']*len(names)

        dtype['formats'][9] = 'i'
        dtype['formats'][10] = 'i'
        dtype['formats'][11] = 'S8'

        tmp2 = np.genfromtxt(data, dtype=dtype)

        self.df_cg5 = pd.DataFrame(tmp2)
        self.cg5file.setText(filename)

    def get_gps(self, filename=''):
        """
        Get GPS filename.

        Parameters
        ----------
        filename : str, optional
            GPS filename (csv). The default is ''.

        Returns
        -------
        None.

        """
        ext = ('GPS comma delimited (*.csv)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        df2 = pd.read_csv(filename)
#        df2.columns = map(str.lower, df2.columns)
        df2.columns = df2.columns.str.lower()

        self.df_gps = df2

        self.gpsfile.setText(filename)

        ltmp = list(df2.columns)

        xind = 0
        yind = 1
        zind = 2
        lind = 0
        sind = 0
        for i, tmp in enumerate(ltmp):
            if 'lon' in tmp.lower():
                xind = i
            elif 'lat' in tmp.lower():
                yind = i
            elif 'elev' in tmp.lower() or 'alt' in tmp.lower() or 'height' in tmp.lower():
                zind = i
            elif 'stat' in tmp.lower():
                sind = i
            elif 'line' in tmp.lower():
                lind = i

        self.line.addItems(ltmp)
        self.station.addItems(ltmp)
        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)
        self.zchan.addItems(ltmp)

        self.line.setCurrentIndex(lind)
        self.station.setCurrentIndex(sind)
        self.xchan.setCurrentIndex(xind)
        self.ychan.setCurrentIndex(yind)
        self.zchan.setCurrentIndex(zind)

        self.line.setEnabled(True)
        self.station.setEnabled(True)
        self.xchan.setEnabled(True)
        self.ychan.setEnabled(True)
        self.zchan.setEnabled(True)
