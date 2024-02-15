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
"""Import Gravity Data."""

import os
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import geopandas as gpd

from pygmi import menu_default
from pygmi.misc import BasicModule


class ImportCG5(BasicModule):
    """
    Import CG-5 data.

    This class imports CG-5 gravimeter data with associated GPS data.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.df_cg5 = None
        self.df_gps = None
        self.is_import = True

        self.cmb_line = QtWidgets.QComboBox()
        self.cmb_station = QtWidgets.QComboBox()
        self.cmb_xchan = QtWidgets.QComboBox()
        self.cmb_ychan = QtWidgets.QComboBox()
        self.cmb_zchan = QtWidgets.QComboBox()
        self.le_cg5file = QtWidgets.QLineEdit('')
        self.le_gpsfile = QtWidgets.QLineEdit('')
        self.le_basethres = QtWidgets.QLineEdit('10000')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        lbl_line = QtWidgets.QLabel('Line:')
        lbl_station = QtWidgets.QLabel('Station:')
        lbl_xchan = QtWidgets.QLabel('Longitude:')
        lbl_ychan = QtWidgets.QLabel('Latitude:')
        lbl_zchan = QtWidgets.QLabel('Ellipsoid (GPS) Elevation:')
        lbl_bthres = QtWidgets.QLabel('Minimum Base Station Number:')
        pb_cg5 = QtWidgets.QPushButton('Load CG-5 File')
        pb_gps = QtWidgets.QPushButton('Load GPS File')

        pixmapi = QtWidgets.QStyle.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_gps.setIcon(icon)
        pb_cg5.setIcon(icon)
        pb_gps.setStyleSheet('text-align:left;')
        pb_cg5.setStyleSheet('text-align:left;')

        self.cmb_line.setEnabled(False)
        self.cmb_station.setEnabled(False)
        self.cmb_xchan.setEnabled(False)
        self.cmb_ychan.setEnabled(False)
        self.cmb_zchan.setEnabled(False)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Import CG-5 Data')

        gl_main.addWidget(self.le_cg5file, 0, 1, 1, 1)
        gl_main.addWidget(pb_cg5, 0, 0, 1, 1)

        gl_main.addWidget(self.le_gpsfile, 1, 1, 1, 1)
        gl_main.addWidget(pb_gps, 1, 0, 1, 1)

        gl_main.addWidget(lbl_line, 2, 0, 1, 1)
        gl_main.addWidget(self.cmb_line, 2, 1, 1, 1)

        gl_main.addWidget(lbl_station, 3, 0, 1, 1)
        gl_main.addWidget(self.cmb_station, 3, 1, 1, 1)

        gl_main.addWidget(lbl_xchan, 4, 0, 1, 1)
        gl_main.addWidget(self.cmb_xchan, 4, 1, 1, 1)

        gl_main.addWidget(lbl_ychan, 5, 0, 1, 1)
        gl_main.addWidget(self.cmb_ychan, 5, 1, 1, 1)

        gl_main.addWidget(lbl_zchan, 6, 0, 1, 1)
        gl_main.addWidget(self.cmb_zchan, 6, 1, 1, 1)

        gl_main.addWidget(lbl_bthres, 7, 0, 1, 1)
        gl_main.addWidget(self.le_basethres, 7, 1, 1, 1)

        gl_main.addWidget(helpdocs, 8, 0, 1, 1)
        gl_main.addWidget(buttonbox, 8, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_cg5.pressed.connect(self.get_cg5)
        pb_gps.pressed.connect(self.get_gps)

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
        if not nodialog:
            tmp = self.exec()

            if tmp != 1 or self.df_cg5 is None or self.df_gps is None:
                return False

        if self.cmb_line.currentText() == self.cmb_station.currentText():
            self.showlog('Your line column cannot be the same as your '
                         'station column')
            return False

        tmp = [self.cmb_line.currentText(),
               self.cmb_station.currentText(),
               self.cmb_xchan.currentText(),
               self.cmb_ychan.currentText(),
               self.cmb_zchan.currentText()]

        if len(set(tmp)) != len(tmp):
            self.showlog('Unable to import, two of your GPS file '
                         'columns are the same. Make sure you have a '
                         'line column in your GPS file, and that you '
                         'did not specify the same column twice.')
            return False

        # Rename columns
        cren = {}
        cren[self.cmb_line.currentText()] = 'line'
        cren[self.cmb_station.currentText()] = 'station'
        cren[self.cmb_xchan.currentText()] = 'longitude'
        cren[self.cmb_ychan.currentText()] = 'latitude'
        cren[self.cmb_zchan.currentText()] = 'elevation'

        self.df_gps.rename(columns=cren, inplace=True)

        if self.df_gps.latitude.dtype == 'O':
            filt = self.df_gps.latitude.str.contains('S')
            self.df_gps.latitude.loc[filt] = '-'+self.df_gps.latitude[filt]
            self.df_gps.latitude.replace('S', '', inplace=True, regex=True)
            self.df_gps.latitude.replace('N', '', inplace=True, regex=True)
            try:
                self.df_gps.latitude = pd.to_numeric(self.df_gps.latitude)
            except ValueError:
                self.showlog('You have characters in your latitude'
                             ' string which could not be converted.')
                return False

        if self.df_gps.longitude.dtype == 'O':
            filt = self.df_gps.longitude.str.contains('W')
            self.df_gps.longitude.loc[filt] = '-'+self.df_gps.longitude[filt]
            self.df_gps.longitude.replace('W', '', inplace=True, regex=True)
            self.df_gps.longitude.replace('E', '', inplace=True, regex=True)
            try:
                self.df_gps.longitude = pd.to_numeric(self.df_gps.longitude)
            except ValueError:
                self.showlog('You have characters in your longitude'
                             ' string which could not be converted.')
                return False

        # Get rid of text in line columns
        if self.df_gps['line'].dtype == object:
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

        # eliminate ordinary stations (not base stations) without coordinates
        filt = dfmerge['STATION'] < float(self.le_basethres.text())

        filt = filt & dfmerge['longitude'].isna()

        dfmerge = dfmerge[~filt]

        x = dfmerge['longitude']
        y = dfmerge['latitude']
        dfmerge = gpd.GeoDataFrame(dfmerge, geometry=gpd.points_from_xy(x, y))

        dfmerge['line'] = dfmerge['line'].astype(str)
        dfmerge.attrs['Gravity'] = True
        dfmerge.attrs['source'] = str(self.le_cg5file.text())
        self.outdata['Vector'] = [dfmerge]

        # Check for duplicates
        dtest = dfmerge.duplicated(['LINE', 'STATION'])
        dlist = dfmerge[['LINE', 'STATION']].loc[dtest]
        dlist = dlist[~dlist.duplicated()]
        dlist = dlist[dlist.STATION < float(self.le_basethres.text())]

        if dlist.size > 0:
            self.showlog('Warning, the following are duplicated:')
            self.showlog(dlist.to_string(index=False))

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_line)
        self.saveobj(self.cmb_station)
        self.saveobj(self.cmb_xchan)
        self.saveobj(self.cmb_ychan)
        self.saveobj(self.cmb_zchan)

        self.saveobj(self.le_cg5file)
        self.saveobj(self.le_gpsfile)
        self.saveobj(self.le_basethres)

    def get_cg5(self, filename=''):
        """
        Get CG-5 filename and load data.

        Parameters
        ----------
        filename : str, optional
            CG-5 filename submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        ext = 'CG-5 ASCII (*.txt *.xyz)'

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        with open(filename, encoding='utf-8') as fno:
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
        self.le_cg5file.setText(filename)

    def get_gps(self, filename=''):
        """
        Get GPS filename and load data.

        Parameters
        ----------
        filename : str, optional
            GPS filename (csv). The default is ''.

        Returns
        -------
        None.

        """
        ext = 'GPS comma delimited (*.csv)'

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        df2 = pd.read_csv(filename)
        df2.columns = df2.columns.str.lower()

        self.df_gps = df2

        self.le_gpsfile.setText(filename)

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
            elif ('elev' in tmp.lower() or 'alt' in tmp.lower() or
                  'height' in tmp.lower() or tmp.lower() == 'z'):
                zind = i
            elif 'stat' in tmp.lower():
                sind = i
            elif 'line' in tmp.lower():
                lind = i

        self.cmb_line.addItems(ltmp)
        self.cmb_station.addItems(ltmp)
        self.cmb_xchan.addItems(ltmp)
        self.cmb_ychan.addItems(ltmp)
        self.cmb_zchan.addItems(ltmp)

        self.cmb_line.setCurrentIndex(lind)
        self.cmb_station.setCurrentIndex(sind)
        self.cmb_xchan.setCurrentIndex(xind)
        self.cmb_ychan.setCurrentIndex(yind)
        self.cmb_zchan.setCurrentIndex(zind)

        self.cmb_line.setEnabled(True)
        self.cmb_station.setEnabled(True)
        self.cmb_xchan.setEnabled(True)
        self.cmb_ychan.setEnabled(True)
        self.cmb_zchan.setEnabled(True)
