# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""Import Borehole Data."""

import os
from PyQt5 import QtWidgets
import pandas as pd


class ImportData():
    """
    Import Data.

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
    ext : str
        filename extension
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.name = 'Import Data: '
        self.ext = ''
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.hfile = ''

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = ('Common formats (*.xls *.xlsx *.csv);;'
               'Excel (*.xls *.xlsx);;'
               'Comma Delimited (*.csv)')

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open CGS Lithology File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))
        self.ifile = str(filename)
        self.ext = filename[-3:]
        self.ext = self.ext.lower()

        filename, filt = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open CGS Header File', '.', ext)
        if filename == '':
            return False

        self.hfile = str(filename)
        self.ext = filename[-3:]
        self.ext = self.ext.lower()

        dat = get_CGS(self.ifile, self.hfile)

        if dat is None:
            if 'CGS' in filt:
                QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                              'Could not import dataset. '
                                              'Please make sure it not '
                                              'another format.',
                                              QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'Borehole'

        self.outdata[output_type] = dat
        return True


def get_CGS(lithfile, headerfile):
    """
    Borehole Import.

    Parameters
    ----------
    lithfile : str
        filename to import
    headerfile : str
        filename to import

    Returns
    -------
    dat : dictionary
        dictionary of Pandas dataframes

    """
    xl = pd.ExcelFile(lithfile)
    df = xl.parse(xl.sheet_names[0])
    xl.close()

    xl = pd.ExcelFile(headerfile)
    hdf = xl.parse(xl.sheet_names[0])
    xl.close()

    dat = {}
    for i in hdf['Boreholeid']:
        blog = df[df['Boreholeid'] == i]
        bhead = hdf[hdf['Boreholeid'] == i]
        dat[str(i)] = {'log': blog, 'header': bhead}

    return dat
