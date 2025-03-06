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
"""Import borehole data, currently supports Council for Geoscience data."""

import os
from PyQt5 import QtWidgets
import pandas as pd

from pygmi.misc import BasicModule


class ImportData(BasicModule):
    """
    Import borehole data.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.hfile = ''
        self.is_import = True

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
        filt = ''
        if not nodialog:
            ext = ('Common formats (*.xls *.xlsx *.csv);;'
                   'Excel (*.xls *.xlsx);;'
                   'Comma Delimited (*.csv)')

            filename, filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open CGS Lithology File', '.', ext)
            if filename == '':
                return False
            os.chdir(os.path.dirname(filename))
            self.ifile = str(filename)

            filename, filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open CGS Header File', '.', ext)
            if filename == '':
                return False

            self.hfile = str(filename)

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

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)
        self.saveobj(self.hfile)


def get_CGS(lithfile, headerfile):
    """
    Borehole Import.

    Parameters
    ----------
    lithfile : str
        Filename to import.
    headerfile : str
        Filename to import.

    Returns
    -------
    dat : dictionary
        Dictionary of Pandas dataframes.

    """
    xl = pd.ExcelFile(lithfile)
    df = xl.parse(xl.sheet_names[0])
    xl.close()
    df = df.dropna(subset=['Depth from', 'Depth to'])

    xl = pd.ExcelFile(headerfile)
    hdf = xl.parse(xl.sheet_names[0])
    xl.close()

    dat = {}
    for i in hdf['Boreholeid']:
        blog = df[df['Boreholeid'] == i]
        bhead = hdf[hdf['Boreholeid'] == i]
        dat[str(i)] = {'log': blog, 'header': bhead}

    return dat
