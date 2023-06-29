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
"""Import and export EDI data."""

import os
from PyQt5 import QtWidgets
import mtpy.core.mt
import numpy as np

from pygmi.misc import ContextModule, BasicModule

# The lines below are a temporary fix for mtpy. Removed in future.
np.float = float
np.complex = complex


class ImportEDI(BasicModule):
    """
    Import Data.

    Attributes
    ----------
    ifilelist : list
        list of input file names.
    """

    def __init__(self, parent=None):
        super().__init__(parent
                         )
        self.ifilelist = []

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
            ext = 'EDI (*.edi)'

            self.ifilelist, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self.parent, 'Open EDI Files (single or multiple)', '.', ext)
            if not self.ifilelist:
                return False

        os.chdir(os.path.dirname(self.ifilelist[0]))

        dat = get_EDI(self.ifilelist)

        self.ifile = os.path.dirname(self.ifilelist[0]) + r'\EDI List'

        if dat is None:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import dataset. ',
                                          QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'MT - EDI'

        self.outdata[output_type] = dat
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
        self.ifilelist = projdata['ifilelist']

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifilelist'] = self.ifilelist

        return projdata


def get_EDI(ifiles):
    """
    EDI Import.

    Parameters
    ----------
    ifiles : list
        filenames to import

    Returns
    -------
    dat : EDI data.
        Dataset imported
    """
    dat = {}

    for edi_file in ifiles:
        mt_obj = mtpy.core.mt.MT(edi_file)

        bname = os.path.basename(edi_file)
        bname = bname[:-4]

        dat[bname] = mt_obj

    return dat


class ExportEDI(ContextModule):
    """
    Export Data.

    Attributes
    ----------
    ofile : str
        output file name.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ofile = ''

    def run(self):
        """
        Run.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.parent.process_is_active(True)

        if 'MT - EDI' in self.indata:
            data = self.indata['MT - EDI']
        else:
            self.showlog('No EDI data')
            self.parent.process_is_active(False)
            return False

        ext = 'EDI (*.edi)'

        self.ofile, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)

        if self.ofile == '':
            self.parent.process_is_active(False)
            return False
        os.chdir(os.path.dirname(self.ofile))

        ext = self.ofile[-3:]

        self.showlog('Export Data Busy...')

        # Pop up save dialog box
        if ext == 'edi':
            self.export_edi(data)

        self.showlog('Export EDI Finished!')
        self.parent.process_is_active(False)
        return True

    def export_edi(self, dat):
        """
        Export to EDI format.

        Parameters
        ----------
        dat : EDI Data
            dataset to export

        Returns
        -------
        None.

        """
        savepath = os.path.dirname(self.ofile)
        basename = os.path.basename(self.ofile)[:-4]
        for i in dat:
            dat[i].write_mt_file(save_dir=savepath,
                                 fn_basename=basename+'_'+i,
                                 file_type='edi',
                                 longitude_format='LONG',
                                 latlon_format='dd')
