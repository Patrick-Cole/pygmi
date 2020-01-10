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
from PyQt5 import QtWidgets
import mtpy.core.mt


class ImportEDI():
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

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'EDI (*.edi)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.parent, 'Open EDI Files (single or multiple)', '.', ext)
        if not filename:
            return False

        os.chdir(os.path.dirname(filename[0]))
        self.ifile = filename

        dat = get_EDI(filename)

        self.ifile = os.path.dirname(filename[0]) + r'\EDI List'

        if dat is None:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import dataset. ',
                                          QtWidgets.QMessageBox.Ok)
            return False

        output_type = 'MT - EDI'

        self.outdata[output_type] = dat
        return True


def get_EDI(ifiles):
    """
    EDI Import.

    Parameters
    ----------
    ifiles : list
        filenames to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    dat = {}


    for edi_file in ifiles:
        mt_obj = mtpy.core.mt.MT(edi_file)

        bname = os.path.basename(edi_file)
        bname = bname[:-4]

        dat[bname] = mt_obj

    return dat


class ExportEDI():
    """
    Export Data

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
    def __init__(self, parent):
        self.ifile = ''
        self.name = 'Export Data: '
        self.ext = ''
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def run(self):
        """ Show Info """
        self.parent.process_is_active(True)

        if 'MT - EDI' in self.indata:
            data = self.indata['MT - EDI']
        else:
            self.parent.showprocesslog('No EDI data')
            self.parent.process_is_active(False)
            return False

        ext = 'EDI (*.edi)'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)

        if filename == '':
            self.parent.process_is_active(False)
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)
        self.ext = filename[-3:]

        self.parent.showprocesslog('Export Data Busy...')

    # Pop up save dialog box
        if self.ext == 'edi':
            self.export_edi(data)

        self.parent.showprocesslog('Export EDI Finished!')
        self.parent.process_is_active(False)
        return True

    def export_edi(self, dat):
        """
        Export to EDI format

        Parameters
        ----------
        dat : EDI Data
            dataset to export
        """

        savepath = os.path.dirname(self.ifile)
        basename = os.path.basename(self.ifile)[:-4]
        for i in dat:
            dat[i].write_mt_file(save_dir=savepath,
                                 fn_basename=basename+'_'+i,
                                 file_type='edi',
#                                 new_Z_obj=new_Z_obj,
#                                 new_Tipper_obj=new_Tipper_obj,
                                 longitude_format='LONG',
                                 latlon_format='dd')


