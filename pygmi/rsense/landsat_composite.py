# -----------------------------------------------------------------------------
# Name:        landsat_composite.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2022 Council for Geoscience
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
"""Calculate Landsat composite scenes."""

import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore

from pygmi.rsense.iodefs import get_data
from pygmi.raster.dataprep import lstack
from pygmi.misc import ProgressBarText, BasicModule
import pygmi.menu_default as menu_default


class LandsatComposite(BasicModule):
    """
    Landsat Composite Interface.

    Attributes
    ----------
    idir : str
        Input directory.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.idir = ''

        self.tday = QtWidgets.QSpinBox()
        self.idirlist = QtWidgets.QLineEdit('')

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
        helpdocs = menu_default.HelpButton('pygmi.rsense.landsat_composite')
        pb_idirlist = QtWidgets.QPushButton('Batch Directory')

        lbl_tday = QtWidgets.QLabel('Target Day:')

        self.tday.setMinimum(1)
        self.tday.setMaximum(366)
        self.tday.setValue(1)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Landsat Temporal Composite')

        gridlayout_main.addWidget(pb_idirlist, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.idirlist, 1, 1, 1, 1)
        gridlayout_main.addWidget(lbl_tday, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.tday, 2, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_idirlist.pressed.connect(self.get_idir)

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
            tmp = self.exec_()
            if tmp != 1:
                return False

        if self.idir == '':
            self.showlog('Error: No input directory')
            return False

        os.chdir(self.idir)

        ifiles = glob.glob(os.path.join(self.idir, '**/*MTL.txt'),
                           recursive=True)

        if not ifiles:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'No *MTL.txt in the directory or '
                                          'subdirectories.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        mean = self.tday.value()
        dat = composite(self.idir, 10, showlog=self.showlog,
                        piter=self.piter, mean=mean)

        self.outdata['Raster'] = dat

        return True

    def get_idir(self):
        """
        Get the input directory.

        Returns
        -------
        None.

        """
        self.idir = QtWidgets.QFileDialog.getExistingDirectory(
             self.parent, 'Select Directory')

        self.idirlist.setText(self.idir)

        if self.idir == '':
            self.idir = None
            return

        ifiles = glob.glob(os.path.join(self.idir, '**/*MTL.txt'),
                           recursive=True)

        if not ifiles:
            self.showlog('Error: No *MTL.txt in the directory.')
            return

        allday = []
        for ifile in ifiles:
            sdate = os.path.basename(ifile).split('_')[3]
            sdate = datetime.strptime(sdate, '%Y%m%d')
            datday = sdate.timetuple().tm_yday
            allday.append(datday)
            self.showlog(f'Scene name: {os.path.basename(ifile)}')
            self.showlog(f'Scene day of year: {datday}')

        allday = np.array(allday)
        mean = int(allday.mean())

        self.showlog(f'Mean day: {mean}')

        self.tday.setValue(mean)

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.idir)
        self.saveobj(self.tday)
        self.saveobj(self.idirlist)


def composite(idir, dreq=10, mean=None, showlog=print, piter=None):
    """
    Create a Landsat composite.

    Parameters
    ----------
    idir : str
        Input directory.
    dreq : int, optional
        Distance to cloud in pixels. The default is 10.
    mean : float, optional
        The mean or target day. If not specified, it is calculated
        automatically. The default is None.
    showlog : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Progress bar iterable. The default is None.

    Returns
    -------
    datfin : list of PyGMI Data.
        List of PyGMI Data.

    """
    if piter is None:
        piter = ProgressBarText().iter

    ifiles = glob.glob(os.path.join(idir, '**/*MTL.txt'), recursive=True)

    allday = []
    for ifile in ifiles:
        sdate = os.path.basename(ifile).split('_')[3]
        sdate = datetime.strptime(sdate, '%Y%m%d')
        allday.append(sdate.timetuple().tm_yday)

    allday = np.array(allday)
    if mean is None:
        mean = allday.mean()
    std = allday.std()

    dat1 = import_and_score(ifiles[0], dreq, mean, std, piter=piter,
                            showlog=showlog)

    for ifile in ifiles[1:]:
        dat2 = import_and_score(ifile, dreq, mean, std, piter=piter,
                                showlog=showlog)

        tmp1 = {}
        tmp2 = {}

        for band in dat1:
            tmp1[band], tmp2[band] = lstack([dat1[band], dat2[band]],
                                            showlog=showlog, piter=piter)

        filt = (tmp1['score'].data < tmp2['score'].data)

        for band in tmp1:
            tmp1[band].data[filt] = tmp2[band].data[filt]

        dat1 = tmp1
        del tmp1
        del tmp2
        del dat2

    datfin = []

    del dat1['score']
    for key, data in dat1.items():
        datfin.append(data)
        datfin[-1].dataid = key

    showlog(f'Range of days for scenes: {allday}')
    showlog(f'Mean day {mean}')
    showlog(f'Standard deviation {std:.2f}')

    return datfin


def import_and_score(ifile, dreq, mean, std, showlog=print, piter=None):
    """
    Import data and score it.

    Parameters
    ----------
    ifile : str
        Input filename.
    dreq : int, optional
        Distance to cloud in pixels. The default is 10.
    mean : float
        The mean or target day.
    std : float
        The standard deviation of all days.
    showlog : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Progress bar iterable. The default is None.

    Returns
    -------
    dat : dictionary.
        Dictionary of bands imported.

    """
    if piter is None:
        piter = ProgressBarText().iter

    bands = [f'B{i+1}' for i in range(11)]

    dat = {}
    tmp = get_data(ifile, piter=piter, showlog=showlog)

    for i in tmp:
        if i.dataid in bands:
            i.data = i.data.astype(np.float32)
            dat[i.dataid] = i
        if 'ST_CDIST' in i.dataid:
            dat['cdist'] = i

    del tmp

    # CDist calculations
    dmin = 0

    cdist2 = dat['cdist'].data.copy()
    cdist2[cdist2 > dreq] = dreq

    cdist2 = 1/(1+np.exp(-0.2*(cdist2-(dreq-dmin)/2)))

    # Get day of year
    sdate = os.path.basename(ifile).split('_')[3]
    sdate = datetime.strptime(sdate, '%Y%m%d')
    datday = sdate.timetuple().tm_yday
    cdistscore = dat['cdist'].copy()
    cdistscore.data = np.ma.masked_equal(cdist2.filled(0), 0)
    cdistscore.nodata = 0

    dayscore = (1/(std*np.sqrt(2*np.pi)) *
                np.exp(-0.5*((datday-mean)/std)**2))
    dat['score'] = cdistscore
    dat['score'].data += dayscore

    showlog(f'Scene name: {os.path.basename(ifile)}')
    showlog(f'Scene day of year: {datday}')

    filt = (dat['cdist'].data == 0)
    for data in dat.values():
        data.data[filt] = 0
        data.data = np.ma.masked_equal(data.data, 0)

    del dat['cdist']

    return dat


def _testfn():
    """Test routine."""
    import sys

    idir = r'C:\WorkProjects\Landsat_Summer'

    app = QtWidgets.QApplication(sys.argv)

    gui = LandsatComposite()
    gui.idir = idir
    gui.settings()


if __name__ == "__main__":
    _testfn()
