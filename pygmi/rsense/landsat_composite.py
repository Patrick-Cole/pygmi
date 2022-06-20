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
import copy
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

from pygmi.rsense.iodefs import get_data
from pygmi.raster.dataprep import lstack
from pygmi.raster.iodefs import export_raster
from pygmi.misc import ProgressBarText


class LandsatComposite():
    """
    Landsat Composite Interface.

    Attributes
    ----------
    parent : parent
        reference to the parent routine.
    idir : str
        Input directory.
    ifile : str
        Input file.
    indata : dictionary
        dictionary of input datasets.
    outdata : dictionary
        dictionary of output datasets.
    """

    def __init__(self, parent=None):
        self.ifile = ''
        self.idir = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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
        if not nodialog or self.idir == '':
            self.idir = QtWidgets.QFileDialog.getExistingDirectory(
                self.parent, 'Select Directory')
            if self.idir == '':
                return False
        os.chdir(self.idir)

        ifiles = glob.glob(os.path.join(self.idir, '**/*MTL.txt'),
                           recursive=True)

        if not ifiles:
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          'No *MTL.txt in the directory.',
                                          QtWidgets.QMessageBox.Ok)
            return False

        dat = composite(self.idir, 10, pprint=self.showprocesslog,
                        piter=self.piter)

        self.outdata['Raster'] = dat

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
        self.idir = projdata['idir']

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

        projdata['idir'] = self.idir

        return projdata


def composite(idir, dreq=10, mean=None, pprint=print, piter=None):
    """
    Create a landsat composite.

    Parameters
    ----------
    idir : str
        Input directory.
    dreq : int, optional
        Distance to cloud in pixels. The default is 10.
    mean : float, optional
        The mean or target day. If not specified, it is calculated
        automatically. The default is None.
    pprint : function, optional
        Function for printing text. The default is print.
    piter : iter, optional
        Progress bar iterable. The default is None.

    Returns
    -------
    datfin : list
        List of PyGMI Data objects.

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
                            pprint=pprint)

    for ifile in ifiles[1:]:
        dat2 = import_and_score(ifile, dreq, mean, std, piter=piter,
                                pprint=pprint)

        tmp1 = {}
        tmp2 = {}

        for band in dat1:
            tmp1[band], tmp2[band] = lstack([dat1[band], dat2[band]])

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

    pprint(f'Range of days for scenes: {allday}')
    pprint(f'Mean day {mean}')
    pprint(f'Standard deviation {std:.2f}')

    return datfin


def import_and_score(ifile, dreq, mean, std, pprint=print, piter=None):
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
    pprint : function, optional
        Function for printing text. The default is print.
    piter : iter, optional
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
    tmp = get_data(ifile, alldata=True, piter=piter, showprocesslog=pprint)

    for i in tmp:
        if i.dataid in bands:
            i.data = i.data.astype(np.float32)
            dat[i.dataid] = i
        if 'ST_CDIST' in i.dataid:
            dat['cdist'] = i

    del tmp

    # CDist calculations
    # dmin = dat['cdist'].data.min()
    dmin = 0

    cdist2 = dat['cdist'].data.copy()
    cdist2[cdist2 > dreq] = dreq

    cdist2 = 1/(1+np.exp(-0.2*(cdist2-(dreq-dmin)/2)))

    # Get day of year
    sdate = os.path.basename(ifile).split('_')[3]
    sdate = datetime.strptime(sdate, '%Y%m%d')
    datday = sdate.timetuple().tm_yday
    cdistscore = copy.deepcopy(dat['cdist'])
    cdistscore.data = np.ma.masked_equal(cdist2.filled(0), 0)
    cdistscore.nodata = 0

    dayscore = (1/(std*np.sqrt(2*np.pi)) *
                np.exp(-0.5*((datday-mean)/std)**2))
    dat['score'] = cdistscore
    dat['score'].data += dayscore

    pprint(f'Scene name: {os.path.basename(ifile)}')
    pprint(f'Scene day of year: {datday}')

    filt = (dat['cdist'].data == 0)
    for band, data in dat.items():
        data.data[filt] = 0
        data.data = np.ma.masked_equal(data.data, 0)

    del dat['cdist']

    return dat


def plot_rgb(dat, title='RGB'):
    """
    Plot RGB map.

    Parameters
    ----------
    dat : list
        List of PyGMI datasets.
    title : str
        Title for plot.

    Returns
    -------
    None.

    """
    blue = dat['B2'].data[5000:6000, 2000:3000]
    green = dat['B3'].data[5000:6000, 2000:3000]
    red = dat['B4'].data[5000:6000, 2000:3000]
    alpha = np.logical_not(red.mask)
    rgb = np.array([red, green, blue, alpha])
    rgb = np.moveaxis(rgb, 0, 2)

    plt.figure(dpi=150)
    plt.title(title)
    plt.imshow(rgb)
    plt.show()


def _testfn():
    """Test routine."""
    idir = r'C:\WorkProjects\Landsat_Summer'
    ofile = os.path.join(idir, 'landsat_composite.tif')

    dat = composite(idir, 10)
    # dat = composite(idir, 10, 87)

    # for i in [dato[0], dat[0]]:
    #     # plot_rgb(i)
    #     plt.imshow(i.data[5000:6000, 2000:3000], extent=i.extent)
    #     plt.colorbar()
    #     plt.show()

    # print(dato[0].data[4000,4000])
    # print(dat[0].data[4000,4000])

    export_raster(ofile, dat, 'GTiff')


if __name__ == "__main__":
    _testfn()
