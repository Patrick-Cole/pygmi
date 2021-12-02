# -----------------------------------------------------------------------------
# Name:        smooth.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
"""Smooth Data."""

import copy
import warnings
from PyQt5 import QtWidgets
import numpy as np
import scipy.signal as ssig

import pygmi.menu_default as menu_default
from pygmi.misc import ProgressBarText


class Smooth(QtWidgets.QDialog):
    """Smooth."""

    def __init__(self, parent=None):
        super().__init__(parent)

        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        if parent is not None:
            self.piter = self.parent.pbar.iter
        else:
            self.piter = ProgressBarText().iter

        self.label = QtWidgets.QLabel('X:')
        self.spinbox_x = QtWidgets.QSpinBox()
        self.label_2 = QtWidgets.QLabel('Y:')
        self.label_3 = QtWidgets.QLabel('Radius in Samples:')
        self.spinbox_y = QtWidgets.QSpinBox()
        self.spinbox_radius = QtWidgets.QSpinBox()
        self.label_4 = QtWidgets.QLabel('Standard Deviation:')
        self.spinbox_stddev = QtWidgets.QSpinBox()

        self.radiobutton_2dmedian = QtWidgets.QRadioButton('2D Median')
        self.radiobutton_2dmean = QtWidgets.QRadioButton('2D Mean')
        self.radiobutton_box = QtWidgets.QRadioButton('Box Window')
        self.radiobutton_disk = QtWidgets.QRadioButton('Disk Window')
        self.radiobutton_gaussian = QtWidgets.QRadioButton('Gaussian Window')
        self.tablewidget = QtWidgets.QTableWidget()

        self.setupui()

        self.fmat = None

        self.choosefilter()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        groupbox = QtWidgets.QGroupBox('Filter Size')
        gridlayout_2 = QtWidgets.QGridLayout(groupbox)
        groupbox_2 = QtWidgets.QGroupBox('Filter Type')
        groupbox_3 = QtWidgets.QGroupBox('Filter Shape')
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.smooth')

        self.spinbox_x.setMinimum(1)
        self.spinbox_x.setMaximum(999999)
        self.spinbox_x.setProperty('value', 5)
        self.spinbox_y.setMinimum(1)
        self.spinbox_y.setMaximum(9999999)
        self.spinbox_y.setProperty('value', 5)
        self.spinbox_radius.setMinimum(1)
        self.spinbox_radius.setMaximum(99999)
        self.spinbox_radius.setProperty('value', 5)
        self.spinbox_stddev.setMinimum(1)
        self.spinbox_stddev.setMaximum(99999)
        self.spinbox_stddev.setProperty('value', 5)
        self.radiobutton_2dmean.setChecked(True)
        self.radiobutton_box.setChecked(True)
        self.tablewidget.setRowCount(5)
        self.tablewidget.setColumnCount(5)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Smoothing Filters')

        verticallayout = QtWidgets.QVBoxLayout(groupbox_2)
        verticallayout.addWidget(self.radiobutton_2dmean)
        verticallayout.addWidget(self.radiobutton_2dmedian)
        verticallayout_2 = QtWidgets.QVBoxLayout(groupbox_3)
        verticallayout_2.addWidget(self.radiobutton_box)
        verticallayout_2.addWidget(self.radiobutton_disk)
        verticallayout_2.addWidget(self.radiobutton_gaussian)

        gridlayout_2.addWidget(self.label, 0, 0, 1, 1)
        gridlayout_2.addWidget(self.spinbox_x, 0, 1, 1, 1)
        gridlayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        gridlayout_2.addWidget(self.spinbox_y, 1, 1, 1, 1)
        gridlayout_2.addWidget(self.label_3, 2, 0, 1, 1)
        gridlayout_2.addWidget(self.spinbox_radius, 2, 1, 1, 1)
        gridlayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        gridlayout_2.addWidget(self.spinbox_stddev, 3, 1, 1, 1)

        gridlayout.addWidget(self.tablewidget, 1, 0, 1, 3)
        gridlayout.addWidget(groupbox_2, 2, 0, 1, 1)
        gridlayout.addWidget(groupbox_3, 2, 1, 1, 1)
        gridlayout.addWidget(groupbox, 2, 2, 1, 1)
        gridlayout.addWidget(buttonbox, 3, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 3, 0, 1, 1)

        self.radiobutton_2dmean.clicked.connect(self.choosefilter)
        self.radiobutton_2dmedian.clicked.connect(self.choosefilter)
        self.radiobutton_box.clicked.connect(self.choosefilter)
        self.radiobutton_disk.clicked.connect(self.choosefilter)
        self.radiobutton_gaussian.clicked.connect(self.choosefilter)
        self.spinbox_x.valueChanged.connect(self.choosefilter)
        self.spinbox_y.valueChanged.connect(self.choosefilter)
        self.spinbox_radius.valueChanged.connect(self.choosefilter)
        self.spinbox_stddev.valueChanged.connect(self.choosefilter)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

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
            temp = self.exec_()
            if temp == 0:
                return False
            self.parent.process_is_active(True)

        self.showprocesslog('Smoothing ')
        data = copy.deepcopy(self.indata['Raster'])
        if self.radiobutton_2dmean.isChecked():
            for i, _ in enumerate(data):
                data[i].data = self.mov_win_filt(data[i].data, self.fmat,
                                                 '2D Mean', data[i].dataid)
                data[i].dataid = data[i].dataid+' 2D Mean'

        if self.radiobutton_2dmedian.isChecked():
            for i, _ in enumerate(data):
                data[i].data = self.mov_win_filt(data[i].data, self.fmat,
                                                 '2D Median', data[i].dataid)
                data[i].dataid = data[i].dataid+' 2D Median'

        if not nodialog:
            self.parent.process_is_active(False)
        self.outdata['Raster'] = data
        self.showprocesslog('Finished!', True)

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
        if projdata['ftype'] == '2D Mean':
            self.radiobutton_2dmean.setChecked(True)
        else:
            self.radiobutton_2dmedian.setChecked(True)

        if projdata['fshape'] == 'box':
            self.radiobutton_box.setChecked(True)
            self.spinbox_x.setValue(projdata['fsize'][0])
            self.spinbox_y.setValue(projdata['fsize'][1])
        if projdata['fshape'] == 'disc':
            self.radiobutton_disk.setChecked(True)
            self.spinbox_radius.setValue(projdata['frad'])
        if projdata['fshape'] == 'gaussian':
            self.radiobutton_gaussian.setChecked(True)
            self.spinbox_stddev.setValue(projdata['fsigma'])

        self.choosefilter()

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

        box_x = self.spinbox_x.value()
        box_y = self.spinbox_y.value()
        rad = self.spinbox_radius.value()
        sigma = self.spinbox_stddev.value()

        if self.radiobutton_2dmean.isChecked():
            projdata['ftype'] = '2D Mean'
        elif self.radiobutton_2dmedian.isChecked():
            projdata['ftype'] = '2D Median'

        if self.radiobutton_box.isChecked():
            projdata['fshape'] = 'box'
            projdata['fsize'] = (box_x, box_y)
        elif self.radiobutton_disk.isChecked():
            projdata['fshape'] = 'disc'
            projdata['frad'] = rad
        elif self.radiobutton_gaussian.isChecked():
            projdata['fshape'] = 'gaussian'
            projdata['fsigma'] = sigma

        return projdata

    def choosefilter(self):
        """
        Section to choose the filter.

        Returns
        -------
        None.

        """
        # Do not need to check whether inputs are greater than zero,
        # since the spinboxes will not permit it.

        box_x = self.spinbox_x.value()
        box_y = self.spinbox_y.value()
        rad = self.spinbox_radius.value()
        sigma = self.spinbox_stddev.value()
        fmat = None

        if self.radiobutton_2dmean.isChecked():
            self.radiobutton_gaussian.setVisible(True)
            if self.radiobutton_box.isChecked():
                fmat = filters2d('average', [box_y, box_x])
            elif self.radiobutton_disk.isChecked():
                fmat = filters2d('disc', rad)
            elif self.radiobutton_gaussian.isChecked():
                fmat = filters2d('gaussian', [box_x, box_y], sigma)
        else:
            self.radiobutton_gaussian.setVisible(False)
            if self.radiobutton_gaussian.isChecked():
                self.radiobutton_box.setChecked(True)
            if self.radiobutton_box.isChecked():
                fmat = np.ones((box_y, box_x))
            elif self.radiobutton_disk.isChecked():
                fmat = filters2d('disc', rad)
                fmat[fmat > 0] = 1

        if self.radiobutton_box.isChecked():
            self.spinbox_x.setVisible(True)
            self.spinbox_y.setVisible(True)
            self.spinbox_radius.setVisible(False)
            self.spinbox_stddev.setVisible(False)
            self.label.setVisible(True)
            self.label_2.setVisible(True)
            self.label_3.setVisible(False)
            self.label_4.setVisible(False)
        elif self.radiobutton_disk.isChecked():
            self.spinbox_x.setVisible(False)
            self.spinbox_y.setVisible(False)
            self.spinbox_radius.setVisible(True)
            self.spinbox_stddev.setVisible(False)
            self.label.setVisible(False)
            self.label_2.setVisible(False)
            self.label_3.setVisible(True)
            self.label_4.setVisible(False)
        elif self.radiobutton_gaussian.isChecked():
            self.spinbox_x.setVisible(False)
            self.spinbox_y.setVisible(False)
            self.spinbox_radius.setVisible(False)
            self.spinbox_stddev.setVisible(True)
            self.label.setVisible(False)
            self.label_2.setVisible(False)
            self.label_3.setVisible(False)
            self.label_4.setVisible(True)

        self.fmat = fmat
        self.updatetable()

    def updatetable(self):
        """
        Update table.

        Returns
        -------
        None.

        """
        if self.fmat is None:
            return

        red = [0]*85 + list(range(0, 256, 3))+[255]*85
        green = list(range(0, 256, 3)) + 85*[255] + list(range(255, 0, -3))
        blue = [255]*85 + list(range(255, 0, -3)) + [0]*86
        fmin = self.fmat.min()
        frange = self.fmat.ptp()

        self.tablewidget.setRowCount(self.fmat.shape[0])
        self.tablewidget.setColumnCount(self.fmat.shape[1])

        for row in range(self.fmat.shape[0]):
            for col in range(self.fmat.shape[1]):
                if frange == 0:
                    i = 127
                else:
                    i = int(255*(self.fmat[row, col]-fmin)/frange)
                ltmp = QtWidgets.QLabel('{:g}'.format(self.fmat[row, col]))
                ltmp.setStyleSheet(
                    'Background: rgb' +
                    str((red[i], green[i], blue[i])) + '; Color: rgb' +
                    str((255-red[i], 255-green[i], 255-blue[i])))
                self.tablewidget.setCellWidget(row, col, ltmp)

        self.tablewidget.resizeColumnsToContents()

    def msgbox(self, title, message):
        """
        Message box.

        Parameters
        ----------
        title : str
            Title for message box.
        message : str
            Text for message box.

        Returns
        -------
        None.

        """
        QtWidgets.QMessageBox.warning(self.parent, title, message,
                                      QtWidgets.QMessageBox.Ok)

    def mov_win_filt(self, dat, fmat, itype, title):
        """
        Apply moving window filter function to data.

        Parameters
        ----------
        dat : numpy array.
            Data for a PyGMI raster dataset.
        fmat : TYPE
            DESCRIPTION.
        itype : str
            Filter type. Can be '2D Mean' or '2D Median'.
        title : str
            Text for reportback function.

        Returns
        -------
        out : numpy array
            Data for a PyGMI raster dataset.

        """
        out = dat.tolist()

        rowf = fmat.shape[0]
        colf = fmat.shape[1]
        rowd = dat.shape[0]
        cold = dat.shape[1]
        drr = round(rowf/2.0)
        dcc = round(colf/2.0)

        dummy = np.zeros((rowd+rowf-1, cold+colf-1))*np.nan
        dummy[drr-1:drr-1+rowd, dcc-1:dcc-1+cold] = dat
        dummy = np.ma.masked_invalid(dummy)
        dummy.mask[drr-1:drr-1+rowd, dcc-1:dcc-1+cold] = dat.mask

        dummy.data[dummy.mask] = np.nan
        dat = dat.astype(float)
        dat.data[dat.mask] = np.nan

        if itype == '2D Mean':
            out = ssig.correlate(dat, fmat, 'same', method='direct')

        elif itype == '2D Median':
            self.showprocesslog('Calculating Median...')
            out = np.ma.zeros([rowd, cold])*np.nan
            out.mask = np.ma.getmaskarray(dat)
            fmat = fmat.astype(bool)
            dummy = dummy.data

            for i in self.piter(range(rowd)):
                for j in range(cold):
                    tmp1 = dummy[i:i+rowf, j:j+colf][fmat]
                    if np.isnan(tmp1).min() == False:
                        out[i, j] = np.nanmedian(tmp1)

        out = np.ma.masked_invalid(out)
        out.shape = out.shape[0:2]
        out.mask[:rowf//2] = True
        out.mask[-rowf//2:] = True
        out.mask[:, :colf//2] = True
        out.mask[:, -colf//2:] = True
        return out


def filters2d(filtertype, sze, *sigma):
    """
    Filter 2D.

    These filter definitions have been translated from the octave function
    'fspecial'.

    Copyright (C) 2005 Peter Kovesi

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

    FSPECIAL - Create spatial filters for image processing

    Usage:  f = fspecial(filtertype, optional parameters)

    filtertype can be

    |   'average'   - Rectangular averaging filter
    |   'disc'      - Circular averaging filter.
    |   'gaussian'  - Gaussian filter.

    The parameters that need to be specified depend on the filtertype

    Examples of use and associated default values:

    |   f = fspecial('average',sze)           sze can be a 1 or 2 vector
    |                                         default is [3 3].
    |   f = fspecial('disk',radius)           default radius = 5
    |   f = fspecial('gaussian',sze, sigma)   default sigma is 0.5

    Where sze is specified as a single value the filter will be square.

    Author:   Peter Kovesi <pk@csse.uwa.edu.au>
    Keywords: image processing, spatial filters
    Created:  August 2005

    Parameters
    ----------
    filtertype : str
        Type of filter. Can be 'average', 'disc' or 'gaussian'.
    sze : numpy array or integer)
        This is a integer radius for 'disc' or a vector containing rows and
        columns otherwise.
    sigma : numpy array
        numpy array containing std deviation. Used in 'gaussian'.

    Returns
    -------
    f : numpy array
        Returns the filter to be used.

    """
    if filtertype == 'disc':
        radius = sze
        sze = [2*radius+1, 2*radius+1]

    rows = sze[0]
    cols = sze[1]
    r2 = (rows-1)/2
    c2 = (cols-1)/2

    if filtertype == 'average':
        rows = sze[0]
        cols = sze[1]
        f = np.ones(sze)/(rows*cols)

    elif filtertype == 'disc':
        [x, y] = np.mgrid[-c2: c2, -r2: r2]
        rad = np.sqrt(x**2 + y**2)
        f = rad <= radius
        f = f/np.sum(f[:])

    elif filtertype == 'gaussian':
        [x, y] = np.mgrid[-c2: c2, -r2:r2]
        radsqrd = x**2 + y**2
        f = np.exp(-radsqrd/(2*sigma[0]**2))
        f = f/np.sum(f[:])
    else:
        warnings.warn('Unrecognized filter type')

    return f
