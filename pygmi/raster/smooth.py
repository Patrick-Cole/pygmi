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

import warnings
from PyQt5 import QtWidgets
import numpy as np
import scipy.signal as ssig

from pygmi import menu_default
from pygmi.misc import BasicModule


class Smooth(BasicModule):
    """Smooth."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QtWidgets.QLabel('X:')
        self.sb_x = QtWidgets.QSpinBox()
        self.lbl_2 = QtWidgets.QLabel('Y:')
        self.lbl_3 = QtWidgets.QLabel('Radius in Samples:')
        self.sb_y = QtWidgets.QSpinBox()
        self.sb_radius = QtWidgets.QSpinBox()
        self.lbl_4 = QtWidgets.QLabel('Standard Deviation:')
        self.sb_stddev = QtWidgets.QSpinBox()

        self.rb_2dmedian = QtWidgets.QRadioButton('2D Median')
        self.rb_2dmean = QtWidgets.QRadioButton('2D Mean')
        self.rb_box = QtWidgets.QRadioButton('Box Window')
        self.rb_disk = QtWidgets.QRadioButton('Disk Window')
        self.rb_gaussian = QtWidgets.QRadioButton('Gaussian Window')
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

        self.sb_x.setMinimum(1)
        self.sb_x.setMaximum(999999)
        self.sb_x.setProperty('value', 5)
        self.sb_y.setMinimum(1)
        self.sb_y.setMaximum(9999999)
        self.sb_y.setProperty('value', 5)
        self.sb_radius.setMinimum(1)
        self.sb_radius.setMaximum(99999)
        self.sb_radius.setProperty('value', 5)
        self.sb_stddev.setMinimum(1)
        self.sb_stddev.setMaximum(99999)
        self.sb_stddev.setProperty('value', 5)
        self.rb_2dmean.setChecked(True)
        self.rb_box.setChecked(True)
        self.tablewidget.setRowCount(5)
        self.tablewidget.setColumnCount(5)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Smoothing Filters')

        verticallayout = QtWidgets.QVBoxLayout(groupbox_2)
        verticallayout.addWidget(self.rb_2dmean)
        verticallayout.addWidget(self.rb_2dmedian)
        verticallayout_2 = QtWidgets.QVBoxLayout(groupbox_3)
        verticallayout_2.addWidget(self.rb_box)
        verticallayout_2.addWidget(self.rb_disk)
        verticallayout_2.addWidget(self.rb_gaussian)

        gridlayout_2.addWidget(self.label, 0, 0, 1, 1)
        gridlayout_2.addWidget(self.sb_x, 0, 1, 1, 1)
        gridlayout_2.addWidget(self.lbl_2, 1, 0, 1, 1)
        gridlayout_2.addWidget(self.sb_y, 1, 1, 1, 1)
        gridlayout_2.addWidget(self.lbl_3, 2, 0, 1, 1)
        gridlayout_2.addWidget(self.sb_radius, 2, 1, 1, 1)
        gridlayout_2.addWidget(self.lbl_4, 3, 0, 1, 1)
        gridlayout_2.addWidget(self.sb_stddev, 3, 1, 1, 1)

        gridlayout.addWidget(self.tablewidget, 1, 0, 1, 3)
        gridlayout.addWidget(groupbox_2, 2, 0, 1, 1)
        gridlayout.addWidget(groupbox_3, 2, 1, 1, 1)
        gridlayout.addWidget(groupbox, 2, 2, 1, 1)
        gridlayout.addWidget(buttonbox, 3, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 3, 0, 1, 1)

        self.rb_2dmean.clicked.connect(self.choosefilter)
        self.rb_2dmedian.clicked.connect(self.choosefilter)
        self.rb_box.clicked.connect(self.choosefilter)
        self.rb_disk.clicked.connect(self.choosefilter)
        self.rb_gaussian.clicked.connect(self.choosefilter)
        self.sb_x.valueChanged.connect(self.choosefilter)
        self.sb_y.valueChanged.connect(self.choosefilter)
        self.sb_radius.valueChanged.connect(self.choosefilter)
        self.sb_stddev.valueChanged.connect(self.choosefilter)
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
        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
            return False

        self.choosefilter()
        if not nodialog:
            temp = self.exec_()
            if temp == 0:
                return False
            self.parent.process_is_active(True)

        self.showlog('Smoothing ')
        data = [i.copy() for i in self.indata['Raster']]

        if self.rb_2dmean.isChecked():
            filt = '2D Mean'
        else:
            filt = '2D Median'

        for dat in data:
            dat.data = self.mov_win_filt(dat.data, self.fmat, filt)
            dat.dataid = dat.dataid+' '+filt

        if not nodialog:
            self.parent.process_is_active(False)
        self.outdata['Raster'] = data
        self.showlog('Finished!', True)

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.sb_x)
        self.saveobj(self.sb_y)
        self.saveobj(self.sb_radius)
        self.saveobj(self.sb_stddev)

        self.saveobj(self.rb_2dmedian)
        self.saveobj(self.rb_2dmean)
        self.saveobj(self.rb_box)
        self.saveobj(self.rb_disk)
        self.saveobj(self.rb_gaussian)

    def choosefilter(self):
        """
        Section to choose the filter.

        Returns
        -------
        None.

        """
        # Do not need to check whether inputs are greater than zero,
        # since the spinboxes will not permit it.

        box_x = self.sb_x.value()
        box_y = self.sb_y.value()
        rad = self.sb_radius.value()
        sigma = self.sb_stddev.value()
        fmat = None

        if self.rb_2dmean.isChecked():
            self.rb_gaussian.setVisible(True)
            if self.rb_box.isChecked():
                fmat = filters2d('average', [box_y, box_x])
            elif self.rb_disk.isChecked():
                fmat = filters2d('disc', rad)
            elif self.rb_gaussian.isChecked():
                fmat = filters2d('gaussian', [box_x, box_y], sigma)
        else:
            self.rb_gaussian.setVisible(False)
            if self.rb_gaussian.isChecked():
                self.rb_box.setChecked(True)
            if self.rb_box.isChecked():
                fmat = np.ones((box_y, box_x))
            elif self.rb_disk.isChecked():
                fmat = filters2d('disc', rad)
                fmat[fmat > 0] = 1

        if self.rb_box.isChecked():
            self.sb_x.setVisible(True)
            self.sb_y.setVisible(True)
            self.sb_radius.setVisible(False)
            self.sb_stddev.setVisible(False)
            self.label.setVisible(True)
            self.lbl_2.setVisible(True)
            self.lbl_3.setVisible(False)
            self.lbl_4.setVisible(False)
        elif self.rb_disk.isChecked():
            self.sb_x.setVisible(False)
            self.sb_y.setVisible(False)
            self.sb_radius.setVisible(True)
            self.sb_stddev.setVisible(False)
            self.label.setVisible(False)
            self.lbl_2.setVisible(False)
            self.lbl_3.setVisible(True)
            self.lbl_4.setVisible(False)
        elif self.rb_gaussian.isChecked():
            self.sb_x.setVisible(False)
            self.sb_y.setVisible(False)
            self.sb_radius.setVisible(False)
            self.sb_stddev.setVisible(True)
            self.label.setVisible(False)
            self.lbl_2.setVisible(False)
            self.lbl_3.setVisible(False)
            self.lbl_4.setVisible(True)

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
                ltmp = QtWidgets.QLabel(f'{self.fmat[row, col]:g}')
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

    def mov_win_filt(self, dat, fmat, itype):
        """
        Apply moving window filter function to data.

        Parameters
        ----------
        dat : numpy masked array.
            Data for a PyGMI raster dataset.
        fmat : numpy array
            Filter matrix.
        itype : str
            Filter type. Can be '2D Mean' or '2D Median'.

        Returns
        -------
        out : numpy masked array
            Data for a PyGMI raster dataset.

        """
        out = dat.tolist()

        rowf = fmat.shape[0]
        colf = fmat.shape[1]
        rowd = dat.shape[0]
        cold = dat.shape[1]
        drr = round(rowf/2.0)
        dcc = round(colf/2.0)

        dummy = np.ma.masked_all((rowd+rowf-1, cold+colf-1))
        dummy[drr-1:drr-1+rowd, dcc-1:dcc-1+cold] = dat

        dummy.data[dummy.mask] = np.nan
        dat = dat.astype(float)
        dat.data[dat.mask] = np.nan

        if itype == '2D Mean':
            out = ssig.correlate(dat, fmat, 'same', method='direct')

        elif itype == '2D Median':
            self.showlog('Calculating Median...')
            out = np.ma.zeros([rowd, cold])*np.nan
            out.mask = np.ma.getmaskarray(dat)
            fmat = fmat.astype(bool)
            dummy = dummy.data

            for i in self.piter(range(rowd)):
                for j in range(cold):
                    tmp1 = dummy[i:i+rowf, j:j+colf][fmat]
                    if np.any(~np.isnan(tmp1)):  # if any are False
                        out[i, j] = np.nanmedian(tmp1)

        out = np.ma.masked_invalid(out)
        out = np.ma.array(out)

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
