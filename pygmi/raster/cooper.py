# -----------------------------------------------------------------------------
# Name:        cooper.py (part of PyGMI)
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
"""A collection of routines by Gordon Cooper.

|    School of Geosciences, University of the Witwatersrand
|    Johannesburg, South Africa
|    cooperg@geosciences.wits.ac.za
|    http://www.wits.ac.za/science/geophysics/gc.htm
"""

import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np

import pygmi.menu_default as menu_default
from pygmi.misc import ProgressBarText


class Gradients(QtWidgets.QDialog):
    """
    Class used to gather information via a GUI, for function gradients.

    Attributes
    ----------
    parent : parent
    indata : dictionary
        PyGMI input data in a dictionary
    outdata :
        PyGMI input data in a dictionary
    azi : float
        Azimuth/filter direction (degrees)
    elev : float
        Elevation (for sunshading, degrees from horizontal)
    order : int
        Order of DR filter - see paper. Try 1 first.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.azi = 45.
        self.elev = 45.
        self.order = 1
        if parent is not None:
            self.piter = self.parent.pbar.iter
        else:
            self.piter = ProgressBarText().iter

        self.sb_order = QtWidgets.QSpinBox()
        self.sb_azi = QtWidgets.QSpinBox()
        self.rb_ddir = QtWidgets.QRadioButton('Directional Derivative')
        self.rb_vgrad = QtWidgets.QRadioButton('Vertical Derivative')
        self.rb_dratio = QtWidgets.QRadioButton('Derivative Ratio')
        self.rb_thg = QtWidgets.QRadioButton('Total Horizonal Gradient')
        self.label_or = QtWidgets.QLabel('Strength Factor')
        self.label_az = QtWidgets.QLabel('Azimuth')

        self.setupui()

        self.sb_azi.setValue(self.azi)
        self.sb_order.setValue(self.order)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.cooper.gradients')

        self.sb_order.setMinimum(1)
        self.sb_azi.setPrefix('')
        self.sb_azi.setMinimum(-360)
        self.sb_azi.setMaximum(360)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.rb_ddir.setChecked(True)
        self.sb_order.hide()
        self.label_or.hide()

        self.setWindowTitle('Gradient Calculation')

        gridlayout.addWidget(self.rb_ddir, 0, 0, 1, 1)
        gridlayout.addWidget(self.rb_dratio, 1, 0, 1, 1)
        gridlayout.addWidget(self.rb_vgrad, 2, 0, 1, 1)
        gridlayout.addWidget(self.rb_thg, 3, 0, 1, 1)
        gridlayout.addWidget(self.label_az, 4, 0, 1, 1)
        gridlayout.addWidget(self.sb_azi, 4, 1, 1, 1)
        gridlayout.addWidget(self.label_or, 5, 0, 1, 1)
        gridlayout.addWidget(self.sb_order, 5, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 6, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.rb_ddir.clicked.connect(self.radiochange)
        self.rb_dratio.clicked.connect(self.radiochange)
        self.rb_vgrad.clicked.connect(self.radiochange)
        self.rb_thg.clicked.connect(self.radiochange)

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

        self.azi = self.sb_azi.value()
        self.order = self.sb_order.value()

        data = copy.deepcopy(self.indata['Raster'])

        for i in self.piter(range(len(data))):
            if self.rb_ddir.isChecked():
                data[i].data = gradients(data[i].data, self.azi, data[i].xdim,
                                         data[i].ydim)
            elif self.rb_dratio.isChecked():
                data[i].data = derivative_ratio(data[i].data, self.azi,
                                                self.order)
            elif self.rb_thg.isChecked():
                data[i].data = thgrad(data[i].data, data[i].xdim, data[i].ydim)
            else:
                if data[i].xdim != data[i].ydim:
                    self.showprocesslog('X and Y dimension are different. '
                                        'Please resample')
                    return False

                mask = np.ma.getmaskarray(data[i].data)
                dxy = data[i].xdim
                data[i].data = np.ma.array(vertical(data[i].data, xint=dxy))
                data[i].data.mask = mask
            data[i].units = ''

        self.outdata['Raster'] = data

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
        self.azi = projdata['azim']
        self.order = projdata['order']

        self.sb_azi.setValue(projdata['azim'])
        self.sb_order.setValue(projdata['order'])

        if projdata['type'] == 'dratio':
            self.rb_dratio.setChecked(True)
        if projdata['type'] == 'ddir':
            self.rb_dratio.setChecked(True)
        if projdata['type'] == 'vgrad':
            self.rb_dratio.setChecked(True)

        self.radiochange()

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

        self.azi = self.sb_azi.value()
        self.order = self.sb_order.value()
        projdata['azim'] = self.azi
        projdata['order'] = self.order

        if self.rb_dratio.isChecked():
            projdata['type'] = 'dratio'
        elif self.rb_ddir.isChecked():
            projdata['type'] = 'ddir'
        else:
            projdata['type'] = 'vgrad'

        return projdata

    def radiochange(self):
        """
        Check radio button state.

        Returns
        -------
        None.

        """
        self.sb_order.hide()
        self.label_or.hide()
        self.sb_azi.hide()
        self.label_az.hide()

        if self.rb_dratio.isChecked():
            self.sb_order.show()
            self.label_or.show()
            self.sb_azi.show()
            self.label_az.show()
        elif self.rb_ddir.isChecked():
            self.sb_azi.show()
            self.label_az.show()


def gradients(data, azi, xint, yint):
    """
    Gradients.

    Compute directional derivative of image data. Based on code by
    Gordon Cooper.

    Parameters
    ----------
    data : numpy array
        input numpy data array
    azi : float
        Filter direction (degrees)
    xint : float
        X interval/distance.
    yint : float
        Y interval/distance.

    Returns
    -------
    dt1 : float
        returns directional derivative
    """
    azi = np.deg2rad(azi)
    dy, dx = np.gradient(data, yint, xint)
    dt1 = -dy*np.sin(azi)-dx*np.cos(azi)

    return dt1


def thgrad(data, xint, yint):
    """
    Gradients.

    Compute total horizontal gradient.

    Parameters
    ----------
    data : numpy array
        input numpy data array
    xint : float
        X interval/distance.
    yint : float
        Y interval/distance.

    Returns
    -------
    dt1 : float
        returns gradient.
    """
    dy, dx = np.gradient(data, yint, xint)
    dt1 = np.sqrt(dx**2+dy**2)

    return dt1


def derivative_ratio(data, azi, order):
    """
    Compute derivative ratio of image data. Based on code by Gordon Cooper.

    Parameters
    ----------
    data : numpy array
        input numpy data array
    azi : float
        Filter direction (degrees)
    order : int
        Order of DR filter - see paper. Try 1 first.

    Returns
    -------
    dr : float
        returns derivative ratio
    """
    # Directional derivative

    azi = np.deg2rad(azi)
    dx, dy = np.gradient(data)
    dt1 = -dy*np.sin(azi)-dx*np.cos(azi)

    # Derivative ratio

    dt2 = -dy*np.sin(azi+np.pi/2)-dx*np.cos(azi+np.pi/2)
    dt2 = dt2.astype(np.float64)
    dr = np.arctan2(dt1, abs(dt2)**order)

    return dr


class Visibility2d(QtWidgets.QDialog):
    """
    Class used to gather information via a GUI, for function visibility2d.

    Attributes
    ----------
    parent : parent
    indata : dictionary
        PyGMI input data in a dictionary
    outdata :
        PyGMI input data in a dictionary
    wsize : int
        window size, must be odd
    dh : float
        height of observer above surface
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.wsize = 11
        self.dh = 10

        if parent is not None:
            self.piter = self.parent.pbar.iter
        else:
            self.piter = ProgressBarText().iter

        self.sb_dh = QtWidgets.QSpinBox()
        self.sb_wsize = QtWidgets.QSpinBox()

        self.setupui()

        self.sb_wsize.setValue(self.wsize)
        self.sb_dh.setValue(self.dh)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.cooper.visibility')
        label = QtWidgets.QLabel('Viewing Height (% std dev)')
        label_2 = QtWidgets.QLabel('Window Size (Odd)')

        self.sb_dh.setMinimum(1)
        self.sb_dh.setMaximum(10000)
        self.sb_wsize.setPrefix('')
        self.sb_wsize.setMinimum(3)
        self.sb_wsize.setMaximum(100000)
        self.sb_wsize.setSingleStep(2)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Visibility')

        gridlayout.addWidget(label_2, 0, 0, 1, 1)
        gridlayout.addWidget(self.sb_wsize, 0, 1, 1, 1)
        gridlayout.addWidget(label, 1, 0, 1, 1)
        gridlayout.addWidget(self.sb_dh, 1, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 2, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 2, 1, 1, 1)

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

        self.wsize = self.sb_wsize.value()
        self.dh = self.sb_dh.value()

        data = copy.deepcopy(self.indata['Raster'])
        data2 = []

        for i, datai in enumerate(data):
            self.showprocesslog(datai.dataid+':')

            vtot, vstd, vsum = visibility2d(datai.data, self.wsize,
                                            self.dh*data[i].data.std()/100.,
                                            self.piter)
            data2.append(copy.deepcopy(datai))
            data2.append(copy.deepcopy(datai))
            data2.append(copy.deepcopy(datai))
            data2[-3].data = vtot
            data2[-2].data = vstd
            data2[-1].data = vsum
            data2[-3].dataid += ' Total Visibility'
            data2[-2].dataid += ' Visibility Variation'
            data2[-1].dataid += ' Visibility Vector Resultant'

        self.outdata['Raster'] = data2
        self.showprocesslog('Finished!')

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
        self.sb_wsize.setValue(projdata['wsize'])
        self.sb_dh.setValue(projdata['vheight'])

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

        projdata['wsize'] = self.sb_wsize.value()
        projdata['vheight'] = self.sb_dh.value()

        return projdata


def visibility2d(data, wsize, dh, piter=iter):
    """
    Compute visibility as a textural measure.

    Compute vertical derivatives by calculating the visibility at different
    heights above the surface (see paper)

    Parameters
    ----------
    data : numpy array
        input dataset - numpy MxN array
    wsize : int
        window size, must be odd
    dh : float
        height of observer above surface
    piter : iter
        Progress bar iterable. The default is iter.

    Returns
    -------
    vtot : numpy array
        Total visibility.
    vstd : numpy array
        Visibility variation.
    vsum : numpy array
        Visibility vector resultant.

    """
    nr, nc = np.shape(data)
    wsize = abs(np.real(wsize))
    w2 = int(np.floor(wsize/2))
    vn = np.zeros([nr, nc])
    vs = np.zeros([nr, nc])
    ve = np.zeros([nr, nc])
    vw = np.zeros([nr, nc])
    vd1 = np.zeros([nr, nc])
    vd2 = np.zeros([nr, nc])
    vd3 = np.zeros([nr, nc])
    vd4 = np.zeros([nr, nc])
    vstd = np.zeros([nr, nc])
    mask = np.ma.getmaskarray(data)
    mean = data.mean()
    data = data.data
    data[mask] = mean

    for j in piter(range(nc)):    # Columns
        for i in range(w2, nr-w2):
            dtmp = data[i-w2:i+w2+1, j]
            vn[i, j] = __visible1(dtmp, wsize, w2+1, dh)
            vs[i, j] = __visible2(dtmp, w2+1, dh)

    for j in piter(range(w2, nc-w2)):    # Rows
        for i in range(nr):
            dtmp = data[i, j-w2:j+w2+1]
            ve[i, j] = __visible1(dtmp, wsize, w2+1, dh)
            vw[i, j] = __visible2(dtmp, w2+1, dh)

    for j in piter(range(w2, nc-w2)):
        for i in range(w2, nr-w2):
            dtmp = np.zeros(wsize)
            for k in range(wsize):
                dtmp[k] = data[i-w2+k, j-w2+k]
            vd1[i, j] = __visible1(dtmp, wsize, w2+1, dh)
            vd2[i, j] = __visible2(dtmp, w2+1, dh)
            dtmp = np.zeros(wsize)
            for k in range(wsize):
                dtmp[k] = data[i+w2-k, j-w2+k]
            vd3[i, j] = __visible1(dtmp, wsize, w2+1, dh)
            vd4[i, j] = __visible2(dtmp, w2+1, dh)

    vtot = vn+vs+ve+vw+vd1+vd2+vd3+vd4
    vtot = vtot[w2:nr-w2, w2:nc-w2]

    for j in piter(range(nc)):
        for i in range(nr):
            vstd[i, j] = np.std([vn[i, j], vs[i, j], ve[i, j], vw[i, j],
                                 vd1[i, j], vd2[i, j], vd3[i, j],
                                 vd4[i, j]], ddof=1)

    vstd = vstd[w2:nr-w2, w2:nc-w2]

    dtr = np.pi/180
    c45 = np.cos(45*dtr)
    s45 = np.sin(45*dtr)
    vsumx = ve-vw+vd1*c45-vd2*c45+vd3*c45-vd4*c45
    vsumy = vn-vs+vd1*s45-vd2*s45-vd3*s45+vd4*s45
    vsum = np.sqrt(vsumx*vsumx+vsumy*vsumy)
    vsum = vsum[w2:nr-w2, w2:nc-w2]

    vtot = np.ma.array(vtot)
    vstd = np.ma.array(vstd)
    vsum = np.ma.array(vsum)
    vtot.mask = mask[w2:-w2, w2:-w2]
    vstd.mask = mask[w2:-w2, w2:-w2]
    vsum.mask = mask[w2:-w2, w2:-w2]

    return vtot, vstd, vsum


def __visible1(dat, nr, cp, dh):
    """
    Visible 1.

    Parameters
    ----------
    dat : numpy array
        Input vector.
    nr : int
        Window size. Must be odd.
    cp : int
        Center point.
    dh : float
        Observer height.

    Returns
    -------
    num : int
        Output.

    """
    num = 1

    if cp < nr-1 and dat.size > 0:
        num = 2
        cpn = cp-1
        thetamax = float(dat[cpn+1]-dat[cpn]-dh)
        for i in range(cpn+2, nr):
            theta = ((dat[i]-dat[cpn]-dh)/float(i-cpn))
            if theta >= thetamax:
                num = num + 1
                thetamax = theta

    return num


def __visible2(dat, cp, dh):
    """
    Visible 2.

    Parameters
    ----------
    dat : numpy array
        Input vector.
    cp : int
        Center point.
    dh : float
        Observer height.

    Returns
    -------
    num : int
        Output.

    """
    num = 0

    if cp > 2 and dat.size > 0:
        num = 1
        cpn = cp-1
        thetamax = (dat[cpn-1]-dat[cpn]-dh)
        for i in range(cpn-2, -1, -1):
            theta = ((dat[i]-dat[cpn]-dh)/(cpn-i))
            if theta >= thetamax:
                num = num + 1
                thetamax = theta
    return num


def nextpow2(n):
    """
    Next power of 2.

    Parameters
    ----------
    n : float or numpy array
        Current value.

    Returns
    -------
    m_i : float or numpy array
        Output.

    """
    m_i = np.ceil(np.log2(np.abs(n)))
    return m_i


def vertical(data, npts=None, xint=1, order=1):
    """
    Vertical derivative.

    Parameters
    ----------
    data : numpy array
        Input data.
    npts : int, optional
        Number of points. The default is None.
    xint : float, optional
        X interval. The default is 1.
    order : int, optional
        Order. The default is 1.

    Returns
    -------
    dz : numpy array
        Output data

    """
    nr, nc = data.shape

    z = data-np.ma.median(data)
    if np.ma.is_masked(z):
        z = z.filled(0.)

    if npts is None:
        nmax = np.max([nr, nc])
        npts = int(2**nextpow2(nmax))

    cdiff = int(np.floor((npts-nc)/2))
    rdiff = int(np.floor((npts-nr)/2))
    cdiff2 = npts-cdiff-nc
    rdiff2 = npts-rdiff-nr
    data1 = np.pad(z, [[rdiff, rdiff2], [cdiff, cdiff2]], 'edge')

    f = np.fft.fft2(data1)
    fz = f
    wn = 2.0*np.pi/(xint*(npts-1))
    f = np.fft.fftshift(f)
    cx = npts/2+1
    cy = cx
    for i in range(npts):
        freqx = (i+1-cx)*wn
        for j in range(npts):
            freqy = (j+1-cy)*wn
            freq = np.sqrt(freqx*freqx+freqy*freqy)
            fz[i, j] = f[i, j]*freq**order
    fz = np.fft.fftshift(fz)
    fzinv = np.fft.ifft2(fz)
    dz = np.real(fzinv[rdiff:nr+rdiff, cdiff:nc+cdiff])

    return dz


class AGC(QtWidgets.QDialog):
    """
    Class used to gather information via a GUI, for function AGC.

    Attributes
    ----------
    parent : parent
    indata : dictionary
        PyGMI input data in a dictionary
    outdata :
        PyGMI input data in a dictionary
    wsize : int
        window size, must be odd
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.wsize = 11

        self.sb_wsize = QtWidgets.QSpinBox()
        self.rb_mean = QtWidgets.QRadioButton('Mean')
        self.rb_median = QtWidgets.QRadioButton('Median')
        self.rb_rms = QtWidgets.QRadioButton('RMS')

        self.setupui()

        self.sb_wsize.setValue(self.wsize)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.cooper.AGC')
        label_2 = QtWidgets.QLabel('Window Size (Odd)')

        self.sb_wsize.setPrefix('')
        self.sb_wsize.setMinimum(3)
        self.sb_wsize.setMaximum(100000)
        self.sb_wsize.setSingleStep(2)
        self.sb_wsize.setValue(self.wsize)
        self.rb_mean.setChecked(True)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Automatic Gain Control')

        gridlayout.addWidget(self.rb_mean, 0, 0, 1, 1)
        gridlayout.addWidget(self.rb_median, 1, 0, 1, 1)
        gridlayout.addWidget(self.rb_rms, 2, 0, 1, 1)
        gridlayout.addWidget(label_2, 3, 0, 1, 1)
        gridlayout.addWidget(self.sb_wsize, 3, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 4, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 4, 1, 1, 1)

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

        atype = 'mean'
        if self.rb_median.isChecked():
            atype = 'median'
        if self.rb_rms.isChecked():
            atype = 'rms'

        self.wsize = self.sb_wsize.value()

        data = copy.deepcopy(self.indata['Raster'])
        data2 = []

        for i, datai in enumerate(data):
            self.showprocesslog(datai.dataid+':')

            agcdata = agc(datai.data, self.wsize, atype, piter=self.piter)
            data2.append(copy.deepcopy(datai))
            data2[-1].data = agcdata
            data2[-1].dataid += ' AGC'

        self.outdata['Raster'] = data2
        self.showprocesslog('Finished!')

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
        self.sb_wsize.setValue(projdata['wsize'])
        self.sb_dh.setValue(projdata['vheight'])

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

        projdata['wsize'] = self.sb_wsize.value()
        projdata['vheight'] = self.sb_dh.value()

        return projdata


def agc(data, wsize, atype='mean', piter=iter):
    """
    AGC for map data, based on code by Gordon Cooper.

    Parameters
    ----------
    data : numpy array
        Raster data.
    wsize : int
        Window size, must be odd.
    atype : str
        AGC type - can be median or mean.

    Returns
    -------
    agcdata : numpy array
        Output AGC data
    """
    data = data.copy()-data.min()
    nr, nc = data.shape

    weight = np.ones((nr, nc))
    w2 = int(np.floor(wsize/2))

    for i in piter(range(w2, nr-w2)):
        for j in range(w2, nc-w2):
            w = data[i-w2:i+w2+1, j-w2:j+w2+1]
            if atype == 'mean':
                weight[i, j] = np.mean(np.abs(w))
            elif atype == 'median':
                weight[i, j] = np.median(np.abs(w))
            elif atype == 'rms':
                weight[i, j] = np.sqrt(np.mean(w**2))

    agcdata = data/weight

    return agcdata
