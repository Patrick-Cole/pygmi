# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
"""A set of Magnetic Data routines."""

import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np
from scipy.signal import tukey
import scipy.interpolate as si
from scipy import signal
from pygmi import menu_default

from pygmi.raster.datatypes import Data
from pygmi.misc import BasicModule


class Tilt1(BasicModule):
    """
    Class used to gather information via a GUI, for function tilt1.

    Attributes
    ----------
    azi : float
        directional filter azimuth in degrees from East
    smooth : int
        size of smoothing matrix to use - must be odd input 0 for no smoothing
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.azi = 75
        self.smooth = 0

        self.sb_azi = QtWidgets.QSpinBox()
        self.sb_s = QtWidgets.QSpinBox()

        self.setupui()

        self.sb_s.setValue(self.smooth)
        self.sb_azi.setValue(self.azi)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.cooper.tilt')
        label = QtWidgets.QLabel('Azimuth (degrees from east)')
        label_2 = QtWidgets.QLabel('Smoothing Matrix Size (Odd, 0 for None)')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.sb_azi.setMinimum(-360)
        self.sb_azi.setMaximum(360)
        self.sb_azi.setProperty('value', 0)
        self.sb_s.setPrefix('')
        self.sb_s.setMinimum(0)
        self.sb_s.setMaximum(100000)
        self.sb_s.setSingleStep(1)

        self.setWindowTitle('Tilt Angle')

        gridlayout.addWidget(label_2, 0, 0, 1, 1)
        gridlayout.addWidget(self.sb_s, 0, 1, 1, 1)
        gridlayout.addWidget(label, 1, 0, 1, 1)
        gridlayout.addWidget(self.sb_azi, 1, 1, 1, 1)
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
        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
            return False

        if not nodialog:
            temp = self.exec_()
            if temp == 0:
                return False

        self.smooth = self.sb_s.value()
        self.azi = self.sb_azi.value()

        data = copy.deepcopy(self.indata['Raster'])
        data2 = []

        for i in self.piter(range(len(data))):
            t1, th, t2, ta, tdx, tahg = tilt1(data[i].data, self.azi,
                                              self.smooth)
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2[-6].data = t1
            data2[-5].data = th
            data2[-4].data = t2
            data2[-3].data = ta
            data2[-2].data = tdx
            data2[-1].data = tahg
            data2[-6].dataid += ' Standard Tilt Angle'
            data2[-5].dataid += ' Hyperbolic Tilt Angle'
            data2[-4].dataid += ' 2nd Order Tilt Angle'
            data2[-3].dataid += ' Tilt Based Directional Derivative'
            data2[-2].dataid += ' Total Derivative'
            data2[-1].dataid += ' Tilt Angle of the Horizontal Gradient'

        for i in data2:
            if i.nodata is None:
                continue
            i.data.data[i.data.mask] = i.nodata

        self.outdata['Raster'] = data2
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
        self.sb_s.setValue(projdata['smooth'])
        self.sb_azi.setValue(projdata['azi'])

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

        projdata['smooth'] = self.sb_s.value()
        projdata['azi'] = self.sb_azi.value()

        return projdata


def tilt1(data, azi, s):
    """
    Tilt angle calculations.

    Based on work by Gordon Cooper (School of Geosciences, University of the
                                    Witwatersrand, Johannesburg, South Africa)

    Parameters
    ----------
    data : numpy masked array
        matrix of double to be filtered
    azi : float
        directional filter azimuth in degrees from East
    s : int
        size of smoothing matrix to use - must be odd input 0 for no smoothing

    Returns
    -------
    t1 : numpy masked array
        Standard tilt angle
    th : numpy masked array
        Hyperbolic tilt angle
    t2 : numpy masked array
        Second order tilt angle
    ta : numpy masked array
        Tilt Based Directional Derivative
    tdx : numpy masked array
        Total Derivative
    """
    dmin = data.min()
    dmax = data.max()
    dm = 0.5*(dmin+dmax)
    data.data[data.mask] = dm
    data[np.isnan(data)] = dm
    data[np.isinf(data)] = dm

    if s > 0:
        se = np.ones((s, s))/(s*s)
        data2 = signal.convolve2d(data, se, 'valid')  # smooth
        mask = signal.convolve2d(data.mask, se, 'valid')
        data = np.ma.array(data2, mask=mask)

    nr, nc = data.shape
    dtr = np.pi/180.0
    azi = azi*dtr

    dy, dx = np.gradient(data)
    dxtot = np.ma.sqrt(dx*dx+dy*dy)
    nmax = np.max([nr, nc])
    npts = int(2**nextpow2(nmax))
    dz = vertical(data, npts, 1)
    t1 = np.ma.arctan(dz/dxtot)
    th = np.real(np.arctanh(np.nan_to_num(dz/dxtot)+(0+0j)))

    tdx = np.real(np.ma.arctan(dxtot/abs(dz)))

    dx1 = dx*np.cos(azi)+dy*np.sin(azi)  # Standard directional derivative
    dx2 = dx*np.cos(azi+np.pi/2)+dy*np.sin(azi+np.pi/2)
    dxz = np.ma.sqrt(dx2*dx2+dz*dz)
    ta = np.ma.arctan(dx1/dxz)         # Tilt directional derivative

    # 2nd order Tilt angle

    s = max(s, 3)
    se = np.ones([s, s])/(s*s)
    ts = signal.convolve2d(t1.filled(t1.mean()), se, 'same')
    ts = np.ma.array(ts, mask=t1.mask)

    [dxs, dys] = np.gradient(ts)
    dzs = vertical(ts, npts, 1)
    dxtots = np.ma.sqrt(dxs*dxs+dys*dys)
    t2 = np.ma.arctan(dzs/dxtots)

    # Standard tilt angle, hyperbolic tilt angle, 2nd order tilt angle,
    # Tilt Based Directional Derivative, Total Derivative

    data = dxtot
    nr, nc = data.shape
    dy, dx = np.gradient(data)
    dxtot = np.ma.sqrt(dx*dx+dy*dy)
    nmax = np.max([nr, nc])
    npts = int(2**nextpow2(nmax))
    dz = vertical(data, npts, 1)
    tahg = np.ma.arctan(dz/dxtot)

    return t1, th, t2, ta, tdx, tahg


def nextpow2(n):
    """
    Next power of 2.

    Based on work by Gordon Cooper (School of Geosciences, University of the
                                    Witwatersrand, Johannesburg, South Africa).

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

    Based on work by Gordon Cooper (School of Geosciences, University of the
                                    Witwatersrand, Johannesburg, South Africa).

    Parameters
    ----------
    data : numpy array
        Input data.
    npts : int, optional
        Number of points. The default is None.
    xint : float, optional
        X interval. The default is 1.
    order : int
        Order of derivative. The default is 1.

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


class RTP(BasicModule):
    """Perform Reduction to the Pole on Magnetic data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataid = QtWidgets.QComboBox()
        self.dsb_inc = QtWidgets.QDoubleSpinBox()
        self.dsb_dec = QtWidgets.QDoubleSpinBox()

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
        helpdocs = menu_default.HelpButton('pygmi.raster.dataprep.rtp')
        label_band = QtWidgets.QLabel('Band to Reduce to the Pole:')
        label_inc = QtWidgets.QLabel('Inclination of Magnetic Field:')
        label_dec = QtWidgets.QLabel('Declination of Magnetic Field:')

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        self.dsb_inc.setValue(-62.5)
        self.dsb_dec.setValue(-16.75)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Reduction to the Pole')

        gridlayout_main.addWidget(label_band, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.dataid, 0, 1, 1, 1)

        gridlayout_main.addWidget(label_inc, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_inc, 1, 1, 1, 1)
        gridlayout_main.addWidget(label_dec, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.dsb_dec, 2, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 3, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

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
        tmp = []
        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
            return False

        for i in self.indata['Raster']:
            tmp.append(i.dataid)

        self.dataid.clear()
        self.dataid.addItems(tmp)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        self.acceptall()

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
        self.dataid.setCurrentText(projdata['band'])
        self.dsb_inc.setValue(projdata['inc'])
        self.dsb_dec.setValue(projdata['dec'])

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

        projdata['band'] = self.dataid.currentText()
        projdata['inc'] = self.dsb_inc.value()
        projdata['dec'] = self.dsb_dec.value()

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        I_deg = self.dsb_inc.value()
        D_deg = self.dsb_dec.value()

        newdat = []
        for data in self.piter(self.indata['Raster']):
            if data.dataid != self.dataid.currentText():
                continue
            dat = rtp(data, I_deg, D_deg)
            newdat.append(dat)

        self.outdata['Raster'] = newdat


def fftprep(data):
    """
    FFT Preparation.

    Parameters
    ----------
    data : numpy array
        Input dataset.

    Returns
    -------
    zfin : numpy array.
        Output prepared data.
    rdiff : int
        rows divided by 2.
    cdiff : int
        columns divided by 2.
    datamedian : float
        Median of data.

    """
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    nr, nc = data.data.shape
    cdiff = nc//2
    rdiff = nr//2

    # Section to pad data

    nr, nc = data.data.shape

    z1 = np.zeros((nr+2*rdiff, nc+2*cdiff))-999
    x1, y1 = np.mgrid[0: nr+2*rdiff, 0: nc+2*cdiff]
    z1[rdiff:-rdiff, cdiff:-cdiff] = ndat.filled(-999)

    z1[0] = 0
    z1[-1] = 0
    z1[:, 0] = 0
    z1[:, -1] = 0

    x = x1.flatten()
    y = y1.flatten()
    z = z1.flatten()

    x = x[z != -999]
    y = y[z != -999]
    z = z[z != -999]

    points = np.transpose([x, y])

    zfin = si.griddata(points, z, (x1, y1), method='linear', fill_value=0.)

    nr, nc = zfin.shape
    zfin *= tukey(nc)
    zfin *= tukey(nr)[:, np.newaxis]

    return zfin, rdiff, cdiff, datamedian


def fft_getkxy(fftmod, xdim, ydim):
    """
    Get KX and KY.

    Parameters
    ----------
    fftmod : numpy array
        FFT data.
    xdim : float
        cell x dimension.
    ydim : float
        cell y dimension.

    Returns
    -------
    KX : numpy array
        x sample frequencies.
    KY : numpy array
        y sample frequencies.

    """
    ny, nx = fftmod.shape
    kx = np.fft.fftfreq(nx, xdim)*2*np.pi
    ky = np.fft.fftfreq(ny, ydim)*2*np.pi

    KX, KY = np.meshgrid(kx, ky)
    KY = -KY
    return KX, KY


def rtp(data, I_deg, D_deg):
    """
    Reduction to the pole.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI raster data.
    I_deg : float
        Magnetic inclination.
    D_deg : float
        Magnetic declination.

    Returns
    -------
    dat : PyGMI Data
        PyGMI raster data.

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprep(data)
    fftmod = np.fft.fft2(ndat)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    I = np.deg2rad(I_deg)
    D = np.deg2rad(D_deg)
    alpha = np.arctan2(KY, KX)

    filt = 1/(np.sin(I)+1j*np.cos(I)*np.sin(D+alpha))**2

    zout = np.real(np.fft.ifft2(fftmod*filt))
    zout = zout[rdiff:-rdiff, cdiff:-cdiff]
    zout = zout + datamedian

    zout[data.data.mask] = data.data.fill_value

    dat = Data()
    dat.data = np.ma.masked_invalid(zout)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nodata = data.data.fill_value
    dat.dataid = 'RTP_'+data.dataid
    dat.set_transform(transform=data.transform)
    dat.crs = data.crs

    return dat


def _testfn_rtp():
    """RTP testing routine."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from pygmi.pfmod.grvmag3d import quick_model, calc_field
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    # quick model
    finc = -57
    fdec = 50

    lmod = quick_model(numx=300, numy=300, numz=30, finc=finc, fdec=fdec)
    lmod.lith_index[100:200, 100:200, 0:10] = 1
    lmod.mht = 100
    calc_field(lmod, magcalc=True)

    # Calculate the field

    magval = lmod.griddata['Calculated Magnetics'].data
    plt.imshow(magval, cmap=colormaps['jet'])
    plt.show()

    dat2 = rtp(lmod.griddata['Calculated Magnetics'], finc, fdec)
    plt.imshow(dat2.data, cmap=colormaps['jet'])
    plt.show()


def _testfn():
    """RTP testing routine."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster

    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    ifile = 'd:/Workdata/bugs/detlef/TMI_norm_wdw.tif'
    ifile = r'C:/Workdata/raster/ER Mapper/magmicrolevel.PD.ers'

    dat = get_raster(ifile)[0]

    t1, th, t2, ta, tdx = tilt1(dat.data, 75, 0)

    plt.figure(dpi=150)
    plt.imshow(t2)
    plt.colorbar()
    plt.show()

    plt.figure(dpi=150)
    plt.imshow(t1)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    _testfn()
