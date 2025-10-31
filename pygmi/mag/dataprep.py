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

from PySide6 import QtWidgets
import numpy as np
from scipy import signal

from pygmi.raster.dataprep import verticalp
from pygmi.misc import BasicModule
from pygmi.raster.misc import lstack
from pygmi.raster.fft import fftprep, fft_getkxy


class ASig(BasicModule):
    """
    Class used to gather information via a GUI.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    azi : float
        directional filter azimuth in degrees from East
    smooth : int
        size of smoothing matrix to use - must be odd input 0 for no smoothing
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_1 = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = 'mag.dm.asig'

        self.setWindowTitle('Analytic Signal')

        gl_1.addWidget(self.buttonbox, 3, 0, 1, 2)

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
            temp = self.exec()
            if temp == 0:
                return False

        data = [i.copy() for i in self.indata['Raster']]
        data2 = []

        for i in self.piter(range(len(data))):
            asignal = asig(data[i])
            data2.append(data[i].copy())
            data2[-1].data = asignal
            data2[-1].dataid += ' Analytic Signal'

        for i in data2:
            if i.nodata is None:
                continue
            i.data.data[i.data.mask] = i.nodata

        self.outdata['Raster'] = data2
        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.sb_s)
        self.saveobj(self.sb_azi)
        self.saveobj(self.sb_k)


def asig(data1, showlog=print, piter=iter):
    """
    Tilt angle calculations.

    Based on work by Gordon Cooper (School of Geosciences, University of the
                                    Witwatersrand, Johannesburg, South Africa)

    Parameters
    ----------
    data1 : pygmi.raster.datatypes.Data
        data with matrix of double to be filtered

    Returns
    -------
    asig1 : numpy masked array
        Analytic signal
    """
    data = data1.data
    dmin = data.min()
    dmax = data.max()
    dm = 0.5 * (dmin + dmax)
    data.data[data.mask] = dm
    data[np.isnan(data)] = dm
    data[np.isinf(data)] = dm

    dy, dx = np.gradient(data, data1.ydim, data1.xdim)

    dz = verticalp(data1, showlog=showlog, piter=piter)
    asig1 = np.ma.sqrt(dx * dx + dy * dy + dz * dz)

    return asig1


class Tilt1(BasicModule):
    """
    Class used to gather information via a GUI, for function tilt1.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

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
        self.sb_k = QtWidgets.QSpinBox()

        self.setupui()

        self.sb_s.setValue(self.smooth)
        self.sb_azi.setValue(self.azi)
        self.sb_k.setValue(2)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_1 = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = 'mag.dm.tilt'

        lbl_1 = QtWidgets.QLabel('Azimuth (degrees from east)')
        lbl_2 = QtWidgets.QLabel('Smoothing Matrix Size (Odd, 0 for None)')
        lbl_3 = QtWidgets.QLabel('EHGA k factor (2 or greater)')

        self.sb_azi.setMinimum(-360)
        self.sb_azi.setMaximum(360)
        self.sb_azi.setProperty('value', 0)
        self.sb_s.setPrefix('')
        self.sb_s.setMinimum(0)
        self.sb_s.setMaximum(100000)
        self.sb_s.setSingleStep(1)
        self.sb_k.setMinimum(1)
        self.sb_k.setMaximum(1000)

        self.setWindowTitle('Tilt Angle')

        gl_1.addWidget(lbl_2, 0, 0, 1, 1)
        gl_1.addWidget(self.sb_s, 0, 1, 1, 1)
        gl_1.addWidget(lbl_1, 1, 0, 1, 1)
        gl_1.addWidget(self.sb_azi, 1, 1, 1, 1)
        gl_1.addWidget(lbl_3, 2, 0, 1, 1)
        gl_1.addWidget(self.sb_k, 2, 1, 1, 1)
        gl_1.addWidget(self.buttonbox, 3, 0, 1, 2)

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
            temp = self.exec()
            if temp == 0:
                return False

        self.smooth = self.sb_s.value()
        self.azi = self.sb_azi.value()
        kval = self.sb_k.value()

        data = [i.copy() for i in self.indata['Raster']]
        data2 = []

        for i in self.piter(range(len(data))):
            t1, th, t2, ta, tdx, tahg, ehga = tilt1(data[i], self.azi,
                                                    self.smooth, kval)
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2.append(data[i].copy())
            data2[-7].data = t1
            data2[-6].data = th
            data2[-5].data = t2
            data2[-4].data = ta
            data2[-3].data = tdx
            data2[-2].data = tahg
            data2[-1].data = ehga
            data2[-7].dataid += ' Standard Tilt Angle'
            data2[-6].dataid += ' Hyperbolic Tilt Angle'
            data2[-5].dataid += ' 2nd Order Tilt Angle'
            data2[-4].dataid += ' Tilt Based Directional Derivative'
            data2[-3].dataid += ' Total Derivative'
            data2[-2].dataid += ' Tilt Angle of the Horizontal Gradient'
            data2[-1].dataid += ' Enhanced Horizontal Gradient Amplitude'

        for i in data2:
            if i.nodata is None:
                continue
            i.data.data[i.data.mask] = i.nodata

        self.outdata['Raster'] = data2
        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.sb_s)
        self.saveobj(self.sb_azi)
        self.saveobj(self.sb_k)


def tilt1(data1, azi, s, k=2, showlog=print, piter=iter):
    """
    Tilt angle calculations.

    Based on work by Gordon Cooper (School of Geosciences, University of the
                                    Witwatersrand, Johannesburg, South Africa)

    Parameters
    ----------
    data1 : pygmi.raster.datatypes.Data
        data with matrix of double to be filtered
    azi : float
        directional filter azimuth in degrees from East
    s : int
        size of smoothing matrix to use - must be odd input 0 for no smoothing
    k : int
        Factor for EHGA filter. Must be > 0. Optional.

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
    tahg : numpy masked array
        Tilt Angle of the Horizontal Gradient
    ehga : numpy masked array
        Enhanced Horizontal Gradient Amplitude
    """
    data = data1.data
    dmin = data.min()
    dmax = data.max()
    dm = 0.5 * (dmin + dmax)
    data.data[data.mask] = dm
    data[np.isnan(data)] = dm
    data[np.isinf(data)] = dm

    if s > 0:
        se = np.ones((s, s)) / (s * s)
        data2 = signal.convolve2d(data, se, 'valid')  # smooth
        mask = np.ma.getmaskarray(data.data)
        mask = signal.convolve2d(mask, se, 'valid')
        data = np.ma.array(data2, mask=mask)

    dtr = np.pi / 180.0
    azi = azi * dtr

    dy, dx = np.gradient(data, data1.ydim, data1.xdim)
    dxtot = np.ma.sqrt(dx * dx + dy * dy)
    dz = verticalp(data1, showlog=showlog, piter=piter)
    t1 = np.ma.arctan2(dz, dxtot)
    th = np.real(np.arctanh(np.nan_to_num(dz / dxtot) + (0 + 0j)))

    tdx = np.real(np.ma.arctan2(dxtot, abs(dz)))

    # Standard directional derivative
    dx1 = dx * np.cos(azi) + dy * np.sin(azi)
    dx2 = dx * np.cos(azi + np.pi / 2) + dy * np.sin(azi + np.pi / 2)
    dxz = np.ma.sqrt(dx2 * dx2 + dz * dz)
    ta = np.ma.arctan2(dx1, dxz)         # Tilt directional derivative

    # 2nd order Tilt angle

    s = max(s, 3)
    se = np.ones([s, s]) / (s * s)
    ts = signal.convolve2d(t1.filled(t1.mean()), se, 'same')
    ts = np.ma.array(ts, mask=t1.mask)
    ts = data1.copy(ts)

    dxs, dys = np.gradient(ts.data, data1.ydim, data1.xdim)
    dzs = verticalp(ts, showlog=showlog, piter=piter)
    dxtots = np.ma.sqrt(dxs * dxs + dys * dys)
    t2 = np.ma.arctan(dzs, dxtots)

    # Standard tilt angle, hyperbolic tilt angle, 2nd order tilt angle,
    # Tilt Based Directional Derivative, Total Derivative

    data = data1.copy(dxtot)
    dy, dx = np.gradient(data.data, data1.ydim, data1.xdim)
    dxtot = np.ma.sqrt(dx * dx + dy * dy)
    dz = verticalp(data, showlog=showlog, piter=piter)
    tahg = np.ma.arctan2(dz, dxtot)

    dxyztot = np.ma.sqrt(dx * dx + dy * dy + dz * dz)

    ehga = np.ma.arcsin(k * (dz / dxyztot - 1) + 1)

    ehga = k * (dz / dxyztot - 1) + 1
    ehga[ehga < -1.0] = -1.0
    ehga = np.ma.arcsin(ehga)

    return t1, th, t2, ta, tdx, tahg, ehga


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


class RTP(BasicModule):
    """
    Perform Reduction to the Pole on Magnetic data.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmb_dataid = QtWidgets.QComboBox()
        self.dsb_inc = QtWidgets.QDoubleSpinBox()
        self.dsb_dec = QtWidgets.QDoubleSpinBox()
        self.dsb_inca = QtWidgets.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = 'mag.dm.rtp'

        lbl_band = QtWidgets.QLabel('Band to Reduce to the Pole:')
        lbl_inc = QtWidgets.QLabel('Inclination of Magnetic Field:')
        lbl_inca = QtWidgets.QLabel(
            'Amplitude Correction Inclination for low latitudes:')
        lbl_dec = QtWidgets.QLabel('Declination of Magnetic Field:')

        self.dsb_inc.setMaximum(90.0)
        self.dsb_inc.setMinimum(-90.0)
        self.dsb_inca.setMaximum(90.0)
        self.dsb_inca.setMinimum(-90.0)
        self.dsb_dec.setMaximum(360.0)
        self.dsb_dec.setMinimum(-360.0)
        self.dsb_inc.setValue(-62.5)
        self.dsb_inca.setValue(20)
        self.dsb_dec.setValue(-16.75)

        self.setWindowTitle('Reduction to the Pole')

        gl_main.addWidget(lbl_band, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_dataid, 0, 1, 1, 1)

        gl_main.addWidget(lbl_inc, 1, 0, 1, 1)
        gl_main.addWidget(self.dsb_inc, 1, 1, 1, 1)
        gl_main.addWidget(lbl_dec, 2, 0, 1, 1)
        gl_main.addWidget(self.dsb_dec, 2, 1, 1, 1)
        gl_main.addWidget(lbl_inca, 3, 0, 1, 1)
        gl_main.addWidget(self.dsb_inca, 3, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 4, 0, 1, 4)

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

        self.cmb_dataid.clear()
        self.cmb_dataid.addItems(tmp)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.dsb_inc)
        self.saveobj(self.dsb_dec)

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
        Ia = self.dsb_inca.value()

        newdat = []
        for data in self.piter(self.indata['Raster']):
            if data.dataid != self.cmb_dataid.currentText():
                continue
            dat = rtp(data, I_deg, D_deg, Ia, self.showlog, self.piter)
            newdat.append(dat)

        self.outdata['Raster'] = newdat


def rtp(data, I_deg, D_deg, Ia=20, showlog=print, piter=iter):
    """
    Reduction to the pole.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        PyGMI raster data.
    I_deg : float
        Magnetic inclination.
    D_deg : float
        Magnetic declination.
    Ia : float
        Amplitude correction inclination Ia in degree. The default is 20.
    showlog : function, optional
        Show information using a function. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    dat : pygmi.raster.datatypes.Data
        PyGMI raster data.

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, datamedian = fftprep(data, showlog=showlog, piter=piter)

    fftmod = np.fft.fft2(ndat.data)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    # Ia = min(90, max(0, ((int((0.008 * I_deg**2 - 1.71 * abs(I_deg) + 80) / 2.5)) * 2.5)))

    Ia = np.deg2rad(Ia)
    I = np.deg2rad(I_deg)
    D = np.deg2rad(D_deg)
    alpha = np.arctan2(KX, KY)

    if abs(I) >= abs(Ia):
        Ia = I

    filt = (np.sin(I) - 1j * np.cos(I) * np.cos(D - alpha))**2
    filt2 = (np.sin(Ia)**2 + np.cos(Ia)**2 * np.cos(D - alpha)**2)
    filt2 = filt2 * (np.sin(I)**2 + np.cos(I)**2 * np.cos(D - alpha)**2)
    filt = filt / filt2

    zout = np.real(np.fft.ifft2(fftmod * filt))

    zout = zout + datamedian
    dat = ndat.copy()
    dat.data = np.ma.array(zout)
    dat.dataid = 'RTP_' + data.dataid
    dat = lstack([dat, data], piter=piter,
                 showlog=showlog,
                 masterid=data.dataid, commonmask=True)[0]

    return dat


def gradient2D(daty, datx):
    """Perform 2D gradient where spacing is inconsistent in 2D."""
    rows, cols = daty.data.shape
    dx = []
    # dy = []
    dx = daty.copy()
    for i in range(rows):
        mask = daty[i].mask
        tmpy = daty[i].compressed()
        if tmpy.size == 0:
            continue
        elif tmpy.size == 1:
            dx[i][~mask] = 0
            continue
        tmpx = datx[i][~mask]
        dx[i][~mask] = np.gradient(tmpy, tmpx)

    pass
    # dx = np.ma.array(dx)

    # for i in range(cols):
    #     dy.append(np.gradient(daty.data[:, i], datx[:, i]))

    # dy = np.ma.masked_invalid(dy)

    # return dx, dy
    return dx


def _testfn_rtp():
    """RTP testing routine."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from pygmi.pfmod.grvmag3d import quick_model, calc_field

    # quick model
    finc = -57
    fdec = 50

    # finc = 0
    # fdec = 50

    lmod = quick_model(numx=300, numy=300, numz=30, finc=finc, fdec=fdec)
    lmod.lith_index[100:200, 100:200, 0:10] = 1
    lmod.mht = 100
    calc_field(lmod, magcalc=True)

    # Calculate the field

    # magval = lmod.griddata['Calculated Magnetics'].data

    ndata = lmod.griddata['Calculated Magnetics'].copy()
    ndata.data += np.random.normal(0, .5, ndata.data.shape)
    dat2 = rtp(ndata, finc, fdec)

    plt.subplot(121)
    plt.imshow(ndata.data, cmap=colormaps['jet'])
    plt.subplot(122)
    plt.imshow(dat2.data, cmap=colormaps['jet'])
    plt.show()


def _testfn():
    """RTP testing routine."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster
    from pygmi.raster.modest_image import imshow

    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\RTP\Whole_mag_residual_modelregional_utm35s.hdr"

    dat1 = get_raster(ifile)[0]
    dat = rtp(dat1, -62.08, -14.23)
    # dat = dat1

    t1, th, t2, ta, tdx, tahg, ehga = tilt1(dat, 75, 0)

    # dy, dx = np.gradient(dat.data, dat.ydim, dat.xdim)
    # dxtot = np.ma.sqrt(dx * dx + dy * dy)
    # dz = verticalp(dat)
    # t1 = np.ma.arctan2(dz, dxtot)

    plt.figure()
    # ax = plt.subplot(221)
    # dat1.plot(ax)
    # ax = plt.subplot(222)
    # dat.plot(ax)
    # plt.subplot(223)
    # vmin = dz.mean() - dz.std()
    # vmax = dz.mean() + dz.std()
    # plt.imshow(dz, interpolation='none', vmin=vmin, vmax=vmax)
    # plt.subplot(224)
    # vmin = dxtot.mean() - dxtot.std()
    # vmax = dxtot.mean() + dxtot.std()
    # plt.imshow(dxtot, interpolation='none', vmin=vmin, vmax=vmax)
    # plt.subplot(224)
    ax = plt.gca()
    vmin = t1.mean() - 2.5 * t1.std()
    vmax = t1.mean() + 2.5 * t1.std()
    imshow(ax, t1, extent=dat.extent, interpolation='none', vmin=vmin,
           vmax=vmax)
    plt.show()


def _testfn_vert():
    """RTP testing routine."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster
    from pygmi.raster.dataprep import verticalp

    ifile = r"D:\Workdata\PyGMI Test Data\Magnetics\tilt\tilt.tif"
    ifile = r"D:\mergemag5_IGRFremoved_RTP.hdr"

    zout = get_raster(ifile)[0]

    dzp = verticalp(zout)

    plt.figure(dpi=150)
    plt.imshow(dzp, interpolation='none', vmin=-1, vmax=1.5)
    plt.colorbar()


def _testfn2():
    from pygmi.raster.iodefs import get_raster
    from pygmi.mag.igrf import calc_igrf
    import matplotlib.pyplot as plt
    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\RTP\Whole_mag_residual_modelregional_utm35s.hdr"
    dfile = r"D:\workdata\PyGMI Test Data\Magnetics\RTP\Areas_A_and_B_DTM_utm35s.hdr"

    datm = get_raster(ifile)[0]
    datd = get_raster(ifile)[0]
    # dat = rtp(dat1, -62.5, -16.75)
    # sdate = sdate.year() + sdate.dayOfYear() / sdate.daysInYear()

    dat = calc_igrf(datd, 2007 + 335 / 365, igrfonly=False, sen_alt=80)

    igrf, inc, dec = dat[0]
    fmean, imean, dmean = dat[1:]

    dinc = inc.data - imean
    ddec = dec.data - dmean

    datr = rtp(datm, imean, dmean)
    datr.nodata = 0
    datr.set_mask()
    dinc = dinc.filled(0)
    dinc = np.ma.array(dinc, mask=datr.data.mask)
    ddec = ddec.filled(0)
    ddec = np.ma.array(ddec, mask=datr.data.mask)

    drdix = gradient2D(datr.data, inc.data)
    drdiy = gradient2D(datr.data.T, inc.data.T)
    d2rdi2x = gradient2D(drdix, inc.data)
    d2rdi2y = gradient2D(drdiy, inc.data.T)

    drddx = gradient2D(datr.data, dec.data)
    drddy = gradient2D(datr.data.T, dec.data.T)
    d2rdd2x = gradient2D(drddx, dec.data)
    d2rdd2y = gradient2D(drddy, dec.data.T)

    rtpx = datr.data + (dinc * drdix + .5 * dinc**2 * d2rdi2x +
                        ddec * drddx + .5 * ddec**2 * d2rdd2x)

    plt.figure()
    plt.subplot(121)
    plt.imshow(datr.data)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(rtpx)
    plt.colorbar()
    plt.show()

    pass


if __name__ == "__main__":
    _testfn_vert()
