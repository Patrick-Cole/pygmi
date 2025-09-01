# -----------------------------------------------------------------------------
# Name:        matchedfilt.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2025 Council for Geoscience
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
"""
Quick start routine to start the GUI form of PyGMI.

This routine is used as a convenience function, typically if you do NOT
formally install PyGMI as a library and prefer to run it from within the
default extracted directory structure.
"""
import numpy as np
from scipy import signal
import pwlf
from PySide6 import QtWidgets
from matplotlib import style, gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT

from pygmi.raster.misc import lstack
from pygmi.misc import BasicModule
from pygmi.raster.fft import fftprep, calculate_raps


class MatchedFilt(BasicModule):
    """
    Primary class for matched filtering.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    self.mmc : FigureCanvas
        main canvas containing the image
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.datapad = None
        self.fftdata = None
        self.depth = None
        self.filt = None
        self.datamedian = 0
        self.sos = None

        self.figure = Figure()
        self.mmc = FigureCanvasQTAgg(self.figure)

        self.cmb_band1 = QtWidgets.QComboBox()
        self.cmb_dtype = QtWidgets.QComboBox()
        self.sb_nsegs = QtWidgets.QSpinBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.buttonbox.htmlfile = 'mag.dm.tiltdepth'

        lbl_1 = QtWidgets.QLabel('Band to perform Filtering:')
        # lbl_2 = QtWidgets.QLabel('Data Type:')
        lbl_3 = QtWidgets.QLabel('Number of depth slices:')

        pb_calculate = QtWidgets.QPushButton('Recalculate')

        self.cmb_dtype.addItems(['Magnetic', 'Gravity'])
        self.sb_nsegs.setMinimum(2)
        self.sb_nsegs.setProperty('value', 2)

        vbl_raster = QtWidgets.QVBoxLayout()
        hbl_all = QtWidgets.QHBoxLayout(self)
        vbl_right = QtWidgets.QVBoxLayout()

        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        spacer = QtWidgets.QSpacerItem(20, 40,
                                       QtWidgets.QSizePolicy.Policy.Minimum,
                                       QtWidgets.QSizePolicy.Policy.Expanding)

        self.setWindowTitle('Matched Filtering')

        vbl_raster.addWidget(lbl_1)
        vbl_raster.addWidget(self.cmb_band1)
        # vbl_raster.addWidget(lbl_2)
        # vbl_raster.addWidget(self.cmb_dtype)
        vbl_raster.addWidget(lbl_3)
        vbl_raster.addWidget(self.sb_nsegs)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(pb_calculate)
        vbl_raster.addWidget(self.buttonbox)

        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)

        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.sb_nsegs.valueChanged.connect(self.calculate)
        self.cmb_band1.currentIndexChanged.connect(self.fftprep)
        self.cmb_dtype.currentIndexChanged.connect(self.calculate)
        pb_calculate.pressed.connect(self.calculate)

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

        self.indata['Raster'] = lstack(self.indata['Raster'])

        data = self.indata['Raster']
        blist = []
        for i in data:
            blist.append(i.dataid)

        self.cmb_band1.currentIndexChanged.disconnect()
        self.cmb_band1.clear()
        self.cmb_band1.addItems(blist)
        self.cmb_band1.currentIndexChanged.connect(self.fftprep)

        self.fftprep()

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        odat = []
        nsegs = self.sb_nsegs.value()
        for i in range(nsegs):
            zout = np.real(np.fft.ifft2(self.fftdata * self.filt[i]))
            zout = zout + self.datamedian
            tmp = self.datapad.copy()
            tmp.data = np.ma.array(zout)
            tmp.dataid = f'depth {self.depth[i]:.2f}'
            tmp = lstack([tmp, self.data], piter=self.piter,
                         showlog=self.showlog,
                         masterid=self.data.dataid, commonmask=True)[0]

            odat.append(tmp)

        self.outdata['Raster'] = odat

        return True

    def calculate(self):
        """Calculate matched filter."""
        nsegs = self.sb_nsegs.value()

        # Calculate the radially averaged power spectrum
        x_data, power_data, k, self.fftdata = calculate_raps(self.datapad)

        # n = -2.9
        n = 0
        y_data = np.log(power_data / x_data**(n))

        my_pwlf = pwlf.PiecewiseLinFit(x_data, y_data)

        m = []
        i = nsegs
        while len(m) < nsegs:
            breaks = my_pwlf.fit(i)
            m1 = my_pwlf.calc_slopes()
            m = np.array(m1)
            breaks1 = breaks[:-1][m < 0]
            breaks2 = breaks[1:][m < 0]
            m = m[m < 0]
            i += 1

        # x0 = breaks1
        # logy0 = my_pwlf.predict(x0)

        d = -m / 2
        # c = np.sqrt(np.exp(logy0 - m * x0))

        # Filter just for plot
        f = getbutter(breaks1, breaks2, x_data)

        # fsum = 0
        # for i in range(nsegs):
        #     fsum += c[i] * x_data**(n / 2) * np.exp(-x_data * d[i])

        # f = []
        # for i in range(nsegs):
        #     f.append(c[i] * x_data**(n / 2) * np.exp(-x_data * d[i]) / fsum)

        # f1 = 1 / (1 + c[1] / c[0] * np.exp(x_data * (d[0] - d[1])))
        # f2 = 1 - f1
        # f = [f1, f2]

        # Filter to apply to data
        # fsum = 0
        # for i in range(nsegs):
        #     fsum += c[i] * k**(n / 2) * np.exp(-k * d[i])

        self.filt = []
        for i in range(nsegs):
            # self.filt.append(c[i] * k**(n / 2) * np.exp(-k * d[i]) / fsum)
            self.filt.append(np.interp(k, x_data, f[i]))

        self.depth = d

        # Plotting
        style.use('bmh')
        self.figure.clear()
        gs = gridspec.GridSpec(3, 1)
        axes = self.figure.add_subplot(gs[:2, 0])
        axes.scatter(x_data, y_data, label="Data", color="blue", s=2)

        for i in range(nsegs):
            xtmp = [breaks1[i], breaks2[i]]
            axes.plot(xtmp, my_pwlf.predict(xtmp), f'C{i + 1}',
                      label=f'Depth: {d[i]:.2f}')

        for i in breaks1:
            axes.axvline(x=i, color="green", linestyle="--")
        for i in breaks2:
            axes.axvline(x=i, color="green", linestyle="--")

        axes.legend(fontsize=8)
        axes.set_xlabel("$Wavenumbers (k)$", fontsize=10)
        # axes.set_ylabel(r"$\ln(Power/k^{-2.9})$", fontsize=10)
        axes.set_ylabel(r"$\ln(Power)$", fontsize=10)
        axes.set_title("Piecewise Linear Fit", fontsize=10)
        axes.tick_params(axis='x', labelsize=8)
        axes.tick_params(axis='y', labelsize=8)

        axes = self.figure.add_subplot(gs[2, 0])
        axes.tick_params(axis='x', labelsize=8)
        axes.tick_params(axis='y', labelsize=8)
        axes.set_title("FFT Filters", fontsize=10)
        for i, fi in enumerate(f):
            axes.plot(x_data, abs(fi), f'C{i + 1}')
        axes.set_xlabel("$Wavenumbers (k)$", fontsize=10)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def fftprep(self):
        """FFT preparation when choosing band."""
        txt = str(self.cmb_band1.currentText())
        for i in self.indata['Raster']:
            if i.dataid == txt:
                self.data = i
                break

        self.datapad, self.datamedian = fftprep(
            self.data, self.showlog, self.piter)
        self.calculate()

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_band1)


def getbutter(lowcut, highcut, f, order=5):
    """
    Create Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : list of floats
        Low cutoff frequencys.
    highcut : list of floats
        High cutoff frequencys.
    f : numpy array
        List of frequencies, ending in nyquist frequency.
    order : int
        Order of the filter.

    Returns
    -------
    filt : list
        List of 1D butterworth filters.

    """
    filt = []
    nq = f[-1]
    fs = nq * 2
    for i, low in enumerate(lowcut):
        high = highcut[i]
        if high / nq == 1.0:
            sos = signal.butter(
                order, low / nq, btype='highpass', output='sos')
        elif low == 0.0:
            sos = signal.butter(
                order, high / nq, btype='lowpass', output='sos')
        else:
            sos = signal.butter(
                order, [low / nq, high / nq], btype='bandpass', output='sos')

        _, h = signal.freqz_sos(sos, fs=fs, worN=f)

        filt.append(h)

    return filt


def _testfn():
    """Testing routine."""
    import sys
    from pygmi.raster.iodefs import get_raster

    ifile = r"c:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"
    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\Tilt\tilt.tif"
    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\Matched Filtering\mod400200.tif"
    ifile = r"D:\Heliium_Highresmag_utm35s.hdr"

    dat = get_raster(ifile)

    _ = QtWidgets.QApplication(sys.argv)

    tmp1 = MatchedFilt()
    tmp1.indata['Raster'] = dat

    tmp1.settings()

    dat = tmp1.outdata


def _testfft():
    """Test FFT."""
    from scipy import signal
    import matplotlib.pyplot as plt

    b, a = signal.butter(4, [.2, .4], 'band')
    w, h = signal.freqz(b, a, fs=100)
    plt.plot(w, abs(h))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')  # cutoff frequency
    plt.show()


if __name__ == "__main__":
    _testfn()
    # _testfft()
