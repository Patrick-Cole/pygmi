# -----------------------------------------------------------------------------
# Name:        graphs.py (part of PyGMI)
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
"""Plot Data using Matplotlib."""

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT


class GraphWindow(QtWidgets.QDialog):
    """Graph Window - Main QT Dialog class for graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel('Bands:')
        self.label2 = QtWidgets.QLabel('Bands:')

        hbl.addWidget(self.label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(self.label2)
        hbl.addWidget(self.combobox2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """Combo box to choose band."""


class MyMplCanvas(FigureCanvasQTAgg):
    """
    MPL Canvas class.

    This routine will also allow the picking and movement of nodes of data.
    """

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.line = None
        self.ind = None
        self.background = None

        super().__init__(fig)

        self.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self.motion_notify_callback)

    def button_release_callback(self, event):
        """
        Mouse button release callback.

        Parameters
        ----------
        event : event
            event variable.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.ind = None

    def motion_notify_callback(self, event):
        """
        Move mouse callback.

        Parameters
        ----------
        event : event
            event variable.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return

        dtmp = self.line.get_data()
        dtmp[1][self.ind] = event.ydata
        self.line.set_data(dtmp[0], dtmp[1])

        self.figure.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.figure.canvas.update()

    def onpick(self, event):
        """
        Picker event.

        Parameters
        ----------
        event : event
            event variable.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if event.mouseevent.inaxes is None:
            return False
        if event.mouseevent.button != 1:
            return False
        if event.artist != self.line:
            return True

        self.ind = event.ind
        self.ind = self.ind[len(self.ind) // 2]  # get center-ish value

        return True

    def update_line(self, data, ival, itype):
        """
        Update the plot from point data.

        Parameters
        ----------
        data : EDI data object
            EDI data.
        ival : str
            dictionary key.
        itype : str
            dictionary key.

        Returns
        -------
        None.

        """
        data1 = data[ival]

        self.figure.clear()

        ax1 = self.figure.add_subplot(411, label='Profile')

        # ax1.set_title(ival)
        self.axes = ax1
        x = 1/data1.Z.freq

        if itype == 'xy, yx':
            res1 = data1.Z.resistivity[:, 0, 1]
            res1_err = data1.Z.resistivity_err[:, 0, 1]
            res2 = data1.Z.resistivity[:, 1, 0]
            res2_err = data1.Z.resistivity_err[:, 1, 0]
            pha1 = data1.Z.phase[:, 0, 1]
            pha1_err = data1.Z.phase_err[:, 0, 1]
            pha2 = data1.Z.phase[:, 1, 0]
            pha2_err = data1.Z.phase_err[:, 1, 0]
            label1 = r'$\rho_{xy}$'
            label2 = r'$\rho_{yx}$'
            label3 = r'$\varphi_{xy}$'
            label4 = r'$\varphi_{yx}$'

        else:
            res1 = data1.Z.resistivity[:, 0, 0]
            res1_err = data1.Z.resistivity_err[:, 0, 1]
            res2 = data1.Z.resistivity[:, 1, 1]
            res2_err = data1.Z.resistivity_err[:, 1, 0]
            pha1 = data1.Z.phase[:, 0, 0]
            pha1_err = data1.Z.phase_err[:, 0, 1]
            pha2 = data1.Z.phase[:, 1, 1]
            pha2_err = data1.Z.phase_err[:, 1, 0]
            label1 = r'$\rho_{xx}$'
            label2 = r'$\rho_{yy}$'
            label3 = r'$\varphi_{xx}$'
            label4 = r'$\varphi_{yy}$'

        ax1.errorbar(x, res1, yerr=res1_err, label=label1,
                     ls=' ', marker='.', mfc='b', mec='b', ecolor='b')
        ax1.errorbar(x, res2, yerr=res2_err, label=label2,
                     ls=' ', marker='.', mfc='r', mec='r', ecolor='r')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')
        ax1.tick_params(labelbottom=False)
        ax1.grid(True)

        ax2 = self.figure.add_subplot(412, sharex=ax1)

        ax2.errorbar(x, pha1, yerr=pha1_err, label=label3,
                     ls=' ', marker='.', mfc='b', mec='b', ecolor='b')
        ax2.errorbar(x, pha2, yerr=pha2_err, label=label4,
                     ls=' ', marker='.', mfc='r', mec='r', ecolor='r')

        ax2.set_ylim(-180., 180.)

        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.legend(loc='upper left')
        ax2.set_ylabel(r'Phase (Degrees)')
        ax2.tick_params(labelbottom=False)
        ax2.grid(True)

        ax3 = self.figure.add_subplot(413, sharex=ax1)

        ax3.plot(x, data1.Tipper.mag_real, 'b.', label='real')
        ax3.plot(x, data1.Tipper.mag_imag, 'r.', label='imaginary')

        # ax3.set_ylim(-180., 180.)

        ax3.set_xscale('log')
        ax3.set_yscale('linear')
        ax3.legend(loc='upper left')
        ax3.set_ylabel(r'Tipper Magnitude')
        ax3.tick_params(labelbottom=False)
        ax3.grid(True)

        ax4 = self.figure.add_subplot(414, sharex=ax1)
        ax4.plot(x, data1.Tipper.angle_real, 'b.', label='real')
        ax4.plot(x, data1.Tipper.angle_imag, 'r.', label='imaginary')
        # ax4.set_ylim(-180., 180.)

        ax4.set_xscale('log')
        ax4.set_yscale('linear')
        ax4.legend(loc='upper left')
        ax4.set_xlabel('Period (s)')
        ax4.set_ylabel(r'Tipper Angle (Degrees)')
        ax4.grid(True)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        self.figure.tight_layout()
        self.figure.canvas.draw()


class PlotPoints(GraphWindow):
    """Plot points class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """
        Combo to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['MT - EDI']
        i = self.combobox1.currentText()
        i2 = self.combobox2.currentText()
        self.mmc.update_line(data, i, i2)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['MT - EDI']
        for i in data:
            self.combobox1.addItem(i)
        for i in ['xy, yx', 'xx, yy']:
            self.combobox2.addItem(i)

        self.label1.setText('Station Name:')
        self.label2.setText('Graph Type:')
        self.combobox1.setCurrentIndex(0)
        self.combobox2.setCurrentIndex(0)


def _testfn():
    """Test routine."""
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    from mtpy.core.mt import MT
    from mtpy.imaging.plotresponse import PlotResponse

    datadir = r'd:\Work\workdata\MT\\'
    allfiles = glob.glob(datadir+'\\*.edi')

    edi_file = allfiles[2]
    print(edi_file)
    data1 = MT(edi_file)
    PlotResponse(fn=edi_file, plot_tipper='yr')
    plt.show()

    arrow_direction = 0  # this is 0 for toward a conductor, and 1 for away

    txr = data1.Tipper.mag_real*np.sin(data1.Tipper.angle_real*np.pi/180 +
                                       np.pi*arrow_direction)
    tyr = data1.Tipper.mag_real*np.cos(data1.Tipper.angle_real*np.pi/180 +
                                       np.pi*arrow_direction)

    txi = data1.Tipper.mag_imag*np.sin(data1.Tipper.angle_imag*np.pi/180 +
                                       np.pi*arrow_direction)
    tyi = data1.Tipper.mag_imag*np.cos(data1.Tipper.angle_imag*np.pi/180 +
                                       np.pi*arrow_direction)

    num = len(txr)

    x = 1/data1.Z.freq
    x10 = np.log10(x)

    ax1 = plt.gca()
    for i in range(num):
        # tx0 = np.log10(x[i])
        # tx1 = txr[i]*np.log10(x[i])

        plt.arrow(x10[i], 0, txr[i], tyr[i])
        # print(tx0[i], 0, tx1[i], tyr[i])

    plt.tight_layout()
    plt.grid(True)
    plt.show()

    ax1 = plt.gca()
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('linear')

    for i in range(num):
        plt.plot(x[i], tyr[i])

    ax, bx = ax1.get_xlim()

    ax1.set_ylim((-0.02, 0.02))

    ay, by = ax1.get_ylim()

    ty = (tyr-ay)/(by-ay)
    ty0 = -ay/(by-ay)

    for i in range(num):
        tx0 = (np.log(x[i])-np.log(ax))/(np.log(bx)-np.log(ax))
        # tx1 = txr[i]*tx0
        # tx1 = (txr[i]*np.log(x[i])-np.log(ax))/(np.log(bx)-np.log(ax))

        plt.arrow(tx0, ty0, 2*txr[i], 2*tyr[i], transform=ax1.transAxes)

    plt.tight_layout()
    plt.grid(True)
    plt.show()

    ax1 = plt.gca()
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('linear')

    plt.plot(x, data1.Tipper.mag_real, 'b.-', label='real')
    plt.plot(x, data1.Tipper.mag_imag, 'r.-', label='imaginary')

    plt.tight_layout()
    plt.grid(True)
    plt.show()

    ax1 = plt.gca()
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('linear')

    plt.plot(x, data1.Tipper.angle_real, 'b.-', label='real')
    plt.plot(x, data1.Tipper.angle_imag, 'r.-', label='imaginary')

    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    _testfn()
