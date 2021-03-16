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
"""Plot Vector Data using Matplotlib."""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import segyio


class GraphWindow(QtWidgets.QDialog):
    """Graph Window - Main QT Dialog class for graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)

        self.setFocus()

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """


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

    def update_segy(self, data):
        """
        Update the plot from point data.

        Parameters
        ----------
        data : segyio object
            SEG-Y data

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(111, label='SEG-Y')

        ax1.set_title('SEG-Y')
        self.axes = ax1

        samplecnt = data.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT]
        sampleint = data.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]/1000.
        rectime = samplecnt*sampleint

        y = np.linspace(0, rectime, samplecnt)
        start = 0
        finish = samplecnt
        offset = 0
        gdist = 1  # in meters
        tracemult = 2.

        ax1.invert_yaxis()

        tmax = -1
        for trace in data.trace:
            tmp = np.max(np.abs([trace.max(), trace.min()]))
            if tmax < tmp:
                tmax = tmp
        tracemult = 1/tmax

        for trace in data.trace:
            x = tracemult*trace+offset
            ax1.plot(x[start:finish], y[start:finish], 'k-')

            ax1.fill_betweenx(y, offset, x, where=(x > offset), color='k')

            ax1.set_xlabel('Trace Number')
            ax1.set_ylabel('Time (ms)')

            offset += gdist

        self.figure.tight_layout()
        self.figure.canvas.draw()


class PlotSEGY(GraphWindow):
    """Plot SEG-Y Class."""

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
        data = self.indata['ESEIS']
        self.mmc.update_segy(data)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        self.change_band()
