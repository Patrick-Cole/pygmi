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

import os
import copy
import glob
import numpy as np
from PySide6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt import NavigationToolbar2QT

from pygmi.misc import ContextModule
from pygmi.mt.mtpyold.imaging.phase_tensor_maps import PlotPhaseTensorMaps
from pygmi.mt.mtpyold.utils.shapefiles_creator import create_tensor_tipper_shapefiles
from pygmi.mt.mtpyold.core.edi_collection import EdiCollection

# The lines below are a temporary fix for pygmi.mt.mtpyold.
np.float = float
np.complex = complex


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas widget for the actual plot.

    This routine will also allow the picking and movement of nodes of data.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(layout='constrained', figsize=(width, height), dpi=dpi)
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

        self.axes = ax1
        x = 1 / data1.Z.freq

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

        ax3.set_xscale('log')
        ax3.set_yscale('linear')
        ax3.legend(loc='upper left')
        ax3.set_ylabel(r'Tipper Magnitude')
        ax3.tick_params(labelbottom=False)
        ax3.grid(True)

        ax4 = self.figure.add_subplot(414, sharex=ax1)
        ax4.plot(x, data1.Tipper.angle_real, 'b.', label='real')
        ax4.plot(x, data1.Tipper.angle_imag, 'r.', label='imaginary')

        ax4.set_xscale('log')
        ax4.set_yscale('linear')
        ax4.legend(loc='upper left')
        ax4.set_xlabel('Period (s)')
        ax4.set_ylabel(r'Tipper Angle (Degrees)')
        ax4.grid(True)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(ax1.bbox)

        self.figure.canvas.draw()

    def update_phase(self, edi_list, plot_freq, plot_tipper, ellipse_colorby,
                     ellipse_size):
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
        edicol = EdiCollection(edi_list)
        ptdict = edicol.get_phase_tensor_tippers(1 / plot_freq)

        tmp = []
        tmp2 = {'skew': 'skew',
                'phimin': 'phi_min',
                'phimax': 'phi_max'}
        for i in ptdict:
            tmp.append(i[tmp2[ellipse_colorby]])

        tmp = np.array(tmp)
        ellipse_range = [tmp.min().round() - 1, tmp.max().round() + 1, 1]

        self.figure.clear()

        ptmap = PlotPhaseTensorMaps(fn_list=edi_list,
                                    plot_freq=plot_freq,
                                    fig_size=(4, 3),
                                    # pad around stations
                                    xpad=0.02, ypad=0.02,
                                    # 'y' + 'r' and/or 'i' to plot
                                    # real and/or imaginary
                                    plot_tipper=plot_tipper,
                                    # a matplotlib colour or None for no
                                    # borders
                                    edgecolor='k',
                                    lw=0.5,  # linewidth for the ellipses
                                    minorticks_on=False,
                                    # 'phimin', 'phimax', or 'skew'
                                    ellipse_colorby=ellipse_colorby,
                                    ellipse_range=ellipse_range,
                                    # scaling factor for the ellipses
                                    ellipse_size=ellipse_size,
                                    arrow_size=0.1,
                                    # scaling for arrows (head width)
                                    arrow_head_width=0.002,
                                    # scaling for arrows (head length)
                                    arrow_head_length=0.002,
                                    ellipse_cmap='bwr',  # matplotlib colormap
                                    # station_dict={'id': (5, 7)},
                                    font_size=10,
                                    plot_yn='n'
                                    )

        ptmap.plot(fig=self.figure, show=False)

        ptmap.update_plot()


class PlotPoints(ContextModule):
    """Plot points class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.lbl_1 = QtWidgets.QLabel('Station Name:')
        self.lbl_2 = QtWidgets.QLabel('Graph Type:')

        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = 'mt.cm.showgraphs'

        hbl.addWidget(self.buttonbox)
        hbl.addWidget(self.lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(self.lbl_2, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        hbl.addWidget(self.cmb_2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo to choose band.

        Returns
        -------
        None.

        """
        data = self.indata['MT - EDI']
        i = self.cmb_1.currentText()
        i2 = self.cmb_2.currentText()
        self.mmc.update_line(data, i, i2)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['MT - EDI']
        for i in data:
            self.cmb_1.addItem(i)
        for i in ['xy, yx', 'xx, yy']:
            self.cmb_2.addItem(i)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_2.setCurrentIndex(0)


class PlotPhaseTensor(ContextModule):
    """Plot phase tensor."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.cursoln = 0
        self.edi_list = []

        self.setWindowTitle('Plot Phase Tensor')
        self.buttonbox.htmlfile = 'mt.dm.occam'

        vbl = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout(self)
        gl_1 = QtWidgets.QGridLayout()
        gl_1.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cb_realtipper = QtWidgets.QCheckBox(r'Plot real tipper')
        self.cb_imagtipper = QtWidgets.QCheckBox(r'Plot imaginary tipper')
        self.dsb_esize = QtWidgets.QDoubleSpinBox()
        pb_export = QtWidgets.QPushButton('Export current frequency to '
                                          'shapefiles')

        self.cmb_ecol = QtWidgets.QComboBox()
        self.cmb_ecol.addItems(['skew', 'phimin', 'phimax'])
        self.cmb_ecol.setCurrentIndex(0)
        self.dsb_esize.setValue(0.01)
        self.dsb_esize.setMinimum(0.001)
        self.dsb_esize.setSingleStep(0.001)
        self.dsb_esize.setDecimals(3)
        self.cb_realtipper.setChecked(True)

        lbl_1 = QtWidgets.QLabel('Frequency:')
        lbl_3 = QtWidgets.QLabel('Ellipse color by:')
        lbl_4 = QtWidgets.QLabel('Ellipse scale factor:')

        spacer = QtWidgets.QSpacerItem(20, 40,
                                       QtWidgets.QSizePolicy.Policy.Minimum,
                                       QtWidgets.QSizePolicy.Policy.Expanding)

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        gl_1.addWidget(lbl_1, 1, 0)
        gl_1.addWidget(self.cmb_1, 1, 1)
        gl_1.addWidget(lbl_3, 2, 0)
        gl_1.addWidget(self.cmb_ecol, 2, 1)
        gl_1.addWidget(lbl_4, 3, 0)
        gl_1.addWidget(self.dsb_esize, 3, 1)
        gl_1.addWidget(self.cb_realtipper, 4, 0, 1, 2)
        gl_1.addWidget(self.cb_imagtipper, 5, 0, 1, 2)
        gl_1.addWidget(pb_export, 6, 0, 1, 2)
        gl_1.addItem(spacer, 16, 0, 1, 1)
        gl_1.addWidget(self.buttonbox, 17, 0, 1, 2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gl_1)
        hbl.addLayout(vbl)

        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_ecol.currentIndexChanged.connect(self.change_band)
        self.cb_realtipper.checkStateChanged.connect(self.change_band)
        self.cb_imagtipper.checkStateChanged.connect(self.change_band)
        self.dsb_esize.valueChanged.connect(self.change_band)
        pb_export.clicked.connect(self.export)

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()
        self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])
        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        freq = float(self.cmb_1.currentText())

        tipper = ''
        if self.cb_realtipper.isChecked():
            tipper = 'r'
        if self.cb_imagtipper.isChecked():
            tipper += 'i'
        if len(tipper) > 0:
            tipper = 'y' + tipper

        ecol = self.cmb_ecol.currentText()

        esize = self.dsb_esize.value()

        self.mmc.update_phase(self.edi_list, freq, tipper, ecol, esize)

    def export(self):
        """Export to shapefile."""

        odir = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent, 'Select Output Directory')

        if not odir:
            return

        datadir = os.path.dirname(self.edi_list[0])
        period = 1. / float(self.cmb_1.currentText())
        create_tensor_tipper_shapefiles(datadir, odir, period)

        QtWidgets.QMessageBox.information(self.parent, 'Information',
                                          'Export completed!')

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        None.

        """
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
            self.edi_list = [self.data[i].fn for i in self.data]
        else:
            self.showlog('No EDI data')
            return

        edicol = EdiCollection(self.edi_list)

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_1.clear()

        for i in edicol.all_frequencies:
            self.cmb_1.addItem(str(i))

        self.cmb_1.setCurrentIndex(0)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        self.change_band()

        self.exec()

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_1)
        self.saveobj(self.cmb_ecol)
        self.saveobj(self.dsb_esize)
        self.saveobj(self.cb_imagtipper)
        self.saveobj(self.cb_realtipper)


def _testfn_phase():
    """Test."""
    import sys
    from pygmi.mt.iodefs import get_EDI

    datadir = r'D:\workdata\PyGMI Test Data\MT\mtpy-develop\examples\data\edi2'
    edi_list = glob.glob(os.path.join(datadir, '*.edi'))

    dat = get_EDI(edi_list)

    _ = QtWidgets.QApplication(sys.argv)
    tmp = PlotPhaseTensor()
    tmp.indata['MT - EDI'] = dat
    tmp.run()


def _testfn():
    """Test routine."""
    import sys
    from pygmi.mt.mtpyold.core.mt import MT

    datadir = r'd:\workdata\MT\\'
    edi_file = datadir + r"synth02.edi"

    # Create an MT object
    mt_obj = MT(edi_file)

    print('loading complete')

    _ = QtWidgets.QApplication(sys.argv)
    tmp = PlotPoints()
    tmp.indata['MT - EDI'] = {'SYNTH02': mt_obj}
    tmp.run()
    tmp.exec()


if __name__ == "__main__":
    _testfn()
