# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
"""A set of Data Preparation routines."""

import os
import sys
import copy
import glob
import platform
from contextlib import redirect_stdout
from PyQt5 import QtWidgets, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.patches import Rectangle
from mtpy.modeling import occam1d
from mtpy.core.mt import MT
from mtpy.core.z import Z, Tipper

from pygmi import menu_default
from pygmi.misc import BasicModule, ContextModule

# The lines below are a temporary fix for mtpy. Removed in future.
np.float = float
np.complex = complex


class Metadata(ContextModule):
    """
    Edit Metadata.

    This class allows the editing of the metadata for MT data using a GUI.

    Attributes
    ----------
    banddata : dictionary
        band data
    bandid : dictionary
        dictionary of strings containing band names.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''

        self.cmb_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton('Rename Station Name')
        self.le_lat = QtWidgets.QLineEdit()
        self.le_lon = QtWidgets.QLineEdit()
        self.le_elev = QtWidgets.QLineEdit()
        self.le_utmx = QtWidgets.QLineEdit()
        self.le_utmy = QtWidgets.QLineEdit()
        self.le_utmzone = QtWidgets.QLineEdit()
        self.le_rot = QtWidgets.QLineEdit()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        gbox = QtWidgets.QGroupBox('Dataset')

        gl_1 = QtWidgets.QGridLayout(gbox)
        lbl_utmx = QtWidgets.QLabel('UTM X Coordinate:')
        lbl_utmy = QtWidgets.QLabel('UTM Y Coordinate:')
        lbl_lat = QtWidgets.QLabel('Latitude:')
        lbl_lon = QtWidgets.QLabel('Longitude:')
        lbl_elev = QtWidgets.QLabel('Elevation:')
        lbl_utmzone = QtWidgets.QLabel('UTM Zone:')
        lbl_rot = QtWidgets.QLabel('Rotation:')
        lbl_bandid = QtWidgets.QLabel('Station Name:')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        gbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Metadata')

        gl_main.addWidget(lbl_bandid, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_bandid, 0, 1, 1, 3)
        gl_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gl_main.addWidget(gbox, 2, 0, 1, 2)
        gl_main.addWidget(buttonbox, 4, 0, 1, 4)

        gl_1.addWidget(lbl_lat, 0, 0, 1, 1)
        gl_1.addWidget(self.le_lat, 0, 1, 1, 1)
        gl_1.addWidget(lbl_lon, 1, 0, 1, 1)
        gl_1.addWidget(self.le_lon, 1, 1, 1, 1)
        gl_1.addWidget(lbl_elev, 2, 0, 1, 1)
        gl_1.addWidget(self.le_elev, 2, 1, 1, 1)
        gl_1.addWidget(lbl_utmx, 3, 0, 1, 1)
        gl_1.addWidget(self.le_utmx, 3, 1, 1, 1)
        gl_1.addWidget(lbl_utmy, 4, 0, 1, 1)
        gl_1.addWidget(self.le_utmy, 4, 1, 1, 1)
        gl_1.addWidget(lbl_utmzone, 5, 0, 1, 1)
        gl_1.addWidget(self.le_utmzone, 5, 1, 1, 1)
        gl_1.addWidget(lbl_rot, 6, 0, 1, 1)
        gl_1.addWidget(self.le_rot, 6, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)
        self.pb_rename_id.clicked.connect(self.rename_id)

    def acceptall(self):
        """
        Accept option. Updates self.indata.

        Returns
        -------
        None.

        """
        self.update_vals()
        self.indata['MT - EDI'] = self.banddata

    def rename_id(self):
        """
        Rename station name.

        Returns
        -------
        None.

        """
        ctxt = str(self.cmb_bandid.currentText())
        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent, 'Rename Station Name',
            'Please type in the new name for the station',
            QtWidgets.QLineEdit.Normal, ctxt)

        if isokay:
            self.cmb_bandid.currentIndexChanged.disconnect()
            indx = self.cmb_bandid.currentIndex()
            txt = self.cmb_bandid.itemText(indx)
            self.banddata[skey] = self.banddata.pop(txt)
            self.dataid[skey] = self.dataid.pop(txt)
            self.oldtxt = skey
            self.cmb_bandid.setItemText(indx, skey)
            self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

    def update_vals(self):
        """
        Update the values on the interface.

        Returns
        -------
        None.

        """
        odata = self.banddata[self.oldtxt]

        try:
            odata.lat = float(self.le_lat.text())
            odata.lon = float(self.le_lon.text())
            if self.le_utmx.text() != 'None':
                odata.east = float(self.le_utmx.text())
            if self.le_utmy.text() != 'None':
                odata.north = float(self.le_utmy.text())
            odata.elev = float(self.le_elev.text())
            odata.rotation_angle = float(self.le_rot.text())
        except ValueError:
            self.showlog('Value error - abandoning changes')

        indx = self.cmb_bandid.currentIndex()
        txt = self.cmb_bandid.itemText(indx)
        self.oldtxt = txt
        idata = self.banddata[txt]

        self.le_lat.setText(str(idata.lat))
        self.le_lon.setText(str(idata.lon))
        self.le_elev.setText(str(idata.elev))
        if np.isinf(idata.east):
            self.le_utmx.setText('None')
        else:
            self.le_utmx.setText(str(idata.east))
        if np.isinf(idata.north):
            self.le_utmy.setText('None')
        else:
            self.le_utmy.setText(str(idata.north))
        self.le_utmzone.setText(str(idata.utm_zone))
        self.le_rot.setText(str(idata.rotation_angle))

    def run(self):
        """
        Run.

        Returns
        -------
        bool.
            True if successful, False otherwise.

        """
        bandid = []

        for i in self.indata['MT - EDI']:
            bandid.append(i)
            self.banddata[i] = copy.deepcopy(self.indata['MT - EDI'][i])
            self.dataid[i] = i

        self.cmb_bandid.currentIndexChanged.disconnect()
        self.cmb_bandid.clear()
        self.cmb_bandid.addItems(bandid)
        indx = self.cmb_bandid.currentIndex()
        self.oldtxt = self.cmb_bandid.itemText(indx)
        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

        idata = self.banddata[self.oldtxt]

        self.le_lat.setText(str(idata.lat))
        self.le_lon.setText(str(idata.lon))
        self.le_elev.setText(str(idata.elev))
        if np.isinf(idata.east):
            self.le_utmx.setText('None')
        else:
            self.le_utmx.setText(str(idata.east))
        if np.isinf(idata.north):
            self.le_utmy.setText('None')
        else:
            self.le_utmy.setText(str(idata.north))
        self.le_utmzone.setText(str(idata.utm_zone))
        self.le_rot.setText(str(idata.rotation_angle))

        self.update_vals()

        tmp = self.exec()

        if tmp != 1:
            return False

        self.acceptall()

        return True


class MyMplCanvas(FigureCanvasQTAgg):
    """MPL Canvas class."""

    def __init__(self, parent=None):
        fig = Figure(layout='constrained')
        super().__init__(fig)

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

        ax1 = self.figure.add_subplot(211, label='Profile')

        ax1.set_title(ival)
        ax1.grid(True, 'both')
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
        ax1.set_xlabel('Period (s)')
        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.grid(True, 'both')

        ax2.errorbar(x, pha1, yerr=pha1_err, label=label3,
                     ls=' ', marker='.', mfc='b', mec='b', ecolor='b')
        ax2.errorbar(x, pha2, yerr=pha2_err, label=label4,
                     ls=' ', marker='.', mfc='r', mec='r', ecolor='r')

        ax2.set_ylim(-180., 180.)

        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.legend(loc='upper left')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel(r'Phase (Degrees)')

        self.figure.canvas.draw()


class StaticShiftEDI(BasicModule):
    """Static shift EDI data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.setWindowTitle('Remove Static Shift')
        helpdocs = menu_default.HelpButton('pygmi.mt.static')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl_2 = QtWidgets.QHBoxLayout()
        hbl_3 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_2.addItems(['xy, yx', 'xx, yy'])
        self.cmb_2.setCurrentIndex(0)

        self.dsb_shiftx = QtWidgets.QDoubleSpinBox()
        self.dsb_shiftx.setMinimum(0.)
        self.dsb_shiftx.setMaximum(100000.)
        self.dsb_shiftx.setValue(1.)
        self.dsb_shifty = QtWidgets.QDoubleSpinBox()
        self.dsb_shifty.setMinimum(0.)
        self.dsb_shifty.setMaximum(100000.)
        self.dsb_shifty.setValue(1.)
        lbl_1 = QtWidgets.QLabel('Station Name:')
        lbl_2 = QtWidgets.QLabel('Graph Type:')
        lbl_3 = QtWidgets.QLabel('Shift X:')
        lbl_4 = QtWidgets.QLabel('Shift Y:')
        self.cb_applyall = QtWidgets.QCheckBox('Apply to all stations:')
        pb_apply = QtWidgets.QPushButton('Remove Static Shift')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(lbl_2)
        hbl.addWidget(self.cmb_2)

        hbl_3.addWidget(lbl_3)
        hbl_3.addWidget(self.dsb_shiftx)
        hbl_3.addWidget(lbl_4)
        hbl_3.addWidget(self.dsb_shifty)

        hbl_2.addWidget(helpdocs)
        hbl_2.addWidget(pb_reset)
        hbl_2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.cb_applyall)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl_3)
        vbl.addLayout(hbl_2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.cmb_2.currentIndexChanged.connect(self.change_band)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        self.outdata['MT - EDI'] = self.data

    def apply(self):
        """
        Apply static shift.

        Returns
        -------
        None.

        """
        ssx = self.dsb_shiftx.value()
        ssy = self.dsb_shifty.value()

        if self.cb_applyall.isChecked():
            for i in self.data:
                self.data[i].Z = self.data[i].remove_static_shift(ssx, ssy)
        else:
            i = self.cmb_1.currentText()
            self.data[i].Z = self.data[i].remove_static_shift(ssx, ssy)

        self.change_band()

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()

        if self.cb_applyall.isChecked():
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])

        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()
        i2 = self.cmb_2.currentText()
        self.mmc.update_line(self.data, i, i2)

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
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showlog('No EDI data')
            return False

        self.cmb_1.currentIndexChanged.disconnect()

        self.cmb_1.clear()
        for i in self.data:
            self.cmb_1.addItem(i)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        self.change_band()

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
        self.saveobj(self.dsb_shiftx)
        self.saveobj(self.dsb_shifty)
        self.saveobj(self.cb_applyall)


class RotateEDI(BasicModule):
    """Rotate EDI data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.setWindowTitle('Rotate EDI data')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl_2 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_2.addItems(['xy, yx', 'xx, yy'])
        self.cmb_2.setCurrentIndex(0)

        self.dsb_rotangle = QtWidgets.QDoubleSpinBox()
        self.dsb_rotangle.setMinimum(0.)
        self.dsb_rotangle.setMaximum(360.)
        lbl_1 = QtWidgets.QLabel('Station Name:')
        lbl_2 = QtWidgets.QLabel('Graph Type:')
        lbl_3 = QtWidgets.QLabel('Rotate Z (0 is North):')
        self.cb_applyall = QtWidgets.QCheckBox('Apply to all stations:')
        pb_apply = QtWidgets.QPushButton('Apply rotation')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        helpdocs = menu_default.HelpButton('pygmi.mt.rotate')

        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(lbl_2)
        hbl.addWidget(self.cmb_2)
        hbl.addWidget(lbl_3)
        hbl.addWidget(self.dsb_rotangle)

        hbl_2.addWidget(helpdocs)
        hbl_2.addWidget(pb_reset)
        hbl_2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.cb_applyall)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl_2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        self.outdata['MT - EDI'] = self.data

    def apply(self):
        """
        Apply rotation to data.

        Returns
        -------
        None.

        """
        rotZ = self.dsb_rotangle.value()

        if self.cb_applyall.isChecked():
            for i in self.data:
                self.data[i].Z.rotate(rotZ)
                self.data[i].Tipper.rotate(rotZ)
        else:
            i = self.cmb_1.currentText()
            self.data[i].Z.rotate(rotZ)
            self.data[i].Tipper.rotate(rotZ)

        self.change_band()

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()

        if self.cb_applyall.isChecked():
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])

        self.dsb_rotangle.setValue(self.data[i].rotation_angle)

        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()
        i2 = self.cmb_2.currentText()
        self.mmc.update_line(self.data, i, i2)

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
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showlog('No EDI data')
            return False

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_1.clear()

        for i in self.data:
            self.cmb_1.addItem(i)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        i = self.cmb_1.currentText()
        self.dsb_rotangle.setValue(self.data[i].rotation_angle)

        self.change_band()

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
        self.saveobj(self.dsb_rotangle)
        self.saveobj(self.cb_applyall)


class MyMplCanvasPick(FigureCanvasQTAgg):
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
        self.itype = 'xy, yx'
        self.ival = None
        self.data = None
        self.maskrange = None
        self.axes2 = None
        self.line2 = None

        super().__init__(fig)

        self.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.figure.canvas.mpl_connect('button_press_event',
                                       self.button_press_callback)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self.motion_notify_callback)
        self.figure.canvas.mpl_connect('resize_event', self.revent)

    def button_press_callback(self, event):
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

        rect = None
        if event.button != 1:
            xxx, yyy = self.line.get_data()
            xxx = [event.xdata, event.xdata]
            self.line.set_data(xxx, yyy)

            xxx, yyy = self.line2.get_data()
            xxx = [event.xdata, event.xdata]
            self.line2.set_data(xxx, yyy)
        elif event.button == 1:
            xxx, yyy = self.line.get_data()
            rect = Rectangle((xxx[0], yyy[0]), event.xdata-xxx[0],
                             yyy[1]-yyy[0])
            xxx, yyy = self.line2.get_data()
            rect2 = Rectangle((xxx[0], yyy[0]), event.xdata-xxx[0],
                              yyy[1]-yyy[0])
            self.maskrange = np.sort([xxx[0], event.xdata])

        self.figure.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.axes.draw_artist(self.line2)
        if rect is not None:
            [p.remove() for p in reversed(self.axes.patches)]
            [p.remove() for p in reversed(self.axes2.patches)]
            self.axes.add_patch(rect)
            self.axes2.add_patch(rect2)

        self.figure.canvas.draw()

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

    def revent(self, width):
        """
        Resize event.

        Parameters
        ----------
        width : event
            unused.

        Returns
        -------
        None.

        """
        if self.data is None:
            return
        self.update_line(self.data, self.ival, self.itype)

    def update_line(self, data, ival=None, itype=None):
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
        self.ival = ival
        self.itype = itype
        self.data = data

        data1 = data[ival]

        self.figure.clear()

        ax1 = self.figure.add_subplot(211, label='Profile')
        ax1.grid(True, 'both')

        ax1.set_title(ival)
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
        ax1.set_xlabel('Period (s)')
        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')

        ax2 = self.figure.add_subplot(212, sharex=ax1)
        ax2.grid(True, 'both')

        self.axes2 = ax2

        ax2.errorbar(x, pha1, yerr=pha1_err, label=label3,
                     ls=' ', marker='.', mfc='b', mec='b', ecolor='b')
        ax2.errorbar(x, pha2, yerr=pha2_err, label=label4,
                     ls=' ', marker='.', mfc='r', mec='r', ecolor='r')

        ax2.set_ylim(-180., 180.)

        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.legend(loc='upper left')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel(r'Phase (Degrees)')

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)

        x0 = self.axes.get_xlim()[0]
        y0, y1 = self.axes.get_ylim()
        self.line, = self.axes.plot([x0, x0], [y0, y1])

        x0 = self.axes2.get_xlim()[0]
        y0, y1 = self.axes2.get_ylim()
        self.line2, = self.axes2.plot([x0, x0], [y0, y1])

        self.figure.canvas.draw()


class EditEDI(BasicModule):
    """Edit EDI Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.setWindowTitle('Mask and Interpolate')
        helpdocs = menu_default.HelpButton('pygmi.mt.edit')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl_2 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvasPick(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_2.addItems(['xy, yx', 'xx, yy'])
        self.cmb_2.setCurrentIndex(0)

        lbl_1 = QtWidgets.QLabel('Station Name:')
        lbl_2 = QtWidgets.QLabel('Graph Type:')
        pb_apply = QtWidgets.QPushButton('Mask and Interpolate')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        hbl.addWidget(lbl_1)
        hbl.addWidget(self.cmb_1)
        hbl.addWidget(lbl_2)
        hbl.addWidget(self.cmb_2)

        hbl_2.addWidget(helpdocs)
        hbl_2.addWidget(pb_reset)
        hbl_2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl_2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cmb_2.currentIndexChanged.connect(self.change_band)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        self.outdata['MT - EDI'] = self.data

    def apply(self):
        """
        Apply edited data.

        Returns
        -------
        None.

        """
        if self.mmc.maskrange is None:
            return

        i = self.cmb_1.currentText()

        x0 = self.mmc.maskrange[0]
        x1 = self.mmc.maskrange[1]

        xcrds = 1/self.data[i].Z.freq

        mask = ~np.logical_and(xcrds > x0, xcrds < x1)

        mt_obj = self.data[i]

        mt_obj.Z = Z(z_array=mt_obj.Z.z[mask],
                     z_err_array=mt_obj.Z.z_err[mask],
                     freq=mt_obj.Z.freq[mask])

        mt_obj.Tipper = Tipper(tipper_array=mt_obj.Tipper.tipper[mask],
                               tipper_err_array=mt_obj.Tipper.tipper_err[mask],
                               freq=mt_obj.Tipper.freq[mask])

        if x1 < xcrds.max():
            new_freq_list = 1/xcrds
            new_Z_obj, new_Tipper_obj = mt_obj.interpolate(new_freq_list)
            self.data[i].Z = new_Z_obj
            self.data[i].Tipper = new_Tipper_obj

        self.change_band()

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
        Combo to choose band.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentText()
        i2 = self.cmb_2.currentText()
        self.mmc.update_line(self.data, i, i2)

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
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showlog('No EDI data')
            return False

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_1.clear()

        for i in self.data:
            self.cmb_1.addItem(i)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        self.change_band()

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


class MySlider(QtWidgets.QSlider):
    """
    My Slider.

    Custom class which allows clicking on a horizontal slider bar with slider
    moving to click in a single step.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """
        Mouse press event.

        Parameters
        ----------
        event : event
            event variable.

        Returns
        -------
        None.

        """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))

    def mouseMoveEvent(self, event):
        """
        Jump to pointer position while moving.

        Parameters
        ----------
        event : event
            event variable.

        Returns
        -------
        None.

        """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))


class MyMplCanvas2(FigureCanvasQTAgg):
    """MPL Canvas class."""

    def __init__(self, parent=None):
        fig = Figure(layout='constrained')
        super().__init__(fig)

    def update_line(self, x, pdata, rdata, depths=None, res=None, title=None):
        """
        Update the plot from data.

        Parameters
        ----------
        x : numpy array
            X coordinates (period).
        pdata : numpy array
            Phase data.
        rdata : numpy array
            Apparent resistivity data.
        depths : numpy array, optional
            Model depths. The default is None.
        res : numpy array, optional
            Resistivities. The default is None.
        title : str or None, optional
            Title text. The default is None.

        Returns
        -------
        None.

        """
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 3)

        ax1 = self.figure.add_subplot(gs[:2, :2], label='Profile')
        self.figure.suptitle(title)
        ax1.grid(True, 'both')

        res1 = rdata[0]
        res2 = rdata[1]
        pha1 = pdata[0]
        pha2 = pdata[1]
        label1 = r'Measured'
        label2 = r'Modelled'

        ax1.plot(x, res1, 'b.', label=label1)
        ax1.plot(x, res2, 'r.', label=label2)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Period (s)')
        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')

        ax2 = self.figure.add_subplot(gs[2:, :2], sharex=ax1)
        ax2.grid(True, 'both')

        ax2.plot(x, pha1, 'b.')
        ax2.plot(x, pha2, 'r.')

        ax2.set_ylim(-180., 180.)

        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel(r'Phase (Degrees)')

        ax3 = self.figure.add_subplot(gs[:, 2])
        ax3.grid(True, 'both')
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.set_xlabel(r'Res. ($\Omega.m$)')
        ax3.set_ylabel(r'Depth (km)')

        if depths is not None:
            ax3.plot(res, np.array(depths)/1000)

        self.figure.canvas.draw()


class Occam1D(BasicModule):
    """Occam 1D inversion."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.cursoln = 0

        self.setWindowTitle('Occam 1D Inversion')
        helpdocs = menu_default.HelpButton('pygmi.mt.occam1d')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)

        vbl = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout(self)
        hbl_2 = QtWidgets.QHBoxLayout()
        gl_1 = QtWidgets.QGridLayout()
        gl_1.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.mmc = MyMplCanvas2(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.le_occfile = QtWidgets.QLineEdit('')
        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(['TE', 'TM', 'DET'])
        self.cmb_mode.setCurrentIndex(0)
        self.le_errres = QtWidgets.QLineEdit('data')
        self.le_errres.setSizePolicy(sizepolicy)
        self.le_errphase = QtWidgets.QLineEdit('data')
        self.le_errphase.setSizePolicy(sizepolicy)
        self.le_errfloorres = QtWidgets.QLineEdit('4.')
        self.le_errfloorres.setSizePolicy(sizepolicy)
        self.le_errfloorphase = QtWidgets.QLineEdit('2.')
        self.le_errfloorphase.setSizePolicy(sizepolicy)
        self.cb_remove_out_quad = QtWidgets.QCheckBox(r'Remove Resistivity/'
                                                      r'Phase values out of '
                                                      r'1st/3rd Quadrant')

        self.le_targetdepth = QtWidgets.QLineEdit('40000.')
        self.le_targetdepth.setSizePolicy(sizepolicy)
        self.le_nlayers = QtWidgets.QLineEdit('100')
        self.le_nlayers.setSizePolicy(sizepolicy)
        self.le_bottomlayer = QtWidgets.QLineEdit('100000.')
        self.le_bottomlayer.setSizePolicy(sizepolicy)
        self.le_airlayer = QtWidgets.QLineEdit('10000.')
        self.le_airlayer.setSizePolicy(sizepolicy)
        self.le_z1layer = QtWidgets.QLineEdit('10.')
        self.le_z1layer.setSizePolicy(sizepolicy)
        self.le_maxiter = QtWidgets.QLineEdit('200')
        self.le_maxiter.setSizePolicy(sizepolicy)
        self.le_targetrms = QtWidgets.QLineEdit('1.')
        self.le_targetrms.setSizePolicy(sizepolicy)
        self.cb_remove_out_quad.setChecked(True)

        self.hs_profnum = MySlider()
        self.hs_profnum.setOrientation(QtCore.Qt.Horizontal)

        pb_occ = QtWidgets.QPushButton('Occam executable location')
        lbl_1 = QtWidgets.QLabel('Station Name:')
        lbl_1.setSizePolicy(sizepolicy)
        lbl_3 = QtWidgets.QLabel('Mode:')
        lbl_4 = QtWidgets.QLabel('Resistivity Errorbar (Data or %):')
        lbl_5 = QtWidgets.QLabel('Phase Errorbar (Data or %):')
        lbl_6 = QtWidgets.QLabel('Resistivity Error Floor (%):')
        lbl_7 = QtWidgets.QLabel('Phase Error Floor (degrees):')
        lbl_8 = QtWidgets.QLabel('Height of air layer:')
        lbl_9 = QtWidgets.QLabel('Bottom of model:')
        lbl_10 = QtWidgets.QLabel('Depth of target to investigate:')
        lbl_11 = QtWidgets.QLabel('Depth of first layer:')
        lbl_12 = QtWidgets.QLabel('Number of layers:')
        lbl_13 = QtWidgets.QLabel('Maximum Iterations:')
        lbl_14 = QtWidgets.QLabel('Target RMS:')

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        pb_apply = QtWidgets.QPushButton('Invert Station')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gl_1.addWidget(pb_occ, 0, 0)
        gl_1.addWidget(self.le_occfile, 0, 1)
        gl_1.addWidget(lbl_1, 1, 0)
        gl_1.addWidget(self.cmb_1, 1, 1)
        gl_1.addWidget(lbl_3, 2, 0)
        gl_1.addWidget(self.cmb_mode, 2, 1)
        gl_1.addWidget(lbl_4, 3, 0)
        gl_1.addWidget(self.le_errres, 3, 1)
        gl_1.addWidget(lbl_5, 4, 0)
        gl_1.addWidget(self.le_errphase, 4, 1)
        gl_1.addWidget(lbl_6, 5, 0)
        gl_1.addWidget(self.le_errfloorres, 5, 1)
        gl_1.addWidget(lbl_7, 6, 0)
        gl_1.addWidget(self.le_errfloorphase, 6, 1)
        gl_1.addWidget(lbl_8, 7, 0)
        gl_1.addWidget(self.le_airlayer, 7, 1)
        gl_1.addWidget(lbl_9, 8, 0)
        gl_1.addWidget(self.le_bottomlayer, 8, 1)
        gl_1.addWidget(lbl_10, 9, 0)
        gl_1.addWidget(self.le_targetdepth, 9, 1)
        gl_1.addWidget(lbl_11, 10, 0)
        gl_1.addWidget(self.le_z1layer, 10, 1)
        gl_1.addWidget(lbl_12, 11, 0)
        gl_1.addWidget(self.le_nlayers, 11, 1)
        gl_1.addWidget(lbl_13, 12, 0)
        gl_1.addWidget(self.le_maxiter, 12, 1)
        gl_1.addWidget(lbl_14, 13, 0)
        gl_1.addWidget(self.le_targetrms, 13, 1)
        gl_1.addWidget(self.cb_remove_out_quad, 14, 0, 1, 2)

        gl_1.addWidget(pb_apply, 15, 0, 1, 2)
        gl_1.addWidget(buttonbox, 16, 0, 1, 2)

        hbl_2.addWidget(helpdocs)
        hbl_2.addWidget(self.lbl_profnum)
        hbl_2.addWidget(self.hs_profnum)

        vbl.addWidget(self.mmc)
        vbl.addLayout(hbl_2)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gl_1)
        hbl.addLayout(vbl)

        pb_occ.pressed.connect(self.get_occfile)
        pb_apply.clicked.connect(self.apply)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cmb_1.currentIndexChanged.connect(self.change_band)
        self.hs_profnum.valueChanged.connect(self.snum)

    def snum(self):
        """
        Change solution graph.

        Returns
        -------
        None.

        """
        self.lbl_profnum.setText('Solution: ' +
                                 str(self.hs_profnum.sliderPosition()))
        self.change_band()

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        self.outdata['MT - EDI'] = self.data

    def apply(self):
        """
        Apply.

        Returns
        -------
        None.

        """
        parm = {}

        parm['tdepth'] = tonumber(self.le_targetdepth.text())
        parm['nlayers'] = tonumber(self.le_nlayers.text())
        parm['blayer'] = tonumber(self.le_bottomlayer.text())
        parm['alayer'] = tonumber(self.le_airlayer.text())
        parm['z1layer'] = tonumber(self.le_z1layer.text())
        parm['miter'] = tonumber(self.le_maxiter.text())
        parm['trms'] = tonumber(self.le_targetrms.text())
        parm['rerr'] = tonumber(self.le_errres.text(), 'data')
        parm['perr'] = tonumber(self.le_errphase.text(), 'data')
        parm['perrflr'] = tonumber(self.le_errfloorphase.text())
        parm['rerrflr'] = tonumber(self.le_errfloorres.text())
        parm['routq'] = self.cb_remove_out_quad.isChecked()

        if -999 in parm.values():
            return

        mode = self.cmb_mode.currentText()
        i = self.cmb_1.currentText()
        edi_file = self.data[i].fn

        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            r = glob.glob(save_path+r'\*')
            for i in r:
                os.remove(i)
        else:
            os.makedirs(save_path)

        with redirect_stdout(self.stdout_redirect):
            d1 = occam1d.Data()
            d1.write_data_file(edi_file=edi_file,
                               mode=mode,
                               save_path=save_path,
                               res_err=parm['rerr'],
                               phase_err=parm['perr'],
                               res_errorfloor=parm['rerrflr'],
                               phase_errorfloor=parm['perrflr'],
                               remove_outofquadrant=parm['routq']
                               )

            m1 = occam1d.Model(target_depth=parm['tdepth'],
                               n_layers=parm['nlayers'],
                               bottom_layer=parm['blayer'],
                               z1_layer=parm['z1layer'],
                               air_layer_height=parm['alayer']
                               )
            m1.write_model_file(save_path=d1.save_path)

            s1 = occam1d.Startup(data_fn=d1.data_fn,
                                 model_fn=m1.model_fn,
                                 max_iter=parm['miter'],
                                 target_rms=parm['trms'])

            s1.write_startup_file()

            occam_path = os.path.dirname(__file__)[:-2]+r'\bin\occam1d'
            if platform.system() == 'Windows':
                occam_path += '.exe'

            occam_path = self.le_occfile.text()

            if not os.path.exists(occam_path):
                text = ('No Occam1D executable found. Please place it in the '
                        'bin directory. You may need to obtain the source '
                        'code from '
                        'https://marineemlab.ucsd.edu/Projects/Occam/1DCSEM/ '
                        'and compile it. It should be called occam1d for '
                        'non-windows platforms and occam1d.exe for windows.')
                QtWidgets.QMessageBox.warning(self.parent, 'Error', text,
                                              QtWidgets.QMessageBox.Ok)
                return

            self.mmc.figure.clear()
            self.mmc.figure.set_facecolor('r')
            self.mmc.figure.suptitle('Busy, please wait...', fontsize=14,
                                     y=0.5)
            self.mmc.figure.canvas.draw()
            QtWidgets.QApplication.processEvents()

            occam1d.Run(s1.startup_fn, occam_path, mode=mode)

        self.mmc.figure.set_facecolor('w')

        allfiles = glob.glob(save_path+r'\*.resp')
        self.hs_profnum.setMaximum(len(allfiles))
        self.hs_profnum.setMinimum(1)

        self.change_band()

    def get_occfile(self, filename=''):
        """
        Get Occam executable filename.

        Parameters
        ----------
        filename : str, optional
            Occam executable filename. The default is ''.

        Returns
        -------
        None.

        """
        ext = 'Occam executable (*.exe *.)'

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        self.le_occfile.setText(filename)

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
        i = self.cmb_1.currentText()
        mode = self.cmb_mode.currentText()
        n = self.hs_profnum.value()

        edi_file = self.data[i].fn
        save_path = edi_file[:-4]+'-'+mode

        if not os.path.exists(save_path):
            return
        if os.path.exists(save_path):
            r = glob.glob(save_path+r'\*.resp')
            if len(r) == 0:
                return

        iterfn = os.path.join(save_path, mode+'_'+f'{n:03}'+'.iter')
        respfn = os.path.join(save_path, mode+'_'+f'{n:03}'+'.resp')
        model_fn = os.path.join(save_path, 'Model1D')
        data_fn = os.path.join(save_path, 'Occam1d_DataFile_'+mode+'.dat')

        oc1m = occam1d.Model(model_fn=model_fn)
        oc1m.read_iter_file(iterfn)

        oc1d = occam1d.Data(data_fn=data_fn)
        oc1d.read_resp_file(respfn)

        rough = float(oc1m.itdict['Roughness Value'])
        rms = float(oc1m.itdict['Misfit Value'])
        rough = f'{rough:.1f}'
        rms = f'{rms:.1f}'

        title = 'RMS: '+rms+'    Roughness: '+rough

        depths = []
        res = []

        for i, val in enumerate(oc1m.model_res[:, 1]):
            if i == 0:
                continue
            if i > 1:
                depths.append(-oc1m.model_depth[i-1])
                res.append(val)

            depths.append(-oc1m.model_depth[i])
            res.append(val)

        x = 1/oc1d.freq
        rdata = [oc1d.data['resxy'][0], oc1d.data['resxy'][2]]
        pdata = [oc1d.data['phasexy'][0], oc1d.data['phasexy'][2]]

        self.mmc.update_line(x, pdata, rdata, depths, res, title)

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
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showlog('No EDI data')
            return False

        occam_path = os.path.dirname(__file__)[:-2]+r'\bin\occam1d'
        if platform.system() == 'Windows':
            occam_path += '.exe'

        if os.path.exists(occam_path):
            self.le_occfile.setText(occam_path)

        self.cmb_1.currentIndexChanged.disconnect()
        self.cmb_1.clear()

        for i in self.data:
            self.cmb_1.addItem(i)

        self.cmb_1.setCurrentIndex(0)
        self.cmb_1.currentIndexChanged.connect(self.change_band)

        i = self.cmb_1.currentText()
        mode = self.cmb_mode.currentText()
        edi_file = self.data[i].fn
        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            allfiles = glob.glob(save_path+r'\*.resp')
            if len(allfiles) > 0:
                self.hs_profnum.setMaximum(len(allfiles))
                self.hs_profnum.setMinimum(1)

        self.change_band()

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
        self.saveobj(self.le_targetdepth)
        self.saveobj(self.le_nlayers)
        self.saveobj(self.le_bottomlayer)
        self.saveobj(self.le_airlayer)
        self.saveobj(self.le_z1layer)
        self.saveobj(self.le_maxiter)
        self.saveobj(self.le_targetrms)
        self.saveobj(self.le_errres)
        self.saveobj(self.le_errphase)
        self.saveobj(self.le_errfloorphase)
        self.saveobj(self.le_errfloorres)
        self.saveobj(self.cmb_mode)
        self.saveobj(self.cb_remove_out_quad)


def tonumber(test, alttext=None):
    """
    Check if something is a number or matches alttext.

    Parameters
    ----------
    test : str
        Text to test.
    alttext : str, optional
        Alternate text to test. The default is None.

    Returns
    -------
    str or int or float
        Returns a lower case version of alttext, or a number.

    """
    if alttext is not None and test.lower() == alttext.lower():
        return test.lower()

    if not test.replace('.', '', 1).isdigit():
        return -999

    if '.' in test:
        return float(test)

    return int(test)


def _testfn_occam():
    """Test routine."""
    datadir = r'd:\workdata\MT\\'
    edi_file = datadir+r"synth02.edi"

    # Create an MT object
    mt_obj = MT(edi_file)

    print('loading complete')

    app = QtWidgets.QApplication(sys.argv)
    test = Occam1D(None)
    test.indata['MT - EDI'] = {'SYNTH02': mt_obj}
    test.settings()


def _testfn():
    """Test routine."""
    datadir = r'd:\Work\workdata\MT\\'
    allfiles = glob.glob(datadir+'\\*.edi')

    for edi_file in allfiles:
        mt_obj = MT(edi_file)

    print('loading complete')

    app = QtWidgets.QApplication(sys.argv)
    test = Occam1D(None)
    test.indata['MT - EDI'] = {'SYNTH02': mt_obj}
    test.settings()


if __name__ == "__main__":
    _testfn_occam()
