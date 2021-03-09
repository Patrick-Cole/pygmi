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
from PyQt5 import QtWidgets, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.patches import Rectangle
import mtpy.modeling.occam1d as occam1d
from mtpy.core.mt import MT
from mtpy.core.z import Z, Tipper

import pygmi.menu_default as menu_default


class Metadata(QtWidgets.QDialog):
    """
    Edit Metadata.

    This class allows the editing of the metadata for MT data using a GUI.

    Attributes
    ----------
    name : oldtxt
        old text
    banddata : dictionary
        band data
    bandid : dictionary
        dictionary of strings containing band names.
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ''
        self.parent = parent

        self.combobox_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton('Rename Station Name')
        self.dsb_lat = QtWidgets.QLineEdit()
        self.dsb_lon = QtWidgets.QLineEdit()
        self.dsb_elev = QtWidgets.QLineEdit()
        self.dsb_utmx = QtWidgets.QLineEdit()
        self.dsb_utmy = QtWidgets.QLineEdit()
        self.dsb_utmzone = QtWidgets.QLineEdit()
        self.dsb_rot = QtWidgets.QLineEdit()

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
        groupbox = QtWidgets.QGroupBox('Dataset')

        gridlayout = QtWidgets.QGridLayout(groupbox)
        label_utmx = QtWidgets.QLabel('UTM X Coordinate:')
        label_utmy = QtWidgets.QLabel('UTM Y Coordinate:')
        label_lat = QtWidgets.QLabel('Latitude:')
        label_lon = QtWidgets.QLabel('Longitude:')
        label_elev = QtWidgets.QLabel('Elevation:')
        label_utmzone = QtWidgets.QLabel('UTM Zone:')
        label_rot = QtWidgets.QLabel('Rotation:')
        label_bandid = QtWidgets.QLabel('Station Name:')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        groupbox.setSizePolicy(sizepolicy)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Metadata')

        gridlayout_main.addWidget(label_bandid, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.combobox_bandid, 0, 1, 1, 3)
        gridlayout_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gridlayout_main.addWidget(groupbox, 2, 0, 1, 2)
        gridlayout_main.addWidget(buttonbox, 4, 0, 1, 4)

        gridlayout.addWidget(label_lat, 0, 0, 1, 1)
        gridlayout.addWidget(self.dsb_lat, 0, 1, 1, 1)
        gridlayout.addWidget(label_lon, 1, 0, 1, 1)
        gridlayout.addWidget(self.dsb_lon, 1, 1, 1, 1)
        gridlayout.addWidget(label_elev, 2, 0, 1, 1)
        gridlayout.addWidget(self.dsb_elev, 2, 1, 1, 1)
        gridlayout.addWidget(label_utmx, 3, 0, 1, 1)
        gridlayout.addWidget(self.dsb_utmx, 3, 1, 1, 1)
        gridlayout.addWidget(label_utmy, 4, 0, 1, 1)
        gridlayout.addWidget(self.dsb_utmy, 4, 1, 1, 1)
        gridlayout.addWidget(label_utmzone, 5, 0, 1, 1)
        gridlayout.addWidget(self.dsb_utmzone, 5, 1, 1, 1)
        gridlayout.addWidget(label_rot, 6, 0, 1, 1)
        gridlayout.addWidget(self.dsb_rot, 6, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)
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
        ctxt = str(self.combobox_bandid.currentText())
        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent, 'Rename Station Name',
            'Please type in the new name for the station',
            QtWidgets.QLineEdit.Normal, ctxt)

        if isokay:
            self.combobox_bandid.currentIndexChanged.disconnect()
            indx = self.combobox_bandid.currentIndex()
            txt = self.combobox_bandid.itemText(indx)
            self.banddata[skey] = self.banddata.pop(txt)
            self.dataid[skey] = self.dataid.pop(txt)
            self.oldtxt = skey
            self.combobox_bandid.setItemText(indx, skey)
            self.combobox_bandid.currentIndexChanged.connect(self.update_vals)

    def update_vals(self):
        """
        Update the values on the interface.

        Returns
        -------
        None.

        """
        odata = self.banddata[self.oldtxt]

        try:
            odata.lat = float(self.dsb_lat.text())
            odata.lon = float(self.dsb_lon.text())
            if self.dsb_utmx.text() != 'None':
                odata.east = float(self.dsb_utmx.text())
            if self.dsb_utmy.text() != 'None':
                odata.north = float(self.dsb_utmy.text())
            odata.elev = float(self.dsb_elev.text())
#            odata.utm_zone = float(self.dsb_utmzone.text())
            odata.rotation_angle = float(self.dsb_rot.text())
        except ValueError:
            self.showprocesslog('Value error - abandoning changes')

        indx = self.combobox_bandid.currentIndex()
        txt = self.combobox_bandid.itemText(indx)
        self.oldtxt = txt
        idata = self.banddata[txt]

        self.dsb_lat.setText(str(idata.lat))
        self.dsb_lon.setText(str(idata.lon))
        self.dsb_elev.setText(str(idata.elev))
        if np.isinf(idata.east):
            self.dsb_utmx.setText('None')
        else:
            self.dsb_utmx.setText(str(idata.east))
        if np.isinf(idata.north):
            self.dsb_utmy.setText('None')
        else:
            self.dsb_utmy.setText(str(idata.north))
        self.dsb_utmzone.setText(str(idata.utm_zone))
        self.dsb_rot.setText(str(idata.rotation_angle))

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

        self.combobox_bandid.currentIndexChanged.disconnect()
        self.combobox_bandid.addItems(bandid)
        indx = self.combobox_bandid.currentIndex()
        self.oldtxt = self.combobox_bandid.itemText(indx)
        self.combobox_bandid.currentIndexChanged.connect(self.update_vals)

        idata = self.banddata[self.oldtxt]

        self.dsb_lat.setText(str(idata.lat))
        self.dsb_lon.setText(str(idata.lon))
        self.dsb_elev.setText(str(idata.elev))
        if np.isinf(idata.east):
            self.dsb_utmx.setText('None')
        else:
            self.dsb_utmx.setText(str(idata.east))
        if np.isinf(idata.north):
            self.dsb_utmy.setText('None')
        else:
            self.dsb_utmy.setText(str(idata.north))
        self.dsb_utmzone.setText(str(idata.utm_zone))
        self.dsb_rot.setText(str(idata.rotation_angle))

        self.update_vals()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True


class MyMplCanvas(FigureCanvasQTAgg):
    """
    MPL Canvas class.
    """

    def __init__(self, parent=None):
        fig = Figure()
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

        self.figure.tight_layout()
        self.figure.canvas.draw()


class StaticShiftEDI(QtWidgets.QDialog):
    """Static shift EDI data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.data = None
        self.parent = parent

        self.setWindowTitle('Remove Static Shift')
        helpdocs = menu_default.HelpButton('pygmi.mt.static')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()
        hbl3 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.combobox2.addItems(['xy, yx', 'xx, yy'])
        self.combobox2.setCurrentIndex(0)

        self.shiftx = QtWidgets.QDoubleSpinBox()
        self.shiftx.setMinimum(0.)
        self.shiftx.setMaximum(100000.)
        self.shiftx.setValue(1.)
        self.shifty = QtWidgets.QDoubleSpinBox()
        self.shifty.setMinimum(0.)
        self.shifty.setMaximum(100000.)
        self.shifty.setValue(1.)
        label1 = QtWidgets.QLabel('Station Name:')
        label2 = QtWidgets.QLabel('Graph Type:')
        label3 = QtWidgets.QLabel('Shift X:')
        label4 = QtWidgets.QLabel('Shift Y:')
        self.checkbox = QtWidgets.QCheckBox('Apply to all stations:')
        pb_apply = QtWidgets.QPushButton('Remove Static Shift')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        hbl.addWidget(label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(label2)
        hbl.addWidget(self.combobox2)

        hbl3.addWidget(label3)
        hbl3.addWidget(self.shiftx)
        hbl3.addWidget(label4)
        hbl3.addWidget(self.shifty)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(pb_reset)
        hbl2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.checkbox)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl3)
        vbl.addLayout(hbl2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

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
        ssx = self.shiftx.value()
        ssy = self.shifty.value()

        if self.checkbox.isChecked():
            for i in self.data:
                i = self.combobox1.currentText()
                self.data[i].Z = self.data[i].remove_static_shift(ssx, ssy)
        else:
            i = self.combobox1.currentText()
            self.data[i].Z = self.data[i].remove_static_shift(ssx, ssy)

        self.change_band()

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()

        if self.checkbox.isChecked():
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
        i = self.combobox1.currentText()
        i2 = self.combobox2.currentText()
        self.mmc.update_line(self.data, i, i2)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showprocesslog('No EDI data')
            return False

        self.combobox1.currentIndexChanged.disconnect()

        self.combobox1.clear()
        for i in self.data:
            self.combobox1.addItem(i)

        self.combobox1.setCurrentIndex(0)
        self.combobox1.currentIndexChanged.connect(self.change_band)

        self.change_band()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

        self.shiftx.setValue(projdata['shiftx'])
        self.shifty.setValue(projdata['shifty'])
        self.checkbox.setChecked(projdata['checkbox'])

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

        projdata['shiftx'] = self.shiftx.value()
        projdata['shifty'] = self.shifty.value()
        projdata['checkbox'] = self.checkbox.isChecked()

        return projdata


class RotateEDI(QtWidgets.QDialog):
    """Rotate EDI data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.data = None
        self.parent = parent

        self.setWindowTitle('Rotate EDI data')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.combobox2.addItems(['xy, yx', 'xx, yy'])
        self.combobox2.setCurrentIndex(0)

        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setMinimum(0.)
        self.spinbox.setMaximum(360.)
        label1 = QtWidgets.QLabel('Station Name:')
        label2 = QtWidgets.QLabel('Graph Type:')
        label3 = QtWidgets.QLabel('Rotate Z (0 is North):')
        self.checkbox = QtWidgets.QCheckBox('Apply to all stations:')
        pb_apply = QtWidgets.QPushButton('Apply rotation')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        helpdocs = menu_default.HelpButton('pygmi.mt.rotate')

        hbl.addWidget(label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(label2)
        hbl.addWidget(self.combobox2)
        hbl.addWidget(label3)
        hbl.addWidget(self.spinbox)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(pb_reset)
        hbl2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addWidget(self.checkbox)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.combobox2.currentIndexChanged.connect(self.change_band)
        self.combobox1.currentIndexChanged.connect(self.change_band)

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
        rotZ = self.spinbox.value()

        if self.checkbox.isChecked():
            for i in self.data:
                self.data[i].Z.rotate(rotZ)
                self.data[i].Tipper.rotate(rotZ)
        else:
            i = self.combobox1.currentText()
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
        i = self.combobox1.currentText()

        if self.checkbox.isChecked():
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])

        self.spinbox.setValue(self.data[i].rotation_angle)

        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        i2 = self.combobox2.currentText()
        self.mmc.update_line(self.data, i, i2)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showprocesslog('No EDI data')
            return False

        self.combobox1.currentIndexChanged.disconnect()
        self.combobox1.clear()

        for i in self.data:
            self.combobox1.addItem(i)

        self.combobox1.setCurrentIndex(0)
        self.combobox1.currentIndexChanged.connect(self.change_band)

        i = self.combobox1.currentText()
        self.spinbox.setValue(self.data[i].rotation_angle)

        self.change_band()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

        self.spinbox.setValue(projdata['rotz'])
        self.checkbox.setChecked(projdata['checkbox'])

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

        projdata['rotz'] = self.spinbox.value()
        projdata['checkbox'] = self.checkbox.isChecked()

        return projdata


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
        self.figure.canvas.mpl_connect('resize_event',
                                       self.resize)

    def button_press_callback(self, event):
        """
        Mouse button release callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

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
        event : TYPE
            DESCRIPTION.

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
        event : TYPE
            DESCRIPTION.

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
        event : TYPE
            DESCRIPTION.

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

    def resize(self, name, canvas=None):
        """
        Resize event.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        canvas : TYPE, optional
            DESCRIPTION. The default is None.

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

#        ax1 = self.figure.add_subplot(3, 1, 1)
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
#        ax2.set_title('Normalised stacked graphs')

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


class EditEDI(QtWidgets.QDialog):
    """Edit EDI Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.data = None
        self.parent = parent

        self.setWindowTitle('Mask and Interpolate')
        helpdocs = menu_default.HelpButton('pygmi.mt.edit')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvasPick(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.combobox2.addItems(['xy, yx', 'xx, yy'])
        self.combobox2.setCurrentIndex(0)

        label1 = QtWidgets.QLabel('Station Name:')
        label2 = QtWidgets.QLabel('Graph Type:')
        pb_apply = QtWidgets.QPushButton('Mask and Interpolate')
        pb_reset = QtWidgets.QPushButton('Reset data')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        hbl.addWidget(label1)
        hbl.addWidget(self.combobox1)
        hbl.addWidget(label2)
        hbl.addWidget(self.combobox2)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(pb_reset)
        hbl2.addWidget(pb_apply)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(hbl)
        vbl.addLayout(hbl2)
        vbl.addWidget(buttonbox)

        pb_apply.clicked.connect(self.apply)
        pb_reset.clicked.connect(self.reset_data)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.combobox2.currentIndexChanged.connect(self.change_band)
        self.combobox1.currentIndexChanged.connect(self.change_band)

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

        i = self.combobox1.currentText()

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
        i = self.combobox1.currentText()
        self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])
        self.change_band()

    def change_band(self):
        """
        Combo to choose band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        i2 = self.combobox2.currentText()
        self.mmc.update_line(self.data, i, i2)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showprocesslog('No EDI data')
            return False

        self.combobox1.currentIndexChanged.disconnect()
        self.combobox1.clear()

        for i in self.data:
            self.combobox1.addItem(i)

        self.combobox1.setCurrentIndex(0)
        self.combobox1.currentIndexChanged.connect(self.change_band)

        self.change_band()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata


class MySlider(QtWidgets.QSlider):
    """
    My Slider

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
        event : TYPE
            DESCRIPTION.

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
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(),
                                                               self.maximum(),
                                                               event.x(),
                                                               self.width()))


class MyMplCanvas2(FigureCanvasQTAgg):
    """
    MPL Canvas class.
    """

    def __init__(self, parent=None):
        fig = Figure()
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
            DESCRIPTION. The default is None.

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

        gs.tight_layout(self.figure)
        self.figure.canvas.draw()


class Occam1D(QtWidgets.QDialog):
    """Occam 1D inversion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.data = None
        self.parent = parent
        self.cursoln = 0

        self.setWindowTitle('Occam 1D Inversion')
        helpdocs = menu_default.HelpButton('pygmi.mt.occam1d')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)

        vbl = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout(self)
        hbl2 = QtWidgets.QHBoxLayout()
        gbl = QtWidgets.QGridLayout()
        gbl.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.mmc = MyMplCanvas2(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.combomode = QtWidgets.QComboBox()
        self.combomode.addItems(['TE', 'TM', 'DET'])
        self.combomode.setCurrentIndex(0)
        self.errres = QtWidgets.QLineEdit('data')
        self.errres.setSizePolicy(sizepolicy)
        self.errphase = QtWidgets.QLineEdit('data')
        self.errphase.setSizePolicy(sizepolicy)
        self.errfloorres = QtWidgets.QLineEdit('4.')
        self.errfloorres.setSizePolicy(sizepolicy)
        self.errfloorphase = QtWidgets.QLineEdit('2.')
        self.errfloorphase.setSizePolicy(sizepolicy)
        self.remove_out_quad = QtWidgets.QCheckBox(r'Remove Resistivity/Phase values out of 1st/3rd Quadrant')

        self.targetdepth = QtWidgets.QLineEdit('40000.')
        self.targetdepth.setSizePolicy(sizepolicy)
        self.nlayers = QtWidgets.QLineEdit('100')
        self.nlayers.setSizePolicy(sizepolicy)
        self.bottomlayer = QtWidgets.QLineEdit('100000.')
        self.bottomlayer.setSizePolicy(sizepolicy)
        self.airlayer = QtWidgets.QLineEdit('10000.')
        self.airlayer.setSizePolicy(sizepolicy)
        self.z1layer = QtWidgets.QLineEdit('10.')
        self.z1layer.setSizePolicy(sizepolicy)
        self.maxiter = QtWidgets.QLineEdit('200')
        self.maxiter.setSizePolicy(sizepolicy)
        self.targetrms = QtWidgets.QLineEdit('1.')
        self.targetrms.setSizePolicy(sizepolicy)
        self.remove_out_quad.setChecked(True)

        self.hs_profnum = MySlider()
        self.hs_profnum.setOrientation(QtCore.Qt.Horizontal)

        label1 = QtWidgets.QLabel('Station Name:')
        label1.setSizePolicy(sizepolicy)
        label3 = QtWidgets.QLabel('Mode:')
        label4 = QtWidgets.QLabel('Resistivity Errorbar (Data or %):')
        label5 = QtWidgets.QLabel('Phase Errorbar (Data or %):')
        label6 = QtWidgets.QLabel('Resistivity Error Floor (%):')
        label7 = QtWidgets.QLabel('Phase Error Floor (degrees):')
        label7a = QtWidgets.QLabel(r'Remove Resistivity/Phase values out of 1st/3rd Quadrant (True/False):')
        label8 = QtWidgets.QLabel('Height of air layer:')
        label9 = QtWidgets.QLabel('Bottom of model:')
        label10 = QtWidgets.QLabel('Depth of target to investigate:')
        label11 = QtWidgets.QLabel('Depth of first layer:')
        label12 = QtWidgets.QLabel('Number of layers:')
        label13 = QtWidgets.QLabel('Maximum Iterations:')
        label14 = QtWidgets.QLabel('Target RMS:')

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        pb_apply = QtWidgets.QPushButton('Invert Station')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gbl.addWidget(label1, 0, 0)
        gbl.addWidget(self.combobox1, 0, 1)
        gbl.addWidget(label3, 2, 0)
        gbl.addWidget(self.combomode, 2, 1)
        gbl.addWidget(label4, 3, 0)
        gbl.addWidget(self.errres, 3, 1)
        gbl.addWidget(label5, 4, 0)
        gbl.addWidget(self.errphase, 4, 1)
        gbl.addWidget(label6, 5, 0)
        gbl.addWidget(self.errfloorres, 5, 1)
        gbl.addWidget(label7, 6, 0)
        gbl.addWidget(self.errfloorphase, 6, 1)
        gbl.addWidget(label8, 7, 0)
        gbl.addWidget(self.airlayer, 7, 1)
        gbl.addWidget(label9, 8, 0)
        gbl.addWidget(self.bottomlayer, 8, 1)
        gbl.addWidget(label10, 9, 0)
        gbl.addWidget(self.targetdepth, 9, 1)
        gbl.addWidget(label11, 10, 0)
        gbl.addWidget(self.z1layer, 10, 1)
        gbl.addWidget(label12, 11, 0)
        gbl.addWidget(self.nlayers, 11, 1)
        gbl.addWidget(label13, 12, 0)
        gbl.addWidget(self.maxiter, 12, 1)
        gbl.addWidget(label14, 13, 0)
        gbl.addWidget(self.targetrms, 13, 1)
        gbl.addWidget(self.remove_out_quad, 14, 0, 1, 2)

        gbl.addWidget(pb_apply, 15, 0, 1, 2)
        gbl.addWidget(buttonbox, 16, 0, 1, 2)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(self.lbl_profnum)
        hbl2.addWidget(self.hs_profnum)

        vbl.addWidget(self.mmc)
        vbl.addLayout(hbl2)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gbl)
        hbl.addLayout(vbl)

        pb_apply.clicked.connect(self.apply)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.combobox1.currentIndexChanged.connect(self.change_band)
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

        parm['tdepth'] = tonumber(self.targetdepth.text())
        parm['nlayers'] = tonumber(self.nlayers.text())
        parm['blayer'] = tonumber(self.bottomlayer.text())
        parm['alayer'] = tonumber(self.airlayer.text())
        parm['z1layer'] = tonumber(self.z1layer.text())
        parm['miter'] = tonumber(self.maxiter.text())
        parm['trms'] = tonumber(self.targetrms.text())
        parm['rerr'] = tonumber(self.errres.text(), 'data')
        parm['perr'] = tonumber(self.errphase.text(), 'data')
        parm['perrflr'] = tonumber(self.errfloorphase.text())
        parm['rerrflr'] = tonumber(self.errfloorres.text())
        parm['routq'] = self.remove_out_quad.isChecked()

        if -999 in parm.values():
            return

        mode = self.combomode.currentText()
        i = self.combobox1.currentText()
        edi_file = self.data[i].fn

        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            r = glob.glob(save_path+r'\*')
            for i in r:
                os.remove(i)
        else:
            os.makedirs(save_path)

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

        if not os.path.exists(occam_path):
            text = ('No Occam1D executable found. Please place it in the bin '
                    'directory. You may need to obtain the source code from '
                    'https://marineemlab.ucsd.edu/Projects/Occam/1DCSEM/ '
                    'and compile it. It should be called occam1d for '
                    'non-windows platforms and occam1d.exe for windows.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error', text,
                                          QtWidgets.QMessageBox.Ok)
            return

        self.mmc.figure.clear()
        self.mmc.figure.set_facecolor('r')
        self.mmc.figure.suptitle('Busy, please wait...', fontsize=14, y=0.5)
#        ax = self.mmc.figure.gca()
#        ax.text(0.5, 0.5, 'Busy, please wait...')
        self.mmc.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        occam1d.Run(s1.startup_fn, occam_path, mode=mode)

        self.mmc.figure.set_facecolor('w')

        allfiles = glob.glob(save_path+r'\*.resp')
        self.hs_profnum.setMaximum(len(allfiles))
        self.hs_profnum.setMinimum(1)

        self.change_band()

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])
        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        mode = self.combomode.currentText()
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

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            self.showprocesslog('No EDI data')
            return False

        self.combobox1.currentIndexChanged.disconnect()
        self.combobox1.clear()

        for i in self.data:
            self.combobox1.addItem(i)

        self.combobox1.setCurrentIndex(0)
        self.combobox1.currentIndexChanged.connect(self.change_band)

        i = self.combobox1.currentText()
        mode = self.combomode.currentText()
        edi_file = self.data[i].fn
        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            allfiles = glob.glob(save_path+r'\*.resp')
            if len(allfiles) > 0:
                self.hs_profnum.setMaximum(len(allfiles))
                self.hs_profnum.setMinimum(1)

        self.change_band()

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.targetdepth.setText(projdata['tdepth'])
        self.nlayers.setText(projdata['nlayers'])
        self.bottomlayer.setText(projdata['blayer'])
        self.airlayer.setText(projdata['alayer'])
        self.z1layer.setText(projdata['z1layer'])
        self.maxiter.setText(projdata['miter'])
        self.targetrms.setText(projdata['trms'])
        self.errres.setText(projdata['rerr'])
        self.errphase.setText(projdata['perr'])
        self.errfloorphase.setText(projdata['perrflr'])
        self.errfloorres.setText(projdata['rerrflr'])
        self.combomode.setCurrentText(projdata['mode'])
        self.remove_out_quad.setChecked(projdata['routq'])

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

        projdata['tdepth'] = self.targetdepth.text()
        projdata['nlayers'] = self.nlayers.text()
        projdata['blayer'] = self.bottomlayer.text()
        projdata['alayer'] = self.airlayer.text()
        projdata['z1layer'] = self.z1layer.text()
        projdata['miter'] = self.maxiter.text()
        projdata['trms'] = self.targetrms.text()
        projdata['rerr'] = self.errres.text()
        projdata['perr'] = self.errphase.text()
        projdata['perrflr'] = self.errfloorphase.text()
        projdata['rerrflr'] = self.errfloorres.text()
        projdata['mode'] = self.combomode.currentText()
        projdata['routq'] = self.remove_out_quad.isChecked()

        return projdata


def tonumber(test, alttext=None):
    """
    Checks if something is a number or matches alttext

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


def testfn_occam():
    """ main test """
    datadir = r'C:\Work\workdata\MT\\'
    edi_file = datadir+r"synth02.edi"

    # Create an MT object
    mt_obj = MT(edi_file)

    print('loading complete')

    app = QtWidgets.QApplication(sys.argv)
    test = Occam1D(None)
    test.indata['MT - EDI'] = {'SYNTH02': mt_obj}
    test.settings()


def testfn():
    """ main test """
    from mtpy.utils.shapefiles_creator import ShapeFilesCreator

    datadir = r'C:\Work\workdata\MT\\'
    allfiles = glob.glob(datadir+'\\*.edi')

    for edi_file in allfiles:
    # Create an MT object
        mt_obj = MT(edi_file)

    print('loading complete')

    app = QtWidgets.QApplication(sys.argv)
    test = Occam1D(None)
    test.indata['MT - EDI'] = {'SYNTH02': mt_obj}
    test.settings()


if __name__ == "__main__":
    testfn()
