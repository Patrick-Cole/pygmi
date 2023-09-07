# -----------------------------------------------------------------------------
# Name:        tab_param.py (part of PyGMI)
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
"""Parameter Display Tab Routines."""

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np

from pygmi.pfmod import grvmag3d
from pygmi.pfmod import misc
from pygmi import menu_default


class MergeLith(QtWidgets.QDialog):
    """Class to call up a dialog for ranged copying."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lw_lithmaster = QtWidgets.QListWidget()
        self.lw_lithmerge = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_1 = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()

        lbl_1 = QtWidgets.QLabel('Master Lithology')
        lbl_2 = QtWidgets.QLabel('Lithologies To Merge')

        self.lw_lithmaster.setSelectionMode(self.lw_lithmaster.SingleSelection)
        self.lw_lithmerge.setSelectionMode(self.lw_lithmerge.MultiSelection)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Merge Lithologies')

        gl_1.addWidget(lbl_1, 0, 0, 1, 1)
        gl_1.addWidget(self.lw_lithmaster, 0, 1, 1, 1)
        gl_1.addWidget(lbl_2, 1, 0, 1, 1)
        gl_1.addWidget(self.lw_lithmerge, 1, 1, 1, 1)
        gl_1.addWidget(buttonbox, 2, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)


class LithNotes(QtWidgets.QDialog):
    """Class to call up a dialog for lithology descriptions."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.lmod1 = self.parent.lmod1
        self.oldrowtext = None
        self.codelist = {}
        self.noteslist = {}

        self.lithcode = QtWidgets.QSpinBox()
        self.notes = QtWidgets.QTextEdit()
        self.lw_param_defs = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_1 = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()

        lbl_1 = QtWidgets.QLabel('Lithology Code')
        lbl_2 = QtWidgets.QLabel('Notes')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Preferred)
        sizepolicy.setHorizontalStretch(0)
        sizepolicy.setVerticalStretch(0)
        sizepolicy.setHeightForWidth(
            self.lw_param_defs.sizePolicy().hasHeightForWidth())

        self.lw_param_defs.setSizePolicy(sizepolicy)
        self.lw_param_defs.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        self.lithcode.setMaximum(9999999)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Lithology Notes')

        gl_1.addWidget(self.lw_param_defs, 0, 1, 1, 1)
        gl_1.addWidget(lbl_1, 1, 0, 1, 1)
        gl_1.addWidget(self.lithcode, 1, 1, 1, 1)
        gl_1.addWidget(lbl_2, 2, 0, 1, 1)
        gl_1.addWidget(self.notes, 2, 1, 1, 1)
        gl_1.addWidget(buttonbox, 3, 1, 1, 1)

        self.lw_param_defs.currentItemChanged.connect(self.lw_index_change)

        buttonbox.accepted.connect(self.apply_changes)
        buttonbox.rejected.connect(self.reject)

    def apply_changes(self):
        """
        Apply changes.

        Returns
        -------
        None.

        """
        i = self.lw_param_defs.currentRow()
        if i == -1:
            i = 0
        itxt = str(self.lw_param_defs.item(i).text())
        self.codelist[itxt] = self.lithcode.value()
        self.noteslist[itxt] = self.notes.toPlainText()

        for i in self.lmod1.lith_list:
            self.lmod1.lith_list[i].lithcode = self.codelist[i]
            self.lmod1.lith_list[i].lithnotes = self.noteslist[i]

        self.accept()

    def lw_index_change(self):
        """
        List box in parameter tab for definitions.

        Returns
        -------
        None.

        """
        if self.oldrowtext is not None:
            self.codelist[self.oldrowtext] = self.lithcode.value()
            self.noteslist[self.oldrowtext] = self.notes.toPlainText()

        i = self.lw_param_defs.currentRow()
        if i == -1:
            i = 0
        itxt = str(self.lw_param_defs.item(i).text())

        self.lithcode.setValue(self.codelist[itxt])
        self.notes.setPlainText(self.noteslist[itxt])
        self.oldrowtext = itxt

    def tab_activate(self):
        """
        Entry point.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        misc.update_lith_lw(self.lmod1, self.lw_param_defs)
        # Need this to init the first values.
        self.codelist = {}
        self.noteslist = {}
        for i in self.lmod1.lith_list:
            self.codelist[i] = self.lmod1.lith_list[i].lithcode
            self.noteslist[i] = self.lmod1.lith_list[i].lithnotes

        self.exec_()


class ParamDisplay(QtWidgets.QDialog):
    """Widget class to call the main interface."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.lmod1 = parent.lmod1
        self.grid_stretch = 'linear'
        self.showtext = parent.showtext
        self.islmod1 = True

        self.dsb_mht = QtWidgets.QDoubleSpinBox()
        self.dsb_hdec = QtWidgets.QDoubleSpinBox()
        self.dsb_hint = QtWidgets.QDoubleSpinBox()
        self.dsb_hinc = QtWidgets.QDoubleSpinBox()
        self.dsb_ght = QtWidgets.QDoubleSpinBox()
        self.dsb_gregional = QtWidgets.QDoubleSpinBox()

        self.pb_rename_def = QtWidgets.QPushButton('Rename Current Definition')
        self.pb_rem_def = QtWidgets.QPushButton('Remove Current Definition')
        self.pb_merge_def = QtWidgets.QPushButton('Merge Definitions')
        self.pb_add_def = QtWidgets.QPushButton('Add New Lithological'
                                                ' Definition')
        self.lw_param_defs = QtWidgets.QListWidget()

        self.gbox_lithprops = QtWidgets.QGroupBox()
        self.dsb_mdec = QtWidgets.QDoubleSpinBox()
        self.dsb_density = QtWidgets.QDoubleSpinBox()
        self.dsb_minc = QtWidgets.QDoubleSpinBox()
        self.dsb_susc = QtWidgets.QDoubleSpinBox()
        self.dsb_rmi = QtWidgets.QDoubleSpinBox()
        self.dsb_qratio = QtWidgets.QDoubleSpinBox()
        self.dsb_magnetization = QtWidgets.QDoubleSpinBox()

        self.setupui()
        self.init()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.setWindowTitle('Geophysical Parameters')
        helpdocs = menu_default.HelpButton('pygmi.pfmod.param')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Preferred)
        sizepolicy.setHorizontalStretch(0)
        sizepolicy.setVerticalStretch(0)
        sizepolicy.setHeightForWidth(
            self.lw_param_defs.sizePolicy().hasHeightForWidth())

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        # General Properties
        gb_gen_prop = QtWidgets.QGroupBox('General Properties')
        gl_gen_prop = QtWidgets.QGridLayout(gb_gen_prop)

        lbl_1 = QtWidgets.QLabel('Gravity Regional (mGal)')
        lbl_2 = QtWidgets.QLabel('Height of observation - Gravity')
        lbl_3 = QtWidgets.QLabel('Height of observation - Magnetic')
        lbl_4 = QtWidgets.QLabel('Magnetic Field Intensity (nT)')
        lbl_5 = QtWidgets.QLabel('Magnetic Inclination')
        lbl_6 = QtWidgets.QLabel('Magnetic Declination')

        gl_gen_prop.addWidget(lbl_1, 0, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_gregional, 0, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_2, 2, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_ght, 2, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_3, 3, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_mht, 3, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_4, 4, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hint, 4, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_5, 5, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hinc, 5, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_6, 6, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hdec, 6, 1, 1, 1)

        # Lithological Properties
        gb_lith_prop = QtWidgets.QGroupBox('Lithological Properties')
        gl_lith_prop = QtWidgets.QGridLayout(gb_lith_prop)

        pb_applylith = QtWidgets.QPushButton('Apply Changes')

        self.dsb_gregional.setMinimum(-10000.0)
        self.dsb_gregional.setMaximum(10000.0)
        self.dsb_gregional.setSingleStep(1.0)
        self.dsb_gregional.setProperty('value', 0.0)
        self.dsb_ght.setMaximum(999999999.0)
        self.dsb_mht.setMaximum(999999999.0)
        self.dsb_mht.setProperty('value', 100.0)
        self.dsb_hint.setMaximum(999999999.0)
        self.dsb_hint.setProperty('value', 27000.0)
        self.dsb_hinc.setMinimum(-90.0)
        self.dsb_hinc.setMaximum(90.0)
        self.dsb_hinc.setProperty('value', -63.0)
        self.dsb_hdec.setMinimum(-360.0)
        self.dsb_hdec.setMaximum(360.0)
        self.dsb_hdec.setProperty('value', -17.0)

        self.lw_param_defs.setSizePolicy(sizepolicy)
        self.lw_param_defs.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        gl_lith_prop.addWidget(self.pb_add_def, 0, 0, 1, 1)
        gl_lith_prop.addWidget(self.gbox_lithprops, 0, 1, 5, 1)
        gl_lith_prop.addWidget(self.lw_param_defs, 1, 0, 1, 1)
        gl_lith_prop.addWidget(self.pb_rename_def, 2, 0, 1, 1)
        gl_lith_prop.addWidget(self.pb_rem_def, 3, 0, 1, 1)
        gl_lith_prop.addWidget(self.pb_merge_def, 4, 0, 1, 1)

        gl_lithprops = QtWidgets.QGridLayout(self.gbox_lithprops)

        lbl_7 = QtWidgets.QLabel('Magnetic Susceptibility (SI)')
        lbl_8 = QtWidgets.QLabel('Remanent Magnetization Intensity (nT)')
        lbl_9 = QtWidgets.QLabel('Q Ratio')
        lbl_10 = QtWidgets.QLabel('Remanent Magnetization (A/m)')
        lbl_11 = QtWidgets.QLabel('Remanent Inclination')
        lbl_12 = QtWidgets.QLabel('Remanent Declination')
        lbl_13 = QtWidgets.QLabel('Density (g/cm3)')

        self.dsb_susc.setDecimals(7)
        self.dsb_susc.setMaximum(999999999.0)
        self.dsb_susc.setSingleStep(0.01)
        self.dsb_susc.setProperty('value', 0.01)
        self.dsb_rmi.setEnabled(True)
        self.dsb_rmi.setDecimals(5)
        self.dsb_rmi.setMaximum(999999999.0)
        self.dsb_qratio.setEnabled(True)
        self.dsb_qratio.setMaximum(999999999.0)
        self.dsb_qratio.setDecimals(5)
        self.dsb_magnetization.setEnabled(True)
        self.dsb_magnetization.setDecimals(5)
        self.dsb_magnetization.setMaximum(999999999.0)
        self.dsb_minc.setEnabled(True)
        self.dsb_minc.setMinimum(-90.0)
        self.dsb_minc.setMaximum(90.0)
        self.dsb_minc.setProperty('value', -63.0)
        self.dsb_mdec.setEnabled(True)
        self.dsb_mdec.setMinimum(-360.0)
        self.dsb_mdec.setMaximum(360.0)
        self.dsb_mdec.setProperty('value', -17.0)
        self.dsb_density.setDecimals(5)
        self.dsb_density.setSingleStep(0.01)
        self.dsb_density.setProperty('value', 2.75)

        gl_lithprops.addWidget(lbl_13, 3, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_density, 3, 1, 1, 1)
        gl_lithprops.addWidget(lbl_7, 4, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_susc, 4, 1, 1, 1)
        gl_lithprops.addWidget(lbl_8, 5, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_rmi, 5, 1, 1, 1)
        gl_lithprops.addWidget(lbl_10, 6, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_magnetization, 6, 1, 1, 1)
        gl_lithprops.addWidget(lbl_9, 7, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_qratio, 7, 1, 1, 1)
        gl_lithprops.addWidget(lbl_11, 8, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_minc, 8, 1, 1, 1)
        gl_lithprops.addWidget(lbl_12, 9, 0, 1, 1)
        gl_lithprops.addWidget(self.dsb_mdec, 9, 1, 1, 1)
        gl_lithprops.addWidget(pb_applylith, 10, 0, 1, 2)

        hbl.addWidget(helpdocs)
        hbl.addWidget(buttonbox)

        vbl.addWidget(gb_gen_prop)
        vbl.addWidget(gb_lith_prop)
        vbl.addLayout(hbl)

        self.add_defs(deftxt='Background')  # First call is for background
        self.add_defs()  # Second is for the first lithology type

        self.lw_param_defs.currentItemChanged.connect(self.lw_index_change)
        self.lw_param_defs.itemDoubleClicked.connect(self.lw_color_change)
        self.pb_add_def.clicked.connect(self.add_def)
        self.pb_rem_def.clicked.connect(self.rem_defs)
        self.pb_merge_def.clicked.connect(self.merge_defs)
        self.pb_rename_def.clicked.connect(self.rename_defs)
        self.lw_param_defs.itemChanged.connect(self.change_defs)
        self.dsb_susc.valueChanged.connect(self.change_qratio)
        self.dsb_rmi.valueChanged.connect(self.change_rmi)
        self.dsb_magnetization.valueChanged.connect(self.change_magnetization)
        self.dsb_qratio.valueChanged.connect(self.change_qratio)

        pb_applylith.clicked.connect(self.apply_lith)

        buttonbox.accepted.connect(self.apply_changes)
        buttonbox.rejected.connect(self.reject)

    def add_defs(self, deftxt='', getcol=False, lmod=None):
        """
        Add geophysical definitions and make them editable.

        Parameters
        ----------
        deftxt : str, optional
            Definition text. The default is ''.
        getcol : bool, optional
            Get column. The default is False.
        lmod : LithModel, optional
            3D model. The default is None.

        Returns
        -------
        None.

        """
        if lmod is not None:
            self.islmod1 = False
        else:
            lmod = self.lmod1
            self.islmod1 = True

        new_lith_index = 0
        if lmod.lith_list:
            lmod.update_lith_list_reverse()
            new_lith_index = max(lmod.lith_list_reverse.keys())+1

        defnum = self.lw_param_defs.count()
        if deftxt == '':
            deftxt = 'Generic '+str(defnum)

        lmod.lith_list[deftxt] = grvmag3d.GeoData(
            self.parent, lmod.numx, lmod.numy, lmod.numz, lmod.dxy, lmod.d_z,
            lmod.mht, lmod.ght)

        litho = lmod.lith_list['Background']
        lithn = lmod.lith_list[deftxt]
        lithn.hintn = litho.hintn
        lithn.finc = litho.finc
        lithn.fdec = litho.fdec
        lithn.zobsm = litho.zobsm
        lithn.bdensity = litho.bdensity
        lithn.zobsg = litho.zobsg

        lithn.lith_index = new_lith_index

        if deftxt == 'Background':
            lithn.susc = 0
            lithn.density = lithn.bdensity

        if getcol is True:
            col = QtWidgets.QColorDialog.getColor(parent=self.parent)
            lmod.mlut[lithn.lith_index] = [col.red(), col.green(), col.blue()]

        # setup list widgets
        misc.update_lith_lw(self.lmod1, self.lw_param_defs)

        if defnum == 0:
            lithn.susc = 0
            lithn.density = lithn.bdensity

        self.lw_index_change()

    def apply_lith(self):
        """
        Apply lithological changes.

        Returns
        -------
        None.

        """
        lith = self.get_lith()
        lith.density = self.dsb_density.value()
        if lith == self.lmod1.lith_list['Background']:
            for lith2 in list(self.lmod1.lith_list.values()):
                lith2.bdensity = self.dsb_density.value()

        lith.susc = self.dsb_susc.value()
        lith.mstrength = self.dsb_magnetization.value()
        lith.mdec = self.dsb_mdec.value()
        lith.minc = self.dsb_minc.value()
        lith.qratio = self.dsb_qratio.value()
        lith.modified = True

        self.lmod1.lith_index_mag_old[:] = -1
        self.lmod1.lith_index_grv_old[:] = -1

        self.showtext('Lithological changes applied.')

    def apply_changes(self):
        """
        Apply geophysical properties.

        Returns
        -------
        None.

        """
        self.lmod1.gregional = self.dsb_gregional.value()
        self.lmod1.mht = self.dsb_mht.value()
        self.lmod1.ght = self.dsb_ght.value()
        for lith in list(self.lmod1.lith_list.values()):
            lith.zobsg = -self.dsb_ght.value()
            lith.zobsm = -self.dsb_mht.value()
            lith.hintn = self.dsb_hint.value()
            lith.finc = self.dsb_hinc.value()
            lith.fdec = self.dsb_hdec.value()
            lith.modified = True
        self.showtext('Geophysical properties applied.')

        self.accept()

    def change_rmi(self):
        """
        Update spinboxes when rmi is changed.

        Returns
        -------
        None.

        """
        rmi = self.dsb_rmi.value()
        susc = self.dsb_susc.value()
        hintn = self.dsb_hint.value()

        mstrength = rmi/(400*np.pi)
        if (susc*hintn) == 0:
            qratio = 0.0
        else:
            qratio = rmi/(susc*hintn)

        self.disconnect_spin()
        self.dsb_magnetization.setValue(mstrength)
        self.dsb_qratio.setValue(qratio)
        self.connect_spin()

    def change_magnetization(self):
        """
        Update spinboxes when magnetization is changed.

        Returns
        -------
        None.

        """
        mstrength = self.dsb_magnetization.value()
        susc = self.dsb_susc.value()
        hintn = self.dsb_hint.value()

        rmi = 400*np.pi*mstrength

        if (susc*hintn) == 0:
            qratio = 0.0
        else:
            qratio = rmi/(susc*hintn)

        self.disconnect_spin()
        self.dsb_rmi.setValue(rmi)
        self.dsb_qratio.setValue(qratio)
        self.connect_spin()

    def change_qratio(self):
        """
        Update spinboxes when qratio is changed.

        Returns
        -------
        None.

        """
        qratio = self.dsb_qratio.value()
        susc = self.dsb_susc.value()
        hintn = self.dsb_hint.value()

        rmi = qratio*susc*hintn
        mstrength = rmi/(400*np.pi)

        self.disconnect_spin()
        self.dsb_rmi.setValue(rmi)
        self.dsb_magnetization.setValue(mstrength)
        self.connect_spin()

    def disconnect_spin(self):
        """
        Disconnect spin boxes.

        Returns
        -------
        None.

        """
        self.dsb_susc.valueChanged.disconnect()
        self.dsb_rmi.valueChanged.disconnect()
        self.dsb_magnetization.valueChanged.disconnect()
        self.dsb_qratio.valueChanged.disconnect()

    def connect_spin(self):
        """
        Connect spin boxes.

        Returns
        -------
        None.

        """
        self.dsb_susc.valueChanged.connect(self.change_qratio)
        self.dsb_rmi.valueChanged.connect(self.change_rmi)
        self.dsb_magnetization.valueChanged.connect(self.change_magnetization)
        self.dsb_qratio.valueChanged.connect(self.change_qratio)

    def change_defs(self, item):
        """
        Change geophysical definitions.

        Parameters
        ----------
        item : QListWidget item
            Parameter definition QListWidget item.

        Returns
        -------
        None.

        """
        if self.islmod1 is False:
            return
        itxt = item.text()
        itxtlist = []
        for i in range(self.lw_param_defs.count()):
            itxtlist.append(str(self.lw_param_defs.item(i).text()))

        for i in self.lmod1.lith_list.copy():
            if i not in itxtlist:
                if i == 'Background':
                    j = self.lw_param_defs.currentRow()
                    self.lw_param_defs.item(j).setText(i)
                else:
                    self.lmod1.lith_list[itxt] = self.lmod1.lith_list[i]
                    self.lmod1.lith_list.pop(i)

    def get_lith(self):
        """
        Get parameter definitions.

        Returns
        -------
        lith : GeoData
            Lithology data.

        """
        i = self.lw_param_defs.currentRow()
        if i == -1:
            itxt = 'Background'
        else:
            itxt = str(self.lw_param_defs.item(i).text())

        lith = self.lmod1.lith_list[itxt]
        return lith

    def init(self):
        """
        Initialize parameters.

        Returns
        -------
        None.

        """
        # Magetic Parameters
        self.dsb_hdec.setValue(-17.5)
        self.dsb_hinc.setValue(-62.9)
        self.dsb_hint.setValue(28464.0)
        self.dsb_mht.setValue(100.0)
        self.dsb_susc.setValue(0.01)

        # Remanence Parameters
        self.dsb_minc.setValue(-62.9)
        self.dsb_mdec.setValue(-17.5)

        # Gravity Parameters
        self.dsb_ght.setValue(0.0)
        self.dsb_density.setValue(2.75)
        self.dsb_gregional.setValue(0.0)

        # Body Parameters
        self.dsb_mdec.setValue(0.0)
        self.dsb_minc.setValue(0.0)

    def lw_color_change(self):
        """
        Routine to allow lithologies to have their colors changed.

        Returns
        -------
        None.

        """
        ctxt = str(self.lw_param_defs.currentItem().text())
        col = QtWidgets.QColorDialog.getColor()

        lithi = self.lmod1.lith_list[ctxt].lith_index

        self.lmod1.mlut[lithi] = [col.red(), col.green(), col.blue()]
        self.set_lw_colors(self.lw_param_defs)

    def lw_index_change(self):
        """
        List widget in parameter tab for definitions.

        Returns
        -------
        None.

        """
        i = self.lw_param_defs.currentRow()
        if i == -1:
            i = 0
        itxt = str(self.lw_param_defs.item(i).text())
        lith = self.lmod1.lith_list[itxt]

        self.dsb_susc.setValue(lith.susc)
        self.dsb_magnetization.setValue(lith.mstrength)
        self.dsb_qratio.setValue(lith.qratio)
        self.dsb_minc.setValue(lith.minc)
        self.dsb_mdec.setValue(lith.mdec)
        self.dsb_density.setValue(lith.density)

        self.gbox_lithprops.setTitle(itxt)

    def add_def(self):
        """
        Add geophysical definition.

        Returns
        -------
        None.

        """
        self.add_defs(getcol=True)

    def rem_defs(self):
        """
        Remove geophysical definition.

        Returns
        -------
        None.

        """
        crow = self.lw_param_defs.currentRow()
        if crow == -1:
            return
        ctxt = str(self.lw_param_defs.currentItem().text())
        if ctxt == 'Background':
            self.showtext('You cannot delete the background lithology')
            return

        if self.lw_param_defs.count() <= 2:
            self.showtext('You must have at least two lithologies')
            return

        lind = self.lmod1.lith_list[ctxt].lith_index
        del self.lmod1.lith_list[ctxt]
        self.lmod1.lith_index[self.lmod1.lith_index == lind] = 0
        self.lw_param_defs.takeItem(crow)

        misc.update_lith_lw(self.lmod1, self.lw_param_defs)

    def merge_defs(self):
        """
        Merge geophysical definitions.

        Returns
        -------
        None.

        """
        mlith = MergeLith()
        for i in self.lmod1.lith_list:
            mlith.lw_lithmaster.addItem(i)
            mlith.lw_lithmerge.addItem(i)

        tmp = mlith.exec_()

        if tmp == 0:
            return

        lithmaster = mlith.lw_lithmaster.selectedItems()
        lithmerge = mlith.lw_lithmerge.selectedItems()

        index_master = self.lmod1.lith_list[lithmaster[0].text()].lith_index

        for i in lithmerge:
            mtxt = i.text()
            j = self.lmod1.lith_list[mtxt].lith_index
            self.lmod1.lith_index[self.lmod1.lith_index == j] = index_master

            if mtxt != 'Background':
                del self.lmod1.lith_list[mtxt]

        misc.update_lith_lw(self.lmod1, self.lw_param_defs)

    def rename_defs(self):
        """
        Rename a definition.

        Returns
        -------
        None.

        """
        crow = self.lw_param_defs.currentRow()
        if crow == -1:
            return

        ctxt = str(self.lw_param_defs.currentItem().text())

        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent, 'Rename Definition',
            'Please type in the new name for the definition',
            QtWidgets.QLineEdit.Normal, ctxt)

        if isokay:
            self.lw_param_defs.currentItem().setText(skey)

        self.change_defs(self.lw_param_defs.currentItem())

    def set_lw_colors(self, lwidget, lmod=None):
        """
        Set list widget colors.

        Parameters
        ----------
        lwidget : QListWidget
            Lithology list widget..
        lmod : LithModel, optional
            3D Model. The default is None.

        Returns
        -------
        None.

        """
        if lmod is None:
            lmod = self.lmod1
        for i in range(lwidget.count()):
            tmp = lwidget.item(i)
            tindex = lmod.lith_list[tmp.text()].lith_index
            tcol = lmod.mlut[tindex]
            tmp.setBackground(QtGui.QColor(tcol[0], tcol[1], tcol[2], 255))

    def tab_activate(self):
        """
        Entry point.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        misc.update_lith_lw(self.lmod1, self.lw_param_defs)
# Need this to init the first values.
        itxt = str(self.lw_param_defs.item(0).text())
        lith = self.lmod1.lith_list[itxt]

        self.dsb_ght.setValue(-lith.zobsg)
        self.dsb_hint.setValue(lith.hintn)
        self.dsb_hinc.setValue(lith.finc)
        self.dsb_hdec.setValue(lith.fdec)
        self.dsb_mht.setValue(-lith.zobsm)

        self.lw_index_change()
        self.dsb_gregional.setValue(self.lmod1.gregional)
        self.exec_()

        self.parent.profile.lw_prof_defs.setCurrentRow(-1)
        self.parent.profile.change_defs()
