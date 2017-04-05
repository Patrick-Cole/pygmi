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
""" Parameter Display Tab Routines """

from PyQt5 import QtWidgets, QtGui
import numpy as np
from pygmi.pfmod import grvmag3d
from pygmi.pfmod import misc


class ParamDisplay(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        self.parent = parent
        self.lmod1 = parent.lmod1
        self.lmod2 = parent.lmod2
        self.max_lith_index = -1
        self.grid_stretch = 'linear'
        self.showtext = parent.showtext
        self.islmod1 = True

        self.userint = QtWidgets.QWidget()

        self.dsb_mht = QtWidgets.QDoubleSpinBox()
        self.dsb_hdec = QtWidgets.QDoubleSpinBox()
        self.dsb_hint = QtWidgets.QDoubleSpinBox()
        self.dsb_hinc = QtWidgets.QDoubleSpinBox()
        self.pb_autoregional = QtWidgets.QPushButton()
        self.dsb_ght = QtWidgets.QDoubleSpinBox()
        self.pb_apply_prop_changes = QtWidgets.QPushButton()
        self.dsb_gregional = QtWidgets.QDoubleSpinBox()

        self.pb_rename_def = QtWidgets.QPushButton()
        self.pb_rem_def = QtWidgets.QPushButton()
        self.lw_param_defs = QtWidgets.QListWidget()
        self.pb_add_def = QtWidgets.QPushButton()

        self.gbox_lithprops = QtWidgets.QGroupBox()
        self.dsb_mdec = QtWidgets.QDoubleSpinBox()
        self.dsb_density = QtWidgets.QDoubleSpinBox()
        self.dsb_minc = QtWidgets.QDoubleSpinBox()
        self.dsb_susc = QtWidgets.QDoubleSpinBox()
        self.pb_apply_lith_changes = QtWidgets.QPushButton()
        self.dsb_rmi = QtWidgets.QDoubleSpinBox()
        self.dsb_qratio = QtWidgets.QDoubleSpinBox()
        self.dsb_magnetization = QtWidgets.QDoubleSpinBox()

        self.setupui()
        self.init()

    def setupui(self):
        """ Setup UI """
        verticallayout = QtWidgets.QVBoxLayout(self.userint)

        gbox1 = QtWidgets.QGroupBox()
        gbox2 = QtWidgets.QGroupBox()
        glayout = QtWidgets.QGridLayout(gbox1)
        glayout2 = QtWidgets.QGridLayout(gbox2)
        glayout3 = QtWidgets.QGridLayout(self.gbox_lithprops)

        label_1 = QtWidgets.QLabel()
        label_2 = QtWidgets.QLabel()
        label_3 = QtWidgets.QLabel()
        label_4 = QtWidgets.QLabel()
        label_5 = QtWidgets.QLabel()
        label_6 = QtWidgets.QLabel()
        label_7 = QtWidgets.QLabel()
        label_8 = QtWidgets.QLabel()
        label_9 = QtWidgets.QLabel()
        label_10 = QtWidgets.QLabel()
        label_11 = QtWidgets.QLabel()
        label_12 = QtWidgets.QLabel()
        label_13 = QtWidgets.QLabel()

        self.dsb_gregional.setMinimum(-10000.0)
        self.dsb_gregional.setMaximum(10000.0)
        self.dsb_gregional.setSingleStep(1.0)
        self.dsb_gregional.setProperty("value", 100.0)
        self.dsb_ght.setMaximum(999999999.0)
        self.dsb_mht.setMaximum(999999999.0)
        self.dsb_mht.setProperty("value", 100.0)
        self.dsb_hint.setMaximum(999999999.0)
        self.dsb_hint.setProperty("value", 27000.0)
        self.dsb_hinc.setMinimum(-90.0)
        self.dsb_hinc.setMaximum(90.0)
        self.dsb_hinc.setProperty("value", -63.0)
        self.dsb_hdec.setMinimum(-360.0)
        self.dsb_hdec.setMaximum(360.0)
        self.dsb_hdec.setProperty("value", -17.0)

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                       QtWidgets.QSizePolicy.Preferred)
        sizepolicy.setHorizontalStretch(0)
        sizepolicy.setVerticalStretch(0)
        sizepolicy.setHeightForWidth(
            self.lw_param_defs.sizePolicy().hasHeightForWidth())
        self.lw_param_defs.setSizePolicy(sizepolicy)
        self.lw_param_defs.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dsb_susc.setDecimals(4)
        self.dsb_susc.setMaximum(999999999.0)
        self.dsb_susc.setSingleStep(0.01)
        self.dsb_susc.setProperty("value", 0.01)
        self.dsb_rmi.setEnabled(True)
        self.dsb_rmi.setDecimals(3)
        self.dsb_rmi.setMaximum(999999999.0)
        self.dsb_qratio.setEnabled(True)
        self.dsb_qratio.setMaximum(999999999.0)
        self.dsb_magnetization.setEnabled(True)
        self.dsb_magnetization.setDecimals(3)
        self.dsb_magnetization.setMaximum(999999999.0)
        self.dsb_minc.setEnabled(True)
        self.dsb_minc.setMinimum(-90.0)
        self.dsb_minc.setMaximum(90.0)
        self.dsb_minc.setProperty("value", -63.0)
        self.dsb_mdec.setEnabled(True)
        self.dsb_mdec.setMinimum(-360.0)
        self.dsb_mdec.setMaximum(360.0)
        self.dsb_mdec.setProperty("value", -17.0)
        self.dsb_density.setDecimals(3)
        self.dsb_density.setSingleStep(0.01)
        self.dsb_density.setProperty("value", 2.75)

        gbox1.setTitle("General Properties")
        gbox2.setTitle("Lithological Properties")
        label_1.setText("Gravity Regional (mgal)")
        label_2.setText("Height of observation - Gravity")
        label_3.setText("Height of observation - Magnetic")
        label_4.setText("Magnetic Field Intensity (nT)")
        label_5.setText("Magnetic Inclination")
        label_6.setText("Magnetic Declination")
        label_7.setText("Magnetic Susceptibility (SI)")
        label_8.setText("Remanent Magnetization Intensity (nT)")
        label_9.setText("Q Ratio")
        label_10.setText("Remanent Magnetization (A/m)")
        label_11.setText("Remanent Inclination")
        label_12.setText("Remanent Declination")
        label_13.setText("Density (g/cm3)")
        self.gbox_lithprops.setTitle("Lithological Properties")
        self.pb_apply_lith_changes.setText("Apply Changes")
        self.pb_apply_prop_changes.setText("Apply Changes")
        self.pb_autoregional.setText("Lithology Based Regional Estimation")
        self.pb_add_def.setText("Add New Lithological Definition")
        self.pb_rename_def.setText("Rename Current Definition")
        self.pb_rem_def.setText("Remove Current Definition")

        verticallayout.addWidget(gbox1)
        verticallayout.addWidget(gbox2)
        glayout.addWidget(label_1, 0, 0, 1, 1)
        glayout.addWidget(self.dsb_gregional, 0, 1, 1, 1)
        glayout.addWidget(self.pb_autoregional, 1, 0, 1, 2)
        glayout.addWidget(label_2, 2, 0, 1, 1)
        glayout.addWidget(self.dsb_ght, 2, 1, 1, 1)
        glayout.addWidget(label_3, 3, 0, 1, 1)
        glayout.addWidget(self.dsb_mht, 3, 1, 1, 1)
        glayout.addWidget(label_4, 4, 0, 1, 1)
        glayout.addWidget(self.dsb_hint, 4, 1, 1, 1)
        glayout.addWidget(label_5, 5, 0, 1, 1)
        glayout.addWidget(self.dsb_hinc, 5, 1, 1, 1)
        glayout.addWidget(label_6, 6, 0, 1, 1)
        glayout.addWidget(self.dsb_hdec, 6, 1, 1, 1)
        glayout.addWidget(self.pb_apply_prop_changes, 7, 0, 1, 2)
        glayout2.addWidget(self.pb_add_def, 0, 0, 1, 1)
        glayout2.addWidget(self.gbox_lithprops, 0, 1, 4, 1)
        glayout2.addWidget(self.lw_param_defs, 1, 0, 1, 1)
        glayout2.addWidget(self.pb_rename_def, 2, 0, 1, 1)
        glayout2.addWidget(self.pb_rem_def, 3, 0, 1, 1)
        glayout3.addWidget(label_13, 3, 0, 1, 1)
        glayout3.addWidget(self.dsb_density, 3, 1, 1, 1)
        glayout3.addWidget(label_7, 4, 0, 1, 1)
        glayout3.addWidget(self.dsb_susc, 4, 1, 1, 1)
        glayout3.addWidget(label_8, 5, 0, 1, 1)
        glayout3.addWidget(self.dsb_rmi, 5, 1, 1, 1)
        glayout3.addWidget(label_10, 6, 0, 1, 1)
        glayout3.addWidget(self.dsb_magnetization, 6, 1, 1, 1)
        glayout3.addWidget(label_9, 7, 0, 1, 1)
        glayout3.addWidget(self.dsb_qratio, 7, 1, 1, 1)
        glayout3.addWidget(label_11, 8, 0, 1, 1)
        glayout3.addWidget(self.dsb_minc, 8, 1, 1, 1)
        glayout3.addWidget(label_12, 9, 0, 1, 1)
        glayout3.addWidget(self.dsb_mdec, 9, 1, 1, 1)
        glayout3.addWidget(self.pb_apply_lith_changes, 10, 0, 1, 2)

        self.add_defs(deftxt='Background')  # First call is for background
        self.add_defs()  # Second is for the first lithology type

        self.lw_param_defs.currentItemChanged.connect(self.lw_index_change)
        self.lw_param_defs.itemDoubleClicked.connect(self.lw_color_change)
        self.pb_add_def.clicked.connect(self.add_def)
        self.pb_rem_def.clicked.connect(self.rem_defs)
        self.pb_rename_def.clicked.connect(self.rename_defs)
        self.pb_apply_lith_changes.clicked.connect(self.apply_lith_changes)
        self.pb_apply_prop_changes.clicked.connect(self.apply_prop_changes)
        self.lw_param_defs.itemChanged.connect(self.change_defs)
        self.dsb_susc.valueChanged.connect(self.change_qratio)
        self.dsb_rmi.valueChanged.connect(self.change_rmi)
        self.dsb_magnetization.valueChanged.connect(self.change_magnetization)
        self.dsb_qratio.valueChanged.connect(self.change_qratio)
        self.pb_autoregional.clicked.connect(self.autoregional)

    def autoregional(self):
        """ Automatically estimates the regional """
        if 'Gravity Dataset' not in self.lmod1.griddata:
            self.showtext('No gravity dataset!')
            return

        self.parent.grvmag.calc_regional()
        grvmax = self.lmod2.griddata['Calculated Gravity'].data.max()
        grvmean = self.lmod1.griddata['Gravity Dataset'].data.mean()
        grvreg = grvmean-grvmax
        self.dsb_gregional.setValue(grvreg)

        self.showtext('Regional estimated.')
        # self.apply_prop_changes()

    def add_defs(self, deftxt='', getcol=False, lmod=None):
        """ Add geophysical definitions and make them editable"""

        if lmod is not None:
            self.islmod1 = False
        else:
            lmod = self.lmod1
            self.islmod1 = True

#        self.max_lith_index = len(lmod.lith_list)-1
        if len(lmod.lith_list) == 0:
            self.max_lith_index = -1
        else:
            lmod.update_lith_list_reverse()
            self.max_lith_index = max(lmod.lith_list_reverse.keys())

        defnum = self.lw_param_defs.count()
        if deftxt == '':
            deftxt = 'Generic '+str(defnum)

#        self.lw_param_defs.addItem(deftxt)
#        self.lw_3dmod_defs.addItem(deftxt)
#        self.lw_prof_defs.addItem(deftxt)
#        self.lw_editor_defs.addItem(deftxt)

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

        self.max_lith_index += 1
        lithn.lith_index = self.max_lith_index

        if deftxt == 'Background':
            lithn.susc = 0
            lithn.density = lithn.bdensity

        if getcol is True:
            col = QtGui.QColorDialog.getColor()
            lmod.mlut[lithn.lith_index] = [col.red(), col.green(), col.blue()]

# setup list widgets
#        tmpitem = self.lw_param_defs.item(defnum)
#        tmpitem.setFlags(tmpitem.flags() | QtCore.Qt.ItemIsEditable)
        misc.update_lith_lw(self.lmod1, self.lw_param_defs)

#        self.set_lw_colors(self.lw_param_defs, lmod)
#        self.set_lw_colors(self.lw_3dmod_defs, lmod)
#        self.set_lw_colors(self.lw_editor_defs, lmod)
#        self.set_lw_colors(self.lw_prof_defs, lmod)

        if defnum == 0:
            lithn.susc = 0
            lithn.density = lithn.bdensity

        self.lw_index_change()

    def apply_prop_changes(self):
        """ Applies geophysical property changes """

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

    def apply_lith_changes(self):
        """ Applies lithological changes """

        lith = self.get_lith()
        lith.density = self.dsb_density.value()
        if lith == self.lmod1.lith_list['Background']:
            for lith in list(self.lmod1.lith_list.values()):
                lith.bdensity = self.dsb_density.value()

        lith.susc = self.dsb_susc.value()
        lith.mstrength = self.dsb_magnetization.value()
        lith.mdec = self.dsb_mdec.value()
        lith.minc = self.dsb_minc.value()
        lith.qratio = self.dsb_qratio.value()
        lith.modified = True

        self.lmod1.lith_index_old[:] = -1

        self.showtext('Lithological changes applied.')

    def change_rmi(self):
        """ update spinboxes when rmi is changed """
        rmi = self.dsb_rmi.value()
        susc = self.dsb_susc.value()
        hintn = self.dsb_hint.value()

        mstrength = rmi/(400*np.pi)
        qratio = rmi/(susc*hintn)

        self.disconnect_spin()
        self.dsb_magnetization.setValue(mstrength)
        self.dsb_qratio.setValue(qratio)
        self.connect_spin()

    def change_magnetization(self):
        """ update spinboxes when magnetization is changed """
        mstrength = self.dsb_magnetization.value()
        susc = self.dsb_susc.value()
        hintn = self.dsb_hint.value()

        rmi = 400*np.pi*mstrength
        if susc != 0.0:
            qratio = rmi/(susc*hintn)
        else:
            qratio = 0.0

        self.disconnect_spin()
        self.dsb_rmi.setValue(rmi)
        self.dsb_qratio.setValue(qratio)
        self.connect_spin()

    def change_qratio(self):
        """ update spinboxes when qratio is changed """
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
        """ Disconnect spin boxes """
        self.dsb_susc.valueChanged.disconnect()
        self.dsb_rmi.valueChanged.disconnect()
        self.dsb_magnetization.valueChanged.disconnect()
        self.dsb_qratio.valueChanged.disconnect()

    def connect_spin(self):
        """ Connect spin boxes """
        self.dsb_susc.valueChanged.connect(self.change_qratio)
        self.dsb_rmi.valueChanged.connect(self.change_rmi)
        self.dsb_magnetization.valueChanged.connect(self.change_magnetization)
        self.dsb_qratio.valueChanged.connect(self.change_qratio)

    def change_defs(self, item):
        """ Change geophysical definitions """
        if self.islmod1 is False:
            return
        itxt = item.text()
        itxtlist = []
        for i in range(self.lw_param_defs.count()):
            itxtlist.append(str(self.lw_param_defs.item(i).text()))

        for i in self.lmod1.lith_list:
            if i not in itxtlist:
                if i == 'Background':
                    j = self.lw_param_defs.currentRow()
                    self.lw_param_defs.item(j).setText(i)
                else:
                    self.lmod1.lith_list[itxt] = self.lmod1.lith_list[i]
                    self.lmod1.lith_list.pop(i)
#                    self.lw_3dmod_defs.findItems(i,
#                            QtCore.Qt.MatchExactly)[0].setText(itxt)
#                    self.lw_prof_defs.findItems(i,
#                            QtCore.Qt.MatchExactly)[0].setText(itxt)
#                    self.lw_editor_defs.findItems(i,
#                            QtCore.Qt.MatchExactly)[0].setText(itxt)

    def get_lith(self):
        """ Get parameter definitions """
        i = self.lw_param_defs.currentRow()
        if i == -1:
            i = 0
        itxt = str(self.lw_param_defs.item(i).text())
        lith = self.lmod1.lith_list[itxt]
        return lith

    def init(self):
        """ Initialize parameters """
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
        self.dsb_gregional.setValue(100.00)

    # Body Parameters
        self.dsb_mdec.setValue(0.0)
        self.dsb_minc.setValue(0.0)

    def lw_color_change(self):
        """ Routine to allow lithologies to have their colors changed """
        ctxt = str(self.lw_param_defs.currentItem().text())
        col = QtGui.QColorDialog.getColor()

        lithi = self.lmod1.lith_list[ctxt].lith_index

        self.lmod1.mlut[lithi] = [col.red(), col.green(), col.blue()]
        self.set_lw_colors(self.lw_param_defs)

    def lw_index_change(self):
        """ List box in parameter tab for definitions """
        i = self.lw_param_defs.currentRow()
        if i == -1:
            i = 0
        itxt = str(self.lw_param_defs.item(i).text())
        lith = self.lmod1.lith_list[itxt]

        self.dsb_hint.setValue(lith.hintn)
        self.dsb_hinc.setValue(lith.finc)
        self.dsb_hdec.setValue(lith.fdec)
        self.dsb_mht.setValue(-lith.zobsm)
        self.dsb_susc.setValue(lith.susc)
        self.dsb_magnetization.setValue(lith.mstrength)
        self.dsb_qratio.setValue(lith.qratio)
        self.dsb_minc.setValue(lith.minc)
        self.dsb_mdec.setValue(lith.mdec)
        self.dsb_density.setValue(lith.density)
        self.dsb_ght.setValue(-lith.zobsg)

        self.gbox_lithprops.setTitle('Lithological Properties - ' + itxt)

    def add_def(self):
        """ Routine being called by button push """
        self.add_defs(getcol=True)

    def rem_defs(self):
        """ Remove geophysical definitions """
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

#        self.lw_param_defs.takeItem(crow)
#        self.lw_3dmod_defs.takeItem(crow)
#        self.lw_prof_defs.takeItem(crow)
#        self.lw_editor_defs.takeItem(crow)

        lind = self.lmod1.lith_list[ctxt].lith_index
        del self.lmod1.lith_list[ctxt]
        self.lmod1.lith_index[self.lmod1.lith_index == lind] = 0
        self.lw_param_defs.takeItem(crow)

        misc.update_lith_lw(self.lmod1, self.lw_param_defs)

    def rename_defs(self):
        """ Used to rename a definition """

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
        """ Set list widget colors """
        if lmod is None:
            lmod = self.lmod1
        for i in range(lwidget.count()):
            tmp = lwidget.item(i)
            tindex = lmod.lith_list[tmp.text()].lith_index
            tcol = lmod.mlut[tindex]
            tmp.setBackground(QtGui.QColor(tcol[0], tcol[1], tcol[2], 255))

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.lmod1 = self.parent.lmod1
        misc.update_lith_lw(self.lmod1, self.lw_param_defs)
# Need this to init the first values.
        self.lw_index_change()
        self.dsb_gregional.setValue(self.lmod1.gregional)
#        self.parent.mext.update_vals()
