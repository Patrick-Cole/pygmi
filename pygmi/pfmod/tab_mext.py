# -----------------------------------------------------------------------------
# Name:        tab_mext.py (part of PyGMI)
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
""" Model Extension Display Tab Routines """

# pylint: disable=E1101
from PySide import QtGui
import numpy as np
import scipy.interpolate as si


class MextDisplay(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        self.parent = parent
        self.lmod1 = parent.lmod1  # actual model
        self.showtext = parent.showtext
        self.pbars = self.parent.pbars

        mainwindow = QtGui.QWidget()

        self.groupbox = QtGui.QGroupBox(mainwindow)
        self.groupbox2 = QtGui.QGroupBox(mainwindow)
        self.verticallayout = QtGui.QVBoxLayout(mainwindow)

        self.gridlayout_2 = QtGui.QGridLayout(self.groupbox2)
        self.combo_dtm = QtGui.QComboBox(self.groupbox2)
        self.combo_mag = QtGui.QComboBox(self.groupbox2)
        self.combo_grv = QtGui.QComboBox(self.groupbox2)
        self.combo_reggrv = QtGui.QComboBox(self.groupbox2)

        self.gridlayout = QtGui.QGridLayout(self.groupbox)
        self.combo_dataset = QtGui.QComboBox(self.groupbox)
        self.dsb_utlx = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_utly = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_utlz = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_xextent = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_yextent = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_zextent = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_xycell = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_zcell = QtGui.QDoubleSpinBox(self.groupbox)
        self.sb_cols = QtGui.QSpinBox(self.groupbox)
        self.sb_rows = QtGui.QSpinBox(self.groupbox)
        self.sb_layers = QtGui.QSpinBox(self.groupbox)
        self.pb_apply_changes = QtGui.QPushButton(self.groupbox)

        self.userint = mainwindow
        self.setupui()
        self.init()

    def setupui(self):
        """ Setup UI """
        lbl0 = QtGui.QLabel(self.groupbox)
        lbl1 = QtGui.QLabel(self.groupbox)
        lbl2 = QtGui.QLabel(self.groupbox)
        lbl3 = QtGui.QLabel(self.groupbox)
        lbl4 = QtGui.QLabel(self.groupbox)
        lbl5 = QtGui.QLabel(self.groupbox)
        lbl6 = QtGui.QLabel(self.groupbox)
        lbl7 = QtGui.QLabel(self.groupbox)
        lbl8 = QtGui.QLabel(self.groupbox)
        lbl9 = QtGui.QLabel(self.groupbox)
        lbl10 = QtGui.QLabel(self.groupbox)
        lbl11 = QtGui.QLabel(self.groupbox)

        lbl1_2 = QtGui.QLabel(self.groupbox2)
        lbl2_2 = QtGui.QLabel(self.groupbox2)
        lbl3_2 = QtGui.QLabel(self.groupbox2)
        lbl4_2 = QtGui.QLabel(self.groupbox2)

        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Fixed)
        self.groupbox2.setSizePolicy(sizepolicy)
        self.combo_dataset.setSizePolicy(sizepolicy)
        self.combo_mag.setSizePolicy(sizepolicy)
        self.combo_dtm.setSizePolicy(sizepolicy)
        self.combo_grv.setSizePolicy(sizepolicy)

        self.gridlayout_2.addWidget(lbl1_2, 0, 0, 1, 1)
        self.gridlayout_2.addWidget(lbl2_2, 1, 0, 1, 1)
        self.gridlayout_2.addWidget(lbl3_2, 2, 0, 1, 1)
        self.gridlayout_2.addWidget(lbl4_2, 3, 0, 1, 1)
        self.gridlayout_2.addWidget(self.combo_dtm, 0, 1, 1, 1)
        self.gridlayout_2.addWidget(self.combo_mag, 1, 1, 1, 1)
        self.gridlayout_2.addWidget(self.combo_grv, 2, 1, 1, 1)
        self.gridlayout_2.addWidget(self.combo_reggrv, 3, 1, 1, 1)

        self.verticallayout.addWidget(self.groupbox2)

        self.gridlayout.addWidget(lbl0, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.combo_dataset, 0, 1, 1, 1)
        self.gridlayout.addWidget(lbl3, 5, 0, 1, 1)
        self.dsb_utlx.setMinimum(-999999999.0)
        self.dsb_utlx.setMaximum(999999999.0)
        self.gridlayout.addWidget(self.dsb_utlx, 5, 1, 1, 1)
        self.gridlayout.addWidget(lbl4, 6, 0, 1, 1)
        self.dsb_utly.setMinimum(-999999999.0)
        self.dsb_utly.setMaximum(999999999.0)
        self.gridlayout.addWidget(self.dsb_utly, 6, 1, 1, 1)
        self.gridlayout.addWidget(lbl1, 7, 0, 1, 1)
        self.dsb_utlz.setMinimum(-999999999.0)
        self.dsb_utlz.setMaximum(999999999.0)
#        self.dsb_utlz.setEnabled(False)
        self.gridlayout.addWidget(self.dsb_utlz, 7, 1, 1, 1)
        self.gridlayout.addWidget(lbl8, 8, 0, 1, 1)
        self.dsb_xextent.setEnabled(True)
        self.dsb_xextent.setMinimum(0.1)
        self.dsb_xextent.setMaximum(2000000000.0)
        self.gridlayout.addWidget(self.dsb_xextent, 8, 1, 1, 1)
        self.gridlayout.addWidget(lbl9, 9, 0, 1, 1)
        self.dsb_yextent.setEnabled(True)
        self.dsb_yextent.setMinimum(0.1)
        self.dsb_yextent.setMaximum(2000000000.0)
        self.gridlayout.addWidget(self.dsb_yextent, 9, 1, 1, 1)
        self.gridlayout.addWidget(lbl10, 10, 0, 1, 1)
        self.dsb_zextent.setEnabled(True)
        self.dsb_zextent.setMinimum(0.1)
        self.dsb_zextent.setMaximum(2000000000.0)
        self.gridlayout.addWidget(self.dsb_zextent, 10, 1, 1, 1)
        self.gridlayout.addWidget(lbl5, 11, 0, 1, 1)
        self.dsb_xycell.setEnabled(True)
        self.dsb_xycell.setMinimum(0.1)
        self.dsb_xycell.setMaximum(1000000.0)
        self.gridlayout.addWidget(self.dsb_xycell, 11, 1, 1, 1)
        self.gridlayout.addWidget(lbl6, 12, 0, 1, 1)
        self.dsb_zcell.setEnabled(True)
        self.dsb_zcell.setDecimals(2)
        self.dsb_zcell.setMinimum(0.1)
        self.dsb_zcell.setMaximum(1000000.0)
        self.dsb_zcell.setSingleStep(1.0)
        self.gridlayout.addWidget(self.dsb_zcell, 12, 1, 1, 1)
        self.gridlayout.addWidget(lbl7, 13, 0, 1, 1)
        self.sb_cols.setEnabled(False)
        self.sb_cols.setMinimum(1)
        self.sb_cols.setMaximum(1000000)
        self.gridlayout.addWidget(self.sb_cols, 13, 1, 1, 1)
        self.gridlayout.addWidget(lbl11, 14, 0, 1, 1)
        self.sb_rows.setEnabled(False)
        self.sb_rows.setMinimum(1)
        self.sb_rows.setMaximum(1000000)
        self.gridlayout.addWidget(self.sb_rows, 14, 1, 1, 1)
        self.gridlayout.addWidget(lbl2, 15, 0, 1, 1)
        self.sb_layers.setEnabled(False)
        self.sb_layers.setMinimum(1)
        self.sb_layers.setMaximum(1000000)
        self.gridlayout.addWidget(self.sb_layers, 15, 1, 1, 1)
        self.verticallayout.addWidget(self.groupbox)
        self.verticallayout.addWidget(self.pb_apply_changes)

        self.groupbox.setTitle("Model Extent Properties")
        self.groupbox2.setTitle("Dataset Information")
        self.pb_apply_changes.setText("Accept Proposed Changes")
        lbl0.setText("Get Study Area from following Dataset:")
        lbl1.setText("Upper Top Left Z Coordinate (from DTM):")
        lbl2.setText("Number of Layers (Z Direction):")
        lbl3.setText("Upper Top Left X Coordinate:")
        lbl4.setText("Upper Top Left Y Coordinate:")
        lbl5.setText("X and Y Cell Size:")
        lbl6.setText("Z Cell Size:")
        lbl7.setText("Number of Columns (X Direction):")
        lbl8.setText("Total X Extent:")
        lbl9.setText("Total Y Extent:")
        lbl10.setText("Total Z Extent (Depth):")
        lbl11.setText("Number of Rows (Y Direction):")
        lbl1_2.setText("DTM Dataset:")
        lbl2_2.setText("Magnetic Dataset:")
        lbl3_2.setText("Gravity Dataset:")
        lbl4_2.setText("Gravity Regional Dataset:")

        self.pb_apply_changes.clicked.connect(self.apply_changes)
        self.dsb_xycell.valueChanged.connect(self.xycell)
        self.dsb_zcell.valueChanged.connect(self.zcell)
        self.dsb_utlx.valueChanged.connect(self.upd_layers)
        self.dsb_utly.valueChanged.connect(self.upd_layers)
        self.dsb_utlz.valueChanged.connect(self.upd_layers)
        self.dsb_xextent.valueChanged.connect(self.upd_layers)
        self.dsb_yextent.valueChanged.connect(self.upd_layers)
        self.dsb_zextent.valueChanged.connect(self.upd_layers)

        self.combo_dataset.addItems(['None'])
        self.combo_dataset.currentIndexChanged.connect(self.get_area)
        self.combo_mag.addItems(['None'])
        self.combo_grv.addItems(['None'])
        self.combo_reggrv.addItems(['None'])
        self.combo_dtm.addItems(['None'])
        self.combo_dtm.currentIndexChanged.connect(self.choose_dtm)

    def apply_changes(self):
        """ Update when changing from this tab """
        self.pbars.resetall()
        self.showtext('Working...')

        self.choose_combo(self.combo_dtm, 'DTM Dataset')
        self.choose_combo(self.combo_mag, 'Magnetic Dataset')
        self.choose_combo(self.combo_grv, 'Gravity Dataset')
        self.choose_combo(self.combo_reggrv, 'Gravity Regional')

        cols = self.sb_cols.value()
        rows = self.sb_rows.value()
        layers = self.sb_layers.value()
        utlx = self.dsb_utlx.value()
        utly = self.dsb_utly.value()
        utlz = self.dsb_utlz.value()
        dxy = self.dsb_xycell.value()
        d_z = self.dsb_zcell.value()

        self.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z)

        self.update_vals()

        self.pbars.incr()
        self.parent.outdata['Raster'] = list(self.lmod1.griddata.values())
        self.showtext('Changes applied!')

    def choose_combo(self, combo, dtxt):
        """ Combo box choice routine """
        ctxt = combo.currentText()
        if ctxt != 'None' and ctxt != '':
            self.lmod1.griddata[dtxt] = self.parent.inraster[ctxt]
        elif ctxt == 'None' and dtxt in self.lmod1.griddata:
            self.lmod1.griddata.pop(dtxt)

    def choose_dtm(self):
        """ Combo box to choose current dataset """
        ctxt = self.combo_dtm.currentText()
        if ctxt != 'None' and ctxt != '':
            curgrid = self.parent.inraster[ctxt]

            self.dsb_utlz.setValue(curgrid.data.max())
            zextent = curgrid.data.ptp()+self.dsb_zcell.value()
            if zextent > self.dsb_zextent.value():
                self.dsb_zextent.setValue(zextent)

            self.upd_layers()

    def extgrid(self, gdata):
        """ Extrapolates the grid to get rid of nulls. Uses a masked grid """
        gtmp = np.array(gdata)  # gets rid of masked array
        gmask = np.logical_not(np.isnan(gtmp))

        if gmask.min() is True:
            return gdata

        points1 = np.where(gmask)
        z = gdata[gmask]
        outg = np.ones_like(gtmp)
        points2 = np.where(outg)
        outg = si.griddata(points1, z, points2, method='nearest')

        outg.shape = gtmp.shape
        outg[gmask] = gdata[gmask]
        outg = np.ma.array(outg)
        outg.mask = gdata.mask

        return outg

    def get_area(self):
        """ Get area """
        ctxt = self.combo_dataset.currentText()
        if ctxt != 'None' and ctxt != u'':
            curgrid = self.parent.inraster[ctxt]

            utlx = curgrid.tlx
            utly = curgrid.tly
            xextent = curgrid.cols*curgrid.xdim
            yextent = curgrid.rows*curgrid.ydim
            cols = xextent/self.dsb_xycell.value()
            rows = yextent/self.dsb_xycell.value()

            self.dsb_utlx.setValue(utlx)
            self.dsb_utly.setValue(utly)
            self.dsb_xextent.setValue(xextent)
            self.dsb_yextent.setValue(yextent)
            self.sb_cols.setValue(cols)
            self.sb_rows.setValue(rows)

    def init(self):
        """ Initialize parameters """
    # Extent Parameters
        self.dsb_utlx.setValue(0.0)
        self.dsb_utly.setValue(0.0)
        self.dsb_utlz.setValue(0.0)
        self.dsb_xextent.setValue(self.lmod1.numx*self.lmod1.dxy)
        self.dsb_yextent.setValue(self.lmod1.numy*self.lmod1.dxy)
        self.dsb_zextent.setValue(self.lmod1.numz*self.lmod1.d_z)
        self.dsb_xycell.setValue(self.lmod1.dxy)
        self.dsb_zcell.setValue(self.lmod1.d_z)
        self.sb_cols.setValue(self.lmod1.numx)
        self.sb_rows.setValue(self.lmod1.numy)
        self.sb_layers.setValue(self.lmod1.numz)

    def upd_layers(self):
        """ Update the layers """
        xextent = self.dsb_xextent.value()
        yextent = self.dsb_yextent.value()
        zextent = self.dsb_zextent.value()
        dxy = self.dsb_xycell.value()
        d_z = self.dsb_zcell.value()

        numx = int(xextent/dxy)
        numy = int(yextent/dxy)
        numz = int(zextent/d_z)
        self.sb_cols.setValue(numx)
        self.sb_rows.setValue(numy)
        self.sb_layers.setValue(numz)

    def update_combos(self):
        """ Update the combos """

        gkeys = list(self.parent.inraster.keys())
        if 'Calculated Gravity' in gkeys:
            gkeys.remove('Calculated Gravity')
        if 'Calculated Magnetics' in gkeys:
            gkeys.remove('Calculated Magnetics')
        gkeys = ['None'] + gkeys

        if len(gkeys) > 1:
            self.combo_dtm.clear()
            self.combo_dtm.addItems(gkeys)
            self.combo_dtm.setCurrentIndex(0)
            self.combo_mag.clear()
            self.combo_mag.addItems(gkeys)
            self.combo_mag.setCurrentIndex(0)
            self.combo_grv.clear()
            self.combo_grv.addItems(gkeys)
            self.combo_grv.setCurrentIndex(0)
            self.combo_reggrv.clear()
            self.combo_reggrv.addItems(gkeys)
            self.combo_reggrv.setCurrentIndex(0)
            self.combo_dataset.clear()
            self.combo_dataset.addItems(gkeys)
            self.combo_dataset.setCurrentIndex(0)

            lkeys = list(self.lmod1.griddata.keys())
            if 'DTM Dataset' in lkeys:
                tmp = self.lmod1.griddata['DTM Dataset'].bandid
                self.combo_dtm.setCurrentIndex(gkeys.index(tmp))

            if 'Magnetic Dataset' in lkeys:
                tmp = self.lmod1.griddata['Magnetic Dataset'].bandid
                self.combo_mag.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Dataset' in lkeys:
                tmp = self.lmod1.griddata['Gravity Dataset'].bandid
                self.combo_grv.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Regional' in lkeys:
                tmp = self.lmod1.griddata['Gravity Regional'].bandid
                self.combo_reggrv.setCurrentIndex(gkeys.index(tmp))

    def update_vals(self):
        """ Updates the visible model extent parameters"""
        utlx = self.lmod1.xrange[0]
        utly = self.lmod1.yrange[1]
        utlz = self.lmod1.zrange[1]
        xextent = self.lmod1.xrange[1]-self.lmod1.xrange[0]
        yextent = self.lmod1.yrange[1]-self.lmod1.yrange[0]
        zextent = self.lmod1.zrange[1]-self.lmod1.zrange[0]

        self.dsb_utlx.setValue(utlx)
        self.dsb_utly.setValue(utly)
        self.dsb_xextent.setValue(xextent)
        self.dsb_yextent.setValue(yextent)
        self.dsb_xycell.setValue(self.lmod1.dxy)
        self.sb_cols.setValue(self.lmod1.numx)
        self.sb_rows.setValue(self.lmod1.numy)
        self.sb_layers.setValue(self.lmod1.numz)
        self.dsb_utlz.setValue(utlz)
        self.dsb_zextent.setValue(zextent)
        self.dsb_zcell.setValue(self.lmod1.d_z)

    def xycell(self, dxy):
        """ Function to adjust XY dimensions when cell size changes"""
        xextent = self.dsb_xextent.value()
        yextent = self.dsb_yextent.value()

        if dxy > xextent:
            dxy = xextent
            self.dsb_xycell.setValue(dxy)

        if dxy > yextent:
            dxy = yextent
            self.dsb_xycell.setValue(dxy)

        self.upd_layers()

    def zcell(self, d_z):
        """ Function to adjust Z dimension when cell size changes"""
        zextent = self.dsb_zextent.value()

        if d_z > zextent:
            d_z = zextent
            self.dsb_zcell.setValue(d_z)

        self.upd_layers()

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.lmod1 = self.parent.lmod1
        self.update_vals()
        self.update_combos()
