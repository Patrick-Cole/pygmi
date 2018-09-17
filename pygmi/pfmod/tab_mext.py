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

from PyQt5 import QtWidgets, QtCore
import numpy as np
import scipy.interpolate as si


class MextDisplay(QtWidgets.QDialog):
    """ MextDisplay - Widget class to call the main interface """
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.parent = parent
        self.lmod1 = parent.lmod1  # actual model
        self.lmod2 = parent.lmod2
        self.showtext = parent.showtext
        self.pbars = self.parent.pbars

        self.combo_model = QtWidgets.QComboBox()
        self.combo_regional = QtWidgets.QComboBox()
        self.cb_regional = QtWidgets.QCheckBox("Apply Regional Model")
        self.combo_dtm = QtWidgets.QComboBox()
        self.combo_mag = QtWidgets.QComboBox()
        self.combo_grv = QtWidgets.QComboBox()
        self.combo_reggrv = QtWidgets.QComboBox()
        self.combo_dataset = QtWidgets.QComboBox()
        self.dsb_utlx = QtWidgets.QDoubleSpinBox()
        self.dsb_utly = QtWidgets.QDoubleSpinBox()
        self.dsb_utlz = QtWidgets.QDoubleSpinBox()
        self.dsb_xextent = QtWidgets.QDoubleSpinBox()
        self.dsb_yextent = QtWidgets.QDoubleSpinBox()
        self.dsb_zextent = QtWidgets.QDoubleSpinBox()
        self.dsb_xycell = QtWidgets.QDoubleSpinBox()
        self.dsb_zcell = QtWidgets.QDoubleSpinBox()
        self.sb_cols = QtWidgets.QSpinBox()
        self.sb_rows = QtWidgets.QSpinBox()
        self.sb_layers = QtWidgets.QSpinBox()

        self.setupui()
        self.init()

    def setupui(self):
        """ Setup UI """
        self.setWindowTitle("Model Extent Parameters")

        verticallayout = QtWidgets.QVBoxLayout(self)

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Preferred)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

# Current Models Groupbox
        gb_model = QtWidgets.QGroupBox("Current Models")
        gl_model = QtWidgets.QGridLayout(gb_model)

        lbl1_model = QtWidgets.QLabel("Current Model:")
        lbl2_model = QtWidgets.QLabel("Regional Model:")

        self.combo_model.addItems(['None'])
        self.combo_regional.addItems(['None'])

        self.cb_regional.setSizePolicy(sizepolicy)

        gl_model.addWidget(lbl1_model, 0, 0, 1, 1)
        gl_model.addWidget(lbl2_model, 1, 0, 1, 1)
        gl_model.addWidget(self.combo_model, 0, 1, 1, 1)
        gl_model.addWidget(self.combo_regional, 1, 1, 1, 1)
        gl_model.addWidget(self.cb_regional, 0, 2, 2, 1)

# Data Information Groupbox
        gb_data_info = QtWidgets.QGroupBox("Dataset Information")
        gl_data_info = QtWidgets.QGridLayout(gb_data_info)

        self.combo_mag.addItems(['None'])
        self.combo_grv.addItems(['None'])
        self.combo_reggrv.addItems(['None'])
        self.combo_dtm.addItems(['None'])

        gl_data_info.setColumnStretch(0, 1)
        gl_data_info.setColumnStretch(1, 1)
        gl_data_info.setColumnStretch(2, 1)

        lbl1_data_info = QtWidgets.QLabel("DTM Dataset:")
        lbl2_data_info = QtWidgets.QLabel("Magnetic Dataset:")
        lbl3_data_info = QtWidgets.QLabel("Gravity Dataset:")
        lbl4_data_info = QtWidgets.QLabel("Gravity Regional Dataset:")

        gl_data_info.addWidget(lbl1_data_info, 0, 0, 1, 1)
        gl_data_info.addWidget(lbl2_data_info, 1, 0, 1, 1)
        gl_data_info.addWidget(lbl3_data_info, 2, 0, 1, 1)
        gl_data_info.addWidget(lbl4_data_info, 3, 0, 1, 1)
        gl_data_info.addWidget(self.combo_dtm, 0, 1, 1, 1)
        gl_data_info.addWidget(self.combo_mag, 1, 1, 1, 1)
        gl_data_info.addWidget(self.combo_grv, 2, 1, 1, 1)
        gl_data_info.addWidget(self.combo_reggrv, 3, 1, 1, 1)

# Data Extents Groupbox
        gb_extent = QtWidgets.QGroupBox("Model Extent Properties")
        gl_extent = QtWidgets.QGridLayout(gb_extent)

        self.combo_dataset.addItems(['None'])

        lbl0 = QtWidgets.QLabel("Get Study Area from following Dataset:")
        lbl3 = QtWidgets.QLabel("Upper Top Left X Coordinate:")
        lbl4 = QtWidgets.QLabel("Upper Top Left Y Coordinate:")
        lbl1 = QtWidgets.QLabel("Upper Top Left Z Coordinate (from DTM):")
        lbl8 = QtWidgets.QLabel("Total X Extent:")
        lbl9 = QtWidgets.QLabel("Total Y Extent:")
        lbl10 = QtWidgets.QLabel("Total Z Extent (Depth):")
        lbl5 = QtWidgets.QLabel("X and Y Cell Size:")
        lbl6 = QtWidgets.QLabel("Z Cell Size:")

        self.dsb_utlx.setMinimum(-999999999.0)
        self.dsb_utlx.setMaximum(999999999.0)
        self.dsb_utly.setMinimum(-999999999.0)
        self.dsb_utly.setMaximum(999999999.0)
        self.dsb_utlz.setMinimum(-999999999.0)
        self.dsb_utlz.setMaximum(999999999.0)
        self.dsb_xextent.setEnabled(True)
        self.dsb_xextent.setMinimum(0.1)
        self.dsb_xextent.setMaximum(2000000000.0)
        self.dsb_yextent.setEnabled(True)
        self.dsb_yextent.setMinimum(0.1)
        self.dsb_yextent.setMaximum(2000000000.0)
        self.dsb_zextent.setEnabled(True)
        self.dsb_zextent.setMinimum(0.1)
        self.dsb_zextent.setMaximum(2000000000.0)
        self.dsb_xycell.setEnabled(True)
        self.dsb_xycell.setMinimum(0.1)
        self.dsb_xycell.setMaximum(1000000.0)
        self.dsb_zcell.setEnabled(True)
        self.dsb_zcell.setDecimals(2)
        self.dsb_zcell.setMinimum(0.1)
        self.dsb_zcell.setMaximum(1000000.0)
        self.dsb_zcell.setSingleStep(1.0)
        self.sb_cols.setEnabled(False)
        self.sb_cols.setMinimum(1)
        self.sb_cols.setMaximum(1000000)
        self.sb_cols.setPrefix('Columns (X): ')
        self.sb_rows.setEnabled(False)
        self.sb_rows.setMinimum(1)
        self.sb_rows.setMaximum(1000000)
        self.sb_rows.setPrefix('Rows (Y): ')
        self.sb_layers.setEnabled(False)
        self.sb_layers.setMinimum(1)
        self.sb_layers.setMaximum(1000000)
        self.sb_layers.setPrefix('Layers (Z): ')

        gl_extent.addWidget(lbl0, 0, 0, 1, 1)
        gl_extent.addWidget(lbl3, 1, 0, 1, 1)
        gl_extent.addWidget(lbl4, 2, 0, 1, 1)
        gl_extent.addWidget(lbl1, 3, 0, 1, 1)
        gl_extent.addWidget(lbl8, 4, 0, 1, 1)
        gl_extent.addWidget(lbl9, 5, 0, 1, 1)
        gl_extent.addWidget(lbl10, 6, 0, 1, 1)
        gl_extent.addWidget(lbl5, 7, 0, 1, 1)
        gl_extent.addWidget(lbl6, 8, 0, 1, 1)
        gl_extent.addWidget(self.combo_dataset, 0, 1, 1, 1)
        gl_extent.addWidget(self.dsb_utlx, 1, 1, 1, 1)
        gl_extent.addWidget(self.dsb_utly, 2, 1, 1, 1)
        gl_extent.addWidget(self.dsb_utlz, 3, 1, 1, 1)
        gl_extent.addWidget(self.dsb_xextent, 4, 1, 1, 1)
        gl_extent.addWidget(self.dsb_yextent, 5, 1, 1, 1)
        gl_extent.addWidget(self.dsb_zextent, 6, 1, 1, 1)
        gl_extent.addWidget(self.dsb_xycell, 7, 1, 1, 1)
        gl_extent.addWidget(self.dsb_zcell, 8, 1, 1, 1)
        gl_extent.addWidget(self.sb_cols, 1, 2, 1, 1)
        gl_extent.addWidget(self.sb_rows, 2, 2, 1, 1)
        gl_extent.addWidget(self.sb_layers, 3, 2, 1, 1)

# Assign to main layout

        verticallayout.addWidget(gb_model)
        verticallayout.addWidget(gb_data_info)
        verticallayout.addWidget(gb_extent)
        verticallayout.addWidget(buttonbox)

# Link functions
        self.dsb_xycell.valueChanged.connect(self.xycell)
        self.dsb_zcell.valueChanged.connect(self.zcell)
        self.dsb_utlx.valueChanged.connect(self.upd_layers)
        self.dsb_utly.valueChanged.connect(self.upd_layers)
        self.dsb_utlz.valueChanged.connect(self.upd_layers)
        self.dsb_xextent.valueChanged.connect(self.upd_layers)
        self.dsb_yextent.valueChanged.connect(self.upd_layers)
        self.dsb_zextent.valueChanged.connect(self.upd_layers)
        self.combo_dataset.currentIndexChanged.connect(self.get_area)
        self.combo_dtm.currentIndexChanged.connect(self.choose_dtm)
        self.combo_model.currentIndexChanged.connect(self.choose_model)
        self.combo_regional.currentIndexChanged.connect(self.choose_regional)

        buttonbox.accepted.connect(self.apply_changes)
        buttonbox.rejected.connect(self.reject)

    def apply_changes(self):
        """ Update when changing from this tab """
        self.pbars.resetall()

        if self.cb_regional.isChecked():
            self.apply_regional()

        self.showtext('Working...')

        self.choose_combo(self.combo_dtm, 'DTM Dataset')
        self.choose_combo(self.combo_mag, 'Magnetic Dataset')
        self.choose_combo(self.combo_grv, 'Gravity Dataset')
        self.choose_combo(self.combo_reggrv, 'Gravity Regional')
        self.choose_combo(self.combo_dataset, 'Study Area Dataset')

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
        # This line is to avoid duplicates since study area and dtm are often
        # the same dataset
        tmp = [i for i in set(self.lmod1.griddata.values())]
        self.parent.outdata['Raster'] = tmp
        self.showtext('Changes applied.')

        self.accept()

    def apply_regional(self):
        """ Applies the regional model to the current model """
        self.lmod1.lith_index[self.lmod1.lith_index > 899] = 0

        ctxt = str(self.combo_regional.currentText())
        if ctxt == 'None':
            self.showtext('No regional model selected!')
            return

        self.pbars.resetall(self.lmod1.numx)
        for i in range(self.lmod1.numx):
            self.pbars.incr()
            for j in range(self.lmod1.numy):
                for k in range(self.lmod1.numz):
                    x = self.lmod1.xrange[0]+(i+0.5)*self.lmod1.dxy
                    y = self.lmod1.yrange[0]+(j+0.5)*self.lmod1.dxy
                    z = self.lmod1.zrange[-1]-(k+0.5)*self.lmod1.d_z
                    ii = int((x-self.lmod2.xrange[0])/self.lmod2.dxy)
                    jj = int((y-self.lmod2.yrange[0])/self.lmod2.dxy)
                    kk = int((self.lmod2.zrange[-1]-z)/self.lmod2.d_z)
                    tmp = self.lmod2.lith_index[ii, jj, kk]
                    if tmp > 0 and ii > -1 and jj > -1 and kk > -1:
                        self.lmod1.lith_index[i, j, k] = 900+tmp

        for i in np.unique(self.lmod1.lith_index):
            if i > 900:
                self.lmod1.mlut[i] = self.lmod2.mlut[i-900]

        self.showtext('Regional model applied.')

    def choose_combo(self, combo, dtxt):
        """ Combo box choice routine """
        ctxt = str(combo.currentText())
        if ctxt != 'None' and ctxt != '':
            self.lmod1.griddata[dtxt] = self.parent.inraster[ctxt]
        elif ctxt == 'None' and dtxt in self.lmod1.griddata:
            self.lmod1.griddata.pop(dtxt)

    def choose_dtm(self):
        """ Combo box to choose current dataset """
        ctxt = str(self.combo_dtm.currentText())
        if ctxt != 'None' and ctxt != '':
            curgrid = self.parent.inraster[ctxt]

            self.dsb_utlz.setValue(curgrid.data.max())
            zextent = curgrid.data.ptp()+self.dsb_zcell.value()
            if zextent > self.dsb_zextent.value():
                self.dsb_zextent.setValue(zextent)

            self.upd_layers()

    def choose_model(self):
        """ Choose which model file to use """
        ctxt = str(self.combo_model.currentText())
        if ctxt == 'None' or 'Model3D' not in self.parent.indata:
            return
        for i in self.parent.indata['Model3D']:
            if i.name == ctxt:
                self.lmod1 = i
                self.parent.lmod1 = i
                self.update_vals()
                self.update_combos()

    def choose_regional(self):
        """ Choose which regional model file to use """
        ctxt = str(self.combo_regional.currentText())
        if ctxt == 'None' or 'Model3D' not in self.parent.indata:
            return
        for i in self.parent.indata['Model3D']:
            if i.name == ctxt:
                self.lmod2 = i
                self.parent.lmod2 = i

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
        outg.mask = np.ma.getmaskarray(gdata)

        return outg

    def get_area(self):
        """ Get area """
        ctxt = str(self.combo_dataset.currentText())
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

    def update_model_combos(self):
        """ Updates the model combos """
        modnames = ['None']
        if 'Model3D' in self.parent.indata:
            for i in self.parent.indata['Model3D']:
                modnames.append(i.name)

        self.combo_model.currentIndexChanged.disconnect()
        self.combo_regional.currentIndexChanged.disconnect()

        self.combo_model.clear()
        self.combo_model.addItems(modnames)
        self.combo_model.setCurrentIndex(0)
        self.combo_regional.clear()
        self.combo_regional.addItems(modnames)
        self.combo_regional.setCurrentIndex(0)

        if len(modnames) >= 2:
            self.combo_model.setCurrentIndex(1)
        if len(modnames) > 2:
            self.combo_regional.setCurrentIndex(2)

        self.combo_model.currentIndexChanged.connect(self.choose_model)
        self.combo_regional.currentIndexChanged.connect(self.choose_regional)

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
                tmp = self.lmod1.griddata['DTM Dataset'].dataid
                self.combo_dtm.setCurrentIndex(gkeys.index(tmp))

            if 'Magnetic Dataset' in lkeys:
                tmp = self.lmod1.griddata['Magnetic Dataset'].dataid
                self.combo_mag.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Dataset' in lkeys:
                tmp = self.lmod1.griddata['Gravity Dataset'].dataid
                self.combo_grv.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Regional' in lkeys:
                tmp = self.lmod1.griddata['Gravity Regional'].dataid
                self.combo_reggrv.setCurrentIndex(gkeys.index(tmp))

            if 'Study Area Dataset' in lkeys:
                tmp = self.lmod1.griddata['Study Area Dataset'].dataid
                self.combo_dataset.setCurrentIndex(gkeys.index(tmp))

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
        self.update_model_combos()
        self.choose_model()
        self.choose_regional()
        self.update_vals()
        self.update_combos()

        self.exec_()

        self.parent.tab_change()
