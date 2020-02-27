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
"""Model Extents Display Routines."""

from PyQt5 import QtWidgets, QtCore
import numpy as np
import scipy.interpolate as si
import pygmi.menu_default as menu_default
import pygmi.misc as pmisc


class MextDisplay(QtWidgets.QDialog):
    """MextDisplay - Widget class to call the main interface."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.lmod1 = parent.lmod1  # actual model
        self.showtext = parent.showtext
        self.pbar = pmisc.ProgressBar()

        self.combo_model = QtWidgets.QComboBox()
        self.combo_other = QtWidgets.QComboBox()
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
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.setWindowTitle('Model Extent Parameters')
        helpdocs = menu_default.HelpButton('pygmi.pfmod.mext')

        verticallayout = QtWidgets.QVBoxLayout(self)
        hlayout = QtWidgets.QHBoxLayout()

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Preferred)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

# Current Models Groupbox
        h_model = QtWidgets.QHBoxLayout()

#        gb_model = QtWidgets.QGroupBox('Current Models')
#        gl_model = QtWidgets.QGridLayout(gb_model)

        lbl1_model = QtWidgets.QLabel('Current Model:')

        self.combo_model.addItems(['None'])
        self.combo_model.setSizePolicy(sizepolicy)

#        gl_model.addWidget(lbl1_model, 0, 0, 1, 1)
#        gl_model.addWidget(self.combo_model, 0, 1, 1, 1)

        h_model.addWidget(lbl1_model)
        h_model.addWidget(self.combo_model)

# Data Information Groupbox
        gb_data_info = QtWidgets.QGroupBox('Dataset Information')
        gl_data_info = QtWidgets.QGridLayout(gb_data_info)

        self.combo_mag.addItems(['None'])
        self.combo_grv.addItems(['None'])
        self.combo_reggrv.addItems(['None'])
        self.combo_dtm.addItems(['None'])
        self.combo_other.addItems(['None'])

        gl_data_info.setColumnStretch(0, 1)
        gl_data_info.setColumnStretch(1, 1)
        gl_data_info.setColumnStretch(2, 1)

        lbl1_data_info = QtWidgets.QLabel('DTM Dataset:')
        lbl2_data_info = QtWidgets.QLabel('Magnetic Dataset:')
        lbl3_data_info = QtWidgets.QLabel('Gravity Dataset:')
        lbl4_data_info = QtWidgets.QLabel('Gravity Regional Dataset:')
        lbl5_data_info = QtWidgets.QLabel('Other:')

        gl_data_info.addWidget(lbl1_data_info, 0, 0, 1, 1)
        gl_data_info.addWidget(lbl2_data_info, 1, 0, 1, 1)
        gl_data_info.addWidget(lbl3_data_info, 2, 0, 1, 1)
        gl_data_info.addWidget(lbl4_data_info, 3, 0, 1, 1)
        gl_data_info.addWidget(lbl5_data_info, 4, 0, 1, 1)
        gl_data_info.addWidget(self.combo_dtm, 0, 1, 1, 1)
        gl_data_info.addWidget(self.combo_mag, 1, 1, 1, 1)
        gl_data_info.addWidget(self.combo_grv, 2, 1, 1, 1)
        gl_data_info.addWidget(self.combo_reggrv, 3, 1, 1, 1)
        gl_data_info.addWidget(self.combo_other, 4, 1, 1, 1)

# Data Extents Groupbox
        gb_extent = QtWidgets.QGroupBox('Model Extent Properties')
        gl_extent = QtWidgets.QGridLayout(gb_extent)

        self.combo_dataset.addItems(['None'])

        lbl0 = QtWidgets.QLabel('Get Study Area from following Dataset:')
        lbl3 = QtWidgets.QLabel('Upper Top Left X Coordinate:')
        lbl4 = QtWidgets.QLabel('Upper Top Left Y Coordinate:')
        lbl1 = QtWidgets.QLabel('Upper Top Left Z Coordinate (from DTM):')
        lbl8 = QtWidgets.QLabel('Total X Extent:')
        lbl9 = QtWidgets.QLabel('Total Y Extent:')
        lbl10 = QtWidgets.QLabel('Total Z Extent (Depth):')
        lbl5 = QtWidgets.QLabel('X and Y Cell Size:')
        lbl6 = QtWidgets.QLabel('Z Cell Size:')

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

        hlayout.addWidget(helpdocs)
        hlayout.addWidget(self.pbar)
        hlayout.addWidget(buttonbox)

# Assign to main layout

#        verticallayout.addWidget(gb_model)
        verticallayout.addLayout(h_model)
        verticallayout.addWidget(gb_data_info)
        verticallayout.addWidget(gb_extent)
        verticallayout.addLayout(hlayout)

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

        buttonbox.accepted.connect(self.apply_changes)
        buttonbox.rejected.connect(self.reject)

    def apply_changes(self):
        """
        Apply changes.

        Returns
        -------
        None.

        """
        self.showtext('Working...')

        self.choose_combo(self.combo_dtm, 'DTM Dataset')
        self.choose_combo(self.combo_mag, 'Magnetic Dataset')
        self.choose_combo(self.combo_grv, 'Gravity Dataset')
        self.choose_combo(self.combo_reggrv, 'Gravity Regional')
        self.choose_combo(self.combo_dataset, 'Study Area Dataset')
        self.choose_combo(self.combo_other, 'Other')

        cols = self.sb_cols.value()
        rows = self.sb_rows.value()
        layers = self.sb_layers.value()
        utlx = self.dsb_utlx.value()
        utly = self.dsb_utly.value()
        utlz = self.dsb_utlz.value()
        dxy = self.dsb_xycell.value()
        d_z = self.dsb_zcell.value()

        self.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z,
                          pbar=self.pbar)

        self.update_vals()

        # This line is to avoid duplicates since study area and dtm are often
        # the same dataset
        tmp = [i for i in set(self.lmod1.griddata.values())]
        self.parent.outdata['Raster'] = tmp
        self.showtext('Changes applied.')

        self.accept()

    def choose_combo(self, combo, dtxt):
        """
        Combo box choice routine.

        Parameters
        ----------
        combo : QComboBox
            Combo box.
        dtxt : str
            Text to describe new raster data entry.

        Returns
        -------
        None.

        """
        ctxt = str(combo.currentText())
        if ctxt not in ('None', ''):
            self.lmod1.griddata[dtxt] = self.parent.inraster[ctxt]
        elif ctxt == 'None' and dtxt in self.lmod1.griddata:
            self.lmod1.griddata.pop(dtxt)

    def choose_dtm(self):
        """
        Combo box to choose current DTM.

        Returns
        -------
        None.

        """
        ctxt = str(self.combo_dtm.currentText())
        if ctxt not in ('None', ''):
            curgrid = self.parent.inraster[ctxt]

            self.dsb_utlz.setValue(curgrid.data.max())
            zextent = curgrid.data.ptp()+self.dsb_zcell.value()
            if zextent > self.dsb_zextent.value():
                self.dsb_zextent.setValue(zextent)

            self.upd_layers()

    def choose_model(self):
        """
        Choose model file.

        Returns
        -------
        None.

        """
        ctxt = str(self.combo_model.currentText())
        if ctxt == 'None' or 'Model3D' not in self.parent.indata:
            return
        for i in self.parent.indata['Model3D']:
            if i.name == ctxt:
                self.lmod1 = i
                self.parent.lmod1 = i
                self.update_vals()
                self.update_combos()

    def extgrid(self, gdata):
        """
        Extrapolates the grid to get rid of nulls.

        Uses a masked grid.

        Parameters
        ----------
        gdata : numpy array
            Raster dataset.

        Returns
        -------
        numpy masked array
            Output dataset.

        """
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
        """
        Get current grid extents and parameters.

        Returns
        -------
        None.

        """
        ctxt = str(self.combo_dataset.currentText())
        if ctxt not in ('None', u''):
            curgrid = self.parent.inraster[ctxt]

            crows, ccols = curgrid.data.shape

            utlx = curgrid.extent[0]
            utly = curgrid.extent[-1]
            xextent = ccols*curgrid.xdim
            yextent = crows*curgrid.ydim
            cols = xextent/self.dsb_xycell.value()
            rows = yextent/self.dsb_xycell.value()

            self.dsb_utlx.setValue(utlx)
            self.dsb_utly.setValue(utly)
            self.dsb_xextent.setValue(xextent)
            self.dsb_yextent.setValue(yextent)
            self.sb_cols.setValue(cols)
            self.sb_rows.setValue(rows)

    def init(self):
        """
        Initialise parameters.

        Returns
        -------
        None.

        """
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
        """
        Update layers.

        Returns
        -------
        None.

        """
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
        """
        Update model combos.

        Returns
        -------
        None.

        """
        modnames = ['None']
        if 'Model3D' in self.parent.indata:
            for i in self.parent.indata['Model3D']:
                modnames.append(i.name)

        self.combo_model.currentIndexChanged.disconnect()

        self.combo_model.clear()
        self.combo_model.addItems(modnames)
        self.combo_model.setCurrentIndex(0)

        if len(modnames) >= 2:
            self.combo_model.setCurrentIndex(1)

        self.combo_model.currentIndexChanged.connect(self.choose_model)

    def update_combos(self):
        """
        Update combos.

        Returns
        -------
        None.

        """
        gkeys = list(self.parent.inraster.keys())
        if 'Calculated Gravity' in gkeys:
            gkeys.remove('Calculated Gravity')
        if 'Calculated Magnetics' in gkeys:
            gkeys.remove('Calculated Magnetics')
        gkeys = ['None'] + gkeys

        if len(gkeys) > 1:
            self.combo_other.clear()
            self.combo_other.addItems(gkeys)
            self.combo_other.setCurrentIndex(0)
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

            if 'Other' in lkeys:
                tmp = self.lmod1.griddata['Other'].dataid
                self.combo_other.setCurrentIndex(gkeys.index(tmp))

    def update_vals(self):
        """
        Update the visible model extent parameters.

        Returns
        -------
        None.

        """
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
        """
        Adjust XY dimensions when cell size changes.

        Parameters
        ----------
        dxy : float
            Cell dimension.

        Returns
        -------
        None.

        """
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
        """
        Adjust Z dimension when cell size changes.

        Parameters
        ----------
        d_z : float
            Layer thinkcness.

        Returns
        -------
        None.

        """
        zextent = self.dsb_zextent.value()

        if d_z > zextent:
            d_z = zextent
            self.dsb_zcell.setValue(d_z)

        self.upd_layers()

    def tab_activate(self):
        """
        Entry point.

        Returns
        -------
        None.

        """
        self.update_model_combos()
        self.choose_model()
#        self.choose_regional()
        self.update_vals()
        self.update_combos()

        self.exec_()

        # The next line is necessary to update any dataset changes.
        self.parent.profile.tab_activate()  # Link to tab_prof
