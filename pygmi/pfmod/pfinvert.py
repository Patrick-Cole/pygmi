# -----------------------------------------------------------------------------
# Name:        pfinvert.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2022 Council for Geoscience
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
"""
A routine interfacing wito the SimPEG inversion library for magnetic
inversion.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import scipy.interpolate as si
from discretize import TensorMesh
from SimPEG.potential_fields import magnetics
from SimPEG.utils import surface2ind_topo,  model_builder
from SimPEG import (maps, data, inverse_problem, data_misfit,
                    regularization, optimization, directives,
                    inversion)
import matplotlib.pyplot as plt
import sklearn.cluster as skc
from sklearn.metrics import calinski_harabasz_score
import sklearn.preprocessing as skp

from pygmi import menu_default
from pygmi.misc import ProgressBarText, ProgressBar
from pygmi.pfmod.datatypes import LithModel
from pygmi.raster.dataprep import lstack


class MagInvert(QtWidgets.QDialog):
    """MextDisplay - Widget class to call the main interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.lmod1 = LithModel()
        self.pbar = ProgressBar()
        self.inraster = {}
        self.indata = {}
        self.outdata = {}

        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter
        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

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

        self.dsb_mht = QtWidgets.QDoubleSpinBox()
        self.dsb_hdec = QtWidgets.QDoubleSpinBox()
        self.dsb_hint = QtWidgets.QDoubleSpinBox()
        self.dsb_hinc = QtWidgets.QDoubleSpinBox()
        # self.dsb_ght = QtWidgets.QDoubleSpinBox()
        # self.dsb_gregional = QtWidgets.QDoubleSpinBox()

        self.setupui()
        self.init()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.setWindowTitle('Inverse Modelling Parameters')
        helpdocs = menu_default.HelpButton('pygmi.pfmod.mext')

        verticallayout = QtWidgets.QVBoxLayout(self)
        hlayout = QtWidgets.QHBoxLayout()

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Preferred)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        # buttonbox.button(buttonbox.Cancel).setText('No changes')
        buttonbox.button(buttonbox.Ok).setText('Run Inversion')

# Current Models Groupbox
        h_model = QtWidgets.QHBoxLayout()

        lbl1_model = QtWidgets.QLabel('Current Model:')

        self.combo_model.addItems(['None'])
        self.combo_model.setSizePolicy(sizepolicy)

        h_model.addWidget(lbl1_model)
        h_model.addWidget(self.combo_model)


# General Properties
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

        gb_gen_prop = QtWidgets.QGroupBox('General Properties')
        gl_gen_prop = QtWidgets.QGridLayout(gb_gen_prop)

        # label_1 = QtWidgets.QLabel('Gravity Regional (mGal)')
        # label_2 = QtWidgets.QLabel('Height of observation - Gravity')
        label_3 = QtWidgets.QLabel('Height of observation - Magnetic')
        label_4 = QtWidgets.QLabel('Magnetic Field Intensity (nT)')
        label_5 = QtWidgets.QLabel('Magnetic Inclination')
        label_6 = QtWidgets.QLabel('Magnetic Declination')

        # gl_gen_prop.addWidget(label_1, 0, 0, 1, 1)
        # gl_gen_prop.addWidget(self.dsb_gregional, 0, 1, 1, 1)
        # gl_gen_prop.addWidget(label_2, 2, 0, 1, 1)
        # gl_gen_prop.addWidget(self.dsb_ght, 2, 1, 1, 1)
        gl_gen_prop.addWidget(label_3, 3, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_mht, 3, 1, 1, 1)
        gl_gen_prop.addWidget(label_4, 4, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hint, 4, 1, 1, 1)
        gl_gen_prop.addWidget(label_5, 5, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hinc, 5, 1, 1, 1)
        gl_gen_prop.addWidget(label_6, 6, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hdec, 6, 1, 1, 1)

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

        gl_data_info.addWidget(QtWidgets.QLabel('DTM Dataset:'), 0, 0, 1, 1)
        gl_data_info.addWidget(QtWidgets.QLabel('Magnetic Dataset:'),
                               1, 0, 1, 1)
        # gl_data_info.addWidget(QtWidgets.QLabel('Gravity Dataset:'),
        #                        2, 0, 1, 1)
        # gl_data_info.addWidget(QtWidgets.QLabel('Gravity Regional Dataset:'),
        #                        3, 0, 1, 1)
        # gl_data_info.addWidget(QtWidgets.QLabel('Other:'), 4, 0, 1, 1)
        gl_data_info.addWidget(self.combo_dtm, 0, 1, 1, 1)
        gl_data_info.addWidget(self.combo_mag, 1, 1, 1, 1)
        # gl_data_info.addWidget(self.combo_grv, 2, 1, 1, 1)
        # gl_data_info.addWidget(self.combo_reggrv, 3, 1, 1, 1)
        # gl_data_info.addWidget(self.combo_other, 4, 1, 1, 1)

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
        verticallayout.addLayout(h_model)
        verticallayout.addWidget(gb_data_info)
        verticallayout.addWidget(gb_gen_prop)
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
        self.showprocesslog('Working...')

        self.choose_combo(self.combo_dtm, 'DTM Dataset')
        self.choose_combo(self.combo_mag, 'Magnetic Dataset')
        # self.choose_combo(self.combo_grv, 'Gravity Dataset')
        # self.choose_combo(self.combo_reggrv, 'Gravity Regional')
        self.choose_combo(self.combo_dataset, 'Study Area Dataset')
        # self.choose_combo(self.combo_other, 'Other')

        cols = self.sb_cols.value()
        rows = self.sb_rows.value()
        layers = self.sb_layers.value()
        utlx = self.dsb_utlx.value()
        utly = self.dsb_utly.value()
        utlz = self.dsb_utlz.value()
        dxy = self.dsb_xycell.value()
        d_z = self.dsb_zcell.value()

        self.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z,
                          pbar=self.pbar, usedtm=True)

        self.update_vals()

        # This line is to avoid duplicates since study area and dtm are often
        # the same dataset
        # tmp = [i for i in set(self.lmod1.griddata.values())]
        tmp = list(set(self.lmod1.griddata.values()))
        self.outdata['Raster'] = tmp
        self.showprocesslog('Changes applied.')

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
            self.lmod1.griddata[dtxt] = self.inraster[ctxt]
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
            curgrid = self.inraster[ctxt]

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
        if ctxt == 'None' or 'Model3D' not in self.indata:
            return
        for i in self.indata['Model3D']:
            if i.name == ctxt:
                self.lmod1 = i
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
            curgrid = self.inraster[ctxt]

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
            self.sb_cols.setValue(int(cols))
            self.sb_rows.setValue(int(rows))

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
        if 'Model3D' in self.indata:
            for i in self.indata['Model3D']:
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
        self.combo_dataset.currentIndexChanged.disconnect()

        gkeys = list(self.inraster.keys())
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

        self.combo_dataset.currentIndexChanged.connect(self.get_area)

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
            Layer thickness.

        Returns
        -------
        None.

        """
        zextent = self.dsb_zextent.value()

        if d_z > zextent:
            d_z = zextent
            self.dsb_zcell.setValue(d_z)

        self.upd_layers()

    def settings(self, nodialog=False):
        """
        Entry point.

        Returns
        -------
        None.

        """

        datatmp = list(set(self.lmod1.griddata.values()))

        if 'Raster' not in self.indata:
            self.indata['Raster'] = datatmp

        self.inraster = {}
        for i in self.indata['Raster']:
            self.inraster[i.dataid] = i
        if 'Model3D' in self.indata:
            self.lmod1 = self.indata['Model3D'][0]

        self.update_model_combos()
        self.choose_model()
#        self.choose_regional()
        self.update_vals()

        self.choose_dtm()
        self.get_area()

        if nodialog is False:
            self.update_combos()

        tmp = self.exec_()

        if tmp != 1:
            return tmp

        self.acceptall()

    def acceptall(self):
        """
        Accept All.

        Returns
        -------
        None.

        """
        # dxy = self.lmod1.dxy

        # xxx = np.arange(self.lmod1.xrange[0], self.lmod1.xrange[1],
        #                 dxy)
        # yyy = np.arange(self.lmod1.yrange[0], self.lmod1.yrange[1],
        #                 dxy)

        # xy = np.meshgrid(xxx, yyy[::-1])
        # z = np.zeros_like(xy[0])+self.dsb_mht.value()

        # receiver_locations = np.transpose([xy[0].flatten(),
        #                                    xy[1].flatten(),
        #                                    z.flatten()])

        dat = [self.lmod1.griddata['Magnetic Dataset'],
               self.lmod1.griddata['DTM Dataset']]

        dat = lstack(dat, masterid=0)

        mag = dat[0]
        dtm = dat[1]

        plt.imshow(dat[0].data)
        plt.show()
        plt.imshow(dat[1].data)
        plt.show()

        dobs = mag.data.flatten()

        xmin, xmax, ymin, ymax = mag.extent
        xdim = mag.xdim
        ydim = mag.ydim

        xxx = np.arange(xmin, xmax, xdim) + xdim/2
        yyy = np.arange(ymin, ymax, ydim) + ydim/2

        xy = np.meshgrid(xxx, yyy[::-1])
        z = dtm.data + self.dsb_mht.value()

        receiver_locations = np.transpose([xy[0].flatten(),
                                           xy[1].flatten(),
                                           z.flatten()])

        # dtm = self.lmod1.griddata['DTM Dataset']
        # xrange = dtm.extent[:2]
        # yrange = dtm.extent[2:]

        # xxx = np.arange(xrange[0], xrange[1], dtm.xdim)
        # yyy = np.arange(yrange[0], yrange[1], dtm.ydim)

        # xy = np.meshgrid(xxx, yyy[::-1])
        # z = dtm.data.flatten()

        topo_xyz = np.transpose([xy[0].flatten(),
                                 xy[1].flatten(),
                                 dtm.data.flatten()])

        # dir_path = r'd:\Work\Programming\MagInv\magnetics'
        # topo_filename = dir_path + "\\magnetics_topo.txt"
        # data_filename = dir_path + "\\magnetics_data.obs"
        # topo_xyz1 = np.loadtxt(str(topo_filename))
        # dobs1a = np.loadtxt(str(data_filename))
        # receiver_locations1 = dobs1a[:, 0:3]
        # dobs1 = dobs1a[:, -1]

        # Assign Uncertainty
        maximum_anomaly = np.max(np.abs(dobs))
        std = 0.02 * maximum_anomaly * np.ones(len(dobs))

        # Defining the Survey
        components = ["tmi"]
        receiver_list = magnetics.receivers.Point(receiver_locations,
                                                  components=components)
        receiver_list = [receiver_list]

        inclination = self.dsb_hinc.value()
        declination = self.dsb_hdec.value()
        strength = self.dsb_hint.value()

        inducing_field = (strength, inclination, declination)
        source_field = magnetics.sources.SourceField(
            receiver_list=receiver_list, parameters=inducing_field)

        # Define the survey, data and tensor mesh
        survey = magnetics.survey.Survey(source_field)
        data_object = data.Data(survey, dobs=dobs, standard_deviation=std)
        dh = self.lmod1.d_z
        dhxy = self.lmod1.dxy

        hx = [(dhxy, 5, -1.3), (dhxy, self.lmod1.numx), (dhxy, 5, 1.3)]
        hy = [(dhxy, 5, -1.3), (dhxy, self.lmod1.numy), (dhxy, 5, 1.3)]
        hz = [(dh, 5, -1.3), (dh, self.lmod1.numz)]
        mesh = TensorMesh([hx, hy, hz], "CCN")

        # Starting/Reference Model and Mapping on Tensor Mesh
        background_susceptibility = 1e-4
        ind_active = surface2ind_topo(mesh, topo_xyz)
        nC = int(ind_active.sum())
        model_map = maps.IdentityMap(nP=nC)
        starting_model = background_susceptibility * np.ones(nC)

        # Define the Physics
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, model_type="scalar", chiMap=model_map,
            actInd=ind_active)

        # Define Inverse Problem
        dmis = data_misfit.L2DataMisfit(data=data_object,
                                        simulation=simulation)
        reg = regularization.Sparse(mesh, indActive=ind_active,
                                    mapping=model_map, mref=starting_model,
                                    gradientType="total", alpha_s=1, alpha_x=1,
                                    alpha_y=1, alpha_z=1)
        reg.norms = np.c_[0, 2, 2, 2]
        opt = optimization.ProjectedGNCG(maxIter=10, lower=0.0, upper=1.0,
                                         maxIterLS=20, maxIterCG=10,
                                         tolCG=1e-3)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Define Inversion Directives
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        update_IRLS = directives.Update_IRLS(f_min_change=1e-4,
                                             max_irls_iterations=30,
                                             coolEpsFact=1.5, beta_tol=1e-2)
        update_jacobi = directives.UpdatePreconditioner()
        target_misfit = directives.TargetMisfit(chifact=1)
        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        directives_list = [sensitivity_weights, starting_beta, save_iteration,
                           update_IRLS, update_jacobi]

        # Running the Inversion
        inv = inversion.BaseInversion(inv_prob, directives_list)
        recovered_model = inv.run(starting_model)

        ##############################################################
        # Recreate True Model
        # -------------------
        #

        background_susceptibility = 0.0001
        sphere_susceptibility = 0.01

        true_model = background_susceptibility * np.ones(nC)
        ind_sphere = model_builder.getIndicesSphere(
            np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers
        )
        ind_sphere = ind_sphere[ind_active]
        true_model[ind_sphere] = sphere_susceptibility

        self.tmp = [maps, mesh, ind_active, recovered_model, true_model]

        soln_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

        r2 = soln_map * recovered_model
        r2.shape = (mesh.nCz, mesh.nCy, mesh.nCx)

        r2 = r2[::-1]

        r3 = r2[:-5, 5:-5, 5:-5]

        r3 = np.ma.masked_invalid(r3)
        # breakpoint()
        # X = skp.StandardScaler().fit_transform(r3.flatten())

        X = r3.compressed().reshape(-1, 1)

        # cfit = skc.KMeans(n_clusters=i, tol=self.tol, max_iter=self.max_iter).fit(X)
        cfit = skc.KMeans(n_clusters=2).fit(X)

        zout = cfit.labels_

        r3[~r3.mask] = zout

        r4 = np.moveaxis(r3, [0,1,2], [2, 1, 0])
        r4 = r4.filled(-1)
        self.lmod1.lith_index = r4

        breakpoint()



def _testfn():
    """Main."""
    # Plot Recovered Model
    import sys
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mfile = r"d:\Workdata\MagInv\mag.tif"
    dfile = r"d:\Workdata\MagInv\dem.tif"

    mdat = get_raster(mfile)
    ddat = get_raster(dfile)

    mdat[0].dataid = 'mag'
    ddat[0].dataid = 'dem'

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    DM = MagInvert()

    # DM.lmod1.dxy = 10
    # DM.lmod1.d_z = 20
    DM.indata['Raster'] = mdat+ddat

    for i in DM.indata['Raster']:
        DM.inraster[i.dataid] = i
    if 'Model3D' in DM.indata:
        DM.lmod1 = DM.indata['Model3D'][0]

    cols = 40
    rows = 40
    layers = 15
    utlx = -80
    utly = 90
    utlz = 0
    dxy = 5.
    d_z = 5.

    DM.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z)

    DM.update_combos()
    DM.combo_dtm.setCurrentText('dem')
    DM.combo_mag.setCurrentText('mag')
    DM.combo_dataset.setCurrentText('mag')

    DM.choose_dtm()

    DM.dsb_mht.setValue(1)
    DM.dsb_hdec.setValue(0)
    DM.dsb_hint.setValue(50000)
    DM.dsb_hinc.setValue(90)

    DM.settings(True)

    maps, mesh, ind_active, recovered_model, true_model = DM.tmp

    # Plot True Model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plotSlice(
        plotting_map * true_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.nCy / 2),
        grid=True,
        clim=(np.min(true_model), np.max(true_model)),
        pcolorOpts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(true_model),
                                vmax=np.max(true_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.cm.viridis, format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()

    # Plot recovered model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plotSlice(
        plotting_map * recovered_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.nCy / 2),
        grid=True,
        clim=(np.min(recovered_model), np.max(recovered_model)),
        pcolorOpts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(recovered_model),
                                vmax=np.max(recovered_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.cm.viridis, format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()


    r2 = plotting_map * recovered_model
    r2.shape = (mesh.nCz, mesh.nCy, mesh.nCx)

    r2 = r2[::-1]

    plt.imshow(r2[:, r2.shape[1]//2])
    plt.show()

    breakpoint()


if __name__ == "__main__":
    _testfn()