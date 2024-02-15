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
"""Magnetic inversion using the SimPEG inversion library."""

import sys
from contextlib import redirect_stdout
import numpy as np
from PyQt5 import QtWidgets, QtCore
import scipy.interpolate as si
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from SimPEG.potential_fields import magnetics
from SimPEG.utils import model_builder
from SimPEG import (maps, data, inverse_problem, data_misfit,
                    regularization, optimization, directives,
                    inversion)
import sklearn.cluster as skc

from pygmi import menu_default
from pygmi.misc import BasicModule
from pygmi.pfmod.datatypes import LithModel
from pygmi.raster.dataprep import lstack
from pygmi.pfmod.grvmag3d import quick_model


class MagInvert(BasicModule):
    """MextDisplay - Widget class to call the main interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lmod1 = LithModel()
        self.lmod2 = LithModel()
        self.inraster = {}
        self.tmp = []

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_other = QtWidgets.QComboBox()
        self.cmb_dtm = QtWidgets.QComboBox()
        self.cmb_mag = QtWidgets.QComboBox()
        self.cmb_grv = QtWidgets.QComboBox()
        self.cmb_reggrv = QtWidgets.QComboBox()
        self.cmb_dataset = QtWidgets.QComboBox()
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
        self.sb_classes = QtWidgets.QSpinBox()

        self.dsb_mht = QtWidgets.QDoubleSpinBox()
        self.dsb_hdec = QtWidgets.QDoubleSpinBox()
        self.dsb_hint = QtWidgets.QDoubleSpinBox()
        self.dsb_hinc = QtWidgets.QDoubleSpinBox()

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
        helpdocs = menu_default.HelpButton('pygmi.pfmod.minv')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Preferred)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        buttonbox.button(buttonbox.Ok).setText('Run Inversion')

        # Current Models Groupbox
        hbl_model = QtWidgets.QHBoxLayout()

        lbl_model = QtWidgets.QLabel('Current Model:')

        self.cmb_model.addItems(['None'])
        self.cmb_model.setSizePolicy(sizepolicy)

        hbl_model.addWidget(lbl_model)
        hbl_model.addWidget(self.cmb_model)

        # General Properties
        self.dsb_mht.setMaximum(999999999.0)
        self.dsb_mht.setProperty('value', 30.0)
        self.dsb_hint.setMaximum(999999999.0)
        self.dsb_hint.setProperty('value', 28923.0)
        self.dsb_hinc.setMinimum(-90.0)
        self.dsb_hinc.setMaximum(90.0)
        self.dsb_hinc.setProperty('value', -61.22)
        self.dsb_hdec.setMinimum(-360.0)
        self.dsb_hdec.setMaximum(360.0)
        self.dsb_hdec.setProperty('value', -21.35)
        self.sb_classes.setProperty('value', 5)
        self.sb_classes.setMinimum(2)
        self.sb_classes.setMaximum(1000)

        gbox_gen_prop = QtWidgets.QGroupBox('General Properties')
        gl_gen_prop = QtWidgets.QGridLayout(gbox_gen_prop)

        lbl_3 = QtWidgets.QLabel('Height of observation - Magnetic')
        lbl_4 = QtWidgets.QLabel('Magnetic Field Intensity (nT)')
        lbl_5 = QtWidgets.QLabel('Magnetic Inclination')
        lbl_6 = QtWidgets.QLabel('Magnetic Declination')

        gl_gen_prop.addWidget(lbl_3, 3, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_mht, 3, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_4, 4, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hint, 4, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_5, 5, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hinc, 5, 1, 1, 1)
        gl_gen_prop.addWidget(lbl_6, 6, 0, 1, 1)
        gl_gen_prop.addWidget(self.dsb_hdec, 6, 1, 1, 1)

        # Data Information Groupbox
        gbox_data_info = QtWidgets.QGroupBox('Dataset Information')
        gl_data_info = QtWidgets.QGridLayout(gbox_data_info)

        self.cmb_mag.addItems(['None'])
        self.cmb_grv.addItems(['None'])
        self.cmb_reggrv.addItems(['None'])
        self.cmb_dtm.addItems(['None'])
        self.cmb_other.addItems(['None'])

        gl_data_info.setColumnStretch(0, 1)
        gl_data_info.setColumnStretch(1, 1)
        gl_data_info.setColumnStretch(2, 1)

        gl_data_info.addWidget(QtWidgets.QLabel('DTM Dataset:'), 0, 0, 1, 1)
        gl_data_info.addWidget(QtWidgets.QLabel('Magnetic Dataset:'),
                               1, 0, 1, 1)
        gl_data_info.addWidget(self.cmb_dtm, 0, 1, 1, 1)
        gl_data_info.addWidget(self.cmb_mag, 1, 1, 1, 1)

        # Data Extents Groupbox
        gbox_extent = QtWidgets.QGroupBox('Model Extent Properties')
        gl_extent = QtWidgets.QGridLayout(gbox_extent)

        self.cmb_dataset.addItems(['None'])

        lbl_0 = QtWidgets.QLabel('Get Study Area from following Dataset:')
        lbl_3 = QtWidgets.QLabel('Upper Top Left X Coordinate:')
        lbl_4 = QtWidgets.QLabel('Upper Top Left Y Coordinate:')
        lbl_1 = QtWidgets.QLabel('Upper Top Left Z Coordinate (from DTM):')
        lbl_8 = QtWidgets.QLabel('Total X Extent:')
        lbl_9 = QtWidgets.QLabel('Total Y Extent:')
        lbl_10 = QtWidgets.QLabel('Total Z Extent (Depth):')
        lbl_5 = QtWidgets.QLabel('X and Y Cell Size:')
        lbl_6 = QtWidgets.QLabel('Z Cell Size:')
        lbl_99 = QtWidgets.QLabel('Number of output classes:')

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

        gl_extent.addWidget(lbl_0, 0, 0, 1, 1)
        gl_extent.addWidget(lbl_3, 1, 0, 1, 1)
        gl_extent.addWidget(lbl_4, 2, 0, 1, 1)
        gl_extent.addWidget(lbl_1, 3, 0, 1, 1)
        gl_extent.addWidget(lbl_8, 4, 0, 1, 1)
        gl_extent.addWidget(lbl_9, 5, 0, 1, 1)
        gl_extent.addWidget(lbl_10, 6, 0, 1, 1)
        gl_extent.addWidget(lbl_5, 7, 0, 1, 1)
        gl_extent.addWidget(lbl_6, 8, 0, 1, 1)
        gl_extent.addWidget(lbl_99, 9, 0, 1, 1)
        gl_extent.addWidget(self.cmb_dataset, 0, 1, 1, 1)
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
        gl_extent.addWidget(self.sb_classes, 9, 1, 1, 1)

        hbl.addWidget(helpdocs)
        hbl.addWidget(buttonbox)

        # Assign to main layout
        vbl.addWidget(gbox_data_info)
        vbl.addWidget(gbox_gen_prop)
        vbl.addWidget(gbox_extent)
        vbl.addLayout(hbl)

        # Link functions
        self.dsb_xycell.valueChanged.connect(self.xycell)
        self.dsb_zcell.valueChanged.connect(self.zcell)
        self.dsb_utlx.valueChanged.connect(self.upd_layers)
        self.dsb_utly.valueChanged.connect(self.upd_layers)
        self.dsb_utlz.valueChanged.connect(self.upd_layers)
        self.dsb_xextent.valueChanged.connect(self.upd_layers)
        self.dsb_yextent.valueChanged.connect(self.upd_layers)
        self.dsb_zextent.valueChanged.connect(self.upd_layers)
        self.cmb_dataset.currentIndexChanged.connect(self.get_area)
        self.cmb_dtm.currentIndexChanged.connect(self.choose_dtm)
        self.cmb_model.currentIndexChanged.connect(self.choose_model)

        buttonbox.accepted.connect(self.apply_changes)
        buttonbox.rejected.connect(self.reject)

    def apply_changes(self):
        """
        Apply changes.

        Returns
        -------
        None.

        """
        self.showlog('Working...')

        self.choose_combo(self.cmb_dtm, 'DTM Dataset')
        self.choose_combo(self.cmb_mag, 'Magnetic Dataset')
        self.choose_combo(self.cmb_dataset, 'Study Area Dataset')

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
        tmp = list(set(self.lmod1.griddata.values()))
        self.outdata['Raster'] = tmp
        self.showlog('Changes applied.')

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
        ctxt = str(self.cmb_dtm.currentText())
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
        ctxt = str(self.cmb_model.currentText())
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
        ctxt = str(self.cmb_dataset.currentText())
        if ctxt not in ('None', ''):
            curgrid = self.inraster[ctxt]

            crows, ccols = curgrid.data.shape

            dxy = max(curgrid.xdim, curgrid.ydim)
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
            self.dsb_xycell.setValue(dxy)

    def init(self):
        """
        Initialise parameters.

        Returns
        -------
        None.

        """
        # Extent Parameters
        self.dsb_utlx.setValue(self.lmod1.xrange[0])
        self.dsb_utly.setValue(self.lmod1.yrange[-1])
        self.dsb_utlz.setValue(self.lmod1.zrange[-1])
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

        self.cmb_model.currentIndexChanged.disconnect()

        self.cmb_model.clear()
        self.cmb_model.addItems(modnames)
        self.cmb_model.setCurrentIndex(0)

        if len(modnames) >= 2:
            self.cmb_model.setCurrentIndex(1)

        self.cmb_model.currentIndexChanged.connect(self.choose_model)

    def update_combos(self):
        """
        Update combos.

        Returns
        -------
        None.

        """
        self.cmb_dataset.currentIndexChanged.disconnect()

        gkeys = list(self.inraster.keys())
        if 'Calculated Gravity' in gkeys:
            gkeys.remove('Calculated Gravity')
        if 'Calculated Magnetics' in gkeys:
            gkeys.remove('Calculated Magnetics')
        gkeys = ['None'] + gkeys

        if len(gkeys) > 1:
            self.cmb_other.clear()
            self.cmb_other.addItems(gkeys)
            self.cmb_other.setCurrentIndex(0)
            self.cmb_dtm.clear()
            self.cmb_dtm.addItems(gkeys)
            self.cmb_dtm.setCurrentIndex(0)
            self.cmb_mag.clear()
            self.cmb_mag.addItems(gkeys)
            self.cmb_mag.setCurrentIndex(0)
            self.cmb_grv.clear()
            self.cmb_grv.addItems(gkeys)
            self.cmb_grv.setCurrentIndex(0)
            self.cmb_reggrv.clear()
            self.cmb_reggrv.addItems(gkeys)
            self.cmb_reggrv.setCurrentIndex(0)
            self.cmb_dataset.clear()
            self.cmb_dataset.addItems(gkeys)
            self.cmb_dataset.setCurrentIndex(0)

            lkeys = list(self.lmod1.griddata.keys())
            if 'DTM Dataset' in lkeys:
                tmp = self.lmod1.griddata['DTM Dataset'].dataid
                self.cmb_dtm.setCurrentIndex(gkeys.index(tmp))

            if 'Magnetic Dataset' in lkeys:
                tmp = self.lmod1.griddata['Magnetic Dataset'].dataid
                self.cmb_mag.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Dataset' in lkeys:
                tmp = self.lmod1.griddata['Gravity Dataset'].dataid
                self.cmb_grv.setCurrentIndex(gkeys.index(tmp))

            if 'Gravity Regional' in lkeys:
                tmp = self.lmod1.griddata['Gravity Regional'].dataid
                self.cmb_reggrv.setCurrentIndex(gkeys.index(tmp))

            if 'Study Area Dataset' in lkeys:
                tmp = self.lmod1.griddata['Study Area Dataset'].dataid
                self.cmb_dataset.setCurrentIndex(gkeys.index(tmp))

            if 'Other' in lkeys:
                tmp = self.lmod1.griddata['Other'].dataid
                self.cmb_other.setCurrentIndex(gkeys.index(tmp))

        self.cmb_dataset.currentIndexChanged.connect(self.get_area)

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
        self.update_vals()

        self.choose_dtm()
        self.get_area()

        if nodialog is False:
            self.update_combos()

        tmp = self.exec()

        if tmp != 1:
            return tmp

        tmp = self.acceptall()

        if tmp is True:
            self.outdata['Model3D'] = [self.lmod2]

        return tmp

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.cmb_model)
        self.saveobj(self.cmb_other)
        self.saveobj(self.cmb_dtm)
        self.saveobj(self.cmb_mag)
        self.saveobj(self.cmb_grv)
        self.saveobj(self.cmb_reggrv)
        self.saveobj(self.cmb_dataset)

        self.saveobj(self.dsb_utlx)
        self.saveobj(self.dsb_utly)
        self.saveobj(self.dsb_utlz)
        self.saveobj(self.dsb_xextent)
        self.saveobj(self.dsb_yextent)
        self.saveobj(self.dsb_zextent)
        self.saveobj(self.dsb_xycell)
        self.saveobj(self.dsb_zcell)
        self.saveobj(self.sb_cols)
        self.saveobj(self.sb_rows)
        self.saveobj(self.sb_layers)
        self.saveobj(self.sb_classes)
        self.saveobj(self.dsb_mht)
        self.saveobj(self.dsb_hdec)
        self.saveobj(self.dsb_hint)
        self.saveobj(self.dsb_hinc)

    def acceptall(self):
        """
        Accept All.

        Based on the SimPEG example.

        Returns
        -------
        None.

        """
        dat = [self.lmod1.griddata['Magnetic Dataset'],
               self.lmod1.griddata['DTM Dataset']]

        masterid = self.lmod1.griddata['Magnetic Dataset'].dataid

        dat = lstack(dat, masterid=masterid, commonmask=True,
                     piter=self.piter, showlog=self.showlog)

        mag = dat[0]
        dtm = dat[1]

        dobs = mag.data.compressed()

        xmin, xmax, ymin, ymax = mag.extent

        xdim = mag.xdim
        ydim = mag.ydim

        rows, cols = mag.data.shape

        xxx = np.linspace(xmin, xmax, cols, False) + xdim/2
        yyy = np.linspace(ymin, ymax, rows, False) + ydim/2

        xy = np.meshgrid(xxx, yyy[::-1])
        z = dtm.data + self.dsb_mht.value()

        xxx = xy[0][~mag.data.mask]
        yyy = xy[1][~mag.data.mask]
        z = z.compressed()

        receiver_locations = np.transpose([xxx, yyy, z])

        topo_xyz = np.transpose([xxx, yyy, dtm.data.compressed()])

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

        source_field = magnetics.UniformBackgroundField(
            receiver_list=receiver_list, amplitude=strength,
            inclination=inclination, declination=declination)

        # Define the survey, data and tensor mesh
        survey = magnetics.survey.Survey(source_field)
        data_object = data.Data(survey, dobs=dobs, standard_deviation=std)
        dh = self.lmod1.d_z
        dhxy = self.lmod1.dxy

        hx = [(dhxy, 5, -1.3), (dhxy, self.lmod1.numx), (dhxy, 5, 1.3)]
        hy = [(dhxy, 5, -1.3), (dhxy, self.lmod1.numy), (dhxy, 5, 1.3)]
        hz = [(dh, 5, -1.3), (dh, self.lmod1.numz)]

        x0 = xmin-(np.sum([dhxy*1.3**(i+1) for i in range(5)]))
        y0 = ymin-(np.sum([dhxy*1.3**(i+1) for i in range(5)]))
        z0 = -(dh*self.lmod1.numz)-(np.sum([dh*1.3**(i+1) for i in range(5)]))

        mesh = TensorMesh([hx, hy, hz], [x0, y0, z0])

        # Starting/Reference Model and Mapping on Tensor Mesh
        background_susceptibility = 1e-4

        ind_active = active_from_xyz(mesh, topo_xyz)
        nC = int(ind_active.sum())
        model_map = maps.IdentityMap(nP=nC)

        # Define Starting model
        starting_model = background_susceptibility * np.ones(nC)

        # Define the Physics
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, model_type="scalar", chiMap=model_map,
            ind_active=ind_active)

        # Define Inverse Problem
        dmis = data_misfit.L2DataMisfit(data=data_object,
                                        simulation=simulation)
        reg = regularization.Sparse(mesh, active_cells=ind_active,
                                    mapping=model_map,
                                    reference_model=starting_model,
                                    gradient_type="total")
        reg.norms = [0, 0, 0, 0]

        opt = optimization.ProjectedGNCG(maxIter=20, lower=0.0, upper=1.0,
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
        sensitivity_weights = directives.UpdateSensitivityWeights(
            every_iteration=False)
        directives_list = [sensitivity_weights, starting_beta, save_iteration,
                           update_IRLS, update_jacobi]

        # Running the Inversion
        inv = inversion.BaseInversion(inv_prob, directives_list)

        try:
            with redirect_stdout(self.stdout_redirect):
                recovered_model = inv.run(starting_model)
        except Exception as e:
            self.showlog('Error: '+str(e))
            return False

        # Recreate True Model
        background_susceptibility = 0.0001
        sphere_susceptibility = 0.01

        true_model = background_susceptibility * np.ones(nC)
        ind_sphere = model_builder.get_indices_sphere(
            np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers)
        ind_sphere = ind_sphere[ind_active]
        true_model[ind_sphere] = sphere_susceptibility

        self.tmp = [maps, mesh, ind_active, recovered_model, true_model]

        soln_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

        r2 = soln_map * recovered_model
        r2.shape = (mesh.shape_cells[2],
                    mesh.shape_cells[1],
                    mesh.shape_cells[0])
        r2 = r2[::-1]

        r3 = r2[:-5, 5:-5, 5:-5]
        r3 = np.ma.masked_invalid(r3)
        r3 = np.ma.array(r3)

        X = r3.compressed().reshape(-1, 1)

        numclasses = self.sb_classes.value()
        cfit = skc.KMeans(n_clusters=numclasses, n_init='auto').fit(X)

        zout = cfit.labels_
        r3[~r3.mask] = zout

        r4 = np.moveaxis(r3, [0, 1, 2], [2, 1, 0])
        r4 = r4.filled(-1)

        cnt = cfit.labels_.max()+1
        susc = [0.]*cnt
        inputliths = ['']*cnt
        dens = [2.67]*cnt
        for i2 in range(cnt):
            susc[i2] = X[cfit.labels_ == i2].mean()
            inputliths[i2] = str(susc[i2])

        bsusc = np.min(susc)
        bindx = np.nonzero(susc == bsusc)[0][0]

        if bindx != 0:
            r4[r4 == bindx] = 999
            r4[r4 == 0] = bindx
            r4[r4 == 999] = 0

            susc[bindx] = susc[0]
            inputliths[bindx] = inputliths[0]

        susc = susc[1:]
        inputliths = inputliths[1:]

        tlx = self.dsb_utlx.value()
        tly = self.dsb_utly.value()
        tlz = self.dsb_utlz.value()

        mht = self.dsb_mht.value()

        dxy = self.dsb_xycell.value()
        numx = self.sb_cols.value()
        numy = self.sb_rows.value()
        numz = self.sb_layers.value()
        d_z = self.dsb_zcell.value()

        self.lmod2 = quick_model(numx, numy, numz, dxy, d_z, tlx, tly, tlz,
                                 mht, finc=inclination, fdec=declination,
                                 inputliths=inputliths, susc=susc, dens=dens,
                                 hintn=strength)

        self.lmod2.lith_list['Background'].susc = bsusc
        self.lmod2.lith_index = r4.astype(int)
        self.lmod2.name = 'Internal Inverted Model'
        self.lmod2.griddata = self.lmod1.griddata

        return True


def _testfn():
    """Test Function."""
    from pygmi.raster.iodefs import get_raster
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from pygmi.pfmod.pfmod import MainWidget

    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    mfile = r"D:\Workdata\PyGMI Test Data\Potential Field Modelling\MagInv\magnetics\mag.tif"
    dfile = r"D:\Workdata\PyGMI Test Data\Potential Field Modelling\MagInv\magnetics\dem.tif"

    mdat = get_raster(mfile)
    ddat = get_raster(dfile)

    mdat[0].dataid = 'mag'
    ddat[0].dataid = 'dem'

    app = QtWidgets.QApplication(sys.argv)

    DM = MagInvert()
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
    dxy = 10.
    d_z = 5.
    mht = 10.

    DM.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z, mht)

    DM.update_combos()
    DM.cmb_dtm.setCurrentText('dem')
    DM.cmb_mag.setCurrentText('mag')
    DM.cmb_dataset.setCurrentText('mag')

    DM.choose_dtm()

    DM.dsb_mht.setValue(mht)
    DM.dsb_hdec.setValue(0)
    DM.dsb_hint.setValue(50000)
    DM.dsb_hinc.setValue(90)
    DM.dsb_zcell.setValue(5)

    DM.settings(True)

    maps1, mesh, ind_active, recovered_model, true_model = DM.tmp

    # Plot True Model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps1.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plot_slice(
        plotting_map * true_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(true_model), np.max(true_model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(true_model),
                                vmax=np.max(true_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.colormaps['viridis'],
                                     format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()

    # Plot recovered model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps1.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plot_slice(
        plotting_map * recovered_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(recovered_model), np.max(recovered_model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(recovered_model),
                                vmax=np.max(recovered_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.colormaps['viridis'],
                                     format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()

    get_ipython().run_line_magic('matplotlib', 'Qt5')

    tmp = MainWidget()
    tmp.indata = DM.outdata
    tmp.settings()


def _testfn2():
    """Test Function."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pygmi.raster.iodefs import get_raster
    from pygmi.pfmod.pfmod import MainWidget
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    app = QtWidgets.QApplication(sys.argv)

    mfile = r"D:\Workdata\PyGMI Test Data\Potential Field Modelling\MagInv\pcmagdem.tif"
    dfile = r"D:\Workdata\PyGMI Test Data\Potential Field Modelling\MagInv\pcdem.tif"

    mdat = get_raster(mfile)
    ddat = get_raster(dfile)

    mdat[0].dataid = 'mag'
    ddat[0].dataid = 'dem'

    DM = MagInvert()
    DM.indata['Raster'] = mdat+ddat

    for i in DM.indata['Raster']:
        DM.inraster[i.dataid] = i
    if 'Model3D' in DM.indata:
        DM.lmod1 = DM.indata['Model3D'][0]

    cols = 50
    rows = 40
    layers = 25
    utlx = 0
    utly = 0
    utlz = 0
    dxy = 100.
    d_z = 100.
    mht = 100.

    cols = 50
    rows = 40
    layers = 20
    utlx = 0
    utly = 0
    utlz = 58.11
    dxy = 100.
    d_z = 100.
    mht = 100.

    DM.lmod1.update(cols, rows, layers, utlx, utly, utlz, dxy, d_z, mht)

    DM.update_combos()
    DM.combo_dtm.setCurrentText('dem')
    DM.combo_mag.setCurrentText('mag')
    DM.combo_dataset.setCurrentText('mag')

    DM.choose_dtm()

    DM.dsb_mht.setValue(mht)
    DM.settings(True)

    maps1, mesh, ind_active, recovered_model, true_model = DM.tmp

    # Plot True Model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps1.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plot_slice(
        plotting_map * true_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(true_model), np.max(true_model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(true_model),
                                vmax=np.max(true_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.colormaps['viridis'],
                                     format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()

    # Plot recovered model
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plotting_map = maps1.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
    mesh.plot_slice(
        plotting_map * recovered_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(recovered_model), np.max(recovered_model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(recovered_model),
                                vmax=np.max(recovered_model))
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",
                                     cmap=mpl.colormaps['viridis'],
                                     format="%.1e")
    cbar.set_label("SI", rotation=270, labelpad=15, size=12)
    plt.show()

    get_ipython().run_line_magic('matplotlib', 'Qt5')

    tmp = MainWidget()
    tmp.indata = DM.outdata
    tmp.settings()


if __name__ == "__main__":
    _testfn()
