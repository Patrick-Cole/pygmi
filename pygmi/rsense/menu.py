# -----------------------------------------------------------------------------
# Name:        menu.py (part of PyGMI)
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
"""Remote Sensing Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.rsense import change
from pygmi.rsense import iodefs
from pygmi.rsense import ratios


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Remote Sensing')

# Normal menus
        self.menu = QtWidgets.QMenu('Remote Sensing')
        parent.menubar.addAction(self.menu.menuAction())

        self.menu3 = self.menu.addMenu('Import Data')

        # self.action_import_ged = QtWidgets.QAction('Import ASTER Global '
        #                                            'Emissivity Data')
        # self.menu3.addAction(self.action_import_ged)
        # self.action_import_ged.triggered.connect(self.import_ged)

        self.action_import_aster = QtWidgets.QAction('Import ASTER')
        self.menu3.addAction(self.action_import_aster)
        self.action_import_aster.triggered.connect(self.import_aster)

        self.action_import_landsat = QtWidgets.QAction('Import Landsat 4 to 8')
        self.menu3.addAction(self.action_import_landsat)
        self.action_import_landsat.triggered.connect(self.import_landsat)

        self.action_import_sentinel2 = QtWidgets.QAction('Import Sentinel-2')
        self.menu3.addAction(self.action_import_sentinel2)
        self.action_import_sentinel2.triggered.connect(self.import_sentinel2)

        self.action_import_sentinel5p = QtWidgets.QAction('Import Sentinel-5P')
        self.menu3.addAction(self.action_import_sentinel5p)
        self.action_import_sentinel5p.triggered.connect(self.import_sentinel5p)

        self.action_import_modis = QtWidgets.QAction('Import MODIS v6')
        self.menu3.addAction(self.action_import_modis)
        self.action_import_modis.triggered.connect(self.import_modis)

        self.menu3.addSeparator()

        self.action_batch_list = QtWidgets.QAction('Create Batch List')
        self.menu3.addAction(self.action_batch_list)
        self.action_batch_list.triggered.connect(self.batch_list)

        self.menu.addSeparator()

        self.action_calc_ratios = QtWidgets.QAction('Calculate Band Ratios')
        self.menu.addAction(self.action_calc_ratios)
        self.action_calc_ratios.triggered.connect(self.calc_ratios)

        self.menu.addSeparator()

        self.menu2 = self.menu.addMenu('Change Detection')

        self.action_create_list = QtWidgets.QAction('Create Scene List '
                                                    '(Change Detection)')
        self.menu2.addAction(self.action_create_list)
        self.action_create_list.triggered.connect(self.create_scene)

        self.action_load_list = QtWidgets.QAction('Load Scene List')
        self.menu2.addAction(self.action_load_list)
        self.action_load_list.triggered.connect(self.load_scene)

        self.action_data_viewer = QtWidgets.QAction('View Change Data')
        self.menu2.addAction(self.action_data_viewer)
        self.action_data_viewer.triggered.connect(self.view_change)

    def create_scene(self):
        """Create Scene."""
        fnc = change.CreateSceneList(self.parent)
        self.parent.item_insert('Step', 'Create Scene List', fnc)

    def load_scene(self):
        """Load Scene."""
        fnc = change.LoadSceneList(self.parent)
        self.parent.item_insert('Io', 'Import Scene List', fnc)

    def view_change(self):
        """View Change Detection."""
        fnc = change.SceneViewer(self.parent)
        self.parent.item_insert('Step', 'Change Detection Viewer', fnc)

    def calc_ratios(self):
        """View Change Detection."""
        fnc = ratios.SatRatios(self.parent)
        self.parent.item_insert('Step', 'Calculate Band Ratios', fnc)

    def import_sentinel5p(self):
        """Import Sentinel 5P data."""
        fnc = iodefs.ImportSentinel5P(self.parent)
        self.parent.item_insert('Io', 'Import Sentinel-5P', fnc)

    def import_sentinel2(self):
        """Import Sentinel 2 data."""
        fnc = iodefs.ImportData(self.parent, 'Sentinel-2 (*.xml);;')
        self.parent.item_insert('Io', 'Import Sentinel-2', fnc)

    def import_modis(self):
        """Import MODIS data."""
        fnc = iodefs.ImportData(self.parent, 'MODIS (*.hdf);;')
        self.parent.item_insert('Io', 'Import MODIS v6', fnc)

    def import_aster(self):
        """Import ASTER HDF data."""
        fnc = iodefs.ImportData(self.parent, 'ASTER (AST*.hdf AST*.zip);;')
        self.parent.item_insert('Io', 'Import ASTER', fnc)

    def import_hdf(self):
        """Import HDF data."""
        fnc = iodefs.ImportData(self.parent, 'hdf (*.hdf *.h5);;')
        self.parent.item_insert('Io', 'Import HDF', fnc)

    def import_landsat(self):
        """Import Landsat data."""
        fnc = iodefs.ImportData(self.parent,
                                'Landsat (L*.tar.gz L*_MTL.txt);;')
        self.parent.item_insert('Io', 'Import Landsat', fnc)

    def import_ged(self):
        """Import GED data."""
        fnc = iodefs.ImportData(self.parent, 'ASTER GED (*.bin);;')
        self.parent.item_insert('Io', 'Import ASTER Global Emissivity Data',
                                fnc)

    def batch_list(self):
        """Import batch list."""
        fnc = iodefs.ImportBatch(self.parent)
        self.parent.item_insert('Io', 'Import Batch List', fnc)
