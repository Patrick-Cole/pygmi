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
from pygmi.rsense import hyperspec
from pygmi.rsense import transforms


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

        self.action_import_landsat = QtWidgets.QAction('Import Landsat 4 to 9')
        self.menu3.addAction(self.action_import_landsat)
        self.action_import_landsat.triggered.connect(self.import_landsat)

        self.action_import_sentinel2 = QtWidgets.QAction('Import Sentinel-2')
        self.menu3.addAction(self.action_import_sentinel2)
        self.action_import_sentinel2.triggered.connect(self.import_sentinel2)

        self.action_import_sentinel2b = QtWidgets.QAction('Import Sentinel-2 '
                                                          '(bands only)')
        self.menu3.addAction(self.action_import_sentinel2b)
        self.action_import_sentinel2b.triggered.connect(self.import_sentinel2b)

        self.action_import_sentinel5p = QtWidgets.QAction('Import Sentinel-5P')
        self.menu3.addAction(self.action_import_sentinel5p)
        self.action_import_sentinel5p.triggered.connect(self.import_sentinel5p)

        self.action_import_modis = QtWidgets.QAction('Import MODIS v6')
        self.menu3.addAction(self.action_import_modis)
        self.action_import_modis.triggered.connect(self.import_modis)

        self.action_import_hyperion = QtWidgets.QAction('Import Hyperion L1T')
        self.menu3.addAction(self.action_import_hyperion)
        self.action_import_hyperion.triggered.connect(self.import_hyperion)

        self.action_import_wv = QtWidgets.QAction('Import WorldView Tiles')
        self.menu3.addAction(self.action_import_wv)
        self.action_import_wv.triggered.connect(self.import_wv)

        self.action_import_ged = QtWidgets.QAction('Import ASTER Global '
                                                   'Emissivity Database')
        self.menu3.addAction(self.action_import_ged)
        self.action_import_ged.triggered.connect(self.import_ged)

        self.menu3.addSeparator()

        self.action_batch_list = QtWidgets.QAction('Create Batch List')
        self.menu3.addAction(self.action_batch_list)
        self.action_batch_list.triggered.connect(self.batch_list)

        self.menu.addSeparator()

        self.action_calc_ratios = QtWidgets.QAction('Calculate Band Ratios')
        self.menu.addAction(self.action_calc_ratios)
        self.action_calc_ratios.triggered.connect(self.calc_ratios)

        self.action_calc_ci = QtWidgets.QAction('Calculate Condition Indices')
        self.menu.addAction(self.action_calc_ci)
        self.action_calc_ci.triggered.connect(self.calc_ci)

        self.action_mnf = QtWidgets.QAction('MNF Transform')
        self.menu.addAction(self.action_mnf)
        self.action_mnf.triggered.connect(self.mnf)

        self.menu.addSeparator()

        self.menu4 = self.menu.addMenu('Hyperspectral Imaging')

        self.action_anal_spec = QtWidgets.QAction('Analyse Spectra')
        self.menu4.addAction(self.action_anal_spec)
        self.action_anal_spec.triggered.connect(self.anal_spec)

        self.action_proc_features = QtWidgets.QAction('Process Features')
        self.menu4.addAction(self.action_proc_features)
        self.action_proc_features.triggered.connect(self.proc_features)

        self.menu.addSeparator()

        self.menu2 = self.menu.addMenu('Change Detection')

        self.action_create_list = QtWidgets.QAction('Create Scene List ')
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
        self.parent.item_insert('Step', 'Create Scene List',
                                change.CreateSceneList)

    def load_scene(self):
        """Load Scene."""
        self.parent.item_insert('Io', 'Import Scene List',
                                change.LoadSceneList)

    def view_change(self):
        """View Change Detection."""
        self.parent.item_insert('Step', 'Change Detection Viewer',
                                change.SceneViewer)

    def calc_ratios(self):
        """Calculate Ratios."""
        self.parent.item_insert('Step', 'Calculate Band Ratios',
                                ratios.SatRatios)

    def calc_ci(self):
        """Calculate Condition Indices."""
        self.parent.item_insert('Step', 'Calculate Condition Indices',
                                ratios.ConditionIndices)

    def mnf(self):
        """Calculate MNF."""
        self.parent.item_insert('Step', 'MNF Transform',
                                transforms.MNF)

    def anal_spec(self):
        """Analyse Spectra."""
        self.parent.item_insert('Step', 'Analyse Spectra', hyperspec.AnalSpec)

    def proc_features(self):
        """Process Features."""
        self.parent.item_insert('Step', 'Process Features',
                                hyperspec.ProcFeatures)

    def import_sentinel5p(self):
        """Import Sentinel 5P data."""
        self.parent.item_insert('Io', 'Import Sentinel-5P',
                                iodefs.ImportSentinel5P)

    def import_sentinel2(self):
        """Import Sentinel 2 data."""
        self.parent.item_insert('Io', 'Import Sentinel-2', iodefs.ImportData,
                                params='Sentinel-2 (*.xml *.zip);;')

    def import_sentinel2b(self):
        """Import Sentinel 2 data."""
        self.parent.item_insert('Io', 'Import Sentinel-2', iodefs.ImportData,
                                params='Sentinel-2 Bands Only (*.xml *.zip);;')

    def import_modis(self):
        """Import MODIS data."""
        self.parent.item_insert('Io', 'Import MODIS v6', iodefs.ImportData,
                                params='MODIS (*.hdf);;')

    def import_aster(self):
        """Import ASTER HDF data."""
        self.parent.item_insert('Io', 'Import ASTER', iodefs.ImportData,
                                params='ASTER (AST*.hdf AST*.zip);;')

    def import_ged(self):
        """Import HDF data."""
        self.parent.item_insert('Io', 'Import ASTER Global Emissivity Data',
                                iodefs.ImportData,
                                params='h5 (*.h5);;')

    def import_landsat(self):
        """Import Landsat data."""
        self.parent.item_insert('Io', 'Import Landsat', iodefs.ImportData,
                                params='Landsat (L*.tar L*.tar.gz L*_MTL.txt);;')

    def import_hyperion(self):
        """Import Hyperion data."""
        self.parent.item_insert('Io', 'Import Hyperion L1T', iodefs.ImportData,
                                params='Hyperion L1T (*.zip);;')

    def import_wv(self):
        """Import WorldView data."""
        self.parent.item_insert('Io', 'Import WorldView', iodefs.ImportData,
                                params='WorldView (*.xml);;')

    def batch_list(self):
        """Import batch list."""
        self.parent.item_insert('Io', 'Import Batch List', iodefs.ImportBatch)
