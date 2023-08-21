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
from pygmi.rsense.landsat_composite import LandsatComposite


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
        self.parent.add_to_context('RasterFileList')
        context_menu = self.parent.context_menu

        # Normal menus
        self.menu = QtWidgets.QMenu('Remote Sensing')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_sat = QtWidgets.QAction('Import Satellite Data')
        self.menu.addAction(self.action_import_sat)
        self.action_import_sat.triggered.connect(self.import_sat)

        self.action_import_sentinel5p = QtWidgets.QAction('Import Sentinel-5P')
        self.menu.addAction(self.action_import_sentinel5p)
        self.action_import_sentinel5p.triggered.connect(self.import_sentinel5p)

        self.action_batch_list = QtWidgets.QAction('Create Batch List')
        self.menu.addAction(self.action_batch_list)
        self.action_batch_list.triggered.connect(self.batch_list)

        self.menu.addSeparator()

        self.action_calc_ratios = QtWidgets.QAction('Calculate Band Ratios')
        self.menu.addAction(self.action_calc_ratios)
        self.action_calc_ratios.triggered.connect(self.calc_ratios)

        self.action_calc_ci = QtWidgets.QAction('Calculate Condition Indices')
        self.menu.addAction(self.action_calc_ci)
        self.action_calc_ci.triggered.connect(self.calc_ci)

        self.action_lsat_comp = QtWidgets.QAction('Calculate Landsat '
                                                  'Temporal Composite')
        self.menu.addAction(self.action_lsat_comp)
        self.action_lsat_comp.triggered.connect(self.lsat_comp)

        self.action_mnf = QtWidgets.QAction('MNF Transform')
        self.menu.addAction(self.action_mnf)
        self.action_mnf.triggered.connect(self.mnf)

        self.action_pca = QtWidgets.QAction('PCA Transform')
        self.menu.addAction(self.action_pca)
        self.action_pca.triggered.connect(self.pca)
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

        self.action_calc_change = QtWidgets.QAction('Calculate Change Indices')
        self.menu2.addAction(self.action_calc_change)
        self.action_calc_change.triggered.connect(self.calc_change)

        # self.action_create_list = QtWidgets.QAction('Create Scene List ')
        # self.menu2.addAction(self.action_create_list)
        # self.action_create_list.triggered.connect(self.create_scene)

        # self.action_load_list = QtWidgets.QAction('Load Scene List')
        # self.menu2.addAction(self.action_load_list)
        # self.action_load_list.triggered.connect(self.load_scene)

        self.action_data_viewer = QtWidgets.QAction('View Change Data')
        self.menu2.addAction(self.action_data_viewer)
        self.action_data_viewer.triggered.connect(self.view_change)

        # Context menus
        context_menu['RasterFileList'].addSeparator()

        self.action_exportlist = QtWidgets.QAction('Export Raster File List')
        context_menu['RasterFileList'].addAction(self.action_exportlist)
        self.action_exportlist.triggered.connect(self.exportlist)

    def exportlist(self):
        """Export Raster File List."""
        self.parent.launch_context_item(iodefs.ExportBatch)

    def calc_change(self):
        """Calculate change."""
        self.parent.item_insert('Step', 'Calculate Change Indices',
                                change.CalculateChange)

    # def create_scene(self):
    #     """Create Scene."""
    #     self.parent.item_insert('Step', 'Create Scene List',
    #                             change.CreateSceneList)

    # def load_scene(self):
    #     """Load Scene."""
    #     self.parent.item_insert('Io', 'Import Scene List',
    #                             change.LoadSceneList)

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

    def lsat_comp(self):
        """Calculate Landsat Composite."""
        self.parent.item_insert('Io', 'Calculate Landsat Temporal Composite',
                                LandsatComposite)

    def mnf(self):
        """Calculate MNF."""
        self.parent.item_insert('Step', 'MNF Transform',
                                transforms.MNF)

    def pca(self):
        """Calculate PCA."""
        self.parent.item_insert('Step', 'PCA Transform',
                                transforms.PCA)

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

    def import_sat(self):
        """Import Satellite data."""
        self.parent.item_insert('Io', 'Import Satellite', iodefs.ImportData)

    def batch_list(self):
        """Import batch list."""
        self.parent.item_insert('Io', 'Import Batch List', iodefs.ImportBatch)
