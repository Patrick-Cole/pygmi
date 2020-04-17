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

    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Remote Sensing')

# Normal menus
        self.menu = QtWidgets.QMenu('Remote Sensing')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_sentinel5p = QtWidgets.QAction('Import Sentinel-5P to shapefile')
        self.menu.addAction(self.action_import_sentinel5p)
        self.action_import_sentinel5p.triggered.connect(self.import_sentinel5p)

        self.menu.addSeparator()

        self.menu2 = self.menu.addMenu('Change Detection')

        self.action_create_list = QtWidgets.QAction('Create Scene List')
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

    def import_sentinel5p(self):
        """Import sentinel 5P data."""
        fnc = iodefs.ImportSentinel5P(self.parent)
        self.parent.item_insert('Io', 'Import Sentinel-5P', fnc)