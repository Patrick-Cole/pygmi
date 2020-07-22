# -----------------------------------------------------------------------------
# Name:        menu.py (part of PyGMI)
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
"""Potential Field Modelling menus."""

from PyQt5 import QtWidgets
import pygmi.pfmod.pfmod as pfmod
import pygmi.pfmod.cubes as mvis3d
import pygmi.pfmod.iodefs as iodefs
import pygmi.pfmod.misc as misc


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the modelling menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Model3D')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtWidgets.QMenu('Potential Field Modelling')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_mod3d = QtWidgets.QAction('Import 3D Model')
        self.menu.addAction(self.action_import_mod3d)
        self.action_import_mod3d.triggered.connect(self.import_mod3d)

        self.action_pfmod = QtWidgets.QAction('Model Creation and Editing')
        self.menu.addAction(self.action_pfmod)
        self.action_pfmod.triggered.connect(self.pfmod)

        self.menu.addSeparator()

        self.action_merge_mod3d = QtWidgets.QAction('Merge two 3D Models')
        self.menu.addAction(self.action_merge_mod3d)
        self.action_merge_mod3d.triggered.connect(self.merge_mod3d)

# Context Menu
        context_menu['Model3D'].addSeparator()

        self.action_mod3d = QtWidgets.QAction('3D Model Display')
        context_menu['Model3D'].addAction(self.action_mod3d)
        self.action_mod3d.triggered.connect(self.mod3d)

        self.action_export_mod3d = QtWidgets.QAction('Export 3D Model')
        context_menu['Model3D'].addAction(self.action_export_mod3d)
        self.action_export_mod3d.triggered.connect(self.export_mod3d)

    def export_mod3d(self):
        """Export 3D Model."""
        self.parent.launch_context_item(iodefs.ExportMod3D)

    def pfmod(self):
        """Voxel modelling of data."""
        fnc = pfmod.MainWidget(self.parent)
        self.parent.item_insert('Step', 'Potential Field Modelling', fnc)

    def mod3d(self):
        """3D display of data."""
        self.parent.launch_context_item(mvis3d.Mod3dDisplay)

    def import_mod3d(self):
        """Import data."""
        fnc = iodefs.ImportMod3D(self.parent)
        self.parent.item_insert('Io', 'Import 3D Model', fnc)

    def merge_mod3d(self):
        """Merge models."""
        fnc = misc.MergeMod3D(self.parent)
        self.parent.item_insert('Step', 'Merge 3D Models', fnc)
