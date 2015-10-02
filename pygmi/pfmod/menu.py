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
""" Potential Field Modelling """

from PyQt4 import QtGui
import pygmi.pfmod.pfmod as pfmod
import pygmi.pfmod.cubes as mvis3d
import pygmi.pfmod.iodefs as iodefs


class MenuWidget(object):
    """
    Widget class to call the main interface

    This widget class creates the modelling menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : MainWidget
        Reference to MainWidget class found in main.py
    """
    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Model3D')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtGui.QMenu(parent.menubar)
        self.menu.setTitle("Potential Field Modelling")
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_mod3d = QtGui.QAction(parent)
        self.action_import_mod3d.setText("Import 3D Model")
        self.menu.addAction(self.action_import_mod3d)
        self.action_import_mod3d.triggered.connect(self.import_mod3d)

        self.action_prof_pic = QtGui.QAction(parent)
        self.action_prof_pic.setText("Import Profile Picture")
        self.menu.addAction(self.action_prof_pic)
        self.action_prof_pic.triggered.connect(self.import_prof_pic)

        self.menu.addSeparator()

        self.action_pfmod = QtGui.QAction(self.parent)
        self.action_pfmod.setText("Model Creation and Editing")
        self.menu.addAction(self.action_pfmod)
        self.action_pfmod.triggered.connect(self.pfmod)

# Context Menu
        context_menu['Model3D'].addSeparator()

        self.action_mod3d = QtGui.QAction(self.parent)
        self.action_mod3d.setText("3D Model Display")
        context_menu['Model3D'].addAction(self.action_mod3d)
        self.action_mod3d.triggered.connect(self.mod3d)

        self.action_export_mod3d = QtGui.QAction(self.parent)
        self.action_export_mod3d.setText("Export 3D Model")
        context_menu['Model3D'].addAction(self.action_export_mod3d)
        self.action_export_mod3d.triggered.connect(self.export_mod3d)

    def export_mod3d(self):
        """ Export 3D Model """
        self.parent.launch_context_item(iodefs.ExportMod3D)

    def pfmod(self):
        """ voxel modelling of data"""
        fnc = pfmod.MainWidget(self.parent)
        self.parent.item_insert("Step", "Potential\nField\nModelling", fnc)

    def mod3d(self):
        """ 3D display of data"""
        self.parent.launch_context_item(mvis3d.Mod3dDisplay)

    def import_mod3d(self):
        """ Imports data"""
        fnc = iodefs.ImportMod3D(self.parent)
        self.parent.item_insert("Io", "Import 3D Model", fnc)

    def import_prof_pic(self):
        """ Imports data"""
        fnc = iodefs.ImportPicture(self.parent)
        self.parent.item_insert("Io", "Import Profile\n Picture", fnc)
