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
""" Point Menu Routines """

# pylint: disable=E1101
from PyQt4 import QtGui
import pygmi.point.iodefs as iodefs


class MainWidget(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Point')
        context_menu = self.parent.context_menu

        self.menufile = QtGui.QMenu(parent.menubar)
        self.menufile.setTitle("Point")
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_import_point_data = QtGui.QAction(parent)
        self.action_import_point_data.setText("Import Point/Line Data")
        self.menufile.addAction(self.action_import_point_data)
        self.action_import_point_data.triggered.connect(self.import_point_data)

        self.action_import_lemi417_data = QtGui.QAction(parent)
        self.action_import_lemi417_data.setText("Import LEMI-417 Data")
        self.menufile.addAction(self.action_import_lemi417_data)
        self.action_import_lemi417_data.triggered.connect(
            self.import_lemi417_data)

        self.menufile.addSeparator()

# Context menus
        context_menu['Point'].addSeparator()

        self.action_export_point = QtGui.QAction(self.parent)
        self.action_export_point.setText("Export Point Data")
        context_menu['Point'].addAction(self.action_export_point)
        self.action_export_point.triggered.connect(self.export_point)

    def export_point(self):
        """ Show raster data """
        self.parent.launch_context_item(iodefs.ExportPoint)

    def import_point_data(self):
        """ Imports data"""
        fnc = iodefs.ImportPointData(self.parent)
        self.parent.item_insert("Io", "Import Point/Line Data", fnc)

    def import_lemi417_data(self):
        """ Imports LEMI-417 MT data"""
        fnc = iodefs.ImportLEMI417Data(self.parent)
        self.parent.item_insert("Io", "Import LEMI-417 Data", fnc)
