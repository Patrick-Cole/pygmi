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
"""Exploration seismics Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.eseis import iodefs
from pygmi.eseis import graphs


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
        self.parent.add_to_context('ESEIS')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtWidgets.QMenu('Exploration Seismics')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtWidgets.QAction('Import SEG-Y Data')
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

# Context menus
        context_menu['ESEIS'].addSeparator()

        self.action_show_graphs = QtWidgets.QAction('Show Graphs')
        context_menu['ESEIS'].addAction(self.action_show_graphs)
        self.action_show_graphs.triggered.connect(self.show_graphs)

        self.action_export_data = QtWidgets.QAction('Export SEG-Y')
        context_menu['ESEIS'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def import_data(self):
        """Import data."""
        fnc = iodefs.ImportSEGY(self.parent)
        self.parent.item_insert('Io', 'Import SEG-Y Data', fnc)

    def export_data(self):
        """Export data."""
        self.parent.launch_context_item(iodefs.ExportSEGY)

    def show_graphs(self):
        """Show graphs."""
        self.parent.launch_context_item(graphs.PlotSEGY)
