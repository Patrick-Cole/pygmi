# -----------------------------------------------------------------------------
# Name:        menu.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""Borehole Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.bholes import iodefs
from pygmi.bholes import graphs


class MenuWidget():
    """
    Menu widget class for the main interface.

    This widget class creates the menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to MainWidget class found in main.py. Default is None.
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Borehole')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtWidgets.QMenu('Boreholes')
        self.parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtWidgets.QAction('Import Borehole Data')
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.menu.addSeparator()

# Context menus
        context_menu['Borehole'].addSeparator()

        self.action_show_log = QtWidgets.QAction('Show Borehole Log')
        context_menu['Borehole'].addAction(self.action_show_log)
        self.action_show_log.triggered.connect(self.show_log)

    def import_data(self):
        """Import log data."""
        self.parent.item_insert('Io', 'Import Data', iodefs.ImportData)

    def show_log(self):
        """Show log data."""
        self.parent.launch_context_item(graphs.PlotLog)
