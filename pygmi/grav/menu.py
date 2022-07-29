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
"""Gravity Menu Routines."""

from PyQt5 import QtWidgets

from pygmi.grav import iodefs
from pygmi.grav import dataprep


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

        # Normal menus
        self.menu = QtWidgets.QMenu('Gravity')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtWidgets.QAction('Import CG-5 Data')
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.menu.addSeparator()

        self.action_process = QtWidgets.QAction('Process Gravity Data')
        self.menu.addAction(self.action_process)
        self.action_process.triggered.connect(self.process_data)

    def import_data(self):
        """Import data."""
        self.parent.item_insert('Io', 'Import CG-5 Data', iodefs.ImportCG5)

    def process_data(self):
        """Process data."""
        self.parent.item_insert('Step', 'Process Gravity Data',
                                dataprep.ProcessData)
