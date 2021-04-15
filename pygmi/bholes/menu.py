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
from pygmi.bholes import hypercore
from pygmi.bholes import coremask
from pygmi.bholes import coremeta


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
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

        self.action_imagecor = QtWidgets.QAction('Raw Core Imagery '
                                                 'Corrections')
        self.menu.addAction(self.action_imagecor)
        self.action_imagecor.triggered.connect(self.imagecor)

        self.action_coreprep = QtWidgets.QAction('Tray Clipping and Band '
                                                 'Selection')
        self.menu.addAction(self.action_coreprep)
        self.action_coreprep.triggered.connect(self.coreprep)

        self.action_coremask = QtWidgets.QAction('Core Masking')
        self.menu.addAction(self.action_coremask)
        self.action_coremask.triggered.connect(self.coremask)

        self.action_coremeta = QtWidgets.QAction('Core Metadata and Depth '
                                                 'Assignment')
        self.menu.addAction(self.action_coremeta)
        self.action_coremeta.triggered.connect(self.coremeta)

        self.menu.addSeparator()

# Context menus
        context_menu['Borehole'].addSeparator()

        self.action_show_log = QtWidgets.QAction('Show Borehole Log')
        context_menu['Borehole'].addAction(self.action_show_log)
        self.action_show_log.triggered.connect(self.show_log)

    def import_data(self):
        """Import data."""
        self.parent.item_insert('Io', 'Import Data', iodefs.ImportData)

    def coreprep(self):
        """Tray Clipping and Band Selection."""
        self.parent.item_insert('Step',
                                'Tray Clipping and Band Selection',
                                hypercore.CorePrep)

    def coremask(self):
        """Core Masking."""
        self.parent.item_insert('Step',
                                'Core Masking',
                                coremask.CoreMask)

    def coremeta(self):
        """Core Metadata and depth assignment."""
        self.parent.item_insert('Step',
                                'Core Metadata and Depth Assignment',
                                coremeta.CoreMeta)

    def imagecor(self):
        """Raw core imagery corrections."""
        self.parent.item_insert('Io',
                                'Raw Core Imagery Corrections',
                                hypercore.ImageCor)

    def show_log(self):
        """Show log data."""
        self.parent.launch_context_item(graphs.PlotLog)
