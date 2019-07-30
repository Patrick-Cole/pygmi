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
"""Remote Sensing Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.mt import iodefs
from pygmi.mt import dataprep
from pygmi.mt import graphs


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('MT - EDI')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtWidgets.QMenu('MT')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtWidgets.QAction('Import EDI Data')
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.menu.addSeparator()

        self.action_rotate_data = QtWidgets.QAction('Rotate EDI Data')
        self.menu.addAction(self.action_rotate_data)
        self.action_rotate_data.triggered.connect(self.rotate_data)

        self.action_sshift_data = QtWidgets.QAction('Remove Static Shift')
        self.menu.addAction(self.action_sshift_data)
        self.action_sshift_data.triggered.connect(self.sshift_data)

        self.action_mi_data = QtWidgets.QAction('Mask and Interpolate')
        self.menu.addAction(self.action_mi_data)
        self.action_mi_data.triggered.connect(self.mi_data)

        self.menu.addSeparator()

        self.action_occam1d = QtWidgets.QAction('Occam 1D Inversion')
        self.menu.addAction(self.action_occam1d)
        self.action_occam1d.triggered.connect(self.occam1d)


# Context menus
        context_menu['MT - EDI'].addSeparator()

        self.action_metadata = QtWidgets.QAction('Display/Edit Metadata')
        context_menu['MT - EDI'].addAction(self.action_metadata)
        self.action_metadata.triggered.connect(self.metadata)

        self.action_show_graphs = QtWidgets.QAction('Show Graphs')
        context_menu['MT - EDI'].addAction(self.action_show_graphs)
        self.action_show_graphs.triggered.connect(self.show_graphs)

        self.action_export_data = QtWidgets.QAction('Export EDI')
        context_menu['MT - EDI'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def import_data(self):
        """Import data."""
        fnc = iodefs.ImportEDI(self.parent)
        self.parent.item_insert('Io', 'Import EDI Data', fnc)

    def export_data(self):
        """ Export raster data """
        self.parent.launch_context_item(iodefs.ExportEDI)

    def occam1d(self):
        """Rotate data."""
        fnc = dataprep.Occam1D(self.parent)
        self.parent.item_insert('Step', 'Occam 1D\nInversion', fnc)

    def rotate_data(self):
        """Rotate data."""
        fnc = dataprep.RotateEDI(self.parent)
        self.parent.item_insert('Step', 'Rotate EDI Data', fnc)

    def sshift_data(self):
        """Static Shift data."""
        fnc = dataprep.StaticShiftEDI(self.parent)
        self.parent.item_insert('Step', 'Static Shift EDI Data', fnc)

    def mi_data(self):
        """Mask and interpolate data."""
        fnc = dataprep.EditEDI(self.parent)
        self.parent.item_insert('Step', 'Mask and Interpolate\nEDI Data', fnc)

    def metadata(self):
        """ Basic Statistics """
        self.parent.launch_context_item(dataprep.Metadata)

    def show_graphs(self):
        """Show point data."""
        self.parent.launch_context_item(graphs.PlotPoints)