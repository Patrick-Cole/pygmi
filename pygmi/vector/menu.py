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
"""Vector Menu Routines."""

from PyQt5 import QtWidgets

from pygmi.vector import iodefs
from pygmi.vector import graphs
from pygmi.vector import dataprep


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the vector menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Line')
        self.parent.add_to_context('Vector')
        context_menu = self.parent.context_menu

        self.menufile = QtWidgets.QMenu('Vector')
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_import_shape_data = QtWidgets.QAction('Import Shapefile'
                                                          ' Data')
        self.menufile.addAction(self.action_import_shape_data)
        self.action_import_shape_data.triggered.connect(self.import_shape_data)

        self.action_import_line_data = QtWidgets.QAction('Import Point or Line'
                                                         ' Data')
        self.menufile.addAction(self.action_import_line_data)
        self.action_import_line_data.triggered.connect(self.import_line_data)

        self.menufile.addSeparator()

        self.action_cut_data = QtWidgets.QAction('Cut Points using Polygon')
        self.menufile.addAction(self.action_cut_data)
        self.action_cut_data.triggered.connect(self.cut_data)

        self.action_reproject = QtWidgets.QAction('Reproject Vector Data')
        self.menufile.addAction(self.action_reproject)
        self.action_reproject.triggered.connect(self.reproject)

        self.menufile.addSeparator()

        self.action_grid = QtWidgets.QAction('Dataset Gridding')
        self.menufile.addAction(self.action_grid)
        self.action_grid.triggered.connect(self.grid)

        # Context menus
        context_menu['Line'].addSeparator()

        self.action_show_line_data = QtWidgets.QAction('Show Line Data'
                                                       ' Profile')
        context_menu['Line'].addAction(self.action_show_line_data)
        self.action_show_line_data.triggered.connect(self.show_line_data)

        self.action_show_line_data2 = QtWidgets.QAction('Show Line Data Map')
        context_menu['Line'].addAction(self.action_show_line_data2)
        self.action_show_line_data2.triggered.connect(self.show_line_map)

        self.action_show_point_data2 = QtWidgets.QAction('Show Point Data Map')
        context_menu['Line'].addAction(self.action_show_point_data2)
        self.action_show_point_data2.triggered.connect(self.show_point_map)

        self.action_export_line = QtWidgets.QAction('Export Line Data')
        context_menu['Line'].addAction(self.action_export_line)
        self.action_export_line.triggered.connect(self.export_line)

        context_menu['Vector'].addSeparator()

        self.action_show_vector_data = QtWidgets.QAction('Show Vector Data')
        context_menu['Vector'].addAction(self.action_show_vector_data)
        self.action_show_vector_data.triggered.connect(self.show_vector_data)

        self.action_show_rose_diagram = QtWidgets.QAction('Show Rose Diagram')
        context_menu['Vector'].addAction(self.action_show_rose_diagram)
        self.action_show_rose_diagram.triggered.connect(self.show_rose_diagram)

        self.action_export_shape = QtWidgets.QAction('Export Shape Data')
        context_menu['Vector'].addAction(self.action_export_shape)
        self.action_export_shape.triggered.connect(self.export_shape)

    def grid(self):
        """Grid datasets."""
        self.parent.item_insert('Step', 'Dataset Gridding', dataprep.DataGrid)

    def cut_data(self):
        """Cut point data."""
        self.parent.item_insert('Step', 'Cut Points', dataprep.PointCut)

    def reproject(self):
        """Reproject point data."""
        self.parent.item_insert('Step', 'Reproject Vector Data',
                                dataprep.DataReproj)

    def export_line(self):
        """Export line data."""
        self.parent.launch_context_item(iodefs.ExportLine)

    def export_shape(self):
        """Export line data."""
        self.parent.launch_context_item(iodefs.ExportShapeData)

    def import_line_data(self):
        """Import line data."""
        self.parent.item_insert('Io', 'Import Line Data',
                                iodefs.ImportLineData)

    def import_shape_data(self):
        """Import shape data."""
        self.parent.item_insert('Io', 'Import Shapefile Data',
                                iodefs.ImportShapeData)

    def show_point_map(self):
        """Show point data."""
        self.parent.launch_context_item(graphs.PlotPoints)

    def show_line_data(self):
        """Show line data."""
        self.parent.launch_context_item(graphs.PlotLines)

    def show_line_map(self):
        """Show line map."""
        self.parent.launch_context_item(graphs.PlotLineMap)

    def show_vector_data(self):
        """Show vector data."""
        self.parent.launch_context_item(graphs.PlotVector)

    def show_rose_diagram(self):
        """Show rose diagram."""
        self.parent.launch_context_item(graphs.PlotRose)
