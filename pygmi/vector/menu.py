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
from pygmi.vector import structure
from pygmi.vector import show_table


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
        self.parent.add_to_context('Vector')
        context_menu = self.parent.context_menu

        self.menu = QtWidgets.QMenu('Vector')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_vector = QtWidgets.QAction('Import Vector Data')
        self.menu.addAction(self.action_import_vector)
        self.action_import_vector.triggered.connect(self.import_vector)

        self.action_import_xyz = QtWidgets.QAction('Import XYZ Data')
        self.menu.addAction(self.action_import_xyz)
        self.action_import_xyz.triggered.connect(self.import_xyz)

        self.menu.addSeparator()
        self.action_file_split = QtWidgets.QAction('Text File Split')
        self.menu.addAction(self.action_file_split)
        self.action_file_split.triggered.connect(self.file_split)

        self.menu.addSeparator()

        self.action_cut_data = QtWidgets.QAction('Cut Points using Polygon')
        self.menu.addAction(self.action_cut_data)
        self.action_cut_data.triggered.connect(self.cut_data)

        self.action_reproject = QtWidgets.QAction('Reproject Vector Data')
        self.menu.addAction(self.action_reproject)
        self.action_reproject.triggered.connect(self.reproject)

        self.menu.addSeparator()

        self.action_grid = QtWidgets.QAction('Dataset Gridding')
        self.menu.addAction(self.action_grid)
        self.action_grid.triggered.connect(self.grid)

        self.action_scomp = QtWidgets.QAction('Structure Complexity')
        self.menu.addAction(self.action_scomp)
        self.action_scomp.triggered.connect(self.scomp)

        # Context menus
        context_menu['Vector'].addSeparator()

        self.action_metadata = QtWidgets.QAction('Display/Edit Vector '
                                                 'Metadata')
        context_menu['Vector'].addAction(self.action_metadata)
        self.action_metadata.triggered.connect(self.metadata)

        self.action_basic_stats = QtWidgets.QAction('Basic Vector Statistics')
        context_menu['Vector'].addAction(self.action_basic_stats)
        self.action_basic_stats.triggered.connect(self.basic_stats)

        self.action_show_line_data = QtWidgets.QAction('Show Profile Data')
        context_menu['Vector'].addAction(self.action_show_line_data)
        self.action_show_line_data.triggered.connect(self.show_line_data)

        self.action_show_line_data2 = QtWidgets.QAction('Show Profiles on a '
                                                        'Map')
        context_menu['Vector'].addAction(self.action_show_line_data2)
        self.action_show_line_data2.triggered.connect(self.show_line_map)

        self.action_show_vector_data = QtWidgets.QAction('Show Vector Data')
        context_menu['Vector'].addAction(self.action_show_vector_data)
        self.action_show_vector_data.triggered.connect(self.show_vector_data)

        self.action_show_rose_diagram = QtWidgets.QAction('Show Rose Diagram')
        context_menu['Vector'].addAction(self.action_show_rose_diagram)
        self.action_show_rose_diagram.triggered.connect(self.show_rose_diagram)

        self.action_show_hist = QtWidgets.QAction('Show Histogram')
        context_menu['Vector'].addAction(self.action_show_hist)
        self.action_show_hist.triggered.connect(self.show_hist)

        context_menu['Vector'].addSeparator()

        self.action_export_xyz = QtWidgets.QAction('Export XYZ Data')
        context_menu['Vector'].addAction(self.action_export_xyz)
        self.action_export_xyz.triggered.connect(self.export_xyz)

        self.action_export_vector = QtWidgets.QAction('Export Vector Data')
        context_menu['Vector'].addAction(self.action_export_vector)
        self.action_export_vector.triggered.connect(self.export_vector)

    def grid(self):
        """Grid datasets."""
        self.parent.item_insert('Step', 'Dataset Gridding', dataprep.DataGrid)

    def scomp(self):
        """Structure complexity."""
        self.parent.item_insert('Step', 'Structure Complexity',
                                structure.StructComp)

    def cut_data(self):
        """Cut point data."""
        self.parent.item_insert('Step', 'Cut Points', dataprep.PointCut)

    def reproject(self):
        """Reproject point data."""
        self.parent.item_insert('Step', 'Reproject Vector Data',
                                dataprep.DataReproj)

    def export_xyz(self):
        """Export XYZ data."""
        self.parent.launch_context_item(iodefs.ExportXYZ)

    def export_vector(self):
        """Export line data."""
        self.parent.launch_context_item(iodefs.ExportVector)

    def file_split(self):
        """Text file split."""
        self.parent.item_insert('Io', 'Text File Split',
                                dataprep.TextFileSplit)

    def import_xyz(self):
        """Import XYZ data."""
        self.parent.item_insert('Io', 'Import XYZ Data',
                                iodefs.ImportXYZ)

    def import_vector(self):
        """Import shape data."""
        self.parent.item_insert('Io', 'Import Vector Data',
                                iodefs.ImportVector)

    def metadata(self):
        """Metadata."""
        self.parent.launch_context_item(dataprep.Metadata)

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

    def show_hist(self):
        """Show histogram."""
        self.parent.launch_context_item(graphs.PlotHist)

    def basic_stats(self):
        """Display basic statistics."""
        self.parent.launch_context_item(show_table.BasicStats)
