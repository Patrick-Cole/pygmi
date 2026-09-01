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
"""Vector menu routines."""

from PySide6 import QtGui, QtWidgets

from pygmi.vector import (
    dataprep,
    equation_editor,
    graphs,
    iodefs,
    show_table,
    structure,
    vvis3d,
)


class MenuWidget:
    """
    Widget class to call the main interface.

    This widget class creates the vector menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Parameters
    ----------
    parent
        Reference to MainWidget class found in main.py. The default is None.
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context("Vector")
        self.parent.add_to_context("inVector")
        self.parent.add_to_context("pntVector")
        self.parent.add_to_context("lineVector")
        self.parent.add_to_context("Voxel")
        context_menu = self.parent.context_menu

        self.menu = QtWidgets.QMenu("Vector")
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_vector = QtGui.QAction("Import Vector Data")
        self.menu.addAction(self.action_import_vector)
        self.action_import_vector.triggered.connect(self.import_vector)

        self.action_import_xyz = QtGui.QAction("Import XYZ Data")
        self.menu.addAction(self.action_import_xyz)
        self.action_import_xyz.triggered.connect(self.import_xyz)

        self.action_import_voxel = QtGui.QAction("Import Voxel Data")
        self.menu.addAction(self.action_import_voxel)
        self.action_import_voxel.triggered.connect(self.import_voxel)

        self.action_colselect = QtGui.QAction("Select Columns")
        self.menu.addAction(self.action_colselect)
        self.action_colselect.triggered.connect(self.colselect)

        self.menu.addSeparator()
        self.action_file_split = QtGui.QAction("Text File Split")
        self.menu.addAction(self.action_file_split)
        self.action_file_split.triggered.connect(self.file_split)

        self.menu.addSeparator()

        self.action_equation_editor = QtGui.QAction("Vector Equation Editor")
        self.menu.addAction(self.action_equation_editor)
        self.action_equation_editor.triggered.connect(self.equation_editor)

        self.action_cut_data = QtGui.QAction("Cut Points using Polygon")
        self.menu.addAction(self.action_cut_data)
        self.action_cut_data.triggered.connect(self.cut_data)

        self.action_reproject = QtGui.QAction("Reproject Vector Data")
        self.menu.addAction(self.action_reproject)
        self.action_reproject.triggered.connect(self.reproject)

        self.menu.addSeparator()

        self.action_grid = QtGui.QAction("Dataset Gridding")
        self.menu.addAction(self.action_grid)
        self.action_grid.triggered.connect(self.grid)

        self.action_scomp = QtGui.QAction("Structure Complexity")
        self.menu.addAction(self.action_scomp)
        self.action_scomp.triggered.connect(self.scomp)

        # Context menus
        context_menu["inVector"].addSeparator()

        self.action_bandselect = QtGui.QAction("Select Input Columns")
        context_menu["inVector"].addAction(self.action_bandselect)
        self.action_bandselect.triggered.connect(self.colselect2)

        context_menu["Vector"].addSeparator()

        self.action_metadata = QtGui.QAction("Display/Edit Vector Metadata")
        context_menu["Vector"].addAction(self.action_metadata)
        self.action_metadata.triggered.connect(self.metadata)

        self.action_basic_stats = QtGui.QAction("Basic Vector Statistics")
        context_menu["Vector"].addAction(self.action_basic_stats)
        self.action_basic_stats.triggered.connect(self.basic_stats)

        self.action_plot_ccoef = QtGui.QAction("Plot Correlation Coefficients")
        context_menu["pntVector"].addAction(self.action_plot_ccoef)
        self.action_plot_ccoef.triggered.connect(self.plot_ccoef)

        self.action_show_line_data = QtGui.QAction("Show Profile Data")
        context_menu["pntVector"].addAction(self.action_show_line_data)
        self.action_show_line_data.triggered.connect(self.show_line_data)

        self.action_show_line_data2 = QtGui.QAction("Show Profiles on a Map")
        context_menu["pntVector"].addAction(self.action_show_line_data2)
        self.action_show_line_data2.triggered.connect(self.show_line_map)

        self.action_show_vector_data = QtGui.QAction("Show Vector Data")
        context_menu["Vector"].addAction(self.action_show_vector_data)
        self.action_show_vector_data.triggered.connect(self.show_vector_data)

        self.action_show_rose_diagram = QtGui.QAction("Show Rose Diagram")
        context_menu["lineVector"].addAction(self.action_show_rose_diagram)
        self.action_show_rose_diagram.triggered.connect(self.show_rose_diagram)

        self.action_show_hist = QtGui.QAction("Show Histogram")
        context_menu["Vector"].addAction(self.action_show_hist)
        self.action_show_hist.triggered.connect(self.show_hist)

        self.action_export_xyz = QtGui.QAction("Export XYZ Data")
        context_menu["pntVector"].addAction(self.action_export_xyz)
        self.action_export_xyz.triggered.connect(self.export_xyz)

        self.action_export_vector = QtGui.QAction("Export Vector Data")
        context_menu["Vector"].addAction(self.action_export_vector)
        self.action_export_vector.triggered.connect(self.export_vector)

        context_menu["Voxel"].addSeparator()

        self.action_export_voxel = QtGui.QAction("Export Voxel Data")
        context_menu["Voxel"].addAction(self.action_export_voxel)
        self.action_export_voxel.triggered.connect(self.export_voxel)

        self.action_display_voxel = QtGui.QAction("Display Voxel Data")
        context_menu["Voxel"].addAction(self.action_display_voxel)
        self.action_display_voxel.triggered.connect(self.display_voxel)

    def colselect2(self):
        """Select columns via context menu."""
        self.parent.launch_context_item_indata(iodefs.ColumnSelect)

    def colselect(self):
        """Select columns."""
        self.parent.item_insert("Step", "Column Select", iodefs.ColumnSelect)

    def grid(self):
        """Grid datasets."""
        self.parent.item_insert("Step", "Dataset Gridding", dataprep.DataGrid)

    def scomp(self):
        """Structure complexity."""
        self.parent.item_insert("Step", "Structure Complexity", structure.StructComp)

    def cut_data(self):
        """Cut point data."""
        self.parent.item_insert("Step", "Cut Points", dataprep.PointCut)

    def reproject(self):
        """Reproject point data."""
        self.parent.item_insert("Step", "Reproject Vector Data", dataprep.DataReproj)

    def export_xyz(self):
        """Export XYZ data."""
        self.parent.launch_context_item(iodefs.ExportXYZ)

    def export_vector(self):
        """Export line data."""
        self.parent.launch_context_item(iodefs.ExportVector)

    def export_voxel(self):
        """Export voxel data."""
        self.parent.launch_context_item(iodefs.ExportVoxel)

    def display_voxel(self):
        """Display voxel data."""
        self.parent.launch_context_item(vvis3d.Mod3dDisplay)

    def file_split(self):
        """Text file split."""
        self.parent.item_insert("Io", "Text File Split", dataprep.TextFileSplit)

    def import_xyz(self):
        """Import XYZ data."""
        self.parent.item_insert("Io", "Import XYZ Data", iodefs.ImportXYZ)

    def import_voxel(self):
        """Import Voxel data."""
        self.parent.item_insert("Io", "Import Voxel Data", iodefs.ImportVoxel)

    def import_vector(self):
        """Import shape data."""
        self.parent.item_insert("Io", "Import Vector Data", iodefs.ImportVector)

    def metadata(self):
        """Metadata."""
        self.parent.launch_context_item(dataprep.Metadata)

    def plot_ccoef(self):
        """Plot correlation coefficient data."""
        self.parent.launch_context_item(graphs.PlotCCoef)

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

    def equation_editor(self):
        """VectorEquation Editor."""
        self.parent.item_insert(
            "Step", "Vector Equation Editor", equation_editor.EquationEditor
        )
