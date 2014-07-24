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
""" Raster Menu Routines """

# pylint: disable=E1101, C0103
from PyQt4 import QtGui
import pygmi.raster.equation_editor as equation_editor
import pygmi.raster.smooth as smooth
import pygmi.raster.normalisation as normalisation
import pygmi.raster.cooper as cooper
import pygmi.raster.ginterp as ginterp
import pygmi.clust.plot_graphs as plot_graphs
import pygmi.raster.show_table as show_table
import pygmi.raster.dataprep as dataprep
import pygmi.raster.iodefs as iodefs
import pygmi.raster.igrf as igrf


class MainWidget(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Raster')
        self.parent.add_to_context('Point')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtGui.QMenu(parent.menubar)
        self.menu.setTitle("Raster")
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtGui.QAction(parent)
        self.action_import_data.setText("Import Raster Data")
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.menu.addSeparator()

        self.action_equation_editor = QtGui.QAction(self.parent)
        self.action_equation_editor.setText("Equation Editor")
        self.menu.addAction(self.action_equation_editor)
        self.action_equation_editor.triggered.connect(self.equation_editor)

        self.action_smoothing = QtGui.QAction(self.parent)
        self.action_smoothing.setText("Smoothing")
        self.menu.addAction(self.action_smoothing)
        self.action_smoothing.triggered.connect(self.smoothing)

        self.action_normalisation = QtGui.QAction(self.parent)
        self.action_normalisation.setText("Normalisation")
        self.menu.addAction(self.action_normalisation)
        self.action_normalisation.triggered.connect(self.norm_data)

        self.action_gradients = QtGui.QAction(self.parent)
        self.action_gradients.setText("Horizontal Gradients")
        self.menu.addAction(self.action_gradients)
        self.action_gradients.triggered.connect(self.gradients)

        self.action_visibility = QtGui.QAction(self.parent)
        self.action_visibility.setText("Visibility")
        self.menu.addAction(self.action_visibility)
        self.action_visibility.triggered.connect(self.visibility)

        self.action_tilt = QtGui.QAction(self.parent)
        self.action_tilt.setText("Tilt")
        self.menu.addAction(self.action_tilt)
        self.action_tilt.triggered.connect(self.tilt)

        self.action_merge = QtGui.QAction(self.parent)
        self.action_merge.setText("Merge and Resampling")
        self.menu.addAction(self.action_merge)
        self.action_merge.triggered.connect(self.merge)

        self.action_reproj = QtGui.QAction(self.parent)
        self.action_reproj.setText("Reprojection")
        self.menu.addAction(self.action_reproj)
        self.action_reproj.triggered.connect(self.reproj)

        self.action_cut_data = QtGui.QAction(self.parent)
        self.action_cut_data.setText("Cut Data")
        self.menu.addAction(self.action_cut_data)
        self.action_cut_data.triggered.connect(self.cut_data)

        self.action_get_prof = QtGui.QAction(self.parent)
        self.action_get_prof.setText("Get Profile")
        self.menu.addAction(self.action_get_prof)
        self.action_get_prof.triggered.connect(self.get_prof)

        self.action_grid = QtGui.QAction(self.parent)
        self.action_grid.setText("Grid Point Data (Linear)")
        self.menu.addAction(self.action_grid)
        self.action_grid.triggered.connect(self.grid)

        self.action_igrf = QtGui.QAction(self.parent)
        self.action_igrf.setText("Calculate IGRF Corrected Data")
        self.menu.addAction(self.action_igrf)
        self.action_igrf.triggered.connect(self.igrf)

        self.menu.addSeparator()

        self.action_raster_data_interp = QtGui.QAction(self.parent)
        self.action_raster_data_interp.setText("Raster Data Interpretation")
        self.menu.addAction(self.action_raster_data_interp)
        self.action_raster_data_interp.triggered.connect(self.raster_interp)

# Context menus
        context_menu['Raster'].addSeparator()

        self.action_metadata = QtGui.QAction(self.parent)
        self.action_metadata.setText("Display/Edit Metadata")
        context_menu['Raster'].addAction(self.action_metadata)
        self.action_metadata.triggered.connect(self.metadata)

        self.action_basic_statistics = QtGui.QAction(self.parent)
        self.action_basic_statistics.setText("Basic Statistics")
        context_menu['Raster'].addAction(self.action_basic_statistics)
        self.action_basic_statistics.triggered.connect(self.basic_stats)

        self.action_show_raster_data = QtGui.QAction(self.parent)
        self.action_show_raster_data.setText("Show Raster Data")
        context_menu['Raster'].addAction(self.action_show_raster_data)
        self.action_show_raster_data.triggered.connect(self.show_raster_data)

        self.action_show_point_data = QtGui.QAction(self.parent)
        self.action_show_point_data.setText("Show Point Data")
        context_menu['Point'].addAction(self.action_show_point_data)
        self.action_show_point_data.triggered.connect(self.show_point_data)

        self.action_show_scatter_plot = QtGui.QAction(self.parent)
        self.action_show_scatter_plot.setText("Show Hexbin Plot")
        context_menu['Raster'].addAction(self.action_show_scatter_plot)
        self.action_show_scatter_plot.triggered.connect(self.show_scatter_plot)

        self.action_show_histogram = QtGui.QAction(self.parent)
        self.action_show_histogram.setText("Show Histogram")
        context_menu['Raster'].addAction(self.action_show_histogram)
        self.action_show_histogram.triggered.connect(self.show_histogram)

        self.action_show_2d_corr_coef = QtGui.QAction(self.parent)
        self.action_show_2d_corr_coef.setText(
            "Show 2D Correlation Coefficients")
        context_menu['Raster'].addAction(self.action_show_2d_corr_coef)
        self.action_show_2d_corr_coef.triggered.connect(self.show_ccoef)

        self.action_export_data = QtGui.QAction(self.parent)
        self.action_export_data.setText("Export Data")
        context_menu['Raster'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def metadata(self):
        """ Basic Statistics """
        self.parent.launch_context_item(dataprep.Metadata)

    def basic_stats(self):
        """ Basic Statistics """
        self.parent.launch_context_item(show_table.BasicStats)

    def equation_editor(self):
        """ Equation Editor """
        fnc = equation_editor.EquationEditor(self.parent)
        self.parent.item_insert("Step", "Equation\n  Editor", fnc)

    def export_data(self):
        """ Export raster data """
        self.parent.launch_context_item(iodefs.ExportData)

    def cut_data(self):
        """ Export raster data """
        fnc = dataprep.DataCut(self.parent)
        self.parent.item_insert("Step", "Cut\nData", fnc)

    def get_prof(self):
        """ Export raster data """
        fnc = dataprep.GetProf(self.parent)
        self.parent.item_insert("Step", "Get\nProfile", fnc)

    def gradients(self):
        """ Compute different gradients """
        fnc = cooper.Gradients(self.parent)
        self.parent.item_insert("Step", "Gradients", fnc)

    def norm_data(self):
        """ Normalisation of data"""
        fnc = normalisation.Normalisation(self.parent)
        self.parent.item_insert("Step", "Normalisation", fnc)

    def raster_interp(self):
        """ Show raster data """
        fnc = ginterp.PlotInterp(self.parent)
        self.parent.item_insert("Step", "Raster Data \nInterpretation", fnc)

    def show_ccoef(self):
        """ Show 2D correlation coefficients"""
        self.parent.launch_context_item(plot_graphs.PlotCCoef)

    def show_histogram(self):
        """ Show raster data """
        self.parent.launch_context_item(plot_graphs.PlotHist)

    def show_raster_data(self):
        """ Show raster data """
        self.parent.launch_context_item(plot_graphs.PlotRaster)

    def show_point_data(self):
        """ Show raster data """
        self.parent.launch_context_item(plot_graphs.PlotPoints)

    def show_scatter_plot(self):
        """ Show raster data """
        self.parent.launch_context_item(plot_graphs.PlotScatter)

    def smoothing(self):
        """ Smoothing of Data"""
        fnc = smooth.Smooth(self.parent)
        self.parent.item_insert("Step", "Smoothing", fnc)

    def tilt(self):
        """ Compute visibility """
        fnc = cooper.Tilt1(self.parent)
        self.parent.item_insert("Step", "Tilt\nAngle", fnc)

    def visibility(self):
        """ Compute visibility """
        fnc = cooper.Visibility2d(self.parent)
        self.parent.item_insert("Step", "Visibility", fnc)

    def reproj(self):
        """ Reproject a dataset """
        fnc = dataprep.DataReproj(self.parent)
        self.parent.item_insert("Step", "Data\nReprojection", fnc)

    def merge(self):
        """ Merge datasets """
        fnc = dataprep.DataMerge(self.parent)
        self.parent.item_insert("Step", "Data\nMerge", fnc)

    def grid(self):
        """ Grid datasets """
        fnc = dataprep.DataGrid(self.parent)
        self.parent.item_insert("Step", "Grid\nPoint Data", fnc)

    def igrf(self):
        """ Grid datasets """
        fnc = igrf.IGRF(self.parent)
        self.parent.item_insert("Step", "Remove\nIGRF", fnc)

    def import_data(self):
        """ Imports data"""
        fnc = iodefs.ImportData(self.parent)
        self.parent.item_insert("Io", "Import Data", fnc)
