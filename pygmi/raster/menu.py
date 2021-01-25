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
"""Raster Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.raster import smooth
from pygmi.raster import equation_editor
from pygmi.raster import normalisation
from pygmi.raster import cooper
from pygmi.raster import ginterp
from pygmi.raster import graphs
from pygmi.raster import show_table
from pygmi.raster import dataprep
from pygmi.raster import iodefs
from pygmi.raster import anaglyph


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Raster')
        self.parent.add_to_context('inRaster')
        context_menu = self.parent.context_menu

# Normal menus
        self.menu = QtWidgets.QMenu('Raster')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_data = QtWidgets.QAction('Import Raster Data')
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.action_import_rgb_data = QtWidgets.QAction('Import RGB Image')
        self.menu.addAction(self.action_import_rgb_data)
        self.action_import_rgb_data.triggered.connect(self.import_rgb_data)

        self.menu.addSeparator()

        self.action_equation_editor = QtWidgets.QAction('Equation Editor')
        self.menu.addAction(self.action_equation_editor)
        self.action_equation_editor.triggered.connect(self.equation_editor)

        self.action_smoothing = QtWidgets.QAction('Smoothing')
        self.menu.addAction(self.action_smoothing)
        self.action_smoothing.triggered.connect(self.smoothing)

        self.action_normalisation = QtWidgets.QAction('Normalisation')
        self.menu.addAction(self.action_normalisation)
        self.action_normalisation.triggered.connect(self.norm_data)

        self.action_gradients = QtWidgets.QAction('Gradients')
        self.menu.addAction(self.action_gradients)
        self.action_gradients.triggered.connect(self.gradients)

        self.action_visibility = QtWidgets.QAction('Visibility')
        self.menu.addAction(self.action_visibility)
        self.action_visibility.triggered.connect(self.visibility)

        self.action_cont = QtWidgets.QAction('Continuation')
        self.menu.addAction(self.action_cont)
        self.action_cont.triggered.connect(self.cont)

        self.menu.addSeparator()

        self.action_merge = QtWidgets.QAction('Merge and Resampling')
        self.menu.addAction(self.action_merge)
        self.action_merge.triggered.connect(self.merge)

        self.action_reproj = QtWidgets.QAction('Reprojection')
        self.menu.addAction(self.action_reproj)
        self.action_reproj.triggered.connect(self.reproj)

        self.action_cut_data = QtWidgets.QAction('Cut Raster using Polygon')
        self.menu.addAction(self.action_cut_data)
        self.action_cut_data.triggered.connect(self.cut_data)

        self.menu.addSeparator()

        self.action_get_prof = QtWidgets.QAction('Extract Profile from Raster')
        self.menu.addAction(self.action_get_prof)
        self.action_get_prof.triggered.connect(self.get_prof)

        self.menu.addSeparator()

        self.action_raster_data_interp = QtWidgets.QAction('Raster Data '
                                                           'Interpretation')
        self.menu.addAction(self.action_raster_data_interp)
        self.action_raster_data_interp.triggered.connect(self.raster_interp)

# Context menus
        context_menu['inRaster'].addSeparator()

        self.action_bandselect = QtWidgets.QAction('Select Input Raster Bands')
        context_menu['inRaster'].addAction(self.action_bandselect)
        self.action_bandselect.triggered.connect(self.bandselect)

        context_menu['Raster'].addSeparator()

        self.action_metadata = QtWidgets.QAction('Display/Edit Metadata')
        context_menu['Raster'].addAction(self.action_metadata)
        self.action_metadata.triggered.connect(self.metadata)

        self.action_basic_statistics = QtWidgets.QAction('Basic Statistics')
        context_menu['Raster'].addAction(self.action_basic_statistics)
        self.action_basic_statistics.triggered.connect(self.basic_stats)

        self.action_show_raster_data = QtWidgets.QAction('Show Raster Data')
        context_menu['Raster'].addAction(self.action_show_raster_data)
        self.action_show_raster_data.triggered.connect(self.show_raster_data)

        self.action_show_anaglyph = QtWidgets.QAction('Show Anaglyph')
        context_menu['Raster'].addAction(self.action_show_anaglyph)
        self.action_show_anaglyph.triggered.connect(self.show_anaglyph)

        self.action_show_surface = QtWidgets.QAction('Show Surface')
        context_menu['Raster'].addAction(self.action_show_surface)
        self.action_show_surface.triggered.connect(self.show_surface)

        self.action_show_scatter_plot = QtWidgets.QAction('Show Hexbin Plot')
        context_menu['Raster'].addAction(self.action_show_scatter_plot)
        self.action_show_scatter_plot.triggered.connect(self.show_scatter_plot)

        self.action_show_histogram = QtWidgets.QAction('Show Histogram')
        context_menu['Raster'].addAction(self.action_show_histogram)
        self.action_show_histogram.triggered.connect(self.show_histogram)

        self.action_show_2d_corr_coef = QtWidgets.QAction('Show 2D Correlation'
                                                          ' Coefficients')
        context_menu['Raster'].addAction(self.action_show_2d_corr_coef)
        self.action_show_2d_corr_coef.triggered.connect(self.show_ccoef)

        self.action_export_data = QtWidgets.QAction('Export Data')
        context_menu['Raster'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def metadata(self):
        """Metadata."""
        self.parent.launch_context_item(dataprep.Metadata)

    def basic_stats(self):
        """Display basic statistics."""
        self.parent.launch_context_item(show_table.BasicStats)

    def equation_editor(self):
        """Equation Editor."""
        self.parent.item_insert('Step', 'Equation Editor',
                                equation_editor.EquationEditor)

    def export_data(self):
        """Export raster data."""
        self.parent.launch_context_item(iodefs.ExportData)

    def cut_data(self):
        """Cut data."""
        self.parent.item_insert('Step', 'Cut Data', dataprep.DataCut)

    def get_prof(self):
        """Get profile."""
        self.parent.item_insert('Step', 'Get Profile', dataprep.GetProf)

    def gradients(self):
        """Compute different gradients."""
        self.parent.item_insert('Step', 'Gradients', cooper.Gradients)

    def norm_data(self):
        """Normalisation of data."""
        self.parent.item_insert('Step', 'Normalisation',
                                normalisation.Normalisation)

    def raster_interp(self):
        """Show raster data."""
        self.parent.item_insert('Step', 'Raster Data Interpretation',
                                ginterp.PlotInterp)

    def cont(self):
        """Compute Continuation."""
        self.parent.item_insert('Step', 'Continuation', dataprep.Continuation)

    def show_ccoef(self):
        """Show 2D correlation coefficients."""
        self.parent.launch_context_item(graphs.PlotCCoef)

    def show_histogram(self):
        """Show histogram of raster data."""
        self.parent.launch_context_item(graphs.PlotHist)

    def show_raster_data(self):
        """Show raster data."""
        self.parent.launch_context_item(graphs.PlotRaster)

    def show_anaglyph(self):
        """Show anaglyph of raster data."""
        self.parent.launch_context_item(anaglyph.PlotAnaglyph)

    def show_surface(self):
        """Show surface."""
        self.parent.launch_context_item(graphs.PlotSurface)

    def show_scatter_plot(self):
        """Show scatter plot."""
        self.parent.launch_context_item(graphs.PlotScatter)

    def smoothing(self):
        """Smoothing of Data."""
        self.parent.item_insert('Step', 'Smoothing', smooth.Smooth)

    def visibility(self):
        """Compute visibility."""
        self.parent.item_insert('Step', 'Visibility', cooper.Visibility2d)

    def reproj(self):
        """Reproject a dataset."""
        self.parent.item_insert('Step', 'Data Reprojection',
                                dataprep.DataReproj)

    def merge(self):
        """Merge datasets."""
        self.parent.item_insert('Step', 'Data Merge', dataprep.DataMerge)

    def import_data(self):
        """Import data."""
        self.parent.item_insert('Io', 'Import Data', iodefs.ImportData)

    def import_rgb_data(self):
        """Import RGB data."""
        self.parent.item_insert('Io', 'Import RGB Image', iodefs.ImportRGBData)

    def bandselect(self):
        """Select bands."""
        self.parent.launch_context_item_indata(iodefs.ComboBoxBasic)
