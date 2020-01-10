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
from pygmi.raster import igrf
from pygmi.raster import tiltdepth
from pygmi.raster import anaglyph


class MenuWidget():
    """
    Widget class to call the main interface

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """
    def __init__(self, parent):

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

        self.action_tilt = QtWidgets.QAction('Tilt Angle')
        self.menu.addAction(self.action_tilt)
        self.action_tilt.triggered.connect(self.tilt)

        self.action_rtp = QtWidgets.QAction('Reduction to the Pole')
        self.menu.addAction(self.action_rtp)
        self.action_rtp.triggered.connect(self.rtp)

        self.action_igrf = QtWidgets.QAction('Calculate IGRF Corrected Data')
        self.menu.addAction(self.action_igrf)
        self.action_igrf.triggered.connect(self.igrf)

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

        self.action_raster_data_interp = QtWidgets.QAction('Raster Data Interpretation')
        self.menu.addAction(self.action_raster_data_interp)
        self.action_raster_data_interp.triggered.connect(self.raster_interp)

        self.action_depth_susc = QtWidgets.QAction('Tilt Depth Interpretation')
        self.menu.addAction(self.action_depth_susc)
        self.action_depth_susc.triggered.connect(self.depth_susc)

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

        self.action_show_2d_corr_coef = QtWidgets.QAction('Show 2D Correlation Coefficients')
        context_menu['Raster'].addAction(self.action_show_2d_corr_coef)
        self.action_show_2d_corr_coef.triggered.connect(self.show_ccoef)

        self.action_export_data = QtWidgets.QAction('Export Data')
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
        self.parent.item_insert('Step', 'Equation\n  Editor', fnc)

    def export_data(self):
        """ Export raster data """
        self.parent.launch_context_item(iodefs.ExportData)

    def cut_data(self):
        """ Export raster data """
        fnc = dataprep.DataCut(self.parent)
        self.parent.item_insert('Step', 'Cut\nData', fnc)

    def get_prof(self):
        """ Export raster data """
        fnc = dataprep.GetProf(self.parent)
        self.parent.item_insert('Step', 'Get\nProfile', fnc)

    def gradients(self):
        """ Compute different gradients """
        fnc = cooper.Gradients(self.parent)
        self.parent.item_insert('Step', 'Gradients', fnc)

    def norm_data(self):
        """ Normalisation of data"""
        fnc = normalisation.Normalisation(self.parent)
        self.parent.item_insert('Step', 'Normalisation', fnc)

    def raster_interp(self):
        """ Show raster data """
        fnc = ginterp.PlotInterp(self.parent)
        self.parent.item_insert('Step', 'Raster Data \nInterpretation', fnc)

    def depth_susc(self):
        """ Depth and Susceptibility calculations """
        fnc = tiltdepth.TiltDepth(self.parent)
        self.parent.item_insert('Step',
                                'Tilt\nDepth\nInterpretation', fnc)

    def rtp(self):
        """ Compute rtp """
        fnc = dataprep.RTP(self.parent)
        self.parent.item_insert('Step', 'RTP\nAngle', fnc)

    def show_ccoef(self):
        """ Show 2D correlation coefficients"""
        self.parent.launch_context_item(graphs.PlotCCoef)

    def show_histogram(self):
        """ Show raster data """
        self.parent.launch_context_item(graphs.PlotHist)

    def show_raster_data(self):
        """ Show raster data """
        self.parent.launch_context_item(graphs.PlotRaster)

    def show_anaglyph(self):
        """ Show raster data """
        self.parent.launch_context_item(anaglyph.PlotAnaglyph)

    def show_surface(self):
        """ Show surface """
        self.parent.launch_context_item(graphs.PlotSurface)

    def show_scatter_plot(self):
        """ Show raster data """
        self.parent.launch_context_item(graphs.PlotScatter)

    def smoothing(self):
        """ Smoothing of Data"""
        fnc = smooth.Smooth(self.parent)
        self.parent.item_insert('Step', 'Smoothing', fnc)

    def tilt(self):
        """ Compute visibility """
        fnc = cooper.Tilt1(self.parent)
        self.parent.item_insert('Step', 'Tilt\nAngle', fnc)

    def visibility(self):
        """ Compute visibility """
        fnc = cooper.Visibility2d(self.parent)
        self.parent.item_insert('Step', 'Visibility', fnc)

    def reproj(self):
        """ Reproject a dataset """
        fnc = dataprep.DataReproj(self.parent)
        self.parent.item_insert('Step', 'Data\nReprojection', fnc)

    def merge(self):
        """ Merge datasets """
        fnc = dataprep.DataMerge(self.parent)
        self.parent.item_insert('Step', 'Data\nMerge', fnc)

    def igrf(self):
        """ Grid datasets """
        fnc = igrf.IGRF(self.parent)
        self.parent.item_insert('Step', 'Remove\nIGRF', fnc)

    def import_data(self):
        """ Imports data"""
        fnc = iodefs.ImportData(self.parent)
        self.parent.item_insert('Io', 'Import Data', fnc)

    def import_rgb_data(self):
        """ Imports data"""
        fnc = iodefs.ImportRGBData(self.parent)
        self.parent.item_insert('Io', 'Import RGB Image', fnc)

    def bandselect(self):
        """Select bands."""
        self.parent.launch_context_item_indata(iodefs.ComboBoxBasic)

