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
"""Clustering Menu Routines."""

from PySide6 import QtWidgets, QtGui

from pygmi.clust import cluster
from pygmi.clust import graphtool
from pygmi.clust import graphs
from pygmi.raster import show_table
from pygmi.raster import iodefs
from pygmi.clust import crisp_clust
from pygmi.clust import fuzzy_clust
from pygmi.clust import super_class
from pygmi.clust import segmentation


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the clustering menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to MainWidget class found in main.py. Default is None.

    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Cluster')
        self.parent.add_to_context('memCluster')
        self.parent.add_to_context('objCluster')
        context_menu = self.parent.context_menu

        # Normal menus
        self.menu = QtWidgets.QMenu('Classification')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_clustering = QtGui.QAction('Cluster Analysis')
        self.menu.addAction(self.action_clustering)
        self.action_clustering.triggered.connect(self.cluster)

        self.action_crisp_clustering = QtGui.QAction("Crisp Clustering")
        self.menu.addAction(self.action_crisp_clustering)
        self.action_crisp_clustering.triggered.connect(self.crisp_cluster)

        self.action_fuzzy_clustering = QtGui.QAction("Fuzzy Clustering")
        self.menu.addAction(self.action_fuzzy_clustering)
        self.action_fuzzy_clustering.triggered.connect(self.fuzzy_cluster)

        self.menu.addSeparator()

        self.action_segmentation = QtGui.QAction('Image Segmentation')
        self.menu.addAction(self.action_segmentation)
        self.action_segmentation.triggered.connect(self.segmentation)

        self.action_super_class = QtGui.QAction("Supervised Classification")
        self.menu.addAction(self.action_super_class)
        self.action_super_class.triggered.connect(self.super_class)

        self.menu.addSeparator()

        self.action_scatter_plot = QtGui.QAction('Scatter Plot Tool')
        self.menu.addAction(self.action_scatter_plot)
        self.action_scatter_plot.triggered.connect(self.scatter_plot)

# Context menus
        context_menu['Cluster'].addSeparator()

        self.action_cluster_statistics = QtGui.QAction('Cluster Statistics')
        context_menu['Cluster'].addAction(self.action_cluster_statistics)
        self.action_cluster_statistics.triggered.connect(self.cluster_stats)

        self.action_show_class_data = QtGui.QAction('Show Class Data')
        context_menu['Cluster'].addAction(self.action_show_class_data)
        self.action_show_class_data.triggered.connect(self.show_raster_data)

        self.action_show_class_range = QtGui.QAction('Show Class Data Ranges')
        context_menu['Cluster'].addAction(self.action_show_class_range)
        self.action_show_class_range.triggered.connect(self.show_class_range)

        self.action_show_membership_data = QtGui.QAction("Show Membership "
                                                         "Data")
        context_menu['memCluster'].addAction(self.action_show_membership_data)
        self.action_show_membership_data.triggered.connect(
            self.show_membership_data)

        self.action_show_objvrcncexbigraphs = QtGui.QAction("Show OBJ, "
                                                            "VRC, NCE, "
                                                            "XBI Graphs")
        context_menu['objCluster'].addAction(
            self.action_show_objvrcncexbigraphs)
        self.action_show_objvrcncexbigraphs.triggered.connect(
            self.show_vrc_etc)

        self.action_export_data = QtGui.QAction('Export Class Data')
        context_menu['Cluster'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def cluster_stats(self):
        """Calculate Statistics."""
        self.parent.launch_context_item(show_table.ClusterStats)

    def cluster(self):
        """Clustering of data."""
        self.parent.item_insert('Step', 'Cluster Analysis', cluster.Cluster)

    def crisp_cluster(self):
        """Crisp Clustering of data."""
        self.parent.item_insert('Step', 'Crisp Clustering',
                                crisp_clust.CrispClust)

    def fuzzy_cluster(self):
        """Fuzzy Clustering of data."""
        self.parent.item_insert('Step', 'Fuzzy Clustering',
                                fuzzy_clust.FuzzyClust)

    def super_class(self):
        """Supervised Classification."""
        self.parent.item_insert('Step', 'Supervised Classification',
                                super_class.SuperClass)

    def export_data(self):
        """Export raster data."""
        self.parent.launch_context_item(iodefs.ExportData, 'class')

    def scatter_plot(self):
        """Scatter Plot Tool."""
        self.parent.item_insert('Step', 'Scatter Plot Tool',
                                graphtool.ScatterPlot)

    def show_raster_data(self):
        """Show class data."""
        self.parent.launch_context_item(graphs.PlotRaster)

    def show_class_range(self):
        """Show class ranges."""
        self.parent.launch_context_item(graphs.PlotBars)

    def show_membership_data(self):
        """Show membership data."""
        self.parent.launch_context_item(graphs.PlotMembership)

    def show_vrc_etc(self):
        """Show vrc, xbi, obj, nce graphs."""
        self.parent.launch_context_item(graphs.PlotVRCetc)

    def segmentation(self):
        """Image Segmentation."""
        self.parent.item_insert('Step', 'Image Segmentation',
                                segmentation.ImageSeg)
