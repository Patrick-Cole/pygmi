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
""" Clustering Menu Routines """

from PyQt5 import QtWidgets
from pygmi.clust import cluster
from pygmi.clust import graphtool
from pygmi.clust import graphs
from pygmi.raster import show_table
from pygmi.raster import iodefs


class MenuWidget():
    """
    Widget class to call the main interface

    This widget class creates the clustering menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : MainWidget
        Reference to MainWidget class found in main.py
    """
    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Cluster')
        context_menu = self.parent.context_menu

# Normal menus
        self.menuclustering = QtWidgets.QMenu(parent.menubar)
        self.menuclustering.setTitle('Clustering')
        parent.menubar.addAction(self.menuclustering.menuAction())

        self.action_clustering = QtWidgets.QAction(self.parent)
        self.action_clustering.setText('Cluster Analysis')
        self.menuclustering.addAction(self.action_clustering)
        self.action_clustering.triggered.connect(self.cluster)

        self.menuclustering.addSeparator()

        self.action_scatter_plot = QtWidgets.QAction(self.parent)
        self.action_scatter_plot.setText('Scatter Plot Tool')
        self.menuclustering.addAction(self.action_scatter_plot)
        self.action_scatter_plot.triggered.connect(self.scatter_plot)

# Context menus
        context_menu['Cluster'].addSeparator()

        self.action_cluster_statistics = QtWidgets.QAction(self.parent)
        self.action_cluster_statistics.setText('Cluster Statistics')
        context_menu['Cluster'].addAction(self.action_cluster_statistics)
        self.action_cluster_statistics.triggered.connect(self.cluster_stats)

        self.action_show_class_data = QtWidgets.QAction(self.parent)
        self.action_show_class_data.setText('Show Class Data')
        context_menu['Cluster'].addAction(self.action_show_class_data)
        self.action_show_class_data.triggered.connect(self.show_raster_data)

        self.action_show_objvrcncexbigraphs = QtWidgets.QAction(self.parent)
        self.action_show_objvrcncexbigraphs.setText(
            'Show VRC Graphs')
        context_menu['Cluster'].addAction(self.action_show_objvrcncexbigraphs)
        self.action_show_objvrcncexbigraphs.triggered.connect(
            self.show_vrc_etc)

        self.action_export_data = QtWidgets.QAction(self.parent)
        self.action_export_data.setText('Export Data')
        context_menu['Cluster'].addAction(self.action_export_data)
        self.action_export_data.triggered.connect(self.export_data)

    def cluster_stats(self):
        """ Basic Statistics """
        self.parent.launch_context_item(show_table.ClusterStats)

    def cluster(self):
        """ Clustering of data"""
        fnc = cluster.Cluster(self.parent)
        self.parent.item_insert('Step', 'Cluster\nAnalysis', fnc)

    def export_data(self):
        """ Export raster data """
        self.parent.launch_context_item(iodefs.ExportData)

    def scatter_plot(self):
        """ Scatter Plot Tool"""
        fnc = graphtool.ScatterPlot(self.parent)
        self.parent.item_insert('Step', 'Scatter\nPlot\nTool', fnc)

    def show_raster_data(self):
        """ Show raster data """
        self.parent.launch_context_item(graphs.PlotRaster)

    def show_vrc_etc(self):
        """ Show vrc, xbi, obj, nce graphs """
        self.parent.launch_context_item(graphs.PlotVRCetc)
