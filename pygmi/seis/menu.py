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
"""Seis Menu Routines."""

from PyQt5 import QtWidgets
from pygmi.seis import scan_imp
from pygmi.seis import del_rec
from pygmi.seis import iodefs
from pygmi.seis import beachball
from pygmi.seis import graphs
from pygmi.seis import utils


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the seismology menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Seis')
        context_menu = self.parent.context_menu

        self.menu = QtWidgets.QMenu('Seismology')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_import_seisan = QtWidgets.QAction('Import Seisan Data')
        self.menu.addAction(self.action_import_seisan)
        self.action_import_seisan.triggered.connect(self.import_seisan)

        self.action_check_desc = QtWidgets.QAction('Correct Seisan Type 3'
                                                   ' Descriptions')
        self.menu.addAction(self.action_check_desc)
        self.action_check_desc.triggered.connect(self.correct_desc)

        self.action_filter_seisan = QtWidgets.QAction('Filter Seisan Data')
        self.menu.addAction(self.action_filter_seisan)
        self.action_filter_seisan.triggered.connect(self.filter_seisan)

        self.action_import_genfps = QtWidgets.QAction('Import Generic FPS')
        self.menu.addAction(self.action_import_genfps)
        self.action_import_genfps.triggered.connect(self.import_genfps)

#        self.action_import_scans = QtWidgets.QAction('Import Scanned Bulletins')
#        self.menu.addAction(self.action_import_scans)
#        self.action_import_scans.triggered.connect(self.import_scans)

        self.menu.addSeparator()

        self.action_beachball = QtWidgets.QAction('Fault Plane Solutions')
        self.menu.addAction(self.action_beachball)
        self.action_beachball.triggered.connect(self.beachball)

#        self.action_delete_recs = QtWidgets.QAction('Delete Records')
#        self.menu.addAction(self.action_delete_recs)
#        self.action_delete_recs.triggered.connect(self.delete_recs)

        self.action_quarry = QtWidgets.QAction('Remove Quarry Events')
        self.menu.addAction(self.action_quarry)
        self.action_quarry.triggered.connect(self.quarry)

# Context menus

        context_menu['Seis'].addSeparator()

        self.action_show_QC_plots = QtWidgets.QAction('Show QC Plots')
        context_menu['Seis'].addAction(self.action_show_QC_plots)
        self.action_show_QC_plots.triggered.connect(self.show_QC_plots)

        self.action_export_seisan = QtWidgets.QAction('Export Seisan Data')
        context_menu['Seis'].addAction(self.action_export_seisan)
        self.action_export_seisan.triggered.connect(self.export_seisan)

        self.action_export_csv = QtWidgets.QAction('Export to CSV')
        context_menu['Seis'].addAction(self.action_export_csv)
        self.action_export_csv.triggered.connect(self.export_csv)

        self.action_sexport_csv = QtWidgets.QAction('Export Summary to CSV')
        context_menu['Seis'].addAction(self.action_sexport_csv)
        self.action_sexport_csv.triggered.connect(self.sexport_csv)

    def export_seisan(self):
        """Export Seisan data."""
        self.parent.launch_context_item(iodefs.ExportSeisan)

    def export_csv(self):
        """Export Seisan data to csv."""
        self.parent.launch_context_item(iodefs.ExportCSV)

    def sexport_csv(self):
        """Export Summary data to csv."""
        self.parent.launch_context_item(iodefs.ExportSummaryCSV)

    def beachball(self):
        """Create Beachballs from Fault Plane Solutions."""
        fnc = beachball.BeachBall(self.parent)
        self.parent.item_insert('Step', 'Fault Plane Solutions', fnc)

    def import_scans(self):
        """Import scanned records."""
        fnc = scan_imp.SIMP(self.parent)
        self.parent.item_insert('Io', 'Import Scanned Bulletins', fnc)

    def import_seisan(self):
        """Import Seisan."""
        fnc = iodefs.ImportSeisan(self.parent)
        self.parent.item_insert('Io', 'Import Seisan Data', fnc)

    def correct_desc(self):
        """Correct Seisan descriptions."""
        fnc = utils.CorrectDescriptions(self.parent)
        self.parent.item_insert('Step', 'Correct Seisan Descriptions', fnc)

    def filter_seisan(self):
        """Filter Seisan."""
        fnc = iodefs.FilterSeisan(self.parent)
        self.parent.item_insert('Step', 'Filter Seisan Data', fnc)

    def import_genfps(self):
        """Import Generic Fault Plane Solution."""
        fnc = iodefs.ImportGenericFPS(self.parent)
        self.parent.item_insert('Io', 'Import Generic FPS', fnc)

    def delete_recs(self):
        """Delete Records."""
        fnc = del_rec.DeleteRecord(self.parent)
        self.parent.item_insert('Io', 'Delete Records', fnc)

    def quarry(self):
        """Remove quarry events."""
        fnc = del_rec.Quarry(self.parent)
        self.parent.item_insert('Step', 'Remove Quarry Events', fnc)

    def show_QC_plots(self):
        """Show QC plots."""
        self.parent.launch_context_item(graphs.PlotQC)
