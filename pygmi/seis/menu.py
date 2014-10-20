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
""" Seis Menu Routines """

from PyQt4 import QtGui
from . import scan_imp
from . import del_rec
from . import iodefs
from . import beachball


class MenuWidget(object):
    """
    Widget class to call the main interface

    This widget class creates the seismology menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : MainWidget
        Reference to MainWidget class found in main.py
    """
    def __init__(self, parent):

        self.parent = parent
        self.parent.add_to_context('Seis')
        context_menu = self.parent.context_menu

        self.menufile = QtGui.QMenu(parent.menubar)
        self.menufile.setTitle("Seismology")
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_import_seisan = QtGui.QAction(parent)
        self.action_import_seisan.setText("Import Seisan Data")
        self.menufile.addAction(self.action_import_seisan)
        self.action_import_seisan.triggered.connect(self.import_seisan)

        self.action_import_genfps = QtGui.QAction(parent)
        self.action_import_genfps.setText("Import Generic FPS")
        self.menufile.addAction(self.action_import_genfps)
        self.action_import_genfps.triggered.connect(self.import_genfps)

        self.action_beachball = QtGui.QAction(self.parent)
        self.action_beachball.setText("Fault Plane Solutions")
        self.menufile.addAction(self.action_beachball)
        self.action_beachball.triggered.connect(self.beachball)

        self.action_import_scans = QtGui.QAction(parent)
        self.action_import_scans.setText("Import Scanned Bulletins")
        self.menufile.addAction(self.action_import_scans)
        self.action_import_scans.triggered.connect(self.import_scans)

        self.action_delete_recs = QtGui.QAction(parent)
        self.action_delete_recs.setText("Delete Records")
        self.menufile.addAction(self.action_delete_recs)
        self.action_delete_recs.triggered.connect(self.delete_recs)

        self.action_quarry = QtGui.QAction(parent)
        self.action_quarry.setText("Remove Quarry Events")
        self.menufile.addAction(self.action_quarry)
        self.action_quarry.triggered.connect(self.quarry)

        self.menufile.addSeparator()

    def beachball(self):
        """ Create Beachballs from Fault Plane Solutions """
        fnc = beachball.BeachBall(self.parent)
        self.parent.item_insert("Step", "Fault\nPlane\nSolutions", fnc)

    def import_scans(self):
        """ Imports scanned records"""
        fnc = scan_imp.SIMP(self.parent)
        self.parent.item_insert("Io", "Import\nScanned\nBulletins", fnc)

    def import_seisan(self):
        """ Imports Seisan"""
        fnc = iodefs.ImportSeisan(self.parent)
        self.parent.item_insert("Io", "Import\nSeisan\nData", fnc)

    def import_genfps(self):
        """ Imports Generic Fault Plane Solution"""
        fnc = iodefs.ImportGenericFPS(self.parent)
        self.parent.item_insert("Io", "Import\nGeneric\nFPS", fnc)

    def delete_recs(self):
        """ Deletes Records """
        fnc = del_rec.DeleteRecord(self.parent)
        self.parent.item_insert("Io", "Delete Records", fnc)

    def quarry(self):
        """ Removes quarry events """
        fnc = del_rec.Quarry(self.parent)
        self.parent.item_insert("Io", "Remove\nQuarry\nEvents", fnc)
