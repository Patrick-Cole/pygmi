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
"""Magnetic menu routines."""

from PySide6 import QtGui, QtWidgets

from pygmi.maggrv import dataprep, gravproc, igrf, iodefs, matchedfilt, tiltdepth


class MenuWidget:
    """
    Widget class to call the main interface.

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Parameters
    ----------
    parent
        Reference to MainWidget class found in main.py. Default is None.

    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context("Raster")

        # Normal menus
        self.menu = QtWidgets.QMenu("Magnetics and Gravity")
        parent.menubar.addAction(self.menu.menuAction())

        self.action_rtp = QtGui.QAction("Reduction to the Pole")
        self.menu.addAction(self.action_rtp)
        self.action_rtp.triggered.connect(self.rtp)

        self.action_igrf = QtGui.QAction("Calculate IGRF Corrected Data")
        self.menu.addAction(self.action_igrf)
        self.action_igrf.triggered.connect(self.igrf)

        self.menu.addSeparator()

        self.action_asig = QtGui.QAction("Analytic Signal")
        self.menu.addAction(self.action_asig)
        self.action_asig.triggered.connect(self.asig)

        self.action_cont = QtGui.QAction("Continuation")
        self.menu.addAction(self.action_cont)
        self.action_cont.triggered.connect(self.cont)

        self.action_tilt = QtGui.QAction("Tilt Angle and Related Edge Filters")
        self.menu.addAction(self.action_tilt)
        self.action_tilt.triggered.connect(self.tilt)

        self.action_mfilt = QtGui.QAction("Matched Filtering")
        self.menu.addAction(self.action_mfilt)
        self.action_mfilt.triggered.connect(self.mfilt)

        self.action_depth_susc = QtGui.QAction("Tilt Depth Interpretation")
        self.menu.addAction(self.action_depth_susc)
        self.action_depth_susc.triggered.connect(self.depth_susc)

        self.menu.addSeparator()

        self.action_import_data = QtGui.QAction("Import CG-5 or CG-6 Data")
        self.menu.addAction(self.action_import_data)
        self.action_import_data.triggered.connect(self.import_data)

        self.action_process = QtGui.QAction("Process Gravity Data")
        self.menu.addAction(self.action_process)
        self.action_process.triggered.connect(self.process_data)

    def cont(self):
        """Compute Continuation."""
        self.parent.item_insert("Step", "Continuation", dataprep.Continuation)

    def depth_susc(self):
        """Depth and Susceptibility calculations."""
        self.parent.item_insert(
            "Step", "Tilt Depth Interpretation", tiltdepth.TiltDepth
        )

    def rtp(self):
        """Compute RTP."""
        self.parent.item_insert("Step", "RTP", dataprep.RTP)

    def tilt(self):
        """Compute tilt angle."""
        self.parent.item_insert("Step", "Tilt Angle", dataprep.Tilt1)

    def asig(self):
        """Compute analytic signal."""
        self.parent.item_insert("Step", "Analytic Signal", dataprep.ASig)

    def igrf(self):
        """Compute IGRF."""
        self.parent.item_insert("Step", "Remove IGRF", igrf.IGRF)

    def mfilt(self):
        """Compute Matched Filtering."""
        self.parent.item_insert("Step", "Matched Filtering", matchedfilt.MatchedFilt)

    def import_data(self):
        """Import data."""
        self.parent.item_insert("Io", "Import CG-5 or CG-6 Data", iodefs.ImportCG5)

    def process_data(self):
        """Process data."""
        self.parent.item_insert("Step", "Process Gravity Data", gravproc.ProcessData)
