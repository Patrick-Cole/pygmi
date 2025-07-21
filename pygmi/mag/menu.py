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
"""Magnetic Menu Routines."""

from PySide6 import QtWidgets, QtGui

from pygmi.mag import dataprep
from pygmi.mag import igrf
from pygmi.mag import tiltdepth
from pygmi.mag import matchedfilt


class MenuWidget():
    """
    Widget class to call the main interface.

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to MainWidget class found in main.py. Default is None.

    """

    def __init__(self, parent=None):

        self.parent = parent
        self.parent.add_to_context('Raster')
        self.parent.add_to_context('inRaster')

        # Normal menus
        self.menu = QtWidgets.QMenu('Magnetics')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_tilt = QtGui.QAction('Tilt Angle and Related Edge '
                                         'Filters')
        self.menu.addAction(self.action_tilt)
        self.action_tilt.triggered.connect(self.tilt)

        self.action_rtp = QtGui.QAction('Reduction to the Pole')
        self.menu.addAction(self.action_rtp)
        self.action_rtp.triggered.connect(self.rtp)

        self.action_igrf = QtGui.QAction('Calculate IGRF Corrected Data')
        self.menu.addAction(self.action_igrf)
        self.action_igrf.triggered.connect(self.igrf)

        self.action_mfilt = QtGui.QAction('Matched Filtering')
        self.menu.addAction(self.action_mfilt)
        self.action_mfilt.triggered.connect(self.mfilt)

        self.action_depth_susc = QtGui.QAction('Tilt Depth Interpretation')
        self.menu.addAction(self.action_depth_susc)
        self.action_depth_susc.triggered.connect(self.depth_susc)

    def depth_susc(self):
        """Depth and Susceptibility calculations."""
        self.parent.item_insert('Step', 'Tilt Depth Interpretation',
                                tiltdepth.TiltDepth)

    def rtp(self):
        """Compute RTP."""
        self.parent.item_insert('Step', 'RTP', dataprep.RTP)

    def tilt(self):
        """Compute tilt angle."""
        self.parent.item_insert('Step', 'Tilt Angle', dataprep.Tilt1)

    def igrf(self):
        """Compute IGRF."""
        self.parent.item_insert('Step', 'Remove IGRF', igrf.IGRF)

    def mfilt(self):
        """Compute Matched Filtering."""
        self.parent.item_insert(
            'Step', 'Matched Filtering', matchedfilt.MatchedFilt)
