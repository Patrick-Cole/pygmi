# -----------------------------------------------------------------------------
# Name:        quickstart.py (part of PyGMI)
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
""" This is a quick start routine to start the GUI form of PyGMI

This routine is used as a convenience function, typically if you do NOT
formally install PyGMI as a library and prefer to run it from within the
default extracted directory structure.
"""

import traceback
import sys
from PyQt5 import QtCore
from pygmi.main import main


def excepthook(type_, value, traceback_):
    """ Excepthook bugfix for Qt5 debugging """
    traceback.print_exception(type_, value, traceback_)
    QtCore.qFatal('')

if __name__ == "__main__":
    sys.excepthook = excepthook
    main()
