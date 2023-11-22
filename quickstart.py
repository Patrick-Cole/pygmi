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
"""
Quick start routine to start the GUI form of PyGMI.

This routine is used as a convenience function, typically if you do NOT
formally install PyGMI as a library and prefer to run it from within the
default extracted directory structure.
"""
import os
import sys

from pygmi.main import main

__version__ = '3.2.7.29'

# useful regex for find: \.(?!py) in spyder.ini


if __name__ == "__main__":
    ipth = os.path.dirname(__file__)+r'/pygmi/version.py'
    txt = f"__version__ = '{__version__}'"

    with open(ipth, 'r', encoding='utf-8') as io:
        vtxt = io.read()

    if vtxt != txt:
        with open(ipth, 'w', encoding='utf-8') as io:
            io.write(txt)

    nocgs = bool(len(sys.argv) == 2 and 'true' in sys.argv[1].lower())
    main(nocgs=nocgs)
