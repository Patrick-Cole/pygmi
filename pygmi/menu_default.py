# -----------------------------------------------------------------------------
# Name:        menu_default.py (part of PyGMI)
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
Default set of menus for the main interface.

It also includes the about box.
"""

from datetime import date
import os
import webbrowser
from PySide6 import QtWidgets, QtGui


class FileMenu():
    """
    Widget class to call the main interface.

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to MainWidget class found in main.py. Default is None.

    """

    def __init__(self, parent=None):

        self.parent = parent

# File Menu

        self.menu = QtWidgets.QMenu('File')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_save = QtGui.QAction('Save Project')
        self.menu.addAction(self.action_save)
        self.action_save.triggered.connect(parent.save)

        self.action_load = QtGui.QAction('Load Project')
        self.menu.addAction(self.action_load)
        self.action_load.triggered.connect(parent.load)

        self.action_exit = QtGui.QAction('Exit')
        self.menu.addAction(self.action_exit)
        self.action_exit.triggered.connect(parent.close)


class HelpMenu():
    """
    Widget class to call the main interface.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):

        self.parent = parent
        self.webpage = r'http://patrick-cole.github.io/pygmi/'

        self.menu = QtWidgets.QMenu('Help')
        parent.menubar.addAction(self.menu.menuAction())

        self.action_help = QtGui.QAction('Help')
        self.action_about = QtGui.QAction('About')

        self.menu.addAction(self.action_help)
        self.menu.addAction(self.action_about)

        self.action_about.triggered.connect(self.about)
        self.action_help.triggered.connect(self.webhelp)

    def about(self):
        """About box for PyGMI."""
        year = str(date.today().year)

        msg = ('<p>Name: PyGMI - Python Geoscience Modelling and '
               'Interpretation</p>'
               'Version: ' + self.parent.__version__ + '<br>'
               'Author: Patrick Cole<br>'
               'E-Mail: pcole@geoscience.org.za<br>'
               'Copyright: \xa9 2013-' + year +
               ' <a href="https://www.geoscience.org.za/">'
               'Council for Geoscience</a><br>'
               'Licence: <a href="http://www.gnu.org/licenses/gpl-3.0.html">'
               'GPL-3.0</a></p>'
               '<p>PyGMI is free software: you can redistribute it and/or '
               'modify it under the terms of the GNU General Public License '
               'as published by the Free Software Foundation, either version '
               '3 of the License, or (at your option) any later version.</p>'
               '<p>PyGMI is distributed in the hope that it will be useful, '
               'but WITHOUT ANY WARRANTY; without even the implied warranty '
               'of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. '
               'See the GNU General Public License for more details.</p>'
               '<p>You should have received a copy of the GNU General Public '
               'License along with this program. If not, see '
               '<a href="http://www.gnu.org/licenses">'
               'http://www.gnu.org/licenses </a></p>')

        ipth = os.path.dirname(__file__) + r'/images/'

        msg += ('<p style="text-align:right"></p><img alt="CGS Logo" '
                'src="' + ipth + 'cgslogo.png"')

        QtWidgets.QMessageBox.about(self.parent, 'PyGMI', msg)

    def webhelp(self):
        """Help File."""
        ipth = os.path.dirname(__file__) + r'//helpdocs//html//wiki.html'
        webbrowser.open(ipth)


class HelpButton(QtWidgets.QPushButton):
    """
    Help Button.

    Convenience class to add a Help image to a pushbutton

    Parameters
    ----------
    htmlfile : str
        HTML help file name.
    """

    def __init__(self, htmlfile=None):
        super().__init__()

        self.htmlfile = htmlfile

        self.setMinimumHeight(32)
        self.setMinimumWidth(52)

        ipth = os.path.dirname(__file__) + r'/images/'
        self.setIcon(QtGui.QIcon(ipth + 'help.png'))
        self.setIconSize(self.minimumSize())
        self.clicked.connect(self.help_docs)
        self.setFlat(True)

    def help_docs(self):
        """Help Routine."""
        if self.htmlfile is not None:
            ipth = os.path.dirname(__file__) + r'//helpdocs//html'
            hfile = os.path.join(ipth, self.htmlfile + '.html')
            webbrowser.open(hfile)
