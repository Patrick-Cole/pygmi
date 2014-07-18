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
""" File Menu Routines """

# pylint: disable=E1101, C0103
from PySide import QtGui, QtCore
import webbrowser


class FileMenu(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):

        self.parent = parent

# File Menu

        self.menufile = QtGui.QMenu(parent.menubar)
        self.menufile.setTitle("File")
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_open_project = QtGui.QAction(parent)
        self.action_open_project.setText("Open Project")
#        self.menufile.addAction(self.action_open_project)

        self.action_save_project = QtGui.QAction(parent)
        self.action_save_project.setText("Save Project")
#        self.menufile.addAction(self.action_save_project)

        self.action_exit = QtGui.QAction(parent)
#        self.menufile.addSeparator()
        self.action_exit.setText("Exit")
        self.menufile.addAction(self.action_exit)

        QtCore.QObject.connect(self.action_exit, QtCore.SIGNAL("triggered()"),
                               parent.close)


class HelpMenu(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):

        self.parent = parent

# Help Menu

        self.menuhelp = QtGui.QMenu(parent.menubar)
        parent.menubar.addAction(self.menuhelp.menuAction())

        self.action_help = QtGui.QAction(self.parent)
        self.action_about = QtGui.QAction(self.parent)

        self.menuhelp.addAction(self.action_help)
        self.menuhelp.addAction(self.action_about)

        self.menuhelp.setTitle("Help")
        self.action_help.setText("Help")
        self.action_about.setText("About")

        self.action_about.triggered.connect(self.about)
        self.action_help.triggered.connect(self.help)

    def about(self):
        """ About Box """

        msg = (' Name:         PyGMI - Python Geophysical Modelling and '
               'Interpretation \r\n'
               ' Author:       Patrick Cole\r\n'
               ' E-Mail:        pcole@geoscience.org.za\r\n'
               '\r\n'
               ' Copyright:    (c) 2013 Council for Geoscience\r\n'
               ' Licence:       GPL-3.0\r\n'
               '\r\n'
               ' PyGMI is free software: you can redistribute it and/or '
               'modify\r\n'
               ' it under the terms of the GNU General Public License as '
               'published by\r\n'
               ' the Free Software Foundation, either version 3 of the '
               'License, or\r\n'
               ' (at your option) any later version.\r\n'
               '\r\n'
               ' PyGMI is distributed in the hope that it will be '
               'useful,\r\n'
               ' but WITHOUT ANY WARRANTY; without even the implied '
               'warranty of\r\n'
               ' MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  '
               'See the\r\n'
               ' GNU General Public License for more details.\r\n'
               '\r\n'
               ' You should have received a copy of the GNU General '
               'Public License\r\n'
               ' along with this program.  If not, see '
               'http://www.gnu.org/licenses/')

        QtGui.QMessageBox.about(self.parent, 'PyGMI', msg)

    def help(self):
        """ Help File"""
        webbrowser.open(r'https://code.google.com/p/pygmi/wiki/' +
                        'TableOfContents')
