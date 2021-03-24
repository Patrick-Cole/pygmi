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
from PyQt5 import QtWidgets, QtCore, QtGui


class FileMenu():
    """
    Widget class to call the main interface.

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : pygmi.main.MainWidget
        Reference to MainWidget class found in main.py
    """

    def __init__(self, parent=None):

        self.parent = parent

# File Menu

        self.menufile = QtWidgets.QMenu('File')
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_save = QtWidgets.QAction('Save Project')
        self.menufile.addAction(self.action_save)
        self.action_save.triggered.connect(parent.save)

        self.action_load = QtWidgets.QAction('Load Project')
        self.menufile.addAction(self.action_load)
        self.action_load.triggered.connect(parent.load)

        self.action_exit = QtWidgets.QAction('Exit')
        self.menufile.addAction(self.action_exit)
        self.action_exit.triggered.connect(parent.close)


class HelpMenu():
    """
    Widget class to call the main interface.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):

        self.parent = parent
        self.webpage = r'http://patrick-cole.github.io/pygmi/'

        self.menuhelp = QtWidgets.QMenu('Help')
        parent.menubar.addAction(self.menuhelp.menuAction())

        self.action_help = QtWidgets.QAction('Help')
        self.action_about = QtWidgets.QAction('About')

        self.menuhelp.addAction(self.action_help)
        self.menuhelp.addAction(self.action_about)

        self.action_about.triggered.connect(self.about)
        self.action_help.triggered.connect(self.webhelp)

    def about(self):
        """About box for PyGMI."""

        year = str(date.today().year)

        # msg = ('Name:\t\tPyGMI - Python Geoscience Modelling\n\t\tand '
        #        'Interpretation\n'
        #        'Version:\t\t'+self.parent.__version__+'\n'
        #        'Author:\t\tPatrick Cole\n'
        #        'E-Mail:\t\tpcole@geoscience.org.za\n'
        #        'Copyright:\t\xa9 2013-'+year+' Council for Geoscience\n'
        #        'Licence:\t\tGPL-3.0\n\n'
        #        'PyGMI is free software: you can redistribute it and/or '
        #        'modify it under the terms of the GNU General Public License '
        #        'as published by the Free Software Foundation, either version '
        #        '3 of the License, or (at your option) any later version.\n\n'
        #        'PyGMI is distributed in the hope that it will be useful, '
        #        'but WITHOUT ANY WARRANTY; without even the implied warranty '
        #        'of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. '
        #        'See the GNU General Public License for more details.\n\n'
        #        'You should have received a copy of the GNU General Public '
        #        'License along with this program. If not, see\n'
        #        'http://www.gnu.org/licenses/')

        msg = ('<p>Name: PyGMI - Python Geoscience Modelling and Interpretation</p>'
               'Version: '+self.parent.__version__+'<br>'
               'Author: Patrick Cole<br>'
               'E-Mail: pcole@geoscience.org.za<br>'
               'Copyright: \xa9 2013-' + year +
               ' <a href="https://www.geoscience.org.za/">Council for Geoscience</a><br>'
               'Licence: <a href="http://www.gnu.org/licenses/gpl-3.0.html">GPL-3.0</a></p>'
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
               '<a href="http://www.gnu.org/licenses">http://www.gnu.org/licenses </a></p>')

        ipth = os.path.dirname(__file__)+r'/images/'

        msg += ('<p style="text-align:right"></p><img alt="CGS Logo" '
                'src="'+ipth+'cgslogo.png"')

        QtWidgets.QMessageBox.about(self.parent, 'PyGMI', msg)

    def webhelp(self):
        """Help File."""
        webbrowser.open(self.webpage)


class HelpButton(QtWidgets.QPushButton):
    """
    Help Button.

    Convenience class to add an image to a pushbutton
    """

    def __init__(self, htmlfile=None, parent=None):
        super().__init__(parent)

        self.htmlfile = htmlfile

        self.ipth = os.path.dirname(__file__)+r'/images/'
        self.setMinimumHeight(48)
        self.setMinimumWidth(48)

        self.setIcon(QtGui.QIcon(self.ipth+'help.png'))
        self.setIconSize(self.minimumSize())
        self.clicked.connect(self.help_docs)
        self.setFlat(True)

    def help_docs(self):
        """Help Routine."""
        HelpDocs(self, self.htmlfile)


class HelpDocs(QtWidgets.QDialog):
    """
    A basic combo box application.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None, helptxt=None):
        super().__init__(parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}

        ipth = os.path.dirname(__file__)+r'/helpdocs/'
        opth = os.getcwd()

        if helptxt is None:
            helptxt = 'No Help Available.'
        else:
            os.chdir(ipth)
            with open(helptxt+'.html') as itxt:
                helptxt = itxt.read()

        # create GUI
        self.setWindowTitle('Help!')

        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)

        self.text = QtWidgets.QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.append(helptxt)
        self.text.setMinimumWidth(480)
        self.text.setMinimumHeight(360)
        self.text.setFrameShape(QtWidgets.QFrame.NoFrame)
        cursor = QtGui.QTextCursor()
        cursor.setPosition(0)
        self.text.setTextCursor(cursor)

        ptmp = self.text.palette()
        ptmp.setColor(0, 9, ptmp.color(10))
        ptmp.setColor(1, 9, ptmp.color(10))
        ptmp.setColor(2, 9, ptmp.color(10))
        self.text.setPalette(ptmp)

        self.vbox.addWidget(self.text)

        self.buttonbox = QtWidgets.QDialogButtonBox()
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)

        self.vbox.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)

        self.exec_()

        os.chdir(opth)
