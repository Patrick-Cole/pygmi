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
""" This is the default set of menus for the main interface. It also includes
the about box """

import os
import webbrowser
from PyQt4 import QtGui, QtCore


class FileMenu(object):
    """
    Widget class to call the main interface

    This widget class creates the raster menus to be found on the main
    interface. Normal as well as context menus are defined here.

    Attributes
    ----------
    parent : MainWidget
        Reference to MainWidget class found in main.py
    """
    def __init__(self, parent):

        self.parent = parent
        context_menu = self.parent.context_menu

# File Menu

        self.menufile = QtGui.QMenu(parent.menubar)
        self.menufile.setTitle("File")
        parent.menubar.addAction(self.menufile.menuAction())

        self.action_exit = QtGui.QAction(parent)
        self.action_exit.setText("Exit")
        self.menufile.addAction(self.action_exit)

        QtCore.QObject.connect(self.action_exit, QtCore.SIGNAL("triggered()"),
                               parent.close)

# Context menus
        context_menu['Basic'].addSeparator()

        self.action_bandselect = QtGui.QAction(self.parent)
        self.action_bandselect.setText("Select Bands")
        context_menu['Basic'].addAction(self.action_bandselect)
        self.action_bandselect.triggered.connect(self.bandselect)

    def bandselect(self):
        """ Select bands """
        self.parent.launch_context_item_indata(ComboBoxBasic)


class ComboBoxBasic(QtGui.QDialog):
    """
    A basic combo box application

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}

        # create GUI
        self.setWindowTitle('Band Selection')

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.combo = QtGui.QListWidget()
        self.combo.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        self.vbox.addWidget(self.combo)

        self.buttonbox = QtGui.QDialogButtonBox()
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.vbox.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def run(self):
        """ runs class """
        self.parent.scene.selectedItems()[0].update_indata()
        my_class = self.parent.scene.selectedItems()[0].my_class

        data = my_class.indata.copy()

        for j in data.keys():
            if j is 'Model3D' or j is 'Seis':
                continue

            tmp = []
            for i in data[j]:
                tmp.append(i.dataid)
            self.combo.addItems(tmp)

        if len(tmp) == 0:
            return

        tmp = self.exec_()

        if tmp != 1:
            return

        for j in data.keys():
            if j is 'Model3D' or j is 'Seis':
                continue
            atmp = [i.text() for i in self.combo.selectedItems()]

            if len(atmp) > 0:
                dtmp = []
                for i in data[j]:
                    if i.dataid in atmp:
                        dtmp.append(i)
                data[j] = dtmp

        my_class.indata = data

        if hasattr(my_class, 'data_init'):
            my_class.data_init()

        return True


class HelpMenu(object):
    """
    Widget class to call the main interface

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """
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
        self.action_help.triggered.connect(self.webhelp)

    def about(self):
        """ PyGMI About Box """

        msg = '''\
Name:         PyGMI - Python Geophysical Modelling and Interpretation
Version:       '''+self.parent.__version__+'''
Author:       Patrick Cole
E-Mail:        pcole@geoscience.org.za

Copyright:    (c) 2015 Council for Geoscience
Licence:      GPL-3.0

PyGMI is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

PyGMI is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see http://www.gnu.org/licenses/'''

        QtGui.QMessageBox.about(self.parent, 'PyGMI', msg)

    def webhelp(self):
        """ Help File"""
        webbrowser.open(r'http://patrick-cole.github.io/pygmi/')


class HelpButton(QtGui.QPushButton):
    """
    Help Button

    Convenience class to add an image to a pushbutton
    """
    def __init__(self, htmlfile=None, parent=None):
        QtGui.QPushButton.__init__(self, parent)

        self.htmlfile = htmlfile

        self.ipth = os.path.dirname(__file__)+r'/images/'
        self.setMinimumHeight(48)
        self.setMinimumWidth(48)

        self.setIcon(QtGui.QIcon(self.ipth+'help.png'))
        self.setIconSize(self.minimumSize())
        self.clicked.connect(self.help_docs)
        self.setFlat(True)

    def help_docs(self):
        """
        Help Routine
        """
        HelpDocs(self, self.htmlfile)


class HelpDocs(QtGui.QDialog):
    """
    A basic combo box application

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
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}

        ipth = os.path.dirname(__file__)+r'/helpdocs/'
        opth = os.getcwd()

        if helptxt is None:
            helptxt = 'No Help Available.'
        else:
            os.chdir(ipth)
            itxt = open(helptxt+'.html')
            helptxt = itxt.read()
            itxt.close()

        # create GUI
        self.setWindowTitle('Help!')

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.text = QtGui.QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.append(helptxt)
        self.text.setMinimumWidth(480)
        self.text.setMinimumHeight(360)
        self.text.setFrameShape(QtGui.QFrame.NoFrame)
        cursor = QtGui.QTextCursor()
        cursor.setPosition(0)
        self.text.setTextCursor(cursor)

        ptmp = self.text.palette()
        ptmp.setColor(0, 9, ptmp.color(10))
        ptmp.setColor(1, 9, ptmp.color(10))
        ptmp.setColor(2, 9, ptmp.color(10))
        self.text.setPalette(ptmp)

        self.vbox.addWidget(self.text)

        self.buttonbox = QtGui.QDialogButtonBox()
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setCenterButtons(True)
        self.buttonbox.setStandardButtons(QtGui.QDialogButtonBox.Ok)

        self.vbox.addWidget(self.buttonbox)

        self.buttonbox.accepted.connect(self.accept)

        self.exec_()

        os.chdir(opth)
