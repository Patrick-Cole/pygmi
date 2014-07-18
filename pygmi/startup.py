# -----------------------------------------------------------------------------
# Name:        startup.py (part of PyGMI)
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
""" Start up dialog """

# pylint: disable=E1101, C0103
from PySide import QtGui, QtCore


class Startup(QtGui.QDialog):
    """ Gradients """
    def __init__(self, pbarmax, parent=None):
        QtGui.QDialog.__init__(self, parent)
#        self.setWindowFlags(QtCore.Qt.SplashScreen)
        self.setWindowFlags(QtCore.Qt.ToolTip)

        self.gridlayout_main = QtGui.QVBoxLayout(self)
        self.label_info = QtGui.QLabel(self)
        self.label_pic = QtGui.QLabel(self)
        self.label_pic.setPixmap(QtGui.QPixmap('images/logo256.ico'))
        self.label_info.setScaledContents(True)
        self.pbar = QtGui.QProgressBar(self)

        labelText = "<font color='red'>Py</font><font color='blue'>GMI</font>"

        fnt = QtGui.QFont("Arial", 72, QtGui.QFont.Bold)
        self.label_info.setFont(fnt)
        self.label_info.setText(labelText)
#            'Python Geophysical Modelling and Interpretation\n' +
#            '------------------------------------------------------------')
        self.gridlayout_main.addWidget(self.label_info)
        self.gridlayout_main.addWidget(self.label_pic)

        self.pbar.setMaximum(pbarmax - 1)
        self.gridlayout_main.addWidget(self.pbar)

        self.open()

    def update(self, text):
        """ Updates the text on the dialog """
        oldtext = self.label_info.text()
        newtext = oldtext + '\n' + text
#        self.label_info.setText(newtext)
        self.pbar.setValue(self.pbar.value() + 1)
        QtGui.QApplication.processEvents()
