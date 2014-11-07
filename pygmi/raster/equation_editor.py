# -----------------------------------------------------------------------------
# Name:        equation_editor.py (part of PyGMI)
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
""" This is the function which calls the equation editor """

from PyQt4 import QtGui, QtCore
import numpy as np
import copy
import numexpr as ne
from . import dataprep


class EquationEditor(QtGui.QDialog):
    """
    Equation Editor

    This class allows the input of equations using raster datasets as
    variables. This is commonly done in remote sensing applications, where
    there is a requirement for band ratioing etc. It uses the numexpr library.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    equation : str
        string with the equation in it
    bands : dictionary
        dictionary of bands
    bandsall : list
        list of all bands
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.equation = ''
        self.bands = {}
        self.bandsall = []
        self.localdict = {}

        self.bands['iall'] = 'all data'

        self.gridlayout = QtGui.QGridLayout(self)
        self.combobox = QtGui.QComboBox(self)

        self.textbrowser = QtGui.QTextEdit(self)
        self.textbrowser2 = QtGui.QTextBrowser(self)

#        self.pushbutton_add = QtGui.QPushButton(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.label = QtGui.QLabel(self)
        self.label_2 = QtGui.QLabel(self)

        self.setupui()

#        self.pushbutton_add.clicked.connect(self.useband)
        self.combobox.currentIndexChanged.connect(self.combo)
        self.textbrowser.textChanged.connect(self.textchanged)

    def textchanged(self):
        """ Text Changed """
        self.equation = self.textbrowser.toPlainText()

    def setupui(self):
        """ Setup UI """
        self.textbrowser.setEnabled(True)
#        self.setFixedWidth(600)
        self.resize(600 ,480)

        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout.addWidget(self.label_2, 0, 0, 1, 2)
        self.gridlayout.addWidget(self.textbrowser, 2, 0, 1, 7)

        self.gridlayout.addWidget(self.combobox, 4, 0, 1, 1)
        self.gridlayout.addWidget(self.label, 4, 1, 1, 6)

        ptmp = self.textbrowser2.palette()
        ptmp.setColor(0, 9, ptmp.color(10))
        ptmp.setColor(1, 9, ptmp.color(10))
        ptmp.setColor(2, 9, ptmp.color(10))
        self.textbrowser2.setPalette(ptmp)

#        self.textbrowser2.setTextBackgroundColor(QtGui.QColor(1))
        self.textbrowser2.setFrameShape(QtGui.QFrame.NoFrame)
#        self.textbrowser2.setFrameShadow(QtGui.QFrame.Plain)
        self.gridlayout.addWidget(self.textbrowser2, 5, 0, 1, 7)

#        self.gridlayout.addWidget(self.pushbutton_add, 5, 3, 1, 1)
        self.gridlayout.addWidget(self.buttonbox, 6, 0, 1, 7)

        self.setWindowTitle("Equation Editor")
#        self.pushbutton_add.setText("+")
        self.label.setText(": all data")
        self.label_2.setText("Output Equation:")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

        tmp = ('<h1>Instructions:</h1>'
               '<p>Equation editor uses the numexpr library. Use the variables '
               'iall, i1, i2 etc in formulas. The combobox above shows which '
               'band is assigned to each variable.</p>'
               '<h2>Examples</h2>'
               '<p>Sum:</p>'
               '<p>    i1 + 1000</p>'
               '<p>Threshold between values 1 and 98, substituting -999 as a '
               'nodata value:</p>'
               '<p>    where((i1 &gt; 1) &amp; (i1 &lt; 98) , i1, -999)</p>'
               '<h2>Commands</h2>'
               '<ul>'
               ' <li> Logical operators: &amp;, |, ~</li>'
               ' <li> Comparison operators: &lt;, &lt;=, ==, !=, &gt;=, &gt;</li>'
               ' <li> Arithmetic operators: +, -, *, /, **, %, <<, >></li>'
               ' <li> where(bool, number1, number2) : number1 if the bool '
               'condition is true, number2 otherwise.</li>'
               ' <li> sin, cos, tan, arcsin, arccos, arctan, '
               'sinh, cosh, tanh, arctan2, arcsinh, arccosh, arctanh</li>'
               ' <li> log, log10, log1p, exp, expm1</li>'
               ' <li> sqrt, abs</li>'
               '</ul>')

        self.textbrowser2.setHtml(tmp)

    def combo(self):
        """ update combo information """
        txt = self.combobox.currentText()
        if txt == '':
            return
        self.label.setText(': '+self.bands[txt])

    def useband(self):
        """ use band """
        cband = str(self.combobox.currentText())
        if cband == 'all data':
            self.equation += '(all data[:])'
        else:
            self.equation += '(bands["'+cband+'"])'

        self.textbrowser.setText(self.equation)

    def eq_fix(self):
        """ Corrects names in equation to variable names """
        neweq = str(self.equation)
        neweq = neweq.replace('ln', 'log')
        neweq = neweq.replace('^', '**')

        return neweq

    def settings(self):
        """ Settings """
        self.combobox.clear()
        self.combobox.addItem('iall')

        self.bandsall = []

        if 'Cluster' in self.indata:
            intype = 'Cluster'
        elif 'Raster' in self.indata:
            intype = 'Raster'
        else:
            self.parent.showprocesslog('No raster data')
            return

        indata = dataprep.merge(self.indata[intype])

        self.localdict = {}
        j = 0
        for i in indata:
            j += 1
            self.localdict['i'+str(j)] = i.data
            self.bands['i'+str(j)] = i.dataid
            self.bandsall.append(i.data)
            self.combobox.addItem('i'+str(j))
        self.localdict['iall'] = np.ma.array(self.bandsall)

        self.bandsall = np.ma.array(self.bandsall)

        temp = self.exec_()

        if temp == 0:
            return

        self.equation = self.textbrowser.toPlainText()

        if self.equation == '':
            return

        neweq = self.eq_fix()

        findat = ne.evaluate(neweq, self.localdict)
        outdata = []

        if np.size(findat) == 1:
            QtGui.QMessageBox.warning(
                self.parent, 'Warning',
                ' Nothing processed! Your equation outputs a single ' +
                'value instead of a minimum of one band.',
                QtGui.QMessageBox.Ok, QtGui.QMessageBox.Ok)
            return
        elif len(findat.shape) == 2:
            outdata = [copy.copy(indata[0])]
            outdata[0].data = np.ma.array(findat, mask=indata[0].data.mask)
            outdata[0].dataid = 'equation output'
        else:
            for i in range(len(findat)):
                outdata.append(copy.copy(indata[i]))
                outdata[-1].data = np.ma.array(findat[i],
                                               mask=indata[i].data.mask)

        self.outdata[intype] = outdata

        return True


def hmode(data):
    """
    Mode - this uses a histogram to generate a fast mode estimate

    Parameters
    ----------
    data : list
        list of values to generate the mode from.

    Returns
    -------
    mode2 : float
        mode value
    """
    mmin = np.min(data)
    mmax = np.max(data)
    for _ in range(2):
        mhist = np.histogram(data, 255, range=(mmin, mmax))
        mtmp = mhist[0].tolist()
        mind = mtmp.index(max(mtmp))
        mmin = mhist[1][mind]
        mmax = mhist[1][mind+1]

    mode2 = (mmax-mmin)/2 + mmin
#        mcnt = mhist[0][mind]

    return mode2
