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

import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np
import numexpr as ne
import pygmi.raster.dataprep as dataprep


class EquationEditor(QtWidgets.QDialog):
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
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.equation = ''
        self.bands = {}

        self.combobox = QtWidgets.QComboBox()

        self.textbrowser = QtWidgets.QTextEdit()
        self.textbrowser2 = QtWidgets.QTextBrowser()
        self.label = QtWidgets.QLabel()

        self.setupui()

#    def textchanged(self):
#        """ Text Changed """
#        self.equation = self.textbrowser.toPlainText()

    def setupui(self):
        """ Setup UI """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        label_2 = QtWidgets.QLabel()

        self.textbrowser.setEnabled(True)
        self.resize(600, 480)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        ptmp = self.textbrowser2.palette()
        ptmp.setColor(0, 9, ptmp.color(10))
        ptmp.setColor(1, 9, ptmp.color(10))
        ptmp.setColor(2, 9, ptmp.color(10))
        self.textbrowser2.setPalette(ptmp)
        self.textbrowser2.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.setWindowTitle("Equation Editor")
        self.label.setText(": iall")
        label_2.setText("Output Equation:")
        self.textbrowser.setText('iall')
        tmp = ('<h1>Instructions:</h1>'
               '<p>Equation editor uses the numexpr library. Use the variables'
               ' iall, i1, i2 etc in formulas. The combobox above shows which '
               'band is assigned to each variable.</p>'
               '<h2>Examples</h2>'
               '<p>Sum:</p>'
               '<p>    i1 + 1000</p>'
               '<p>Threshold between values 1 and 98, substituting -999 as a '
               'value:</p>'
               '<p>    where((i1 &gt; 1) &amp; (i1 &lt; 98) , i1, -999)</p>'
               '<p>Replacing the value 0 with a nodata or null value:</p>'
               '<p>    where(iall!=0, iall, nodata)</p>'
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
               ' <li> nodata or null value of first band: nodata</li>'
               '</ul>')
        self.textbrowser2.setHtml(tmp)

        gridlayout.addWidget(label_2, 0, 0, 1, 1)
        gridlayout.addWidget(self.textbrowser, 1, 0, 1, 2)
        gridlayout.addWidget(self.combobox, 4, 0, 1, 1)
        gridlayout.addWidget(self.label, 4, 1, 1, 1)
        gridlayout.addWidget(self.textbrowser2, 5, 0, 1, 2)
        gridlayout.addWidget(buttonbox, 6, 0, 1, 2)

        self.combobox.currentIndexChanged.connect(self.combo)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def combo(self):
        """ update combo information """
        txt = self.combobox.currentText()
        if txt == '':
            return
        self.label.setText(': '+self.bands[txt])

    def eq_fix(self, indata, equation):
        """ Corrects names in equation to variable names """
        neweq = str(equation)
        neweq = neweq.replace('ln', 'log')
        neweq = neweq.replace('^', '**')
        neweq = neweq.replace('nodata', str(indata[0].nullvalue))

        return neweq

    def settings(self):
        """ Settings """
        localdict = {}
        bandsall = []
        self.bands = {}
        self.bands['all data'] = 'iall'

        self.combobox.clear()
        self.combobox.addItem('all data')

        if 'Cluster' in self.indata:
            intype = 'Cluster'
        elif 'Raster' in self.indata:
            intype = 'Raster'
        else:
            self.parent.showprocesslog('No raster data')
            return False

        indata = dataprep.merge(self.indata[intype])

        for j, i in enumerate(indata):
            self.combobox.addItem(i.dataid)
            self.bands[i.dataid] = 'i'+str(j)
            bandsall.append(i.data)
            localdict['i'+str(j)] = i.data

        localdict_list = list(localdict.keys())
        localdict['iall'] = np.ma.array(bandsall)

        temp = self.exec_()

        if temp == 0:
            return False

        equation = self.textbrowser.toPlainText()

        if equation == '':
            return False
        usedbands = []
        for i in localdict_list:
            if i in equation:
                usedbands.append(i)

        mask = None
        for i in usedbands:
            if mask is None:
                mask = localdict[i].mask
            else:
                mask = np.logical_or(mask, localdict[i].mask)

        neweq = self.eq_fix(indata, equation)

        try:
            findat = ne.evaluate(neweq, localdict)
        except Exception:
            QtWidgets.QMessageBox.warning(
                self.parent, 'Error',
                ' Nothing processed! Your equation most likely had an error.',
                QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return False

        outdata = []

        if np.size(findat) == 1:
            QtWidgets.QMessageBox.warning(
                self.parent, 'Warning',
                ' Nothing processed! Your equation outputs a single ' +
                'value instead of a minimum of one band.',
                QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return False
        if len(findat.shape) == 2:
            findat.shape = (1, findat.shape[0], findat.shape[1])

        for i, findati in enumerate(findat):
#            mask = np.ma.getmaskarray(indata[i].data)
            findati[mask] = indata[i].nullvalue

            outdata.append(copy.copy(indata[i]))
            outdata[-1].data = np.ma.masked_equal(findati,
                                                  indata[i].nullvalue)
            outdata[-1].nullvalue = indata[i].nullvalue

        # This is needed to get rid of bad, unmasked values etc.
        for i, outdatai in enumerate(outdata):
            outdatai.data.set_fill_value(indata[i].nullvalue)
            outdatai.data = np.ma.fix_invalid(outdatai.data)

        if len(outdata) == 1:
            outdata[0].dataid = equation

        self.outdata[intype] = outdata
#        breakpoint()

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

    return mode2
