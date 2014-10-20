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

# pylint: disable=E1101, C0103
from PyQt4 import QtGui, QtCore
import numpy as np
import copy
# import numexpr as ne
from . import dataprep


class EquationEditor(QtGui.QDialog):
    """
    Equation Editor

    This class allows the input of equations using raster datasets as
    variables. This is commonly done in remote sensing applications, where
    there is a requirement for band ratioing etc.

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
#        self.bands['all data'] = np.array([1, 2, 3, 4])
#        self.combobox.addItems(self.bands.keys())
        self.gridlayout = QtGui.QGridLayout(self)
        self.combobox = QtGui.QComboBox(self)
        self.pushbutton_add = QtGui.QPushButton(self)
#        self.textbrowser = QtGui.QTextBrowser(self)
        self.textbrowser = QtGui.QTextEdit(self)
        self.line = QtGui.QFrame(self)

        self.pushbutton_one = QtGui.QPushButton(self)
        self.pushbutton_two = QtGui.QPushButton(self)
        self.pushbutton_three = QtGui.QPushButton(self)
        self.pushbutton_four = QtGui.QPushButton(self)
        self.pushbutton_five = QtGui.QPushButton(self)
        self.pushbutton_six = QtGui.QPushButton(self)
        self.pushbutton_seven = QtGui.QPushButton(self)
        self.pushbutton_eight = QtGui.QPushButton(self)
        self.pushbutton_nine = QtGui.QPushButton(self)
        self.pushbutton_zero = QtGui.QPushButton(self)
        self.pushbutton_lbracket = QtGui.QPushButton(self)
        self.pushbutton_rbracket = QtGui.QPushButton(self)
        self.pushbutton_power = QtGui.QPushButton(self)
        self.pushbutton_useband = QtGui.QPushButton(self)
        self.pushbutton_backspace = QtGui.QPushButton(self)
        self.pushbutton_decpnt = QtGui.QPushButton(self)
        self.pushbutton_subtract = QtGui.QPushButton(self)
        self.pushbutton_multiply = QtGui.QPushButton(self)
        self.pushbutton_divide = QtGui.QPushButton(self)
        self.pushbutton_ln = QtGui.QPushButton(self)
        self.pushbutton_log = QtGui.QPushButton(self)
        self.pushbutton_abs = QtGui.QPushButton(self)
        self.pushbutton_mode = QtGui.QPushButton(self)
        self.pushbutton_median = QtGui.QPushButton(self)
        self.pushbutton_mean = QtGui.QPushButton(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.label = QtGui.QLabel(self)
        self.label_2 = QtGui.QLabel(self)

        self.setupui()

        self.pushbutton_abs.clicked.connect(self.abs)
        self.pushbutton_backspace.clicked.connect(self.backspace)
        self.pushbutton_one.clicked.connect(self.one)
        self.pushbutton_two.clicked.connect(self.two)
        self.pushbutton_three.clicked.connect(self.three)
        self.pushbutton_four.clicked.connect(self.four)
        self.pushbutton_five.clicked.connect(self.five)
        self.pushbutton_six.clicked.connect(self.six)
        self.pushbutton_seven.clicked.connect(self.seven)
        self.pushbutton_eight.clicked.connect(self.eight)
        self.pushbutton_nine.clicked.connect(self.nine)
        self.pushbutton_zero.clicked.connect(self.zero)
        self.pushbutton_decpnt.clicked.connect(self.decpnt)
        self.pushbutton_add.clicked.connect(self.add)
        self.pushbutton_subtract.clicked.connect(self.subtract)
        self.pushbutton_multiply.clicked.connect(self.multiply)
        self.pushbutton_divide.clicked.connect(self.divide)
        self.pushbutton_power.clicked.connect(self.power)
        self.pushbutton_ln.clicked.connect(self.natlog)
        self.pushbutton_log.clicked.connect(self.log)
        self.pushbutton_mean.clicked.connect(self.mean)
        self.pushbutton_median.clicked.connect(self.median)
        self.pushbutton_mode.clicked.connect(self.mode)
        self.pushbutton_rbracket.clicked.connect(self.rbracket)
        self.pushbutton_lbracket.clicked.connect(self.lbracket)
        self.pushbutton_useband.clicked.connect(self.useband)
#        self.combobox.currentIndexChanged.connect(self.combo)
        self.textbrowser.textChanged.connect(self.textchanged)

    def textchanged(self):
        """ Text Changed """
        self.equation = self.textbrowser.toPlainText()

    def setupui(self):
        """ Setup UI """
        self.textbrowser.setEnabled(True)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)

        self.gridlayout.addWidget(self.label_2, 0, 0, 1, 2)
        self.gridlayout.addWidget(self.textbrowser, 2, 0, 1, 7)
        self.gridlayout.addWidget(self.label, 4, 0, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_seven, 5, 0, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_four, 6, 0, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_one, 7, 0, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_zero, 8, 0, 1, 2)
        self.gridlayout.addWidget(self.pushbutton_backspace, 9, 0, 1, 1)

        self.gridlayout.addWidget(self.combobox, 4, 1, 1, 5)
        self.gridlayout.addWidget(self.pushbutton_eight, 5, 1, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_five, 6, 1, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_two, 7, 1, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_lbracket, 9, 1, 1, 1)

        self.gridlayout.addWidget(self.pushbutton_nine, 5, 2, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_six, 6, 2, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_three, 7, 2, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_decpnt, 8, 2, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_rbracket, 9, 2, 1, 1)

        self.gridlayout.addWidget(self.pushbutton_add, 5, 3, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_subtract, 6, 3, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_multiply, 7, 3, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_divide, 8, 3, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_power, 9, 3, 1, 1)

        self.gridlayout.addWidget(self.line, 5, 4, 6, 1)
        self.gridlayout.addWidget(self.pushbutton_ln, 5, 5, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_log, 6, 5, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_abs, 7, 5, 1, 1)
        self.gridlayout.addWidget(self.buttonbox, 9, 5, 1, 2)

        self.gridlayout.addWidget(self.pushbutton_useband, 4, 6, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_mode, 5, 6, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_median, 6, 6, 1, 1)
        self.gridlayout.addWidget(self.pushbutton_mean, 7, 6, 1, 1)

        self.setWindowTitle("Equation Editor")
        self.pushbutton_add.setText("+")
        self.pushbutton_one.setText("1")
        self.pushbutton_two.setText("2")
        self.pushbutton_three.setText("3")
        self.pushbutton_four.setText("4")
        self.pushbutton_five.setText("5")
        self.pushbutton_six.setText("6")
        self.pushbutton_seven.setText("7")
        self.pushbutton_eight.setText("8")
        self.pushbutton_nine.setText("9")
        self.pushbutton_zero.setText("0")
        self.pushbutton_decpnt.setText(".")
        self.pushbutton_subtract.setText("-")
        self.pushbutton_multiply.setText("*")
        self.pushbutton_divide.setText("/")
        self.pushbutton_ln.setToolTip("Natural Logarithm")
        self.pushbutton_ln.setText("ln")
        self.pushbutton_log.setToolTip("Base 10 Logarithm")
        self.pushbutton_log.setText("log10")
        self.pushbutton_abs.setToolTip("Absolute Value")
        self.pushbutton_abs.setText("abs")
        self.pushbutton_mode.setToolTip("Mode")
        self.pushbutton_mode.setText("mode")
        self.pushbutton_median.setToolTip("Median")
        self.pushbutton_median.setText("median")
        self.pushbutton_mean.setToolTip("Mean")
        self.pushbutton_mean.setText("mean")
        self.pushbutton_backspace.setToolTip("Backspace")
        self.pushbutton_backspace.setText("Backspace")
        self.pushbutton_lbracket.setText("(")
        self.pushbutton_rbracket.setText(")")
        self.pushbutton_power.setToolTip("Raise to a power")
        self.pushbutton_power.setText("^")
        self.pushbutton_useband.setText("Use Band")
        self.label.setText("Band Name:")
        self.label_2.setText("Output Equation:")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

    def abs(self):
        """ abs """
        self.equation += 'abs'
        self.textbrowser.setText(self.equation)

    def backspace(self):
        """ backspace """
        self.equation = self.equation[:-1]
        self.textbrowser.setText(self.equation)
        print('backspace')

    def useband(self):
        """ use band """
        cband = str(self.combobox.currentText())
        if cband == 'all data':
            self.equation += '(all data[:])'
        else:
            self.equation += '(bands["'+cband+'"])'

        self.textbrowser.setText(self.equation)

    def rbracket(self):
        """ right bracket """
        self.equation += ')'
        self.textbrowser.setText(self.equation)

    def lbracket(self):
        """ left bracket """
        self.equation += '('
        self.textbrowser.setText(self.equation)

    def power(self):
        """ power """
        self.equation += '^'
        self.textbrowser.setText(self.equation)

    def decpnt(self):
        """ decimal point """
        self.equation += '.'
        self.textbrowser.setText(self.equation)

    def add(self):
        """ add """
        self.equation += ' + '
        self.textbrowser.setText(self.equation)

    def subtract(self):
        """ subtract """
        self.equation += ' - '
        self.textbrowser.setText(self.equation)

    def multiply(self):
        """ multiply """
        self.equation += ' * '
        self.textbrowser.setText(self.equation)

    def divide(self):
        """ divide """
        self.equation += ' / '
        self.textbrowser.setText(self.equation)

    def one(self):
        """ one """
        self.equation += '1'
        self.textbrowser.setText(self.equation)

    def two(self):
        """ two """
        self.equation += '2'
        self.textbrowser.setText(self.equation)

    def three(self):
        """ three """
        self.equation += '3'
        self.textbrowser.setText(self.equation)

    def four(self):
        """ four """
        self.equation += '4'
        self.textbrowser.setText(self.equation)

    def five(self):
        """ five """
        self.equation += '5'
        self.textbrowser.setText(self.equation)

    def six(self):
        """ six """
        self.equation += '6'
        self.textbrowser.setText(self.equation)

    def seven(self):
        """ seven """
        self.equation += '7'
        self.textbrowser.setText(self.equation)

    def eight(self):
        """ eight """
        self.equation += '8'
        self.textbrowser.setText(self.equation)

    def nine(self):
        """ nine """
        self.equation += '9'
        self.textbrowser.setText(self.equation)

    def zero(self):
        """ zero """
        self.equation += '0'
        self.textbrowser.setText(self.equation)

    def natlog(self):
        """ natural logarithm """
        self.equation += 'ln'
        self.textbrowser.setText(self.equation)

    def log(self):
        """ base 10 logarithm """
        self.equation += 'log10'
        self.textbrowser.setText(self.equation)

    def mean(self):
        """ mean """
        self.equation += 'mean'
        self.textbrowser.setText(self.equation)

    def median(self):
        """ median """
        self.equation += 'median'
        self.textbrowser.setText(self.equation)

    def mode(self):
        """ mode """
        self.equation += 'mode'
        self.textbrowser.setText(self.equation)

    def eq_fix(self):
        """ Corrects names in equation to variable names """
        neweq = str(self.equation)
        neweq = neweq.replace('ln', 'np.log')
        neweq = neweq.replace('log10', 'np.log10')
        neweq = neweq.replace('mean', 'np.mean')
        neweq = neweq.replace('median', 'np.median')
        neweq = neweq.replace('mode', 'hmode')
        neweq = neweq.replace('abs', 'np.abs')
        neweq = neweq.replace('bands', 'self.bands')
        neweq = neweq.replace('all data', 'self.bandsall')
        neweq = neweq.replace('^', '**')
        neweq = neweq.replace(')(', '),(')

        return neweq

    def settings(self):
        """ Settings """
        self.combobox.clear()
        self.combobox.addItem('all data')
        self.bandsall = []

        if 'Cluster' in self.indata:
            intype = 'Cluster'
        elif 'Raster' in self.indata:
            intype = 'Raster'
        else:
            self.parent.showprocesslog('No raster data')
            return

        indata = dataprep.merge(self.indata[intype])

        for i in indata:
            self.bands[i.dataid] = i.data
            self.bandsall.append(i.data)
            self.combobox.addItem(i.dataid)

        self.bandsall = np.ma.array(self.bandsall)

        temp = self.exec_()

        if temp == 0:
            return

        self.equation = self.textbrowser.toPlainText()

        if self.equation == '':
            return

        neweq = self.eq_fix()

        findat = eval(neweq)
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
            outdata[0].data = findat
            outdata[0].dataid = 'equation output'
        else:
            for i in range(len(findat)):
                outdata.append(copy.copy(indata[i]))
                outdata[-1].data = findat[i]

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
