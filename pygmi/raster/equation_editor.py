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
"""Equation editor."""

import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np
import numexpr as ne
import pygmi.raster.dataprep as dataprep


class EquationEditor(QtWidgets.QDialog):
    """
    Equation Editor.

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
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.equation = None
        self.bands = {}

        self.combobox = QtWidgets.QComboBox()

        self.textbrowser = QtWidgets.QTextEdit()
        self.textbrowser2 = QtWidgets.QTextBrowser()
        self.label = QtWidgets.QLabel(': iall')
        self.dtype = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        label_1 = QtWidgets.QLabel('Data Band Key:')
        label_2 = QtWidgets.QLabel('Output Equation:')
        label_3 = QtWidgets.QLabel('Output Data Type:')
        self.dtype.addItems(['auto', 'uint8', 'int16', 'int32',
                             'float32', 'float64'])

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

        self.setWindowTitle('Equation Editor')
        self.textbrowser.setText('iall')
        tmp = ('<h1>Instructions:</h1>'
               '<p>Equation editor uses the numexpr library. Use the variables'
               ' iall, i1, i2 etc in formulas. The combobox above shows which '
               'band is assigned to each variable.</p>'
               '<h2>Examples</h2>'
               '<p>Sum:</p>'
               '<pre>    i1 + 1000</pre>'
               '<p>Mosaic two bands into one:</p>'
               '<pre>    mosaic(i0, i1)</pre>'
               '<p>Threshold between values 1 and 98, substituting -999 as a '
               'value:</p>'
               '<pre>    where((i1 &gt; 1) &amp; (i1 &lt; 98) , i1, -999)'
               '</pre>'
               '<p>Replacing the value 0 with a nodata or null value:</p>'
               '<pre>    where(iall!=0, iall, nodata)</pre>'
               '<h2>Commands</h2>'
               '<ul>'
               ' <li> Logical operators: &amp;, |, ~</li>'
               ' <li> Comparison operators: &lt;, &lt;=, ==, !=, &gt;=, &gt;'
               '</li>'
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
        gridlayout.addWidget(label_1, 3, 0, 1, 1)
        gridlayout.addWidget(self.combobox, 4, 0, 1, 1)
        gridlayout.addWidget(self.label, 4, 1, 1, 1)
        gridlayout.addWidget(self.dtype, 6, 0, 1, 1)
        gridlayout.addWidget(label_3, 5, 0, 1, 1)
        gridlayout.addWidget(self.textbrowser2, 7, 0, 1, 2)
        gridlayout.addWidget(buttonbox, 8, 0, 1, 2)

        self.combobox.currentIndexChanged.connect(self.combo)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def combo(self):
        """
        Update combo information.

        Returns
        -------
        None.

        """
        txt = self.combobox.currentText()
        if txt != '':
            self.label.setText(': '+self.bands[txt])

    def eq_fix(self, indata):
        """
        Corrects names in equation to variable names.

        Parameters
        ----------
        indata : PyGMI Data
            PyGMI raster dataset.

        Returns
        -------
        neweq : str
            Corrected equation.

        """
        neweq = str(self.equation)
        neweq = neweq.replace('ln', 'log')
        neweq = neweq.replace('^', '**')
        neweq = neweq.replace('nodata', str(indata[0].nodata))

        return neweq

    def mosaic(self, eq, localdict):
        """
        Mosaics data into a single band dataset.

        Parameters
        ----------
        eq : str
            Equation with mosaic command.
        localdict : dictionary
            Dictionary of data.

        Returns
        -------
        findat : numpy array
            Output array.

        """
        idx = eq.index('mosaic(')+7
        eq2 = eq[idx:]
        idx = eq2.index(')')
        eq2 = eq2[:idx]
        eq2 = eq2.replace(' ', '')
        eq3 = eq2.split(',')

        localdict_list = list(localdict.keys())

        # Check for problems
        if 'iall' in eq:
            return None

        if len(eq3) < 2:
            return None

        eq4 = []
        mask = []
        for i in eq3:
            usedbands = []
            for j in localdict_list:
                if j in i:
                    usedbands.append(j)
            mask1 = None
            for j in usedbands:
                if mask1 is None:
                    mask1 = localdict[j].mask
                else:
                    mask1 = np.logical_or(mask1, localdict[j].mask)

            mask.append(mask1)
            try:
                eq4.append(ne.evaluate(i, localdict))
            except Exception:
                return None
            eq4[-1] = np.ma.array(eq4[-1], mask=mask[-1])

        # master = np.ma.mean(eq4, 0)
        master = eq4.pop()
        for i in eq4[::-1]:
            master[~i.mask] = i.data[~i.mask]

        return master

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
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
            self.showprocesslog('No raster data')
            return False

        indata = dataprep.lstack(self.indata[intype])

        for j, i in enumerate(indata):
            self.combobox.addItem(i.dataid)
            self.bands[i.dataid] = 'i'+str(j)
            bandsall.append(i.data)
            localdict['i'+str(j)] = i.data

        localdict_list = list(localdict.keys())
        localdict['iall'] = np.ma.array(bandsall)

        if not nodialog:
            temp = self.exec_()

            if temp == 0:
                return False

            self.equation = self.textbrowser.toPlainText()

        if self.equation == '':
            return False

        if 'iall' in self.equation:
            usedbands = localdict_list
        else:
            usedbands = []
            for i in localdict_list:
                if i in self.equation:
                    usedbands.append(i)

        mask = None
        for i in usedbands:
            if mask is None:
                mask = localdict[i].mask
            else:
                mask = np.logical_or(mask, localdict[i].mask)

        neweq = self.eq_fix(indata)

        if 'mosaic' in neweq:
            findat = self.mosaic(neweq, localdict)
            if findat is None:
                QtWidgets.QMessageBox.warning(
                    self.parent, 'Error',
                    'Nothing processed! '
                    'Your equation most likely had an error.',
                    QtWidgets.QMessageBox.Ok)
                return False
            mask = findat.mask

        else:
            try:
                findat = ne.evaluate(neweq, localdict)
            except Exception:
                QtWidgets.QMessageBox.warning(
                    self.parent, 'Error',
                    'Nothing processed! '
                    'Your equation most likely had an error.',
                    QtWidgets.QMessageBox.Ok)
                return False

        outdata = []

        if np.size(findat) == 1:
            QtWidgets.QMessageBox.warning(
                self.parent, 'Warning',
                ' Nothing processed! Your equation outputs a single ' +
                'value instead of a minimum of one band.',
                QtWidgets.QMessageBox.Ok)
            return False
        if findat.ndim == 2:
            findat.shape = (1, findat.shape[0], findat.shape[1])

        for i, findati in enumerate(findat):
            findati[mask] = indata[i].nodata

            outdata.append(copy.deepcopy(indata[i]))
            outdata[-1].data = np.ma.masked_equal(findati,
                                                  indata[i].nodata)
            outdata[-1].nodata = indata[i].nodata

        dtype = self.dtype.currentText()
        # This is needed to get rid of bad, unmasked values etc.
        for i, outdatai in enumerate(outdata):
            outdatai.data.set_fill_value(indata[i].nodata)
            outdatai.data = np.ma.fix_invalid(outdatai.data)
            if dtype != 'auto':
                outdatai.data = outdatai.data.astype(dtype)

        if len(outdata) == 1:
            outdata[0].dataid = self.equation

        self.outdata[intype] = outdata

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.equation = projdata['equation']
        self.textbrowser.setText(self.equation)

        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['equation'] = self.equation

        return projdata


def hmode(data):
    """
    Mode - this uses a histogram to generate a fast mode estimate.

    Parameters
    ----------
    data : list
        list of values to generate the mode from.

    Returns
    -------
    mode2 : float
        mode value.
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
