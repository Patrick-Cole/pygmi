# -----------------------------------------------------------------------------
# Name:        equation_editor.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2025 Council for Geoscience
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

from PySide6 import QtWidgets, QtGui

from pygmi.misc import BasicModule


class EquationEditor(BasicModule):
    """
    Equation Editor.

    This class allows the input of equations using raster datasets as
    variables. This is commonly done in remote sensing applications, where
    there is a requirement for band ratioing etc. It uses the numexpr library.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    equation : str
        string with the equation in it
    bands : dictionary
        dictionary of bands
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.equation = None
        self.bands = {}

        self.cmb_1 = QtWidgets.QComboBox()

        self.textbrowser = QtWidgets.QTextEdit()
        self.textbrowser2 = QtWidgets.QTextBrowser()
        self.lbl_bands = QtWidgets.QLabel(': i0')
        self.le_name = QtWidgets.QLineEdit('Column1')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_1 = QtWidgets.QGridLayout(self)

        lbl_1 = QtWidgets.QLabel('Data Band Key:')
        lbl_2 = QtWidgets.QLabel('Output Equation:')
        lbl_3 = QtWidgets.QLabel('New Column Name:')
        self.buttonbox.htmlfile = 'vector.dm.equationeditor'

        self.textbrowser.setEnabled(True)
        self.resize(600, 480)

        ptmp = self.textbrowser2.palette()

        ptmp.setColor(ptmp.ColorGroup.Active,
                      ptmp.ColorRole.Base,
                      ptmp.color(QtGui.QPalette.ColorRole.Window))
        ptmp.setColor(ptmp.ColorGroup.Disabled,
                      ptmp.ColorRole.Base,
                      ptmp.color(QtGui.QPalette.ColorRole.Window))
        ptmp.setColor(ptmp.ColorGroup.Inactive,
                      ptmp.ColorRole.Base,
                      ptmp.color(QtGui.QPalette.ColorRole.Window))

        self.textbrowser2.setPalette(ptmp)
        self.textbrowser2.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.setWindowTitle('Vector Equation Editor')
        self.textbrowser.setText('i0')
        tmp = ('<h1>Instructions:</h1>'
               '<p>Equation editor uses the numexpr library. Use the variables'
               ' i0, i1, i2 etc in formulas. The combobox above shows which '
               'band is assigned to each variable.</p>'
               '<h2>Examples</h2>'
               '<p>Sum:</p>'
               '<pre>    i1 + 1000</pre>'
               '</pre>'
               '<h2>Commands</h2>'
               '<ul>'
               ' <li> Comparison operators: &lt;, &lt;=, ==, !=, &gt;=, &gt;'
               '</li>'
               ' <li> Arithmetic operators: +, -, *, /, **, %</li>'
               ' <li> sin, cos, tan, arcsin, arccos, arctan, '
               'sinh, cosh, tanh, arctan2, arcsinh, arccosh, arctanh</li>'
               ' <li> log, log10, log1p, exp, expm1</li>'
               ' <li> sqrt, abs</li>'
               '</ul>')
        self.textbrowser2.setHtml(tmp)

        gl_1.addWidget(lbl_2, 0, 0, 1, 1)
        gl_1.addWidget(self.textbrowser, 1, 0, 1, 2)
        gl_1.addWidget(lbl_1, 5, 0, 1, 1)
        gl_1.addWidget(self.cmb_1, 6, 0, 1, 1)
        gl_1.addWidget(self.lbl_bands, 6, 1, 1, 1)
        gl_1.addWidget(lbl_3, 3, 0, 1, 1)
        gl_1.addWidget(self.le_name, 4, 0, 1, 1)
        gl_1.addWidget(self.textbrowser2, 7, 0, 1, 2)
        gl_1.addWidget(self.buttonbox, 8, 0, 1, 2)

        self.cmb_1.currentIndexChanged.connect(self.combo)

    def combo(self):
        """
        Update combo information.

        Returns
        -------
        None.

        """
        if self.bands == {}:
            return
        txt = self.cmb_1.currentText()
        if txt != '':
            self.lbl_bands.setText(': ' + self.bands[txt])

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
        if 'Vector' not in self.indata:
            self.showlog('No vector data.')
            return False

        self.bands = {}
        self.cmb_1.clear()

        indata = self.indata['Vector'][0].copy()

        for j, i in enumerate(indata.columns):
            if indata[i].dtype == object:
                continue
            self.cmb_1.addItem(i)
            self.bands[i] = 'i' + str(j)

        if not nodialog:
            temp = self.exec()

            if temp == 0:
                return False

            self.equation = self.textbrowser.toPlainText()

        if self.equation == '':
            self.showlog('Error: You need to enter an equation.')
            return False

        if self.le_name.text() == '':
            self.showlog('Error: You must have a colum name.')
            return False

        outdata = eqedit(indata, self.equation, self.le_name.text(),
                         self.showlog)

        self.outdata['Vector'] = [outdata]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.equation)
        self.saveobj(self.textbrowser)


def eqedit(data, equation, colname, showlog=print):
    """
    Use equations on raster data.

    Parameters
    ----------
    data : list
        List of PyGMI raster data.
    equation : str
        Equation to compute.
    showlog : function, optional
        Show information using a function. The default is print.

    Returns
    -------
    list
        List of PyGMI raster data.

    """
    outdata = data.copy()
    indata = data.copy()
    cols = []
    ii = -1
    for j, i in enumerate(data):
        if data[i].dtype == object:
            cols.append(i)
        else:
            ii += 1
            cols.append('i' + str(ii))

    indata.columns = cols

    if equation == '':
        return None

    # neweq = eq_fix(indata, equation, showlog)

    try:
        findat = indata.eval(equation)
    except Exception:
        findat = None

    if findat is None:
        showlog('Error: Nothing processed! '
                'Your equation most likely had an error.')
        return False

    outdata[colname] = findat

    return outdata


def eq_fix(indata, equation, showlog=print):
    """
    Corrects names in equation to variable names.

    Parameters
    ----------
    indata : list of PyGMI Data.
        PyGMI raster dataset.
    equation : str
        Equation to fix.
    showlog : function, optional
        Show information using a function. The default is print.

    Returns
    -------
    neweq : str
        Corrected equation.

    """
    neweq = str(equation)
    neweq = neweq.replace('ln', 'log')
    neweq = neweq.replace('^', '**')
    neweq = neweq.replace('nodata', str(indata[0].nodata))

    if 'log' in neweq:
        showlog('Warning, if you have invalid log values, they will '
                'be masked out.')

    if 'sqrt' in neweq:
        showlog('Warning, if you have invalid sqrt values, they will '
                'be masked out.')

    neweq = neweq.strip()

    return neweq


def _test():
    """Test."""
    import sys
    from pygmi.vector.iodefs import ImportVector

    print('Starting')

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    ifile = r"D:\workdata\PyGMI Test Data\Vector\Geochem\geochem_tzaneen.shp"

    IO = ImportVector()
    IO.ifile = ifile
    # IO.filt = 'Comma Delimited (*.csv)'
    IO.settings(True)

    EE = EquationEditor()
    EE.indata = IO.outdata

    EE.settings()


if __name__ == "__main__":
    _test()
