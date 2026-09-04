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
"""Equation editor for vector data."""

import pandas as pd
from PySide6 import QtGui, QtWidgets

from pygmi.misc import BasicModule


class EquationEditor(BasicModule):
    """
    Equation Editor.

    This class allows the input of equations using raster datasets as
    variables. This is commonly done in remote sensing applications, where
    there is a requirement for band ratioing etc. It uses the numexpr library.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    equation
        String with the equation in it
    bands
        Dictionary of bands
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.equation = None
        self.bands = {}

        self.cmb_1 = QtWidgets.QComboBox()

        self.textbrowser = QtWidgets.QTextEdit()
        self.textbrowser2 = QtWidgets.QTextBrowser()
        self.lbl_bands = QtWidgets.QLabel(": i0")
        self.le_name = QtWidgets.QLineEdit("Column1")

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gl_1 = QtWidgets.QGridLayout(self)

        lbl_1 = QtWidgets.QLabel("Data Band Key:")
        lbl_2 = QtWidgets.QLabel("Output Equation:")
        lbl_3 = QtWidgets.QLabel("New Column Name:")
        self.buttonbox.htmlfile = "vector.dm.equationeditor"

        self.textbrowser.setEnabled(True)
        self.resize(600, 480)

        ptmp = self.textbrowser2.palette()

        ptmp.setColor(
            ptmp.ColorGroup.Active,
            ptmp.ColorRole.Base,
            ptmp.color(QtGui.QPalette.ColorRole.Window),
        )
        ptmp.setColor(
            ptmp.ColorGroup.Disabled,
            ptmp.ColorRole.Base,
            ptmp.color(QtGui.QPalette.ColorRole.Window),
        )
        ptmp.setColor(
            ptmp.ColorGroup.Inactive,
            ptmp.ColorRole.Base,
            ptmp.color(QtGui.QPalette.ColorRole.Window),
        )

        self.textbrowser2.setPalette(ptmp)
        self.textbrowser2.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.setWindowTitle("Vector Equation Editor")
        self.textbrowser.setText("i0")
        tmp = (
            "<h1>Instructions:</h1>"
            "<p>Equation editor uses the numexpr library. Use the variables"
            " i0, i1, i2 etc in formulas. The combobox above shows which "
            "band is assigned to each variable.</p>"
            "<h2>Examples</h2>"
            "<p>Sum:</p>"
            "<pre>    i1 + 1000</pre>"
            "</pre>"
            "<h2>Commands</h2>"
            "<ul>"
            " <li> Comparison operators: &lt;, &lt;=, ==, !=, &gt;=, &gt;"
            "</li>"
            " <li> Arithmetic operators: +, -, *, /, **, %</li>"
            " <li> sin, cos, tan, arcsin, arccos, arctan, "
            "sinh, cosh, tanh, arctan2, arcsinh, arccosh, arctanh</li>"
            " <li> log, log10, log1p, exp, expm1</li>"
            " <li> sqrt, abs</li>"
            "</ul>"
        )
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
        """Update combo information."""
        if self.bands == {}:
            return
        txt = self.cmb_1.currentText()
        if txt != "":
            self.lbl_bands.setText(": " + self.bands[txt])

    def settings(self, nodialog: bool = False) -> bool:
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
        if "Vector" not in self.indata:
            self.showlog("No vector data.")
            return False

        indata = self.indata["Vector"][0].copy()
        indata = indata.select_dtypes(include=["number"])
        self.bands = {col: f"i{i}" for i, col in enumerate(indata.columns)}

        self.cmb_update(self.cmb_1, self.bands.keys())
        self.combo()

        if not nodialog:
            temp = self.exec()

            if temp == 0:
                return False

            self.equation = self.textbrowser.toPlainText()

        if self.equation == "":
            self.showlog("Error: You need to enter an equation.")
            return False

        if self.le_name.text() == "":
            self.showlog("Error: You must have a column name.")
            return False

        indata = indata.rename(columns=self.bands)

        try:
            outcol = indata.eval(self.equation)
        except pd.errors.UndefinedVariableError:
            self.showlog(
                "Error: Nothing processed! Your equation most likely had an error."
            )
            return False

        outdata = self.indata["Vector"][0].copy()
        outdata[self.le_name.text()] = outcol

        self.outdata["Vector"] = [outdata]

        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.equation)
        self.saveobj(self.textbrowser)
        self.saveobj(self.cmb_1)
        self.saveobj(self.le_name)


def _test():
    """Test."""
    import sys

    from pygmi.vector.iodefs import ImportVector

    print("Starting")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ifile = r"D:\Workdata\PyGMI Test Data\Vector\Line Data\2427AB_portion_Mag.shp"

    IO = ImportVector()
    IO.ifile = ifile
    # IO.filt = 'Comma Delimited (*.csv)'
    IO.settings(True)

    EE = EquationEditor()
    EE.indata = IO.outdata

    EE.settings()


if __name__ == "__main__":
    _test()
