# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""Import borehole data, currently supports Council for Geoscience data."""

import os

import pandas as pd
from PySide6 import QtWidgets

from pygmi.misc import BasicModule


class ImportData(BasicModule):
    """
    Import borehole data.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.hfile = ""
        self.is_import = True

        self.le_lithfile = QtWidgets.QLineEdit("")
        self.le_headfile = QtWidgets.QLineEdit("")
        self.cmb_companyno = QtWidgets.QComboBox()
        self.cmb_boreholeid = QtWidgets.QComboBox()
        self.cmb_depthfrom = QtWidgets.QComboBox()
        self.cmb_depthto = QtWidgets.QComboBox()
        self.cmb_lat = QtWidgets.QComboBox()
        self.cmb_long = QtWidgets.QComboBox()
        self.cmb_drilldate = QtWidgets.QComboBox()
        self.cmb_elevation = QtWidgets.QComboBox()
        self.cmb_lith = QtWidgets.QComboBox()
        self.cmb_strat = QtWidgets.QComboBox()
        self.cmb_rank = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        """
        self.buttonbox.htmlfile = "rsense.dm.importdata"
        lbl_companyno = QtWidgets.QLabel("Company number:")
        lbl_boreholeid = QtWidgets.QLabel("Borehole ID:")
        lbl_depthfrom = QtWidgets.QLabel("Depth from:")
        lbl_depthto = QtWidgets.QLabel("Depth to:")
        lbl_lat = QtWidgets.QLabel("Latitude:")
        lbl_long = QtWidgets.QLabel("Longitude:")
        lbl_drilldate = QtWidgets.QLabel("Drill date:")
        lbl_elevation = QtWidgets.QLabel("Elevation:")
        lbl_lith = QtWidgets.QLabel("Lithology:")
        lbl_strat = QtWidgets.QLabel("Stratigraphy:")
        lbl_rank = QtWidgets.QLabel("Rank:")

        pb_lithfile = QtWidgets.QPushButton(" Lithology Filename")
        pb_headfile = QtWidgets.QPushButton(" Header Filename")

        pixmapi = QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_lithfile.setIcon(icon)
        pb_lithfile.setStyleSheet("text-align:left;")
        pb_headfile.setIcon(icon)
        pb_headfile.setStyleSheet("text-align:left;")

        self.setWindowTitle("Import Borehole Data")

        gl_1 = QtWidgets.QGridLayout(self)

        gl_1.addWidget(pb_lithfile, 2, 0, 1, 1)
        gl_1.addWidget(self.le_lithfile, 2, 1, 1, 1)
        gl_1.addWidget(pb_headfile, 1, 0, 1, 1)
        gl_1.addWidget(self.le_headfile, 1, 1, 1, 1)
        gl_1.addWidget(lbl_companyno, 3, 0, 1, 1)
        gl_1.addWidget(self.cmb_companyno, 3, 1, 1, 1)
        gl_1.addWidget(lbl_boreholeid, 4, 0, 1, 1)
        gl_1.addWidget(self.cmb_boreholeid, 4, 1, 1, 1)
        gl_1.addWidget(lbl_depthfrom, 5, 0, 1, 1)
        gl_1.addWidget(self.cmb_depthfrom, 5, 1, 1, 1)
        gl_1.addWidget(lbl_depthto, 6, 0, 1, 1)
        gl_1.addWidget(self.cmb_depthto, 6, 1, 1, 1)
        gl_1.addWidget(lbl_companyno, 7, 0, 1, 1)
        gl_1.addWidget(self.cmb_companyno, 7, 1, 1, 1)
        gl_1.addWidget(lbl_lat, 8, 0, 1, 1)
        gl_1.addWidget(self.cmb_lat, 8, 1, 1, 1)
        gl_1.addWidget(lbl_long, 9, 0, 1, 1)
        gl_1.addWidget(self.cmb_long, 9, 1, 1, 1)
        gl_1.addWidget(lbl_elevation, 10, 0, 1, 1)
        gl_1.addWidget(self.cmb_elevation, 10, 1, 1, 1)
        gl_1.addWidget(lbl_drilldate, 11, 0, 1, 1)
        gl_1.addWidget(self.cmb_drilldate, 11, 1, 1, 1)
        gl_1.addWidget(lbl_lith, 12, 0, 1, 1)
        gl_1.addWidget(self.cmb_lith, 12, 1, 1, 1)
        gl_1.addWidget(lbl_strat, 13, 0, 1, 1)
        gl_1.addWidget(self.cmb_strat, 13, 1, 1, 1)
        gl_1.addWidget(lbl_rank, 14, 0, 1, 1)
        gl_1.addWidget(self.cmb_rank, 14, 1, 1, 1)

        gl_1.addWidget(self.buttonbox, 19, 0, 1, 2)
        pb_lithfile.pressed.connect(self.get_lithfile)
        pb_headfile.pressed.connect(self.get_headfile)

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
        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        lithfile = self.le_lithfile.text()
        headfile = self.le_headfile.text()
        boreholeid = self.cmb_boreholeid.currentText()

        df = get_CGS(lithfile, headfile, boreholeid)

        if df is None:
            QtWidgets.QMessageBox.warning(
                self.parent,
                "Error",
                "Could not import dataset. Please make sure it not another format.",
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return False

        df = rename_or_create(df, self.cmb_boreholeid.currentText(), "Boreholeid")
        df = rename_or_create(df, self.cmb_companyno.currentText(), "Companyno")
        df = rename_or_create(df, self.cmb_depthfrom.currentText(), "Depth from")
        df = rename_or_create(df, self.cmb_depthto.currentText(), "Depth to")
        df = rename_or_create(df, self.cmb_lat.currentText(), "Declat")
        df = rename_or_create(df, self.cmb_long.currentText(), "Declon")
        df = rename_or_create(df, self.cmb_drilldate.currentText(), "Drill date")
        df = rename_or_create(df, self.cmb_elevation.currentText(), "Elevation")
        df = rename_or_create(df, self.cmb_lith.currentText(), "Lithology")
        df = rename_or_create(df, self.cmb_strat.currentText(), "Stratigraphy")
        df = rename_or_create(df, self.cmb_rank.currentText(), "Rank")

        # df.rename(columns=rencols, inplace=True)
        df = df.dropna(subset=["Depth from", "Depth to"])

        self.outdata["Borehole"] = df
        return True

    def saveproj(self):
        """
        Save project data from class.

        """
        # self.saveobj(self.ifile)
        # self.saveobj(self.hfile)

    def cmb_settext(self, obj, list_a):
        """Set the text in a combobox."""
        list_b = [obj.itemText(i) for i in range(obj.count())]
        set_a_folded = {item.casefold() for item in list_a}

        matches_b = [item for item in list_b if item.casefold() in set_a_folded]

        if matches_b:
            obj.setCurrentText(matches_b[0])
        else:
            obj.setCurrentText("None")

    def fillcombos(self):
        """Load in data and fill combo boxes."""

        lithfile = self.le_lithfile.text()
        headfile = self.le_headfile.text()

        dfl = pd.read_excel(lithfile, nrows=0)
        dfh = pd.read_excel(headfile, nrows=0)

        colsh = dfh.columns.tolist()
        colsl = dfl.columns.tolist()

        allcols = list(set(colsh + colsl))
        allcols.sort()
        allcols = ["None"] + allcols
        self.cmb_update(self.cmb_boreholeid, allcols)
        self.cmb_update(self.cmb_companyno, allcols)
        self.cmb_update(self.cmb_depthfrom, allcols)
        self.cmb_update(self.cmb_depthto, allcols)
        self.cmb_update(self.cmb_lat, allcols)
        self.cmb_update(self.cmb_long, allcols)
        self.cmb_update(self.cmb_drilldate, allcols)
        self.cmb_update(self.cmb_elevation, allcols)
        self.cmb_update(self.cmb_lith, allcols)
        self.cmb_update(self.cmb_strat, allcols)
        self.cmb_update(self.cmb_rank, allcols)

        self.cmb_settext(self.cmb_boreholeid, ["Boreholeid", "Borehole id"])
        self.cmb_settext(self.cmb_companyno, ["Companyno", "Company no"])
        self.cmb_settext(self.cmb_depthfrom, ["Depth from", "Depthfrom"])
        self.cmb_settext(self.cmb_depthto, ["Depth to", "Depthto"])
        self.cmb_settext(self.cmb_lat, ["Declat", "Latitude", "lat"])
        self.cmb_settext(self.cmb_long, ["Declon", "Longitude", "long"])
        self.cmb_settext(self.cmb_drilldate, ["Drill date", "Drilldate"])
        self.cmb_settext(self.cmb_elevation, ["Elevation"])
        self.cmb_settext(self.cmb_lith, ["Lithology"])
        self.cmb_settext(self.cmb_strat, ["Stratigraphy"])
        self.cmb_settext(self.cmb_rank, ["Rank"])

    def get_headfile(self):
        """Get the header filename."""
        self.le_headfile.setText("")

        ext = (
            "Common formats (*.xls *.xlsx *.csv);;"
            "Excel (*.xls *.xlsx);;"
            "Comma Delimited (*.csv)"
        )

        ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Open File", ".", ext
        )

        if not ifile:
            return False
        os.chdir(os.path.dirname(ifile))

        self.le_headfile.setText(ifile)

        if self.le_lithfile.text() != "":
            self.fillcombos()

    def get_lithfile(self):
        """Get the lithology filename."""
        self.le_lithfile.setText("")

        ext = (
            "Common formats (*.xls *.xlsx *.csv);;"
            "Excel (*.xls *.xlsx);;"
            "Comma Delimited (*.csv)"
        )

        ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Open File", ".", ext
        )

        if not ifile:
            return False
        os.chdir(os.path.dirname(ifile))

        self.le_lithfile.setText(ifile)

        if self.le_headfile.text() != "":
            self.fillcombos()


def rename_or_create(df, old_name, new_name, default_value=None):
    """
    Rename or create a column if it does not exist.

    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe
    old_name : str
        Old column name
    new_name : str
        New column name
    default_value : any, optional
        Default value for new column, by default None

    Returns
    -------
    df : Pandas DataFrame
        Output dataframe.
    """
    if old_name in df.columns:
        # Rename the column if it exists
        df = df.rename(columns={old_name: new_name})
    elif new_name not in df.columns:
        # Create the new column if neither name is present
        df[new_name] = default_value

    return df


def get_CGS(lithfile, headerfile, boreholeid):
    """
    Borehole Import.

    Parameters
    ----------
    lithfile : str
        Filename to import.
    headerfile : str
        Filename to import.
        Filename to import.
    boreholeid : str
        Column to merge on.

    Returns
    -------
    df : Pandas DataFrame
        Pandas dataframe with borehole information.

    """
    df = pd.read_excel(lithfile)
    hdf = pd.read_excel(headerfile)

    dropcols = set(df.columns) & set(hdf.columns)
    dropcols.remove(boreholeid)
    hdf.drop(columns=dropcols, inplace=True)

    df = df.merge(hdf, on=boreholeid)

    df[boreholeid] = df[boreholeid].astype(str)

    return df


def _testfn():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    os.chdir(r"D:\workdata\PyGMI Test Data\boreholes")

    tmp1 = ImportData()
    tmp1.settings()


if __name__ == "__main__":
    _testfn()
