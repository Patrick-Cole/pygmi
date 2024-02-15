# -----------------------------------------------------------------------------
# Name:        show_table.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2023 Council for Geoscience
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
"""Routine which displays a table graphically with various stats."""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pandas as pd

from pygmi.misc import ContextModule


class BasicStats(ContextModule):
    """Show a summary of basic stats."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None

        self.resize(640, 480)

        self.tablewidget = QtWidgets.QTableWidget()
        self.pushbutton_save = QtWidgets.QPushButton('Save')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        hbl = QtWidgets.QHBoxLayout(self)
        vbl = QtWidgets.QVBoxLayout()

        vbl.addWidget(self.pushbutton_save)
        hbl.addWidget(self.tablewidget)
        hbl.addLayout(vbl)

        self.setWindowTitle('Basic Vector Statistics')

        self.pushbutton_save.clicked.connect(self.save)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        gdf = self.indata['Vector'][0]

        stats1 = {}
        stats1['mean'] = gdf.mean(numeric_only=True)
        stats1['std'] = gdf.std(numeric_only=True)
        stats1['median'] = gdf.median(numeric_only=True)
        stats1['min'] = gdf.min(numeric_only=True)
        stats1['max'] = gdf.max(numeric_only=True)

        dfstats = pd.DataFrame(stats1)
        self.data = dfstats

        cols = dfstats.columns.tolist()
        rows = dfstats.index.tolist()
        data = dfstats.to_numpy()

        self.tablewidget.setRowCount(data.shape[0])
        self.tablewidget.setColumnCount(data.shape[1])
        self.tablewidget.setHorizontalHeaderLabels(cols)
        self.tablewidget.setVerticalHeaderLabels(rows)

        for i, _ in enumerate(cols):
            item = self.tablewidget.horizontalHeaderItem(i)
            fnt = item.font()
            fnt.setBold(True)
            item.setFont(fnt)

        for i, _ in enumerate(rows):
            item = self.tablewidget.verticalHeaderItem(i)
            fnt = item.font()
            fnt.setBold(True)
            item.setFont(fnt)

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                txt = f' {data[row, col]:,.5f}'
                txt = QtWidgets.QLabel(txt)
                txt.setAlignment(Qt.AlignRight)

                self.tablewidget.setCellWidget(row, col, txt)

        self.tablewidget.resizeColumnsToContents()
        self.show()

    def save(self):
        """
        Save Table.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'CSV Format (*.csv)'
        ifile, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Table',
                                                         '.', ext)
        if ifile == '':
            return False

        self.data.to_csv(ifile)

        return True


def _testfn():
    """Calculate structural complexity."""
    import sys
    from pygmi.vector.iodefs import ImportVector

    sfile = r'D:\Work\Programming\geochem\all_geochem.shp'

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportVector()
    tmp1.ifile = sfile
    tmp1.settings(True)

    tmp2 = BasicStats()
    tmp2.indata = tmp1.outdata
    tmp2.run()

    app.exec()


if __name__ == "__main__":
    _testfn()
