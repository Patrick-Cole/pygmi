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
import numpy as np

from pygmi.misc import ContextModule


class BasicStats3D(ContextModule):
    """Show a summary of basic stats."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.combobox = QtWidgets.QComboBox()
        self.tablewidget = QtWidgets.QTableWidget()
        self.pushbutton_save = QtWidgets.QPushButton('Save')

        self.setupui()
        self.bands = None
        self.cols = None
        self.data = None

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
        vbl.addWidget(self.combobox)

        hbl.addWidget(self.tablewidget)
        hbl.addLayout(vbl)

        self.setWindowTitle('Basic Statistics')

        self.combobox.currentIndexChanged.connect(self.combo)
        self.pushbutton_save.clicked.connect(self.save)

    def combo(self):
        """
        Combo.

        Returns
        -------
        None.

        """
        i = self.combobox.currentIndex()
        data = self.data[i][:, 1:]

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                text = data[row, col]
                if not isinstance(text, str):
                    text = f'{text:,.1f}'
                self.tablewidget.setCellWidget(
                    row, col, QtWidgets.QLabel(text))

        self.tablewidget.resizeColumnsToContents()

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        data = self.indata['Model3D']
        self.bands, self.cols, self.data = basicstats3d_calc(data)

        data = self.data[0][:, 1:]
        rows = self.data[0][:, 0]
        cols = self.cols[1:]

        if len(self.data) == 1:
            self.combobox.hide()

        self.combobox.addItems(self.bands)
        self.tablewidget.setRowCount(data.shape[0])
        self.tablewidget.setColumnCount(data.shape[1])
        self.tablewidget.setHorizontalHeaderLabels(cols)
        self.tablewidget.setVerticalHeaderLabels(rows)

        self.combo()

        tmp = self.exec_()

        if tmp != 1:
            return False

        return True

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

        savetable(ifile, self.bands, self.cols, self.data)

        return True


def basicstats3d_calc(lmod):
    """
    Calculate statistics.

    Parameters
    ----------
    lmod : PyGMI LithModel.
        PyGMI lithology model.

    Returns
    -------
    bands : list
        Band list, currently only 'Data Column'
    cols : list
        Columns for the table
    dattmp : list
        List of arrays containing statistics.

    """
    stats = []

    for i in lmod:
        dxy = i.dxy
        dz = i.d_z
        vol = dxy*dxy*dz

        for j in i.lith_list:
            lindex = i.lith_list[j].lith_index
            lvol = (i.lith_index == lindex).sum()
            lvol = lvol * vol

            density = i.lith_list[j].density * 1000

            mass = lvol * density

            lidx = i.lith_index.copy()
            lidx.shape = (-1, lidx.shape[-1])

            air = (lidx == -1).sum(1)
            thick = (lidx == lindex).sum(1)

            lidx[lidx != lindex] = -1

            depthtotop = np.argmax(lidx, 1) - air
            depthtobot = depthtotop + thick

            depthtotop = depthtotop[thick > 0].mean() * dz
            depthtobot = depthtobot[thick > 0].mean() * dz
            thick = thick[thick > 0].mean() * dz

            srow = [j]
            srow.append(lvol)
            srow.append(mass)
            srow.append(thick)
            srow.append(depthtotop)
            srow.append(depthtobot)
            stats.append(srow)

    bands = ['Data Column']
    cols = ['Lithology', 'Volume (m\u00b3)', 'Mass (kg)',
            'Mean Thickness (m)', 'Mean Depth to Top (m)',
            'Mean Depth to Bottom (m)']
    dattmp = [np.array(stats, dtype=object)]
    return bands, cols, dattmp


def savetable(ofile, bands, cols, data):
    """
    Save tabular data.

    Parameters
    ----------
    ofile : str
        Output file name.
    bands : list
        List of bands.
    cols : list
        List of column headings.
    data : list
        List of arrays containing statistics.

    Returns
    -------
    None.

    """
    with open(ofile, 'a', encoding='utf-8') as fobj:
        htmp = cols[0]
        for i in cols[1:]:
            htmp += ',' + i

        for k, _ in enumerate(bands):
            fobj.write(bands[k]+'\n')
            fobj.write(htmp+'\n')
            for i, _ in enumerate(data[k]):
                rtmp = str(data[k][i][0])
                for j in range(1, len(data[k][0])):
                    rtmp += ','+str(data[k][i][j])
                fobj.write(rtmp+'\n')
            fobj.write('\n')


def _testfn():
    """Test routine."""
    import sys
    from pygmi.pfmod.iodefs import ImportMod3D
    ifile = r"D:\Workdata\PyGMI Test Data\Potential Field Modelling\small_upper.npz"

    app = QtWidgets.QApplication(sys.argv)

    DM = ImportMod3D()
    DM.ifile = ifile
    DM.settings(nodialog=True)

    dat = DM.outdata

    tmp2 = BasicStats3D()
    tmp2.indata = dat
    tmp2.run()


if __name__ == "__main__":
    _testfn()
