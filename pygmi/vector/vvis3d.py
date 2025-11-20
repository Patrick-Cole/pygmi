# -----------------------------------------------------------------------------
# Name:        vvid3d.py (part of PyGMI)
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
"""Code for the 3d model creation."""

import os
import sys
import numpy as np

from PySide6 import QtWidgets

import pyvista as pv
from pyvistaqt import QtInteractor

from pygmi.misc import ContextModule


class Mod3dDisplay(ContextModule):
    """
    Widget class to call the main interface.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lmod1 = None
        self.outdata = self.indata

        if hasattr(parent, 'showtext'):
            self.showtext = parent.showtext
        else:
            self.showtext = sys.stdout

        self.setWindowTitle('3D Voxel Model Display')

        # Back to normal stuff
        self.pb_save = QtWidgets.QPushButton('Save to Image File (JPG or PNG)')
        self.plotter = QtInteractor(self)
        self.cb_volume = QtWidgets.QCheckBox('Slice in Opaque Volume')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = 'pfmod.cm.show3dmodel'
        hbl = QtWidgets.QHBoxLayout(self)
        vbl_cmodel = QtWidgets.QVBoxLayout()
        vbl = QtWidgets.QVBoxLayout()

        vbl_cmodel.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint)

        sizepolicy_pb = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Maximum)

        self.pb_save.setSizePolicy(sizepolicy_pb)

        vbl.addWidget(self.cb_volume)
        vbl.addWidget(self.pb_save)
        vbl.addWidget(self.buttonbox)
        vbl_cmodel.addWidget(self.plotter)
        hbl.addLayout(vbl_cmodel)
        hbl.addLayout(vbl)

        self.pb_save.clicked.connect(self.save)
        self.cb_volume.stateChanged.connect(self.update_plot)

    def closeEvent(self, QCloseEvent):
        """
        Close event.

        Parameters
        ----------
        QCloseEvent : TYPE
            Close event.

        Returns
        -------
        None.

        """
        super().closeEvent(QCloseEvent)
        self.plotter.close()

    def save(self):
        """
        Save a jpg.

        Returns
        -------
        None.

        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'JPG (*.jpg);;PNG (*.png)')
        if filename == '':
            return
        os.chdir(os.path.dirname(filename))

        self.plotter.screenshot(filename)

    def data_init(self):
        """
        Initialise data.

        Returns
        -------
        None.

        """
        self.outdata = self.indata

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Voxel' not in self.indata:
            self.showlog('No 3D voxel model. You may need to execute that '
                         'module first')
            return False

        self.show()
        self.update_plot()

        return True

    def update_plot(self):
        """
        Update 3D model.

        Returns
        -------
        None.

        """
        QtWidgets.QApplication.processEvents()

        vdat = self.indata['Voxel'][0]

        # Update 3D model
        self.spacing = vdat.spacing
        self.origin = vdat.origin
        self.gdata = vdat.data

        # Create the spatial reference
        grid = pv.ImageData()

        values = vdat.data
        # Set the grid dimensions: shape + 1 because we want to inject our
        # values on the CELL data
        grid.dimensions = np.array(values.shape) + 1

        # Edit the spatial reference
        # The bottom left corner of the data set
        grid.origin = vdat.origin
        grid.spacing = vdat.spacing  # These are the cell sizes along each axis

        # Add the data values to the cell data
        grid.cell_data['values'] = values.flatten(
            order='F')  # Flatten the array

        # Get rid of nan values
        grid = grid.threshold()

        # Now plot the grid
        # grid.plot(show_edges=True)

        self.plotter.clear()

        if self.cb_volume.isChecked():
            # self.plotter.add_volume(grid)
            self.plotter.add_mesh(grid, opacity=0.5)
            self.plotter.add_mesh_slice(grid)
        else:
            self.plotter.add_mesh_clip_plane(grid, normal=[-1, 0, 0])

        self.plotter.add_axes()
        # self.plotter.show_grid()
        # p.show()

        # self.plotter.show_grid(use_2d=True)


def _testfn():
    """Test function."""
    from pygmi.vector.iodefs import import_ubc

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    ifile = r"D:\UBC_Files\voxel.msh"

    vdat = import_ubc(ifile)

    # IM = ImportMod3D()
    # IM.ifile = ifile
    # IM.settings(True)

    print('Model loaded')

    M3D = Mod3dDisplay()
    M3D.indata['Voxel'] = [vdat]
    M3D.data_init()
    M3D.run()
    M3D.exec()


if __name__ == "__main__":
    _testfn()
