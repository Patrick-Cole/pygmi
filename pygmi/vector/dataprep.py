# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
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
"""Data preparation for vector data."""

import copy
import glob
import os
from functools import partial

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
from shapely import Polygon

from pygmi.misc import BasicModule, ContextModule, ProgressBarText
from pygmi.raster.datatypes import Data
from pygmi.raster.reproj import GroupProj
from pygmi.vector.datatypes import VoxModel
from pygmi.vector.minc import minc


class PointCut(BasicModule):
    """
    GUI to cut data using shapefiles.

    This class cuts point datasets using a boundary defined by a polygon
    shapefile.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True

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
        if "Vector" in self.indata:
            data = copy.deepcopy(self.indata["Vector"][0])
        else:
            self.showlog("No point or vector data")
            return False

        if not nodialog:
            ext = "Shape file (*.shp)"
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, "Open Shape File", ".", ext
            )
            if self.ifile == "":
                return False

        os.chdir(os.path.dirname(self.ifile))
        data = cut_point(data, self.ifile, self.showlog)

        if data is None:
            return False

        if self.pbar is not None:
            self.pbar.to_max()
        self.outdata["Vector"] = [data]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class DataGrid(BasicModule):
    """
    GUI to grid point data.

    This class grids point data using a nearest neighbourhood technique.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None

        self.le_dxy = QtWidgets.QLineEdit("1.0")
        self.le_null = QtWidgets.QLineEdit("0.0")
        self.le_bdist = QtWidgets.QLineEdit("4.0")

        self.cmb_dataid = QtWidgets.QComboBox()
        self.cmb_grid_method = QtWidgets.QComboBox()
        self.cmb_grid_type = QtWidgets.QComboBox()
        self.cmb_grid_dem = QtWidgets.QComboBox()
        self.lbl_rows = QtWidgets.QLabel("Rows: 0")
        self.lbl_cols = QtWidgets.QLabel("Columns: 0")
        self.lbl_layers = QtWidgets.QLabel("Layers: 0")
        self.lbl_bdist = QtWidgets.QLabel("Blanking Distance:")
        self.lbl_method = QtWidgets.QLabel("Gridding Method:")
        self.lbl_dem = QtWidgets.QLabel("DEM Grid:")

        self.cmb_line = QtWidgets.QComboBox()
        self.cmb_z = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)

        self.buttonbox.htmlfile = "vector.dm.gridding"
        lbl_band = QtWidgets.QLabel("Column to Grid:")
        lbl_dxy = QtWidgets.QLabel("Cell Size:")
        lbl_null = QtWidgets.QLabel("Null Value:")
        lbl_type = QtWidgets.QLabel("Gridding Type:")
        self.lbl_line = QtWidgets.QLabel("Line Number:")
        self.lbl_z = QtWidgets.QLabel("Z Coordinate Value:")

        val = QtGui.QDoubleValidator(1e-300, np.inf, -1)
        val.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.Language.C))
        val2 = QtGui.QDoubleValidator(-np.inf, np.inf, -1)
        val2.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        val2.setLocale(QtCore.QLocale(QtCore.QLocale.Language.C))

        self.le_dxy.setValidator(val)
        self.le_null.setValidator(val2)
        self.le_bdist.setValidator(val)

        self.cmb_grid_method.addItems(
            ["Nearest Neighbour", "Linear", "Cubic", "Minimum Curvature"]
        )
        self.cmb_grid_type.addItems(["Raster", "Section", "Voxel"])
        self.cmb_line.hide()
        self.cmb_z.hide()
        self.lbl_line.hide()
        self.lbl_z.hide()
        self.lbl_layers.hide()
        self.cmb_grid_dem.hide()
        self.lbl_dem.hide()

        self.setWindowTitle("Dataset Gridding")

        gl_main.addWidget(lbl_type, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_grid_type, 0, 1, 1, 1)
        gl_main.addWidget(self.lbl_method, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_grid_method, 1, 1, 1, 1)
        gl_main.addWidget(self.lbl_dem, 12, 0, 1, 1)
        gl_main.addWidget(self.cmb_grid_dem, 12, 1, 1, 1)
        gl_main.addWidget(lbl_dxy, 2, 0, 1, 1)
        gl_main.addWidget(self.le_dxy, 2, 1, 1, 1)
        gl_main.addWidget(self.lbl_rows, 9, 0, 1, 2)
        gl_main.addWidget(self.lbl_cols, 10, 0, 1, 2)
        gl_main.addWidget(self.lbl_layers, 11, 0, 1, 2)
        gl_main.addWidget(lbl_band, 5, 0, 1, 1)
        gl_main.addWidget(self.cmb_dataid, 5, 1, 1, 1)
        gl_main.addWidget(lbl_null, 6, 0, 1, 1)
        gl_main.addWidget(self.le_null, 6, 1, 1, 1)
        gl_main.addWidget(self.lbl_bdist, 7, 0, 1, 1)
        gl_main.addWidget(self.le_bdist, 7, 1, 1, 1)
        gl_main.addWidget(self.lbl_line, 8, 0, 1, 1)
        gl_main.addWidget(self.cmb_line, 8, 1, 1, 1)
        gl_main.addWidget(self.lbl_z, 3, 0, 1, 1)
        gl_main.addWidget(self.cmb_z, 3, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 17, 0, 1, 4)

        self.le_dxy.textChanged.connect(self.dxy_change)
        self.cmb_z.currentIndexChanged.connect(self.dxy_change)
        self.cmb_line.currentIndexChanged.connect(self.dxy_change)
        self.cmb_grid_method.currentIndexChanged.connect(self.grid_method_change)
        self.cmb_grid_type.currentIndexChanged.connect(self.grid_type_change)

    def dxy_change(self):
        """
        When dxy is changed on the interface, this updates rows and columns.

        Returns
        -------
        None.

        """
        txt = str(self.le_dxy.text())
        if txt.replace(".", "", 1).isdigit():
            self.dxy = float(self.le_dxy.text())
        else:
            return

        data = self.indata["Vector"][0]

        x = data.geometry.x.values
        y = data.geometry.y.values
        zcol = self.cmb_z.currentText()
        z = np.array([0, self.dxy])
        if zcol != "":
            z = data[zcol].values

        if self.cmb_grid_type.currentText() == "Section" and zcol != "":
            line = self.cmb_line.currentText()
            if line.lower() not in ["none", ""]:
                data1 = data[data.line == line]
            else:
                data1 = data

            x = data1.geometry.x.values
            y = data1.geometry.y.values
            x = xy_to_r(x, y, self.piter)

            y = data1[zcol].values

        cols = round(np.ptp(x) / self.dxy)
        rows = round(np.ptp(y) / self.dxy)
        layers = round(np.ptp(z) / self.dxy)

        self.lbl_rows.setText("Rows: " + str(rows))
        self.lbl_cols.setText("Columns: " + str(cols))
        self.lbl_layers.setText("Layers: " + str(layers))

    def grid_method_change(self):
        """
        When grid method is changed, this updated hidden controls.

        Returns
        -------
        None.

        """
        txt = self.cmb_grid_type.currentText()

        if txt != "Voxel":
            self.lbl_bdist.show()
            self.le_bdist.show()
        else:
            self.lbl_bdist.hide()
            self.le_bdist.hide()

    def grid_type_change(self):
        """Check whether section is checked."""
        txt = self.cmb_grid_type.currentText()

        if txt == "Section":
            self.cmb_line.show()
            self.cmb_z.show()
            self.lbl_line.show()
            self.lbl_z.show()
            self.lbl_layers.hide()
            self.cmb_grid_method.show()
            self.lbl_method.show()
            self.cmb_grid_dem.hide()
            self.lbl_dem.hide()
        elif txt == "Voxel":
            self.cmb_line.hide()
            self.cmb_z.show()
            self.lbl_line.hide()
            self.lbl_z.show()
            self.lbl_layers.show()
            self.cmb_grid_method.hide()
            self.lbl_method.hide()
            self.cmb_grid_dem.show()
            self.lbl_dem.show()
        else:
            self.cmb_line.hide()
            self.cmb_z.hide()
            self.lbl_line.hide()
            self.lbl_z.hide()
            self.lbl_layers.hide()
            self.cmb_grid_method.show()
            self.lbl_method.show()
            self.cmb_grid_dem.hide()
            self.lbl_dem.hide()

        self.dxy_change()

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
        tmp = []
        if "Vector" not in self.indata:
            self.showlog("No Point Data")
            return False

        data = self.indata["Vector"][0]

        if data.geom_type.iloc[0] != "Point":
            self.showlog("No Point Data")
            return False

        demlist = ["None"]
        if "Raster" in self.indata:
            tmp = [i.dataid for i in self.indata["Raster"]]
            demlist += tmp

        self.cmb_update(self.cmb_grid_dem, demlist)

        if self.dxy is None:
            x = data.geometry.x.values
            y = data.geometry.y.values

            dx = np.ptp(x) / np.sqrt(x.size)
            dy = np.ptp(y) / np.sqrt(y.size)
            self.dxy = max(dx, dy)
            self.dxy = min([np.ptp(x), np.ptp(y), self.dxy])

        self.le_dxy.setText(f"{self.dxy:.8f}")
        self.grid_type_change()

        filt = (data.columns != "geometry") & (data.columns != "line")

        cols = list(data.columns[filt])
        self.cmb_update(self.cmb_dataid, cols)
        self.cmb_update(self.cmb_z, cols)

        lines = data.line[data.line != "nan"].unique()

        self.cmb_update(self.cmb_line, lines)

        self.grid_method_change()
        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        if not self.check_validation():
            return False

        flag = self.acceptall()

        return flag

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.dxy)
        self.saveobj(self.le_dxy)
        self.saveobj(self.le_null)
        self.saveobj(self.le_bdist)
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.cmb_grid_method)
        self.saveobj(self.cmb_grid_type)
        self.saveobj(self.cmb_line)
        self.saveobj(self.cmb_grid_dem)
        self.saveobj(self.cmb_z)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = float(self.le_dxy.text())
        method = self.cmb_grid_method.currentText()
        line = self.cmb_line.currentText()
        nullvalue = float(self.le_null.text())
        bdist = float(self.le_bdist.text())
        data = self.indata["Vector"][0]
        dataid = self.cmb_dataid.currentText()
        zcol = self.cmb_z.currentText()
        demid = self.cmb_grid_dem.currentText()
        scoords = None
        newdat = []

        if bdist < 1:
            bdist = None
            self.showlog("Blanking distance too small.")
        if (
            line.lower() not in ["none", ""]
            and self.cmb_grid_type.currentText() == "Section"
        ):
            data1 = data[data.line == line]
        else:
            data1 = data

        if dataid == zcol:
            data2 = data1[["geometry", dataid]]
        else:
            data2 = data1[["geometry", dataid, zcol]]
        data2 = data2.dropna()

        filt = data2[dataid] != nullvalue
        if filt.ndim > 1:
            filt = filt.iloc[:, 0]

        x = data2.geometry.x.values[filt]
        y = data2.geometry.y.values[filt]
        val = data2[dataid].values[filt]

        if val.ndim > 1:
            val = val[:, 0]

        if self.cmb_grid_type.currentText() == "Section":
            x1 = x
            y1 = y
            x = xy_to_r(x, y, self.piter)

            y = data2[zcol].values
            scoords = np.transpose([x1, y1, x])
            scoords = np.unique(scoords, axis=0)
            sortidx = scoords[:, 2].argsort()
            scoords = scoords[sortidx]

        if self.cmb_grid_type.currentText() == "Voxel":
            z = data2[zcol].values[filt]

            if z.ndim > 1:
                z = z[:, 0]
            ddat = None
            if "Raster" in self.indata:
                for i in self.indata["Raster"]:
                    if i.dataid == demid:
                        ddat = i

            dat = gridvolume(x, y, z, val, dxy, dat=ddat)
            if dat is None:
                return False
        else:
            dat = gridxyz(
                x,
                y,
                val,
                dxy,
                nullvalue=nullvalue,
                method=method,
                bdist=bdist,
                showlog=self.showlog,
            )
        dat.dataid = dataid
        dat.crs = data2.crs

        if self.cmb_grid_type.currentText() == "Section":
            dat.metadata["Raster"]["Section"] = True
            dat.metadata["Raster"]["SectionCoords"] = scoords

        newdat.append(dat)

        if self.cmb_grid_type.currentText() == "Voxel":
            self.outdata["Voxel"] = newdat
        else:
            self.outdata["Raster"] = newdat
        self.outdata["Vector"] = self.indata["Vector"]

        return True


class DataReproj(BasicModule):
    """
    GUI to reproject vector data.

    This class reprojects datasets using the GeoPandas routines.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_wkt = None
        self.targ_wkt = None

        self.in_proj = GroupProj("Input Projection")
        self.out_proj = GroupProj("Output Projection")

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = "vector.dm.reproj"

        self.setWindowTitle("Dataset Reprojection")

        gl_main.addWidget(self.in_proj, 0, 0, 1, 1)
        gl_main.addWidget(self.out_proj, 0, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 1, 0, 1, 2)

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
        if "Vector" not in self.indata:
            self.showlog("No vector data.")
            return False

        if self.indata["Vector"][0].crs is not None:
            self.orig_wkt = self.indata["Vector"][0].crs.to_wkt()

        if self.orig_wkt is None:
            indx = self.in_proj.cmb_datum.findText(r"WGS 84")
            self.in_proj.cmb_datum.setCurrentIndex(indx)
            self.orig_wkt = self.in_proj.wkt
        else:
            self.in_proj.set_current(self.orig_wkt)

        if self.targ_wkt is None:
            indx = self.in_proj.cmb_datum.findText(r"WGS 84")
            self.out_proj.cmb_datum.setCurrentIndex(indx)
            self.targ_wkt = self.out_proj.wkt
        else:
            self.out_proj.set_current(self.targ_wkt)

        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        self.orig_wkt = self.in_proj.wkt
        self.targ_wkt = self.out_proj.wkt

        self.outdata["Vector"] = []
        for ivec in self.indata["Vector"]:
            ivec = ivec.set_crs(self.in_proj.wkt)
            self.outdata["Vector"].append(ivec.to_crs(self.out_proj.wkt))

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.orig_wkt)
        self.saveobj(self.targ_wkt)


class Metadata(ContextModule):
    """
    GUI to display and edit vector metadata.

    This class allows the editing of the metadata for a vector dataset using a
    GUI.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    banddata : dictionary
        band data
    bandid : dictionary
        dictionary of strings containing band names.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmb_bandid = QtWidgets.QComboBox()
        self.proj = GroupProj("Input Projection")

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = "vector.cm.meta"
        lbl_bandid = QtWidgets.QLabel("Source:")

        self.setWindowTitle("Vector Dataset Metadata")

        gl_main.addWidget(lbl_bandid, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_bandid, 0, 1, 1, 3)
        gl_main.addWidget(self.proj, 2, 0, 1, 4)
        gl_main.addWidget(self.buttonbox, 4, 0, 1, 4)

        self.resize(-1, 320)
        self.buttonbox.buttonbox.accepted.connect(self.acceptall)

    def acceptall(self):
        """
        Accept option.

        Returns
        -------
        None.

        """
        wkt = self.proj.wkt

        for tmp in self.indata["Vector"]:
            if wkt == "None":
                tmp.crs = None
            else:
                tmp.crs = CRS.from_wkt(wkt)

        self.accept()

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        bandid = []
        if self.indata["Vector"][0].crs is None:
            self.proj.set_current("None")
        else:
            self.proj.set_current(self.indata["Vector"][0].crs.to_wkt())

        for i in self.indata["Vector"]:
            if "source" in i.attrs:
                bandid.append(i.attrs["source"])
            else:
                bandid.append("Unknown")

        self.cmb_bandid.clear()
        self.cmb_bandid.addItems(bandid)

        self.show()


class TextFileSplit(BasicModule):
    """
    GUI to split a text file into smaller text files.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_import = True

        self.le_ifile = QtWidgets.QLineEdit("")
        self.le_files = QtWidgets.QLineEdit("1")
        self.le_lines = QtWidgets.QLineEdit("1")
        self.le_bytes = QtWidgets.QLineEdit("1")
        self.cb_allfiles = QtWidgets.QCheckBox(
            "Split all text files with same extension in current directory"
        )

        self.cmb_method = QtWidgets.QComboBox()
        self.lbl_totsize = QtWidgets.QLabel("0")
        self.lbl_totlines = QtWidgets.QLabel("0")

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        pb_ifile = QtWidgets.QPushButton(" Filename")
        gl_main = QtWidgets.QGridLayout(self)

        self.buttonbox.htmlfile = "vector.dm.txtfilesplit"
        lbl_files = QtWidgets.QLabel("Number of files:")
        lbl_lines = QtWidgets.QLabel("Max lines per file:")
        lbl_bytes = QtWidgets.QLabel("Max bytes per file:")
        lbl_method = QtWidgets.QLabel("Split Method:")
        self.lbl_totsize = QtWidgets.QLabel("0")
        self.lbl_totlines = QtWidgets.QLabel("0")

        val = QtGui.QIntValidator(1, 2147483647)

        self.le_files.setValidator(val)
        self.le_lines.setValidator(val)
        self.le_bytes.setValidator(val)
        self.le_files.setEnabled(True)
        self.le_lines.setDisabled(True)
        self.le_bytes.setDisabled(True)

        self.cmb_method.addItems(["Files", "Bytes", "Lines"])

        self.setWindowTitle("Text File Split")

        gl_main.addWidget(pb_ifile, 0, 0, 1, 1)
        gl_main.addWidget(self.le_ifile, 0, 1, 1, 1)
        gl_main.addWidget(lbl_method, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_method, 1, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel("Total File Size:"), 2, 0, 1, 1)
        gl_main.addWidget(self.lbl_totsize, 2, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel("Total Lines:"), 3, 0, 1, 1)
        gl_main.addWidget(self.lbl_totlines, 3, 1, 1, 1)
        gl_main.addWidget(lbl_files, 4, 0, 1, 1)
        gl_main.addWidget(self.le_files, 4, 1, 1, 1)
        gl_main.addWidget(lbl_lines, 5, 0, 1, 1)
        gl_main.addWidget(self.le_lines, 5, 1, 1, 1)
        gl_main.addWidget(lbl_bytes, 6, 0, 1, 1)
        gl_main.addWidget(self.le_bytes, 6, 1, 1, 1)
        gl_main.addWidget(self.cb_allfiles, 7, 0, 1, 2)
        gl_main.addWidget(self.buttonbox, 8, 0, 1, 4)

        pb_ifile.pressed.connect(self.get_ifile)

        self.le_files.textChanged.connect(self.change_method)
        self.le_lines.textChanged.connect(self.change_method)
        self.le_bytes.textChanged.connect(self.change_method)
        self.cmb_method.currentIndexChanged.connect(self.change_method)

    def change_method(self):
        """Update fields when method changes."""
        method = self.cmb_method.currentText()

        totlines = int(self.lbl_totlines.text().replace(",", ""))
        totbytes = int(self.lbl_totsize.text().replace(",", ""))

        try:
            numfiles = int(self.le_files.text().replace(",", ""))
            numlines = int(self.le_lines.text().replace(",", ""))
            numbytes = int(self.le_bytes.text().replace(",", ""))
        except ValueError:
            return

        if method == "Files":
            numlines = totlines // numfiles + 1
            numbytes = totbytes // numfiles + 1
            self.le_files.setEnabled(True)
            self.le_lines.setDisabled(True)
            self.le_bytes.setDisabled(True)
        elif method == "Lines":
            numfiles = totlines // numlines + 1
            numbytes = totbytes // numfiles + 1
            self.le_files.setDisabled(True)
            self.le_lines.setEnabled(True)
            self.le_bytes.setDisabled(True)

        elif method == "Bytes":
            numfiles = totbytes // numbytes + 1
            numlines = totlines // numfiles + 1
            self.le_files.setDisabled(True)
            self.le_lines.setDisabled(True)
            self.le_bytes.setEnabled(True)

        self.le_files.blockSignals(True)
        self.le_lines.blockSignals(True)
        self.le_bytes.blockSignals(True)

        self.le_files.setText(f"{numfiles:,}")
        self.le_lines.setText(f"{numlines:,}")
        self.le_bytes.setText(f"{numbytes:,}")

        self.le_files.blockSignals(False)
        self.le_lines.blockSignals(False)
        self.le_bytes.blockSignals(False)

    def get_ifile(self):
        """
        Get input file information.

        Returns
        -------
        None.

        """
        ext = "Common formats (*.txt *.xyz *.csv);;"

        self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Open File", ".", ext
        )

        if not self.ifile:
            return

        self.le_ifile.setText(self.ifile)
        fsize = os.path.getsize(self.ifile)
        tlines = txtlinecnt(self.ifile)

        self.lbl_totsize.setText(f"{fsize:,}")
        self.lbl_totlines.setText(f"{tlines:,}")

        self.le_files.setValidator(QtGui.QIntValidator(1, fsize))
        self.le_lines.setValidator(QtGui.QIntValidator(1, tlines))
        self.le_bytes.setValidator(QtGui.QIntValidator(1, fsize))

        self.change_method()

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
        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.le_ifile)
        self.saveobj(self.le_files)
        self.saveobj(self.le_lines)
        self.saveobj(self.le_bytes)
        self.saveobj(self.cb_allfiles)
        self.saveobj(self.cmb_method)
        self.saveobj(self.lbl_totsize)
        self.saveobj(self.lbl_totlines)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        method = self.cmb_method.currentText()

        try:
            numfiles = int(self.le_files.text().replace(",", ""))
            numlines = int(self.le_lines.text().replace(",", ""))
            numbytes = int(self.le_bytes.text().replace(",", ""))
        except ValueError:
            return

        if method == "Bytes":
            num = numbytes
        elif method == "Lines":
            num = numlines
        else:
            num = numfiles

        if self.cb_allfiles.isChecked():
            _, fext = os.path.splitext(self.ifile)
            fdir = os.path.dirname(self.ifile)
            ifiles = glob.glob(os.path.join(fdir, f"*{fext}"))
        else:
            ifiles = [self.ifile]

        for ifile in ifiles:
            self.showlog(f"Splitting {os.path.basename(ifile)}...")
            filesplit(
                ifile, num, method.lower(), showlog=self.showlog, piter=self.piter
            )


def blanking(gdat, x, y, bdist, extent, dxy, nullvalue):
    """
    Blanks area further than a defined number of cells from input data.

    Parameters
    ----------
    gdat : numpy array
        grid data to blank.
    x : numpy array
        x coordinates.
    y : numpy array
        y coordinates.
    bdist : int
        Blanking distance in units for cell.
    extent : list
        extent of grid.
    dxy : float
        Cell size.
    nullvalue : float
        Null or nodata value.

    Returns
    -------
    gdat : numpy array
        Masked output array.

    """
    if bdist is None:
        return gdat

    mask = np.zeros_like(gdat)

    points = np.transpose([x, y])

    for xy in points:
        col = int((xy[0] - extent[0]) / dxy)
        row = int((xy[1] - extent[2]) / dxy)

        mask[row, col] = 1

    dist = distance_transform_edt(np.logical_not(mask))
    mask = dist > bdist

    gdat[mask] = nullvalue

    return gdat


def cut_point(data, ifile, showlog=print):
    """
    Cuts a point dataset.

    Cut a point dataset using a shapefile.

    Parameters
    ----------
    data : GeoDataFrame
        GeoPandas GeoDataFrame
    ifile : str
        shapefile used to cut data
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    data : GeoDataFrame
        GeoPandas GeoDataFrame
    """
    gdf = gpd.read_file(ifile)
    gdf = gdf[gdf.geometry != None]

    if "Polygon" not in gdf.geom_type.iloc[0]:
        showlog("No polygons in shapefile.")
        return None

    if data.crs is None and gdf.crs is not None:
        showlog(
            "Your vectors need a projection assigned, assuming it is the "
            "same as the shapefile."
        )
        data = data.set_crs(gdf.crs)
    elif data.crs is None:
        showlog("Your vectors need a projection assigned.")
        return None

    if gdf.crs is None:
        showlog(
            "Your shapefile needs a projection assigned, assuming it is "
            "the same as your vectors."
        )
        gdf = gdf.set_crs(data.crs)
    else:
        gdf = gdf.to_crs(data.crs)

    data = gpd.clip(data, gdf)
    data = data.explode()

    if data.size == 0:
        showlog("Nothing found in the clip area.")
        return None

    return data


def txtlinecnt(filename):
    """
    Count lines in text file.

    Parameters
    ----------
    filename : str
        filename of text file.

    Returns
    -------
    linecnt : int
        Total number of lines in a file.

    """
    with open(filename, "rb") as f:
        bufgen = iter(partial(f.raw.read, 1024 * 1024), b"")
        linecnt = sum(buf.count(b"\n") for buf in bufgen)
    return linecnt


def filesplit(ifile, num, mode="bytes", showlog=print, piter=None):
    """
    Split an input file into a number of output files.

    Parameters
    ----------
    ifile : str
        Input filename.
    num : int
        Number of bytes or lines to split by.
    mode : str, optional
        Can be 'bytes', 'files' or 'lines'. The default is 'bytes'.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress iterator. The default is None.

    Returns
    -------
    None.

    """
    if piter is None:
        piter = ProgressBarText().iter

    fsize = os.path.getsize(ifile)
    fname, fext = os.path.splitext(ifile)
    numfiles = 0
    numcnt = 0

    if mode == "files":
        numfiles = num
        numcnt = fsize // num + 1
    elif mode == "bytes":
        numcnt = num
        numfiles = fsize // num + 1
    elif mode == "lines":
        totlines = txtlinecnt(ifile)
        numfiles = totlines // num + 1
        numcnt = num

    txt = None
    with open(ifile, encoding="utf-8") as reader:
        for i in piter(range(numfiles)):
            if txt == "":
                continue

            with open(f"{fname}_{i + 1}{fext}", "w", encoding="utf-8") as writer:
                fread = 0
                while fread < numcnt:
                    txt = reader.readline()
                    if txt == "":
                        break
                    if mode == "lines":
                        fread += 1
                    else:
                        fread += len(txt)

                    writer.write(txt)


def gridxyz(
    x,
    y,
    z,
    dxy,
    *,
    nullvalue=1e20,
    method="Nearest Neighbour",
    bdist=4.0,
    showlog=print,
):
    """
    Grid xyz data.

    Parameters
    ----------
    x : numpy array
        X coordinate values.
    y : numpy array
        Y coordinate values.
    z : numpy array
        Z or data values.
    dxy : float
        Grid cell size, in distance units.
    nullvalue : float, optional
        null or nodata value. The default is 1e+20.
    method : str, optional
        Gridding method. The default is 'Nearest Neighbour'.
    bdist : float, optional
        Blanking distance. The default is 4.0.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    dat : pygmi.raster.datatypes.Data.
        Output raster dataset.

    """
    if bdist is not None and bdist < 1:
        bdist = None
        showlog("Blanking distance too small.")

    if method == "Minimum Curvature":
        gdat = minc(x, y, z, dxy, showlog=showlog, bdist=bdist)
        gdat = np.ma.filled(gdat, fill_value=nullvalue)
    else:
        extent = np.array([x.min(), x.max(), y.min(), y.max()])

        xxx = np.arange(extent[0], extent[1] + dxy / 2, dxy)
        yyy = np.arange(extent[2], extent[3] + dxy / 2, dxy)

        xxx, yyy = np.meshgrid(xxx, yyy)

        points = np.transpose([x.flatten(), y.flatten()])

        if method == "Nearest Neighbour":
            gdat = griddata(points, z, (xxx, yyy), method="nearest")
        elif method == "Linear":
            gdat = griddata(
                points, z, (xxx, yyy), method="linear", fill_value=nullvalue
            )
        else:
            gdat = griddata(points, z, (xxx, yyy), method="cubic", fill_value=nullvalue)

        gdat = blanking(gdat, x, y, bdist, extent, dxy, nullvalue)
        gdat = gdat[::-1]
    gdat = np.ma.masked_equal(gdat, nullvalue)

    # Create dataset
    dat = Data()
    dat.data = gdat
    dat.nodata = nullvalue

    rows, _ = dat.data.shape

    left = x.min() - dxy / 2
    top = y.min() + dxy * rows - dxy / 2

    dat.set_transform(dxy, left, dxy, top)

    return dat


def gridvolume(x, y, z, val, dxy, *, dat=None, showlog=print):
    """
    Grid volume data.

    Parameters
    ----------
    x : numpy array
        X coordinate values.
    y : numpy array
        Y coordinate values.
    z : numpy array
        Z coordinate values.
    val : numpy array
        Data values.
    dxy : float
        Grid cell size, in distance units.
    dat : pygmi.raster.datatypes.Data
        DEM data used to constrain surface. The default is None.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    dat : pygmi.raster.datatypes.Data.
        Output raster dataset.

    """
    points = np.transpose([x, y, z])
    try:
        interpolator = RBFInterpolator(points, val, kernel="linear")
    except np.linalg.LinAlgError:
        showlog("Problem with coordinates, csnnot calculate.")
        return None
    min_limit = np.min(val)
    max_limit = np.max(val)

    xxx = np.arange(x.min(), x.max() + dxy / 2, dxy)
    yyy = np.arange(y.min(), y.max() + dxy / 2, dxy)
    zzz = np.arange(z.min(), z.max() + dxy / 2, dxy)
    xxx, yyy, zzz = np.meshgrid(xxx, yyy, zzz)

    newpoints = np.transpose([xxx.flatten(), yyy.flatten(), zzz.flatten()])
    d_interpolated = interpolator(newpoints)
    d_limited = np.clip(d_interpolated, min_limit, max_limit)
    d_limited = d_limited.reshape(xxx.shape)

    if dat is not None:
        extent = dat.extent
        dx = dat.xdim
        dy = dat.ydim
        xxx1 = np.arange(extent[0], extent[1], dx) + dx / 2
        yyy1 = np.arange(extent[2], extent[3], dy) + dy / 2

        xxx1, yyy1 = np.meshgrid(xxx1, yyy1)
        points = np.transpose([xxx1.flatten(), yyy1.flatten()])
        zz = dat.data.flatten()
        gdat = griddata(points, zz, (xxx, yyy), method="nearest")
        d_limited[zzz > gdat] = np.nan

    out = VoxModel()
    out.data = d_limited
    out.origin = (x.min(), y.min(), z.min())
    out.spacing = (dxy, dxy, dxy)

    return out


def lltomap(lat, lon):
    """
    Convert a latitude and longitude to a 1:50,000 map sheet name.

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    mapsheet : str
        Map sheet number.

    """
    if np.isnan(lat) or np.isnan(lon):
        return ""

    cdict = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}

    latfrac = abs(lat) % 1
    lonfrac = lon % 1

    latf = latfrac // 0.5
    lonf = lonfrac // 0.5
    letter1 = cdict[(latf, lonf)]

    latf = latfrac % 0.5
    lonf = lonfrac % 0.5

    latf = latf // 0.25
    lonf = lonf // 0.25

    letter2 = cdict[(latf, lonf)]

    mapsheet = f"{int(abs(lat))}{int(lon)}{letter1}{letter2}"

    return mapsheet


def maptobounds(mapsheet, crs_to=None, showlog=print):
    """
    Convert a South African map sheet name to bounds.

    Parameters
    ----------
    mapsheet : str
        Map sheet number. Four numbers and up to two letters denoting NE corner
        in latitude and longitude and quadrants (A to D). Eg, 2928AB is
        latitude 29, longitude 28, quadrant B of quadrant A.
    crs_to : CRS, optional
        Destination projection. The default is None.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    bounds : list
        output bounds.

    """
    i = mapsheet
    try:
        lat = float(i[:2])
        lon = float(i[2:4])
    except ValueError:
        showlog("Invalid Map Sheet Number")
        return None

    q1 = "A"
    q2 = "A"
    latincr = 1
    lonincr = 2
    if len(i) > 4:
        q1 = i[4:5]
        lonincr = 0.5
        latincr = 0.5
    if len(i) > 5:
        q2 = i[5:6]
        lonincr = 0.25
        latincr = 0.25

    qlat1 = {"A": 0.0, "B": 0.0, "C": 0.5, "D": 0.5}

    qlon1 = {"A": 0.0, "B": 0.5, "C": 0.0, "D": 0.5}

    qlat2 = {"A": 0.0, "B": 0.0, "C": 0.25, "D": 0.25}

    qlon2 = {"A": 0.0, "B": 0.25, "C": 0.0, "D": 0.25}

    lat = -(lat + qlat1[q1] + qlat2[q2])
    lon = lon + qlon1[q1] + qlon2[q2]

    xmin = lon
    ymin = lat - latincr
    xmax = lon + lonincr
    ymax = lat

    if crs_to is not None:
        crs_from = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
        xmin, ymin = transformer.transform(xmin, ymin)
        xmax, ymax = transformer.transform(xmax, ymax)

    bounds = (xmin, ymin, xmax, ymax)

    return bounds


def maptovector(maplist):
    """
    Create a vector layer from map numbers.

    Parameters
    ----------
    maplist : list
        List of strings containing map sheet numbers.

    Returns
    -------
    data : GeoDataFrame
        GeoPandas GeoDataFrame

    """
    allpolys = []
    for i in maplist:
        bounds = maptobounds(i)
        x0, y0, x1, y1 = bounds

        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

        allpolys.append(poly)

    data = gpd.GeoDataFrame({"geometry": allpolys})
    newgeom = [data.union_all()]
    data = gpd.GeoDataFrame({"geometry": newgeom})

    data = data.set_crs(4326)

    return data


def quickgrid(x, y, z, dxy, *, numits=4, showlog=print):
    """
    Do a quick grid.

    Parameters
    ----------
    x : numpy array
        array of x coordinates
    y : numpy array
        array of y coordinates
    z : numpy array
        array of z values - this is the column being gridded
    dxy : float
        cell size for the grid, in both the x and y direction.
    numits : int
        number of iterations. By default its 4. If this is negative, a maximum
        will be calculated and used.
    showlog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    newz : numpy array
        M x N array of z values
    """
    showlog("Creating Grid")
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    newmask = np.array([1])
    j = -1
    rows = int((ymax - ymin) / dxy) + 1
    cols = int((xmax - xmin) / dxy) + 1

    if numits < 1:
        numits = int(max(np.log2(cols), np.log2(rows)))

    zfin = np.zeros([1, 1])
    while np.max(newmask) > 0 and j < (numits - 1):
        j += 1
        jj = 2**j

        dxy2 = dxy * jj
        rows = int((ymax - ymin) / dxy2) + 1
        cols = int((xmax - xmin) / dxy2) + 1

        newz = np.zeros([rows, cols])
        zdiv = np.zeros([rows, cols])

        xindex = ((x - xmin) / dxy2).astype(int)
        yindex = ((y - ymin) / dxy2).astype(int)

        for i in range(z.size):
            newz[yindex[i], xindex[i]] += z[i]
            zdiv[yindex[i], xindex[i]] += 1

        filt = zdiv > 0
        newz[filt] = newz[filt] / zdiv[filt]

        if j == 0:
            newmask = np.ones([rows, cols])
            for i in range(z.size):
                newmask[yindex[i], xindex[i]] = 0
            zfin = newz
        else:
            xx, yy = newmask.nonzero()
            xx2 = xx // jj
            yy2 = yy // jj
            zfin[xx, yy] = newz[xx2, yy2]
            newmask[xx, yy] = np.logical_not(zdiv[xx2, yy2])

        showlog("Iteration done: " + str(j + 1) + " of " + str(numits))

    showlog("Finished!")

    newz = np.ma.array(zfin)
    newz.mask = newmask
    return newz


def reprojxy(x, y, iwkt, owkt, showlog=print):
    """
    Reproject x and y coordinates.

    Parameters
    ----------
    x : numpy array or float
        x coordinates
    y : numpy array or float
        y coordinates
    iwkt : str, int, CRS
        Input wkt description or EPSG code (int) or CRS
    owkt : str, int, CRS
        Output wkt description or EPSG code (int) or CRS
    showlog : function, optional
        Routine to show text messages. The default is print.

    Returns
    -------
    xout : numpy array
        x coordinates.
    yout : numpy array
        y coordinates.

    """
    if isinstance(iwkt, int):
        crs_from = CRS.from_epsg(iwkt)
    elif isinstance(iwkt, str):
        crs_from = CRS.from_wkt(iwkt)
    else:
        crs_from = iwkt

    if isinstance(owkt, int):
        crs_to = CRS.from_epsg(owkt)
    elif isinstance(iwkt, str):
        crs_to = CRS.from_wkt(owkt)
    else:
        crs_to = owkt

    try:
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    except Exception:
        showlog("Problem reprojecting. Aborting.")
        return None, None

    xout, yout = transformer.transform(x, y)

    return xout, yout


def xy_to_r(x, y, piter=iter):
    """
    Convert x an y values on a section to r.

    This will take into account r being reset for each depth.

    Parameters
    ----------
    x : numpy array or float
        x coordinates
    y : numpy array or float
        y coordinates

    Returns
    -------
    r : numpy array
        r coordinates.
    """
    r1 = np.sqrt(x**2 + y**2)
    r2 = np.diff(r1)
    r2 = np.sign(r2[0]) * r2
    rind = np.where(r2 < 0)[0] + 1
    rind = np.append(rind, r2.size + 1)

    points = np.transpose([x, y])
    points = np.transpose(fast_sort(points, piter))

    x1a = points[0]
    y1a = points[1]

    r = np.sqrt((x1a[1:] - x1a[:-1]) ** 2 + (y1a[1:] - y1a[:-1]) ** 2)
    r = np.concatenate(([np.nan], r))

    x1a = x1a[r != 0]
    y1a = y1a[r != 0]
    r = r[r != 0]
    r[0] = 0

    r0 = np.cumsum(r)

    i0 = 0
    r1 = []
    for i1 in rind:
        x1 = x[i0:i1]
        y1 = y[i0:i1]

        r = []
        for i in range(i1 - i0):
            filt = np.logical_and(x1a == x1[i], y1a == y1[i])
            r.append(r0[filt])
        i0 = i1

        r1 += r

    r = np.array(r1)
    r = r.flatten()

    return r


def fast_sort(points, piter=iter):
    """
    Fast sort of coordinate pairs.

    Parameters
    ----------
    points : numpy array
        Coordinates.
    piter : function, optional
        progress bar iterable, default is iter.

    Returns
    -------
    sorted_pts : list
        Sorted coordinates.
    """
    points = list(points)
    sorted_pts = [points.pop(0)]

    num = len(points)

    for _ in piter(range(num)):
        tree = KDTree(points)
        _, index = tree.query(sorted_pts[-1])
        sorted_pts.append(points.pop(index))
    return sorted_pts


def _testfn():
    """Test routine."""
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ofile = r"D:\mining_guidelines\2430\2430.shp"

    maplist = ["2430DA", "2430DB", "2430DC", "2430DD"]
    data = maptovector(maplist)

    data.to_file(ofile)


def _testfn_pointcut():
    """Test routine."""
    import sys

    from pygmi.vector.iodefs import ImportXYZ  # , ImportVector

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ifile = r"D:\Workdata\PyGMI Test Data\Vector\linecut\test2.csv"
    sfile = r"D:\Workdata\PyGMI Test Data\Vector\linecut\test2_cut_outline.shp"

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = "Comma Delimited (*.csv)"
    IO.settings(True)

    # ifile = r"E:\WorkProjects\ST-2025-1365 Energy Mapping\lineaments\MP_mag_lineaments_utm36s.shp"
    # sfile = r"E:\WorkProjects\ST-2025-1365 Energy Mapping\lineaments\3D study area.shp"

    # IO = ImportVector()
    # IO.ifile = ifile
    # IO.settings(True)

    DR = PointCut()
    DR.indata = IO.outdata
    DR.ifile = sfile
    DR.settings(True)

    # dat = DR.outdata['Vector']


def _testfn_grid():
    """Test routine."""
    import sys

    import matplotlib.pyplot as plt

    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ifile = r"D:\Gravity\Final_RSA_Old_WGS84v3.csv"
    ifile = r"D:\workdata\PyGMI Test Data\Vector\Line Data\MAGARCHIVE.XYZ"
    # ifile = r"D:\UBC_Files\new_PyGMI_test_xyz_data.csv"
    # ifile = r"D:\UBC_Files\line1_segment1_rho_model.csv"
    # ifile = r"D:\workdata\PyGMI Test Data\Vector\Volume grid\all_ert_lines_Res2Dinv_inversion.XYZ"

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = "Comma Delimited (*.csv)"
    IO.filt = "Geosoft XYZ (*.xyz)"
    IO.settings(True)

    DR = DataGrid()
    DR.indata = IO.outdata
    DR.settings()

    data = DR.outdata["Raster"][0]

    plt.imshow(data.data, extent=data.extent)
    plt.show()


def _testfn_vol():
    """Test routine."""
    import sys

    import pyvista as pv

    from pygmi.raster.iodefs import get_raster
    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    ifile = r"D:\workdata\PyGMI Test Data\Vector\Volume grid\all_ert_lines_Res2Dinv_inversion.XYZ"
    dfile = r"D:\workdata\PyGMI Test Data\Vector\Volume grid\SRTM_ER_Mapper.ers"

    dat = get_raster(dfile)[0]

    IO = ImportXYZ()
    IO.ifile = ifile
    IO.filt = "Geosoft XYZ (*.xyz)"
    IO.settings(True)

    IO.outdata["Raster"] = [dat]

    DR = DataGrid()
    DR.indata = IO.outdata
    DR.settings()

    vdat = DR.outdata["Voxel"][0]

    # Create the spatial reference
    grid = pv.ImageData()

    values = vdat.data
    # Set the grid dimensions: shape + 1 because we want to inject our values
    # on the CELL data
    grid.dimensions = np.array(values.shape) + 1

    # Edit the spatial reference
    # The bottom left corner of the data set
    grid.origin = vdat.origin
    grid.spacing = vdat.spacing  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array

    # Get rid of nan values
    grid = grid.threshold()

    # Now plot the grid
    # grid.plot(show_edges=True)

    p = pv.Plotter()
    p.add_mesh_clip_plane(grid, normal=[-1, 0, 0])
    # p.add_volume(grid)
    # p.add_mesh(grid, opacity=0.5)
    # p.add_mesh_slice(grid)
    p.add_axes()
    p.show_grid()
    p.show()


if __name__ == "__main__":
    _testfn()
