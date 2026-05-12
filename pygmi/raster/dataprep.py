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
"""A set of raster data preparation routines."""

import glob
import math
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.merge
from pyproj.crs import CRS
from PySide6 import QtGui, QtWidgets
from rasterio.warp import calculate_default_transform
from shapely import LineString, unary_union

from pygmi.misc import BasicModule, ContextModule
from pygmi.raster.datatypes import Data, numpy_to_pygmi
from pygmi.raster.fft import fft_getkxy, fftprep
from pygmi.raster.iodefs import export_raster
from pygmi.raster.misc import cut_raster, lstack
from pygmi.raster.reproj import GroupProj, data_reproject
from pygmi.rsense.iodefs import get_data, get_from_rastermeta
from pygmi.vector.dataprep import reprojxy


class Continuation(BasicModule):
    """
    GUI to perform upward and downward continuation on potential field data.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmb_dataid = QtWidgets.QComboBox()
        self.cmb_cont = QtWidgets.QComboBox()
        self.dsb_height = QtWidgets.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = "raster.dm.cont"
        lbl_band = QtWidgets.QLabel("Band to perform continuation:")
        lbl_cont = QtWidgets.QLabel("Continuation type:")
        lbl_height = QtWidgets.QLabel("Continuation distance:")

        self.dsb_height.setMaximum(1000000.0)
        self.dsb_height.setMinimum(0.0)
        self.dsb_height.setValue(0.0)
        self.cmb_cont.clear()
        self.cmb_cont.addItems(["Upward", "Downward"])

        self.setWindowTitle("Continuation")

        gl_main.addWidget(lbl_band, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_dataid, 0, 1, 1, 1)

        gl_main.addWidget(lbl_cont, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_cont, 1, 1, 1, 1)
        gl_main.addWidget(lbl_height, 2, 0, 1, 1)
        gl_main.addWidget(self.dsb_height, 2, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 3, 0, 1, 2)

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
        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        items = [i.dataid for i in self.indata["Raster"]]
        self.cmb_update(self.cmb_dataid, items)

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
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.cmb_cont)
        self.saveobj(self.dsb_height)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        h = self.dsb_height.value()
        ctype = self.cmb_cont.currentText()
        data = None

        # Get data
        for i in self.indata["Raster"]:
            if i.dataid == self.cmb_dataid.currentText():
                data = i
                break

        if ctype == "Downward":
            dat = taylorcont(data, h, self.showlog, self.piter)
        else:
            dat = fftcont(data, h, self.showlog, self.piter)

        self.outdata["Raster"] = [dat]


class DataCut(BasicModule):
    """
    GUI to Cut Data using shapefiles.

    This class cuts raster datasets using a boundary defined by a polygon
    shapefile.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

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
        if (
            "Raster" not in self.indata
            and "Cluster" not in self.indata
            and "RasterFileList" not in self.indata
        ):
            self.showlog("No raster data")
            return False

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, "Open Shape File", ".", "Shape file (*.shp)"
            )
            if self.ifile == "":
                return False

        for datatype in ["Raster", "Cluster"]:
            if datatype not in self.indata:
                continue
            data = self.indata[datatype]

            os.chdir(os.path.dirname(self.ifile))
            data = cut_raster(data, self.ifile, showlog=self.showlog)

            if data is None:
                return False

            self.outdata[datatype] = data

        if "RasterFileList" in self.indata:
            flist = self.indata["RasterFileList"]
            for ifile in flist:
                data = get_from_rastermeta(
                    ifile, piter=self.piter, showlog=self.showlog
                )
                os.chdir(os.path.dirname(self.ifile))
                data = cut_raster(data, self.ifile, showlog=self.showlog)

                if data:
                    odir = os.path.dirname(data[0].filename)
                    odir = os.path.join(odir, "cut")

                    os.makedirs(odir, exist_ok=True)

                    ofile = os.path.basename(data[0].filename)
                    ofile = os.path.join(odir, ofile)

                    self.showlog("Exporting to " + ofile)
                    export_raster(
                        ofile,
                        data,
                        drv="GTiff",
                        piter=self.piter,
                        compression="DEFLATE",
                        showlog=self.showlog,
                    )
                    self.outdata["Raster"] = data

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class DataLayerStack(BasicModule):
    """
    Data Layer Stack GUI.

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None
        self.cb_cmask = QtWidgets.QCheckBox("Common mask for all bands")

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.lbl_rows = QtWidgets.QLabel("Rows: 0")
        self.lbl_cols = QtWidgets.QLabel("Columns: 0")

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)

        self.buttonbox.htmlfile = "raster.dm.layerstack"
        lbl_dxy = QtWidgets.QLabel("Cell Size:")

        self.dsb_dxy.setMaximum(9999999999.0)
        self.dsb_dxy.setMinimum(0.00001)
        self.dsb_dxy.setDecimals(5)
        self.dsb_dxy.setValue(40.0)

        self.cb_cmask.setChecked(True)

        self.setWindowTitle("Dataset Layer Stack and Resample")

        gl_main.addWidget(lbl_dxy, 0, 0, 1, 1)
        gl_main.addWidget(self.dsb_dxy, 0, 1, 1, 1)
        gl_main.addWidget(self.lbl_rows, 1, 0, 1, 2)
        gl_main.addWidget(self.lbl_cols, 2, 0, 1, 2)
        gl_main.addWidget(self.cb_cmask, 3, 0, 1, 2)
        gl_main.addWidget(self.buttonbox, 4, 0, 1, 2)

        self.dsb_dxy.valueChanged.connect(self.dxy_change)

    def dxy_change(self):
        """
        Update dxy.

        This is the size of a grid cell in the x and y directions.

        Returns
        -------
        None.

        """
        data = self.indata["Raster"][0]
        dxy = self.dsb_dxy.value()

        xmin0, xmax0, ymin0, ymax0 = data.extent
        xmin, xmax, ymin, ymax = data.extent

        for data in self.indata["Raster"]:
            xmin, xmax, ymin, ymax = data.extent
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)

        cols = int(round((xmax - xmin) / dxy, 9))
        rows = int(round((ymax - ymin) / dxy, 9))

        self.lbl_rows.setText("Rows: " + str(rows))
        self.lbl_cols.setText("Columns: " + str(cols))

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
        if "RasterFileList" in self.indata:
            ifiles = self.indata["RasterFileList"]
            self.showlog(
                "Warning: Layer stacking a file list assumes all datasets overlap in the same area"
            )
            self.indata["Raster"] = []
            for ifile in ifiles:
                self.showlog("Processing " + os.path.basename(ifile))
                dat = get_data(ifile, piter=self.piter, showlog=self.showlog)
                # for i in dat:
                #     i.data = i.data.astype(np.float32)
                #     i.nodata = np.float32(i.nodata)
                self.indata["Raster"] += dat

        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        if not nodialog:
            data = self.indata["Raster"][0]

            if self.dxy is None:
                self.dxy = min(data.xdim, data.ydim)
                for data in self.indata["Raster"]:
                    self.dxy = min(self.dxy, data.xdim, data.ydim)

            self.dsb_dxy.setValue(self.dxy)
            self.dxy_change()

            tmp = self.exec()
            if tmp != 1:
                return False

        self.acceptall()

        if self.outdata["Raster"] is None:
            self.outdata = {}
            return False

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.dxy)
        self.saveobj(self.dsb_dxy)
        self.saveobj(self.cb_cmask)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        dxy = self.dsb_dxy.value()
        self.dxy = dxy
        dat = lstack(
            self.indata["Raster"],
            piter=self.piter,
            dxy=dxy,
            showlog=self.showlog,
            commonmask=self.cb_cmask.isChecked(),
        )
        self.outdata["Raster"] = dat


class DataMerge(BasicModule):
    """
    Data merge or mosaic GUI.

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.idir = None
        self.tmpdir = None
        self.is_import = True
        self.method = "merge_median"
        self.res = None

        self.rb_first = QtWidgets.QRadioButton(
            "First - copy first file over last file at overlap."
        )
        self.rb_last = QtWidgets.QRadioButton(
            "Last - copy last file over first file at overlap."
        )
        self.rb_min = QtWidgets.QRadioButton(
            "Min - copy pixel wise minimum at overlap."
        )
        self.rb_max = QtWidgets.QRadioButton(
            "Max - copy pixel wise maximum at overlap."
        )
        self.rb_median = QtWidgets.QRadioButton(
            "Median - shift last file to median overlap value and copy over first file at overlap."
        )

        self.le_idirlist = QtWidgets.QLineEdit("")
        self.le_sfile = QtWidgets.QLineEdit("")
        self.le_nodata = QtWidgets.QLineEdit("")
        self.le_res = QtWidgets.QLineEdit("")

        self.le_nodata.setValidator(self.qval)
        self.le_res.setValidator(self.qval)

        self.cb_shift_to_median = QtWidgets.QCheckBox(
            "Shift bands to median value before mosaic. May "
            "allow for cleaner mosaic if datasets are offset."
        )

        self.cb_bands_to_files = QtWidgets.QCheckBox(
            'Save each band separately in a "mosaic" subdirectory.'
        )
        self.forcetype = None
        self.singleband = False
        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = "raster.dm.mosaic"
        pb_idirlist = QtWidgets.QPushButton("Batch Directory")
        pb_sfile = QtWidgets.QPushButton("Shapefile or Raster for boundary (optional)")

        pixmapi = QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
        icon = self.style().standardIcon(pixmapi)
        pb_sfile.setIcon(icon)
        pb_idirlist.setIcon(icon)
        pb_sfile.setStyleSheet("text-align:left;")
        pb_idirlist.setStyleSheet("text-align:left;")

        self.cb_shift_to_median.setChecked(False)
        self.rb_median.setChecked(True)

        self.setWindowTitle("Dataset Mosaic")

        gbox_merge_method = QtWidgets.QGroupBox("Mosiac method")
        vbl_merge_method = QtWidgets.QVBoxLayout(gbox_merge_method)

        vbl_merge_method.addWidget(self.rb_median)
        vbl_merge_method.addWidget(self.rb_first)
        vbl_merge_method.addWidget(self.rb_last)
        vbl_merge_method.addWidget(self.rb_min)
        vbl_merge_method.addWidget(self.rb_max)

        gl_main.addWidget(pb_idirlist, 1, 0, 1, 1)
        gl_main.addWidget(self.le_idirlist, 1, 1, 1, 1)
        gl_main.addWidget(pb_sfile, 2, 0, 1, 1)
        gl_main.addWidget(self.le_sfile, 2, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel("Nodata Value (optional):"), 3, 0, 1, 1)
        gl_main.addWidget(self.le_nodata, 3, 1, 1, 1)
        gl_main.addWidget(QtWidgets.QLabel("Output Resolution (optional):"), 4, 0, 1, 1)
        gl_main.addWidget(self.le_res, 4, 1, 1, 1)

        gl_main.addWidget(self.cb_shift_to_median, 5, 0, 1, 2)
        gl_main.addWidget(gbox_merge_method, 6, 0, 1, 2)
        gl_main.addWidget(self.cb_bands_to_files, 7, 0, 1, 2)
        gl_main.addWidget(self.buttonbox, 8, 0, 1, 2)

        pb_idirlist.pressed.connect(self.get_idir)
        pb_sfile.pressed.connect(self.get_sfile)

        self.rb_first.clicked.connect(self.method_change)
        self.rb_last.clicked.connect(self.method_change)
        self.rb_min.clicked.connect(self.method_change)
        self.rb_max.clicked.connect(self.method_change)
        self.rb_median.clicked.connect(self.method_change)

    def method_change(self):
        """
        Change method.

        Returns
        -------
        None.

        """
        if self.rb_first.isChecked():
            self.method = "first"
        if self.rb_last.isChecked():
            self.method = "last"
        if self.rb_min.isChecked():
            self.method = "merge_min"
        if self.rb_max.isChecked():
            self.method = "merge_max"
        if self.rb_median.isChecked():
            self.method = "merge_median"

    def get_idir(self):
        """
        Get the input directory.

        Returns
        -------
        None.

        """
        self.idir = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent, "Select Directory"
        )

        self.le_idirlist.setText(self.idir)

        if self.idir == "":
            self.idir = None

    def get_sfile(self):
        """
        Get the input shapefile.

        Returns
        -------
        None.

        """
        ext = "Common formats (*.shp *.hdr *.tif);;"

        sfile, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, "Open File", ".", ext
        )

        if not sfile:
            return False

        self.le_sfile.setText(sfile)

        return True

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

        if not self.check_validation():
            return False

        tmp = self.merge_different()

        return tmp

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.idir)
        self.saveobj(self.le_idirlist)
        self.saveobj(self.le_nodata)
        self.saveobj(self.le_res)
        self.saveobj(self.cb_shift_to_median)

        self.saveobj(self.rb_first)
        self.saveobj(self.rb_last)
        self.saveobj(self.rb_min)
        self.saveobj(self.rb_max)
        self.saveobj(self.rb_median)

        self.saveobj(self.le_sfile)
        self.saveobj(self.cb_bands_to_files)
        self.saveobj(self.forcetype)
        self.saveobj(self.singleband)

    def merge_different(self):
        """
        Merge files with different numbers of bands and/or band order.

        This uses more memory, but is flexible.

        Returns
        -------
        bool
            Success of routine.

        """
        bfile = self.le_sfile.text()
        bandstofiles = self.cb_bands_to_files.isChecked()
        shifttomedian = self.cb_shift_to_median.isChecked()

        if self.le_nodata.text().strip() == "":
            nodata = None
        else:
            nodata = float(self.le_nodata.text())
        if self.le_res.text().strip() == "":
            res = None
        else:
            res = float(self.le_res.text())

        outdat = mosaic(
            self.indata,
            idir=self.idir,
            bfile=bfile,
            bandstofiles=bandstofiles,
            piter=self.piter,
            showlog=self.showlog,
            singleband=self.singleband,
            forcetype=self.forcetype,
            shifttomedian=shifttomedian,
            tmpdir=self.tmpdir,
            nodata=nodata,
            method=self.method,
            res=res,
        )

        if outdat:
            self.outdata["Raster"] = outdat

        return True


class DataReproj(BasicModule):
    """
    Raster reprojection GUI.

    This class reprojects datasets using the rasterio routines.

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

        self.buttonbox.htmlfile = "raster.dm.reproj"

        self.setWindowTitle("Dataset Reprojection")

        gl_main.addWidget(self.in_proj, 0, 0, 1, 1)
        gl_main.addWidget(self.out_proj, 0, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 1, 0, 1, 2)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        if self.in_proj.wkt == "Unknown" or self.out_proj.wkt == "Unknown":
            self.showlog("Unknown Projection. Could not reproject")
            return

        if self.in_proj.wkt == "" or self.out_proj.wkt == "":
            self.showlog("Unknown Projection. Could not reproject")
            return

        # Input stuff
        src_crs = CRS.from_wkt(self.in_proj.wkt)

        # Output stuff
        dst_crs = CRS.from_wkt(self.out_proj.wkt)

        # Now create virtual dataset
        dat = []
        data2 = None
        for data in self.piter(self.indata["Raster"]):
            if data.isrgb:
                _, _, bands = data.data.shape
                data3 = []
                data1 = [data.copy() for i in range(bands)]
                for i, band in enumerate(data1):
                    band.data = band.data[:, :, i]
                    data2 = data_reproject(band, dst_crs, icrs=src_crs)
                    if data2 is None:
                        return
                    data3.append(data2)
                data2.data = np.transpose([i.data.T for i in data3])
                data2.isrgb = True
                dat.append(data2)
            else:
                data2 = data_reproject(data, dst_crs, icrs=src_crs)
                if data2 is None:
                    return

                dat.append(data2)

        self.orig_wkt = self.in_proj.wkt
        self.targ_wkt = self.out_proj.wkt
        self.outdata["Raster"] = dat

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
        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        if self.indata["Raster"][0].crs is None:
            self.showlog(
                "Your input data has no projection. Please assign one in the metadata summary."
            )
            return False

        if self.orig_wkt is None:
            self.orig_wkt = self.indata["Raster"][0].crs.to_wkt()
        if self.targ_wkt is None:
            self.targ_wkt = self.indata["Raster"][0].crs.to_wkt()

        self.in_proj.set_current(self.orig_wkt)
        self.out_proj.set_current(self.targ_wkt)

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
        self.saveobj(self.orig_wkt)
        self.saveobj(self.targ_wkt)


class GetProf(BasicModule):
    """
    GUI to extract a profile from a raster dataset.

    This class extracts a profile from a raster dataset using a line shapefile.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

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
        if "Raster" in self.indata:
            data = [i.copy() for i in self.indata["Raster"]]
            icrs = data[0].crs
        else:
            self.showlog("No raster data")
            return False

        ext = "Shape file (*.shp)"

        if not nodialog:
            self.ifile, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, "Open Shape File", ".", ext
            )
            if self.ifile == "":
                return False

        os.chdir(os.path.dirname(self.ifile))

        gdf = gpd.read_file(self.ifile, engine="pyogrio")

        gdf = gdf[gdf.geometry != None]

        if gdf.geom_type.iloc[0] != "LineString":
            self.showlog("You need lines in that shape file")
            return False

        data = lstack(data, piter=self.piter, showlog=self.showlog)
        dxy = min(data[0].xdim, data[0].ydim)
        ogdf2 = None

        icnt = 0
        for line in gdf.geometry:
            line2 = redistribute_vertices(line, dxy)
            x, y = line2.coords.xy
            xy = np.transpose([x, y])
            ogdf = None

            for idata in self.piter(data):
                mdata = idata.to_mem()
                z = []
                for pnt in xy:
                    z.append(idata.data[mdata.index(pnt[0], pnt[1])])

                if ogdf is None:
                    ogdf = pd.DataFrame(xy[:, 0], columns=["X"])
                    ogdf["Y"] = xy[:, 1]

                    x = ogdf["X"]
                    y = ogdf["Y"]
                    ogdf = gpd.GeoDataFrame(ogdf, geometry=gpd.points_from_xy(x, y))

                ogdf[idata.dataid] = z

            icnt += 1
            ogdf["line"] = str(icnt)
            ogdf.crs = icrs

            if ogdf2 is None:
                ogdf2 = ogdf
            else:
                ogdf2 = ogdf2.append(ogdf, ignore_index=True)

        self.outdata["Vector"] = [ogdf2]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.ifile)


class Metadata(ContextModule):
    """
    Edit raster metadata.

    This class allows the editing of the metadata for a raster dataset using a
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
        self.banddata = {}
        self.dataid = {}
        self.oldtxt = ""

        self.cmb_bandid = QtWidgets.QComboBox()
        self.pb_rename_id = QtWidgets.QPushButton("Rename Band Name")
        self.lbl_rows = QtWidgets.QLabel()
        self.lbl_cols = QtWidgets.QLabel()
        self.le_txt_null = QtWidgets.QLineEdit()
        self.le_tlx = QtWidgets.QLineEdit()
        self.le_tly = QtWidgets.QLineEdit()
        self.le_xdim = QtWidgets.QLineEdit()
        self.le_ydim = QtWidgets.QLineEdit()
        self.le_led_units = QtWidgets.QLineEdit()
        self.lbl_min = QtWidgets.QLabel()
        self.lbl_max = QtWidgets.QLabel()
        self.lbl_mean = QtWidgets.QLabel()
        self.lbl_dtype = QtWidgets.QLabel()
        self.date = QtWidgets.QDateEdit()

        self.le_txt_null.setValidator(self.qval)
        self.le_tlx.setValidator(QtGui.QDoubleValidator(self))
        self.le_tly.setValidator(QtGui.QDoubleValidator(self))
        self.le_xdim.setValidator(QtGui.QDoubleValidator(1e-300, np.inf, -1))
        self.le_ydim.setValidator(QtGui.QDoubleValidator(1e-300, np.inf, -1))

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
        gbox = QtWidgets.QGroupBox("Dataset")
        self.buttonbox.htmlfile = "raster.cm.meta"

        gl_1 = QtWidgets.QGridLayout(gbox)
        lbl_tlx = QtWidgets.QLabel("Top Left X Coordinate:")
        lbl_tly = QtWidgets.QLabel("Top Left Y Coordinate:")
        lbl_xdim = QtWidgets.QLabel("X Dimension:")
        lbl_ydim = QtWidgets.QLabel("Y Dimension:")
        lbl_null = QtWidgets.QLabel("Null/Nodata value:")
        lbl_rows = QtWidgets.QLabel("Rows:")
        lbl_cols = QtWidgets.QLabel("Columns:")
        lbl_min = QtWidgets.QLabel("Dataset Minimum:")
        lbl_max = QtWidgets.QLabel("Dataset Maximum:")
        lbl_mean = QtWidgets.QLabel("Dataset Mean:")
        lbl_units = QtWidgets.QLabel("Dataset Units:")
        lbl_bandid = QtWidgets.QLabel("Band Name:")
        lbl_dtype = QtWidgets.QLabel("Data Type:")
        lbl_date = QtWidgets.QLabel("Acquisition Date:")

        sizepolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        gbox.setSizePolicy(sizepolicy)
        self.proj.setSizePolicy(sizepolicy)

        self.setWindowTitle("Dataset Metadata")
        self.date.setCalendarPopup(True)

        gl_main.addWidget(lbl_bandid, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_bandid, 0, 1, 1, 3)
        gl_main.addWidget(self.pb_rename_id, 1, 1, 1, 3)
        gl_main.addWidget(gbox, 2, 0, 1, 2)
        gl_main.addWidget(self.proj, 2, 2, 1, 2)
        gl_main.addWidget(self.buttonbox, 4, 0, 1, 4)

        gl_1.addWidget(lbl_tlx, 0, 0, 1, 1)
        gl_1.addWidget(self.le_tlx, 0, 1, 1, 1)
        gl_1.addWidget(lbl_tly, 1, 0, 1, 1)
        gl_1.addWidget(self.le_tly, 1, 1, 1, 1)
        gl_1.addWidget(lbl_xdim, 2, 0, 1, 1)
        gl_1.addWidget(self.le_xdim, 2, 1, 1, 1)
        gl_1.addWidget(lbl_ydim, 3, 0, 1, 1)
        gl_1.addWidget(self.le_ydim, 3, 1, 1, 1)
        gl_1.addWidget(lbl_null, 4, 0, 1, 1)
        gl_1.addWidget(self.le_txt_null, 4, 1, 1, 1)
        gl_1.addWidget(lbl_rows, 5, 0, 1, 1)
        gl_1.addWidget(self.lbl_rows, 5, 1, 1, 1)
        gl_1.addWidget(lbl_cols, 6, 0, 1, 1)
        gl_1.addWidget(self.lbl_cols, 6, 1, 1, 1)
        gl_1.addWidget(lbl_min, 7, 0, 1, 1)
        gl_1.addWidget(self.lbl_min, 7, 1, 1, 1)
        gl_1.addWidget(lbl_max, 8, 0, 1, 1)
        gl_1.addWidget(self.lbl_max, 8, 1, 1, 1)
        gl_1.addWidget(lbl_mean, 9, 0, 1, 1)
        gl_1.addWidget(self.lbl_mean, 9, 1, 1, 1)
        gl_1.addWidget(lbl_units, 10, 0, 1, 1)
        gl_1.addWidget(self.le_led_units, 10, 1, 1, 1)
        gl_1.addWidget(lbl_dtype, 11, 0, 1, 1)
        gl_1.addWidget(self.lbl_dtype, 11, 1, 1, 1)
        gl_1.addWidget(lbl_date, 12, 0, 1, 1)
        gl_1.addWidget(self.date, 12, 1, 1, 1)

        self.buttonbox.buttonbox.accepted.connect(self.acceptall)

        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)
        self.pb_rename_id.clicked.connect(self.rename_id)

    def acceptall(self):
        """
        Accept option.

        Returns
        -------
        None.

        """
        wkt = self.proj.wkt

        self.update_vals()
        for tmp in self.indata["Raster"]:
            for j in self.dataid.items():
                if j[1] == tmp.dataid:
                    i = self.banddata[j[0]]
                    tmp.dataid = j[0]
                    tmp.set_transform(transform=i.transform)
                    tmp.nodata = i.nodata
                    tmp.datetime = i.datetime
                    if wkt == "None":
                        tmp.crs = None
                    else:
                        tmp.crs = CRS.from_wkt(wkt)
                    tmp.units = i.units
                    tmp.data.mask = tmp.data.data == i.nodata

        self.accept()

    def rename_id(self):
        """
        Rename the band name.

        Returns
        -------
        None.

        """
        ctxt = str(self.cmb_bandid.currentText())
        (skey, isokay) = QtWidgets.QInputDialog.getText(
            self.parent,
            "Rename Band Name",
            "Please type in the new name for the band",
            QtWidgets.QLineEdit.EchoMode.Normal,
            ctxt,
        )

        if isokay:
            self.cmb_bandid.currentIndexChanged.disconnect()
            indx = self.cmb_bandid.currentIndex()
            txt = self.cmb_bandid.itemText(indx)
            self.banddata[skey] = self.banddata.pop(txt)
            self.dataid[skey] = self.dataid.pop(txt)
            self.oldtxt = skey
            self.cmb_bandid.setItemText(indx, skey)
            self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

    def update_vals(self):
        """
        Update the values on the interface.

        Returns
        -------
        None.

        """
        tmp = self.check_validation()
        if not tmp:
            self.cmb_bandid.blockSignals(True)
            self.cmb_bandid.setCurrentText(self.oldtxt)
            self.cmb_bandid.blockSignals(False)
            return

        odata = self.banddata[self.oldtxt]

        utxt = self.le_led_units.text()
        if utxt.lower() == "none":
            utxt = ""
        odata.units = utxt

        if self.le_txt_null.text() == "":
            odata.nodata = None
        else:
            odata.nodata = float(self.le_txt_null.text())
        left = float(self.le_tlx.text())
        top = float(self.le_tly.text())
        xdim = float(self.le_xdim.text())
        ydim = float(self.le_ydim.text())

        odata.set_transform(xdim, left, ydim, top)
        odata.datetime = self.date.date().toPyDate()

        indx = self.cmb_bandid.currentIndex()
        txt = self.cmb_bandid.itemText(indx)
        self.oldtxt = txt
        idata = self.banddata[txt]

        irows = idata.data.shape[0]
        icols = idata.data.shape[1]

        self.lbl_cols.setText(str(icols))
        self.lbl_rows.setText(str(irows))
        if idata.nodata is None:
            self.le_txt_null.setText("")
        else:
            self.le_txt_null.setText(str(idata.nodata))

        self.le_tlx.setText(str(idata.extent[0]))
        self.le_tly.setText(str(idata.extent[-1]))
        self.le_xdim.setText(str(idata.xdim))
        self.le_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.data.min()))
        self.lbl_max.setText(str(idata.data.max()))
        self.lbl_mean.setText(str(idata.data.mean()))
        self.le_led_units.setText(str(idata.units))
        self.lbl_dtype.setText(str(idata.data.dtype))
        self.date.setDate(idata.datetime)

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        bandid = []
        if self.indata["Raster"][0].crs is None:
            self.proj.set_current("None")
        else:
            crs = CRS.from_user_input(self.indata["Raster"][0].crs)
            self.proj.set_current(crs.to_wkt(pretty=True))

        for i in self.indata["Raster"]:
            bandid.append(i.dataid)
            self.banddata[i.dataid] = Data()
            tmp = self.banddata[i.dataid]
            self.dataid[i.dataid] = i.dataid
            tmp.data = i.data
            tmp.set_transform(transform=i.transform)
            tmp.nodata = i.nodata
            tmp.crs = i.crs
            tmp.units = i.units
            tmp.datetime = i.datetime

        self.cmb_bandid.currentIndexChanged.disconnect()
        self.cmb_bandid.clear()
        self.cmb_bandid.addItems(bandid)
        indx = self.cmb_bandid.currentIndex()
        self.oldtxt = self.cmb_bandid.itemText(indx)
        self.cmb_bandid.currentIndexChanged.connect(self.update_vals)

        idata = self.banddata[self.oldtxt]

        irows = idata.data.shape[0]
        icols = idata.data.shape[1]

        self.lbl_cols.setText(str(icols))
        self.lbl_rows.setText(str(irows))
        if idata.nodata is None:
            self.le_txt_null.setText("")
        else:
            self.le_txt_null.setText(str(idata.nodata))
        self.le_tlx.setText(str(idata.extent[0]))
        self.le_tly.setText(str(idata.extent[-1]))
        self.le_xdim.setText(str(idata.xdim))
        self.le_ydim.setText(str(idata.ydim))
        self.lbl_min.setText(str(idata.data.min()))
        self.lbl_max.setText(str(idata.data.max()))
        self.lbl_mean.setText(str(idata.data.mean()))
        self.le_led_units.setText(str(idata.units))
        self.lbl_dtype.setText(str(idata.data.dtype))
        self.date.setDate(idata.datetime)

        self.update_vals()

        self.show()


class RasterToVector(BasicModule):
    """
    Raster to vector GUI.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)

        self.buttonbox.htmlfile = "raster.dm.rtov"

        self.setWindowTitle("Raster to Vector")

        gl_main.addWidget(self.buttonbox, 4, 0, 1, 2)

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
        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        gdf = gpd.GeoDataFrame()

        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        data = self.indata["Raster"]
        data = lstack(data, piter=self.piter, showlog=self.showlog)

        xmin = data[0].extent[0]
        ymax = data[0].extent[-1]
        krows, kcols = data[0].data.shape

        x = []
        y = []
        for j in self.piter(range(krows)):
            for i in range(kcols):
                x.append(xmin + (i + 0.5) * data[0].xdim)
                y.append(ymax - (j + 0.5) * data[0].ydim)

        geom = gpd.points_from_xy(x, y)
        gdf = gpd.GeoDataFrame(geometry=geom)
        gdf["x"] = x
        gdf["y"] = y

        for band in self.piter(data):
            gdf[band.dataid] = band.data.flatten()

        gdf = gdf.dropna(subset=gdf.columns[3:].values.tolist(), how="all")
        gdf = gdf.set_crs(data[0].crs)

        self.outdata["Vector"] = [gdf]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """


class RasterToVectorBoundary(BasicModule):
    """
    Raster to vector boundary GUI.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)

        self.buttonbox.htmlfile = "raster.dm.rtov"

        self.setWindowTitle("Raster to Vector (Boundary Dataset)")

        gl_main.addWidget(self.buttonbox, 4, 0, 1, 2)

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
        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        gdf = gpd.GeoDataFrame()

        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        data = self.indata["Raster"]

        geoms = {"Data_ID": [], "geometry": []}

        for band in self.piter(data):
            band.get_boundary()
            geoms["Data_ID"].append(band.dataid)
            geoms["geometry"].append(band.geometry)

        gdf = gpd.GeoDataFrame(geoms)
        gdf = gdf.set_crs(data[0].crs)

        self.outdata["Vector"] = [gdf]

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """


def cluster_to_raster(indata):
    """
    Convert cluster datasets to raster datasets.

    Some routines will not understand the datasets produced by cluster
    analysis routines, since they are designated 'Cluster' and not 'Raster'.
    This provides a work-around for that.

    Parameters
    ----------
    indata : dict
        Dictionary of PyGMI datasets (pygmi.raster.datatypes.Data).

    Returns
    -------
    indata : dict
        Dictionary of PyGMI datasets (pygmi.raster.datatypes.Data).

    """
    if "Cluster" not in indata:
        return indata
    if "Raster" not in indata:
        indata["Raster"] = []

    for i in indata["Cluster"]:
        indata["Raster"].append(i)
        indata["Raster"][-1].data = indata["Raster"][-1].data + 1

    return indata


def fftcont(data, h, showlog=print, piter=iter):
    """
    Continuation.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        PyGMI raster data.
    h : float
        Height.
    showlog : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Progress bar iterable. The default is iter.

    Returns
    -------
    dat : pygmi.raster.datatypes.Data
        PyGMI raster data.

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, datamedian = fftprep(data)

    fftmod = np.fft.fft2(ndat.data)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)
    k = np.sqrt(KX**2 + KY**2)
    k[0, 0] = 1e-10  # to avoid division by zero

    filt = np.exp(-np.abs(k) * h)

    zout = np.real(np.fft.ifft2(fftmod * filt))

    zout = zout + datamedian
    dat = ndat.copy()
    dat.data = np.ma.array(zout)
    dat.dataid = "Upward_" + str(h) + "_" + data.dataid
    dat = lstack(
        [dat, data], piter=piter, showlog=showlog, masterid=data.dataid, commonmask=True
    )[0]

    return dat


def get_shape_bounds(sfile, crs=None, showlog=print):
    """
    Get bounds from a shape file.

    Parameters
    ----------
    sfile : str
        Filename for shapefile.
    crs : rasterio CRS
        target crs for shapefile
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    bounds : list
        Rasterio bounds.

    """
    if sfile == "" or sfile is None:
        return None

    gdf = gpd.read_file(sfile)

    gdf = gdf[gdf.geometry != None]

    if crs is not None:
        gdf = gdf.to_crs(crs)

    if gdf.geom_type.iloc[0] == "MultiPolygon":
        showlog(
            "You have a MultiPolygon. Only the first Polygon of the MultiPolygon will be used."
        )
        poly = gdf["geometry"].iloc[0]
        tmp = poly.geoms[0]

        gdf.geometry.iloc[0] = tmp

    if gdf.geom_type.iloc[0] != "Polygon":
        showlog("You need a polygon in that shape file")
        return None

    bounds = gdf.geometry.iloc[0].bounds

    return bounds


def merge_median(
    merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None
):
    """
    Merge using median for rasterio, taking minimum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    merged_data = np.ma.array(merged_data, mask=merged_mask)
    new_data = np.ma.array(new_data, mask=new_mask)

    mtmp1 = np.logical_and(~merged_mask, ~new_mask)
    mtmp2 = np.logical_and(~merged_mask, new_mask)

    tmp1 = new_data.copy()

    if True in mtmp1:
        tmp1 = tmp1 - np.ma.median(new_data[mtmp1])
        tmp1 = tmp1 + np.ma.median(merged_data[mtmp1])

    tmp1[mtmp2] = merged_data[mtmp2]

    merged_data[:] = tmp1


def merge_min(
    merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None
):
    """
    Merge using minimum for rasterio, taking minimum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    tmp = np.logical_and(~merged_mask, ~new_mask)

    tmp1 = merged_data.copy()
    tmp1[~new_mask] = new_data[~new_mask]
    tmp1[tmp] = np.minimum(merged_data[tmp], new_data[tmp])

    merged_data[:] = tmp1


def merge_max(
    merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None
):
    """
    Merge using maximum for rasterio, taking maximum value.

    Parameters
    ----------
    merged_data : numpy array
        Old data.
    new_data : numpy array
        New data to merge to old data.
    merged_mask : float
        Old mask.
    new_mask : float
        New mask.
    index : int, optional
        index of the current dataset within the merged dataset collection.
        The default is None.
    roff : int, optional
        row offset in base array. The default is None.
    coff : int, optional
        col offset in base array. The default is None.

    Returns
    -------
    None.

    """
    tmp = np.logical_and(~merged_mask, ~new_mask)

    tmp1 = merged_data.copy()
    tmp1[~new_mask] = new_data[~new_mask]
    tmp1[tmp] = np.maximum(merged_data[tmp], new_data[tmp])

    merged_data[:] = tmp1


def merge_order(ifiles, igeoms):
    """
    Sort data in an order which ensures overlaps.

    Parameters
    ----------
    ifiles : list
        list of filenames
    igeoms : list
        list of geometries

    Returns
    -------
    ofiles : list
        list of filenames
    """
    ofiles = []
    ogeoms = []

    ofiles.append(ifiles.pop(0))
    ogeoms.append(igeoms.pop(0))

    while igeoms:
        areas = []
        master = unary_union(ogeoms)
        for i in igeoms:
            areas.append(master.intersection(i).area)

        for i, area in enumerate(areas):
            if area == max(areas):
                ofiles.append(ifiles.pop(i))
                ogeoms.append(igeoms.pop(i))
                break

    return ofiles


def mosaic(
    dat,
    *,
    idir=None,
    bfile=None,
    bandstofiles=False,
    piter=iter,
    showlog=print,
    singleband=False,
    forcetype=None,
    shifttomedian=False,
    tmpdir=None,
    nodata=None,
    method="first",
    res=None,
    ifiles=None,
):
    """
    Merge files with different numbers of bands and/or band order.

    This uses more memory, but is flexible.

    Parameters
    ----------
    dat : list
        List of PyGMI data bands to be merged. Can be empty if idir is
        provided.
    idir : str, optional
        Directory where file to be mosaiced are found. The default is None.
    bfile : str, optional
        Path to boundary file. Can be shapefile or raster. The default is None.
    bandstofiles : bool, optional
        Export output bands to files. The default is False.
    piter : function, optional
        Progress bar iterable. The default is iter.
    showlog : function, optional
        Function for printing text. The default is print.
    singleband : bool, optional
        Ignore band names, since there is only one band. The default is False.
    forcetype : bool, optional
        Force input data type. The default is None.
    shifttomedian : bool, optional
        Shift bands to median value. The default is False.
    tmpdir : str, optional
        Alternate directory for temporary files. The default is None.
    nodata : float, optional
        Nodata value. The default is None.
    method : str, optional
        Mosaic method. Can be 'first', 'last', 'merge_min', 'merge_max' or
        'merge_median'. The default is 'first'.
    res : float, optional
        Output resolution. Can be a tuple. The default is None.
    ifiles : list, optional
        List of input files.

    Returns
    -------
    outdat : PyGMI raster data
        Output mosaiced dataset.

    """
    if method == "merge_min":
        method = merge_min
    if method == "merge_max":
        method = merge_max
    if method == "merge_median":
        method = merge_median

    indata = []
    if "Raster" in dat:
        for i in dat["Raster"]:
            indata.append(i)

    if "RasterFileList" in dat:
        for i in dat["RasterFileList"]:
            indata += get_from_rastermeta(i, piter=iter, metaonly=True)

    if idir is not None or ifiles is not None:
        if ifiles is None:
            ifiles = []
            for ftype in ["*.tif", "*.hdr", "*.img", "*.ers"]:
                ifiles += glob.glob(os.path.join(idir, ftype))

        if not ifiles:
            showlog("No input files in that directory")
            return False

        for ifile in piter(ifiles):
            indata += get_data(ifile, piter=iter, metaonly=True)

        if len(indata) == len(ifiles):
            singleband = True

    if indata is None:
        showlog("No input datasets")
        return False

    # Get projection information
    wkt = []
    crs = []
    for i in indata:
        if i.crs is None:
            showlog(f"{i.dataid} has no projection. Please assign one.")
            return False

        wkt.append(i.crs.to_wkt())
        crs.append(i.crs)
        # nodata = i.nodata

    wkt, iwkt, numwkt = np.unique(wkt, return_index=True, return_counts=True)
    if len(wkt) > 1:
        showlog("Error: Mismatched input projections. Selecting most common projection")

        crs = crs[iwkt[numwkt == numwkt.max()][0]]
    else:
        crs = indata[0].crs

    if bfile is None:
        bounds = None
    elif bfile[-3:] == "shp":
        bounds = get_shape_bounds(bfile, crs, showlog)
    else:
        dattmp = get_data(bfile, piter=iter, metaonly=True)
        if dattmp is None:
            bounds = None
        else:
            bounds = dattmp[0].bounds
            x = [bounds[0], bounds[2]]
            y = [bounds[1], bounds[3]]
            x, y = reprojxy(x, y, dattmp[0].crs, crs)
            bounds = [x[0], y[0], x[1], y[1]]

    # Start Merge
    bandlist = []
    for i in indata:
        bandlist.append(i.dataid)

    bandlist = list(set(bandlist))

    if singleband is True:
        bandlist = ["Band_1"]

    outdat = []
    for dataid in bandlist:
        showlog("Extracting " + dataid + "...")

        if bandstofiles:
            odir = os.path.join(idir, "mosaic")
            os.makedirs(odir, exist_ok=True)
            ofile = dataid + ".tif"
            ofile = ofile.replace(" ", "_")
            ofile = ofile.replace(",", "_")
            ofile = ofile.replace("*", "mult")
            ofile = os.path.join(odir, ofile)

            if os.path.exists(ofile):
                showlog("Output file exists, skipping.")
                continue

        ifiles = []
        allmval = []
        geomlist = []
        metadata = {}
        datetime = None
        ofile = ""

        for i in piter(indata):
            if i.dataid != dataid and singleband is False:
                continue
            metadata = i.metadata
            datetime = i.datetime

            if bounds is not None:
                x = [bounds[0], bounds[2]]
                y = [bounds[1], bounds[3]]
                x, y = reprojxy(x, y, crs, i.crs)
                bounds2 = [x[0], y[0], x[1], y[1]]
            else:
                bounds2 = None

            i2 = get_data(
                i.filename,
                piter=iter,
                tnames=[i.dataid],
                bounds=bounds2,
                showlog=showlog,
            )

            if i2 is None:
                continue

            i2 = i2[0]

            if i2.crs != crs:
                src_height, src_width = i2.data.shape
                try:
                    transform, width, height = calculate_default_transform(
                        i2.crs, crs, src_width, src_height, *i2.bounds
                    )
                except rasterio.errors.CRSError:
                    showlog("Problem with projection,aborting....")
                    return False
                i2 = data_reproject(i2, crs, transform, height, width)

            if method == "merge_median":
                i2.get_boundary()
                geomlist.append(i2.geometry)

            if forcetype is not None:
                i2.data = i2.data.astype(forcetype)

            if shifttomedian:
                mval = np.ma.median(i2.data)
            else:
                mval = 0
            allmval.append(mval)

            if singleband is True:
                i2.dataid = "Band_1"

            trans = rasterio.transform.from_origin(
                i2.extent[0], i2.extent[3], i2.xdim, i2.ydim
            )

            if tmpdir is None:
                tmpdir = tempfile.gettempdir()

            if i.meta["driver"] == "SENTINEL2":
                tmpfile = os.path.join(
                    tmpdir, os.path.basename(os.path.dirname(i.filename))
                )
            else:
                tmpfile = os.path.join(tmpdir, os.path.basename(i.filename))

            tmpid = i2.dataid
            tmpid = tmpid.replace(" ", "_")
            tmpid = tmpid.replace(",", "_")
            tmpid = tmpid.replace("*", "mult")
            tmpid = tmpid.replace(r"/", "div")

            tmpfile = tmpfile[:-4] + "_" + tmpid + ".tif"

            if i2.data.dtype == np.int16:
                i2.data = i2.data.astype(np.int32)

            raster = rasterio.open(
                tmpfile,
                "w",
                driver="GTiff",
                height=i2.data.shape[0],
                width=i2.data.shape[1],
                count=1,
                dtype=i2.data.dtype,
                transform=trans,
            )

            if nodata is None and np.issubdtype(i2.data.dtype, np.floating):
                nodata = 1.0e20
            elif nodata is None:
                nodata = -99999

            if i2.data.dtype == np.float32:
                nodata = np.float32(nodata)

            tmpdat = i2.data
            tmpdat = tmpdat.filled(nodata)
            tmpdat = np.ma.masked_equal(tmpdat, nodata)
            tmpdat = tmpdat - mval

            raster.write(tmpdat, 1)
            raster.write_mask(~np.ma.getmaskarray(i2.data))

            raster.close()
            ifiles.append(tmpfile)
            del i2

        if len(ifiles) < 2:
            showlog("Too few bands of name " + dataid)
            continue

        if geomlist:
            ifiles = merge_order(ifiles, geomlist)
        showlog("Mosaicing " + dataid + "...")

        with rasterio.Env(CPL_DEBUG=True):
            datmos, otrans = rasterio.merge.merge(
                ifiles, nodata=nodata, method=method, res=res, bounds=bounds
            )

        for j in ifiles:
            if os.path.exists(j):
                os.remove(j)
            if os.path.exists(j + ".msk"):
                os.remove(j + ".msk")

        datmos = datmos.squeeze()
        datmos = np.ma.masked_equal(datmos, nodata)
        datmos = datmos + np.median(allmval)
        outdat.append(numpy_to_pygmi(datmos, dataid=dataid))
        outdat[-1].set_transform(transform=otrans)
        outdat[-1].crs = crs
        outdat[-1].nodata = nodata
        outdat[-1].metadata = metadata
        outdat[-1].datetime = datetime

        if bandstofiles:
            export_raster(
                ofile,
                outdat,
                drv="GTiff",
                compression="DEFLATE",
                showlog=showlog,
                piter=piter,
            )

            del outdat
            del datmos
            outdat = []

    if bounds is not None and bfile[-3:] == "shp":
        outdat = cut_raster(outdat, bfile, deepcopy=False)

    return outdat


def redistribute_vertices(geom, distance):
    """
    Redistribute vertices in a geometry.

    From https://stackoverflow.com/questions/34906124/
    interpolating-every-x-distance-along-multiline-in-shapely,
    and by Mike-T.

    Parameters
    ----------
    geom : shapely geometry
        Geometry from geopandas.
    distance : float
        sampling distance.

    Raises
    ------
    ValueError
        Error when there is an unknown geometry.

    Returns
    -------
    shapely geometry
        New geometry.

    """
    if geom.geom_type == "LineString":
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [
                geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )
    if geom.geom_type == "MultiLineString":
        parts = [redistribute_vertices(part, distance) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    raise ValueError(f"unhandled geometry {geom.geom_type}")


def taylorcont(data, h, showlog=print, piter=iter):
    """
    Taylor Continuation.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        PyGMI raster data.
    h : float
        Height.
    showlog : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Progress bar iterable. The default is iter.

    Returns
    -------
    dat : pygmi.raster.datatypes.Data
        PyGMI raster data.

    """
    dz = verticalp(data, order=1, showlog=showlog, piter=piter)
    dz2 = verticalp(data, order=2, showlog=showlog, piter=piter)
    dz3 = verticalp(data, order=3, showlog=showlog, piter=piter)
    zout = (
        data.data
        + h * dz
        + h**2 * dz2 / math.factorial(2)
        + h**3 * dz3 / math.factorial(3)
    )

    dat = Data()
    dat.data = np.ma.masked_invalid(zout)
    dat.data.mask = np.ma.getmaskarray(data.data)
    dat.nodata = data.data.fill_value
    dat.dataid = "Downward_" + str(h) + "_" + data.dataid
    dat.set_transform(transform=data.transform)
    dat.crs = data.crs
    return dat


def trim_raster(olddata):
    """
    Trim nulls from a raster dataset.

    This function trims entire rows or columns of data which are masked,
    and are on the edges of the dataset. Masked values are set to the null
    value.

    Parameters
    ----------
    olddata : list of pygmi.raster.datatypes.Data
        PyGMI dataset.

    Returns
    -------
    olddata : list of pygmi.raster.datatypes.Data
        PyGMI dataset.
    """
    for data in olddata:
        mask = np.ma.getmaskarray(data.data)

        rowstart = 0
        for i in range(mask.shape[0]):
            if bool(mask[i].min()) is False:
                break
            rowstart += 1

        rowend = mask.shape[0]
        for i in range(mask.shape[0] - 1, -1, -1):
            if bool(mask[i].min()) is False:
                break
            rowend -= 1

        colstart = 0
        for i in range(mask.shape[1]):
            if bool(mask[:, i].min()) is False:
                break
            colstart += 1

        colend = mask.shape[1]
        for i in range(mask.shape[1] - 1, -1, -1):
            if bool(mask[:, i].min()) is False:
                break
            colend -= 1

        # drows, dcols = data.data.shape
        data.data = data.data[rowstart:rowend, colstart:colend]
        data.data.mask = mask[rowstart:rowend, colstart:colend]

        xmin = data.extent[0] + colstart * data.xdim
        ymax = data.extent[-1] - rowstart * data.ydim

        data.set_transform(data.xdim, xmin, data.ydim, ymax)

    return olddata


def verticalp(data, order=1, showlog=print, piter=iter):
    """
    Vertical derivative.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        Input data.
    order : float, optional
        Order. The default is 1.
    showlog : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Progress bar iterable. The default is iter.

    Returns
    -------
    dout : numpy array
        Output data

    """
    xdim = data.xdim
    ydim = data.ydim

    ndat, _ = fftprep(data)
    fftmod = np.fft.fft2(ndat.data)

    KX, KY = fft_getkxy(fftmod, xdim, ydim)

    k = np.sqrt(KX**2 + KY**2)
    k[0, 0] = 1e-10  # to avoid division by zero
    filt = k**order

    zout = np.real(np.fft.ifft2(fftmod * filt))

    dat = ndat.copy()
    dat.data = np.ma.array(zout)
    dat.dataid = "VD_" + data.dataid
    dat = lstack(
        [dat, data], piter=piter, showlog=showlog, masterid=data.dataid, commonmask=True
    )[0]

    zout = dat.data

    return zout


def _testdown():
    """Continuation testing routine."""
    import matplotlib.pyplot as plt

    from pygmi.pfmod.grvmag3d import calc_field, quick_model
    # from IPython import get_ipython
    # get_ipython().run_line_magic('matplotlib', 'inline')

    h = 4.0
    dxy = 1.0
    magcalc = True

    # quick model
    lmod = quick_model(
        numx=100, numy=100, numz=10, dxy=dxy, d_z=1, tlx=1000.0, tly=1000.0
    )
    lmod.lith_index[45:55, :, 1] = 1
    lmod.lith_index[45:50, :, 0] = 1
    lmod.ght = 10
    lmod.mht = 10
    calc_field(lmod, magcalc=magcalc)
    if magcalc:
        z = lmod.griddata["Calculated Magnetics"]
        z.data = z.data + 5
    else:
        z = lmod.griddata["Calculated Gravity"]

    # Calculate the field
    lmod = quick_model(
        numx=100, numy=100, numz=10, dxy=dxy, d_z=1, tlx=1000.0, tly=1000
    )
    lmod.lith_index[45:55, :, 1] = 1
    lmod.lith_index[45:50, :, 0] = 1
    lmod.ght = 10 - h
    lmod.mht = 10 - h
    calc_field(lmod, magcalc=magcalc)
    if magcalc:
        downz0 = lmod.griddata["Calculated Magnetics"]
        downz0.data = downz0.data + 5
    else:
        downz0 = lmod.griddata["Calculated Gravity"]

    # downz0, z = z, downz0

    dz = verticalp(z, order=1)
    dz2 = verticalp(z, order=2)
    dz3 = verticalp(z, order=3)

    # normal downward
    zdownn = fftcont(z, h)

    # downward, taylor
    # h = -h
    zdown = (
        z.data
        + h * dz
        + h**2 * dz2 / math.factorial(2)
        + h**3 * dz3 / math.factorial(3)
    )

    # Plotting
    plt.plot(downz0.data[50], "r")
    plt.plot(zdown.data[50], "b")
    plt.plot(zdownn.data[50], "k+")
    plt.show()


def _testfn():
    """Test."""
    import os
    import sys

    import matplotlib.pyplot as plt

    from pygmi.raster.iodefs import get_raster

    ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.tif"
    dat = get_raster(ifile)

    os.chdir(os.path.dirname(ifile))

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    tmp = RasterToVectorBoundary()
    tmp.indata["Raster"] = dat
    tmp.settings()

    gdf = tmp.outdata["Vector"][0]

    gdf.plot()

    plt.show()


def _testmosaic():
    """Test."""
    idir = r"D:\tmp"
    dat = {}

    mosaic(dat, idir=idir)


if __name__ == "__main__":
    _testmosaic()
