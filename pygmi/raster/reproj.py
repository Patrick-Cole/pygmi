# -----------------------------------------------------------------------------
# Name:        reporj.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2024 Council for Geoscience
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
"""Reprojection functions."""

import numpy as np
from PyQt5 import QtWidgets
import pyproj
from pyproj.crs import CRS, ProjectedCRS
from pyproj.crs.coordinate_operation import TransverseMercatorConversion
import rasterio
from rasterio.warp import calculate_default_transform, reproject

from pygmi.raster.datatypes import Data


class GroupProj(QtWidgets.QWidget):
    """
    Group Proj.

    Custom widget
    """

    def __init__(self, title='Projection', parent=None):
        super().__init__(parent)

        self.wkt = ''

        self.gl_1 = QtWidgets.QGridLayout(self)
        self.gbox = QtWidgets.QGroupBox(title)
        self.cmb_datum = QtWidgets.QComboBox()
        self.cmb_proj = QtWidgets.QComboBox()

        self.lbl_wkt = QtWidgets.QTextBrowser()
        self.lbl_wkt.setWordWrapMode(0)

        self.gl_1.addWidget(self.gbox, 1, 0, 1, 2)

        gl_1 = QtWidgets.QGridLayout(self.gbox)
        gl_1.addWidget(self.cmb_datum, 0, 0, 1, 1)
        gl_1.addWidget(self.cmb_proj, 1, 0, 1, 1)
        gl_1.addWidget(self.lbl_wkt, 2, 0, 1, 1)

        self.epsg_proj = getepsgcodes()
        self.epsg_proj[r'Current / Current'] = self.wkt
        self.epsg_proj[r'None / None'] = ''
        tmp = list(self.epsg_proj.keys())
        tmp.sort(key=lambda c: c.lower())

        self.plist = {}
        for i in tmp:
            if r' / ' in i:
                datum, proj = i.split(r' / ')
            else:
                datum = i
                proj = i

            if datum not in self.plist:
                self.plist[datum] = []
            self.plist[datum].append(proj)

        tmp = list(set(self.plist.keys()))
        tmp.sort()
        tmp = ['Current', 'WGS 84']+tmp

        for i in tmp:
            j = self.plist[i]
            if r'Geodetic Geographic' in j and j[0] != r'Geodetic Geographic':
                self.plist[i] = [r'Geodetic Geographic']+self.plist[i]

        self.cmb_datum.addItems(tmp)
        self.cmb_proj.addItem('Current')
        self.cmb_datum.currentIndexChanged.connect(self.combo_datum_change)
        self.cmb_proj.currentIndexChanged.connect(self.combo_change)

    def set_current(self, wkt):
        """
        Set new WKT for current option.

        Parameters
        ----------
        wkt : str
            Well Known Text descriptions for coordinates (WKT).

        Returns
        -------
        None.

        """
        if wkt in ['', 'None']:
            self.cmb_datum.setCurrentText('None')
            return

        self.wkt = wkt
        self.epsg_proj[r'Current / Current'] = self.wkt
        self.combo_change()

    def combo_datum_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        indx = self.cmb_datum.currentIndex()
        txt = self.cmb_datum.itemText(indx)
        self.cmb_proj.currentIndexChanged.disconnect()

        self.cmb_proj.clear()
        self.cmb_proj.addItems(self.plist[txt])

        self.cmb_proj.currentIndexChanged.connect(self.combo_change)

        self.combo_change()

    def combo_change(self):
        """
        Change Combo.

        Returns
        -------
        None.

        """
        dtxt = self.cmb_datum.currentText()
        ptxt = self.cmb_proj.currentText()

        txt = dtxt + r' / '+ptxt

        self.wkt = self.epsg_proj[txt]

        # if self.wkt is not a string, it must be epsg code
        if not isinstance(self.wkt, str):
            self.wkt = CRS.from_epsg(self.wkt).to_wkt(pretty=True)
        elif self.wkt not in ['', 'None']:
            self.wkt = CRS.from_wkt(self.wkt).to_wkt(pretty=True)

        # The next two lines make sure we have spaces after ALL commas.
        wkttmp = self.wkt.replace(', ', ',')
        wkttmp = wkttmp.replace(',', ', ')

        self.lbl_wkt.setText(wkttmp)


def data_reproject(data, ocrs, otransform=None, orows=None,
                   ocolumns=None, icrs=None):
    """
    Reproject dataset.

    Parameters
    ----------
    data : PyGMI Data
        PyGMI dataset.
    ocrs : CRS
        output crs.
    otransform : Affine, optional
        Output affine transform. The default is None.
    orows : int, optional
        output rows. The default is None.
    ocolumns : int, optional
        output columns. The default is None.
    icrs : CRS, optional
        input crs. The default is None.

    Returns
    -------
    data2 : PyGMI Data
        Reprojected dataset.

    """
    if icrs is None:
        icrs = data.crs

    if otransform is None:
        src_height, src_width = data.data.shape

        otransform, ocolumns, orows = calculate_default_transform(
            icrs, ocrs, src_width, src_height, *data.bounds)

    if data.nodata is None:
        nodata = data.data.fill_value
    else:
        nodata = data.nodata

    odata = np.zeros((orows, ocolumns), dtype=data.data.dtype)
    odata, _ = reproject(source=data.data,
                         destination=odata,
                         src_transform=data.transform,
                         src_crs=icrs,
                         dst_transform=otransform,
                         dst_crs=ocrs,
                         src_nodata=nodata,
                         resampling=rasterio.enums.Resampling['bilinear'])

    data2 = Data()
    data2.data = odata
    data2.crs = ocrs
    data2.set_transform(transform=otransform)
    data2.data = data2.data.astype(data.data.dtype)
    data2.dataid = data.dataid
    data2.wkt = CRS.to_wkt(ocrs)
    data2.filename = data.filename[:-4]+'_prj'+data.filename[-4:]

    data2.data = np.ma.masked_equal(data2.data, nodata)
    data2.nodata = nodata
    data2.metadata = data.metadata

    return data2


def getepsgcodes():
    """
    Routine used to get a list of EPSG codes.

    Returns
    -------
    pcodes : dictionary
        Dictionary of codes per projection in WKT format.

    """
    crs_list = pyproj.database.query_crs_info(auth_name='EPSG', pj_types=None)

    pcodes = {}
    for i in crs_list:
        if '/' in i.name:
            pcodes[i.name] = int(i.code)
        else:
            pcodes[i.name+r' / Geodetic Geographic'] = int(i.code)

    pcodes['WGS 84 / Geodetic Geographic'] = 4326

    for datum in [4222, 4148]:
        for clong in range(15, 35, 2):
            geog_crs = CRS.from_epsg(datum)
            proj_crs = ProjectedCRS(name=f'{geog_crs.name} / TM{clong}',
                                    conversion=TransverseMercatorConversion(
                                        latitude_natural_origin=0,
                                        longitude_natural_origin=clong,
                                        false_easting=0,
                                        false_northing=0,
                                        scale_factor_natural_origin=1.0,),
                                    geodetic_crs=geog_crs)

            pcodes[f'{geog_crs.name} / TM{clong}'] = proj_crs.to_wkt(pretty=True)

    return pcodes


def _testfn():
    """Test."""
    from pygmi.raster.iodefs import get_raster

    ifile1 = r"D:\Landslides\JTNdem.tif"

    dat1 = get_raster(ifile1)


if __name__ == "__main__":
    _testfn()
