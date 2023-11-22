# -----------------------------------------------------------------------------
# Name:        structure.py (part of PyGMI)
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
"""Structure complexity routines."""

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.signal import correlate
import shapely
from shapely.geometry import LineString
from rasterio.features import rasterize

from pygmi import menu_default
from pygmi.misc import BasicModule, ProgressBarText
from pygmi.raster.datatypes import Data, bounds_to_transform


class StructComp(BasicModule):
    """Structure complexity."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dxy = None
        self.dataid_text = None

        self.le_dxy = QtWidgets.QLineEdit('1.0')
        self.le_null = QtWidgets.QLineEdit('0.0')
        self.le_bdist = QtWidgets.QLineEdit('4.0')

        self.cmb_dataid = QtWidgets.QComboBox()
        self.cmb_grid_method = QtWidgets.QComboBox()
        self.lbl_rows = QtWidgets.QLabel('Rows: 0')
        self.lbl_cols = QtWidgets.QLabel('Columns: 0')
        self.lbl_bdist = QtWidgets.QLabel('Blanking Distance:')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.vector.dataprep.datagrid')
        lbl_band = QtWidgets.QLabel('Column to Grid:')
        lbl_dxy = QtWidgets.QLabel('Cell Size:')
        lbl_null = QtWidgets.QLabel('Null Value:')
        lbl_method = QtWidgets.QLabel('Gridding Method:')

        val = QtGui.QDoubleValidator(0.0000001, 9999999999.0, 9)
        val.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.C))

        self.le_dxy.setValidator(val)
        self.le_null.setValidator(val)

        self.cmb_grid_method.addItems(['Nearest Neighbour', 'Linear', 'Cubic',
                                       'Minimum Curvature'])

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Dataset Gridding')

        gl_main.addWidget(lbl_method, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_grid_method, 0, 1, 1, 1)
        gl_main.addWidget(lbl_dxy, 1, 0, 1, 1)
        gl_main.addWidget(self.le_dxy, 1, 1, 1, 1)
        gl_main.addWidget(self.lbl_rows, 2, 0, 1, 2)
        gl_main.addWidget(self.lbl_cols, 3, 0, 1, 2)
        gl_main.addWidget(lbl_band, 4, 0, 1, 1)
        gl_main.addWidget(self.cmb_dataid, 4, 1, 1, 1)
        gl_main.addWidget(lbl_null, 5, 0, 1, 1)
        gl_main.addWidget(self.le_null, 5, 1, 1, 1)
        gl_main.addWidget(self.lbl_bdist, 6, 0, 1, 1)
        gl_main.addWidget(self.le_bdist, 6, 1, 1, 1)
        gl_main.addWidget(helpdocs, 7, 0, 1, 1)
        gl_main.addWidget(buttonbox, 7, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.le_dxy.textChanged.connect(self.dxy_change)

    def dxy_change(self):
        """
        When dxy is changed on the interface, this updates rows and columns.

        Returns
        -------
        None.

        """
        txt = str(self.le_dxy.text())
        if txt.replace('.', '', 1).isdigit():
            self.dxy = float(self.le_dxy.text())
        else:
            return

        data = self.indata['Vector'][0]

        x = data.geometry.x.values
        y = data.geometry.y.values

        cols = round(x.ptp()/self.dxy)
        rows = round(y.ptp()/self.dxy)

        self.lbl_rows.setText('Rows: '+str(rows))
        self.lbl_cols.setText('Columns: '+str(cols))

    def grid_method_change(self):
        """
        When grid method is changed, this updated hidden controls.

        Returns
        -------
        None.

        """
        if self.cmb_grid_method.currentText() == 'Minimum Curvature':
            self.lbl_bdist.show()
            self.le_bdist.show()
        else:
            self.lbl_bdist.hide()
            self.le_bdist.hide()

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
        if 'Vector' not in self.indata:
            self.showlog('No Point Data')
            return False

        data = self.indata['Vector'][0]

        if data.geom_type.iloc[0] != 'Point':
            self.showlog('No Point Data')
            return False

        self.cmb_dataid.clear()

        filt = ((data.columns != 'geometry') &
                (data.columns != 'line'))

        cols = list(data.columns[filt])
        self.cmb_dataid.addItems(cols)

        if self.dataid_text is None:
            self.dataid_text = self.cmb_dataid.currentText()
        if self.dataid_text in cols:
            self.cmb_dataid.setCurrentText(self.dataid_text)

        if self.dxy is None:
            x = data.geometry.x.values
            y = data.geometry.y.values

            dx = x.ptp()/np.sqrt(x.size)
            dy = y.ptp()/np.sqrt(y.size)
            self.dxy = max(dx, dy)
            self.dxy = min([x.ptp(), y.ptp(), self.dxy])

        self.le_dxy.setText(f'{self.dxy:.8f}')
        self.dxy_change()

        self.grid_method_change()
        if not nodialog:
            tmp = self.exec()
            if tmp != 1:
                return False

        try:
            float(self.le_dxy.text())
            float(self.le_null.text())
            float(self.le_bdist.text())
        except ValueError:
            self.showlog('Value Error')
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
        self.saveobj(self.le_dxy)
        self.saveobj(self.le_null)
        self.saveobj(self.le_bdist)
        self.saveobj(self.dataid_text)
        self.saveobj(self.cmb_dataid)
        self.saveobj(self.cmb_grid_method)

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
        nullvalue = float(self.le_null.text())
        bdist = float(self.le_bdist.text())
        data = self.indata['Vector'][0]
        dataid = self.cmb_dataid.currentText()
        newdat = []

        if bdist < 1:
            bdist = None
            self.showlog('Blanking distance too small.')

        data2 = data[['geometry', dataid]]
        data2 = data2.dropna()

        filt = (data2[dataid] != nullvalue)
        x = data2.geometry.x.values[filt]
        y = data2.geometry.y.values[filt]
        z = data2[dataid].values[filt]


        newdat.append(dat)

        geom2, Hdat = feature_intersection_density(gdf, 500, 200000, piter=piter)

        Edat = feature_orientation_diversity(gdf, 500, 3, piter=piter)

        Vdat, Ddat = feature_circular_stats(gdf, 500, 3, piter=piter)

        Fdat = feature_fracdim(gdf, 200, 21, piter=piter)


        self.outdata['Raster'] = newdat
        self.outdata['Vector'] = self.indata['Vector']



def extendlines(gdf, length=500, piter=iter):
    """
    Extent lines from GeoPandas dataframe.

    Parameters
    ----------
    gdf : GeoDataFrame
        A dataframe containing LINESTRINGs.
    length : float
        distance in metres to extend the line on either side.
    piter : iter
        Progressbar iterable.

    Returns
    -------
    gdf2 : GeoDataFrame
        A dataframe containing extended LINESTRINGs.

    """
    gdf2 = gdf.copy()

    for i, row in piter(gdf.iterrows()):
        line = np.array(row.geometry.coords)

        p2, p1 = line[:2]
        theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        dy = length*np.sin(theta)
        dx = length*np.cos(theta)
        line[0] = (p2[0]+dx, p2[1]+dy)

        p1, p2 = line[-2:]
        theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        dy = length*np.sin(theta)
        dx = length*np.cos(theta)
        line[-1] = (p2[0]+dx, p2[1]+dy)

        gdf2.loc[i, 'geometry'] = shapely.LineString(line)

    return gdf2


def feature_intersection_density(gdf, dxy, var, extend=500, piter=iter):
    """
    Feature intersection density.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe of linear features.
    dxy : float
        Raster cell size
    var : float
        Variance.
    extend : float
        Distance to extend linear feaatures.
    piter : iter
        Progressbar iterable.

    Returns
    -------
    geom2 : GeoSeries
        New geometry with intersection points.
    dat : PyGMi Data
        Output raster data

    """
    # Extend lines to make sure almost intersections are found
    gdf = extendlines(gdf, extend, piter=piter)

    # Find intersections
    pnts = []
    geom1 = gdf.geometry
    for i, line1 in enumerate(piter(geom1)):
        geom2 = gdf.loc[i+1:, 'geometry']
        for line2 in geom2:
            pnt = line1.intersection(line2)
            if not pnt.is_empty:
                pnts.append(pnt)

    gdf2 = gpd.GeoDataFrame(geometry=pnts)
    geom2 = gdf2.geometry.explode(index_parts=False)

    # Calculate density using intersections
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xcoords = np.arange(xmin, xmax, dxy)
    ycoords = np.arange(ymin, ymax, dxy)

    x, y = np.meshgrid(xcoords, ycoords)
    H = np.zeros_like(x)

    for pnt in piter(geom2):
        G = 1/np.sqrt(2*np.pi*var)
        xdiff = (x-pnt.x)**2/(var*2)
        ydiff = (y-pnt.y)**2/(var*2)
        G = G*np.exp(-(xdiff+ydiff))
        H = H + G

    dat = Data()
    dat.data = np.ma.array(H[::-1])
    xmin = xcoords[0] - dxy/2
    ymax = ycoords[-1] + dxy/2
    dat.set_transform(dxy, xmin, dxy, ymax)
    dat.crs = gdf.crs

    return geom2, dat


def feature_orientation_diversity(gdf, dxy, wsize=3, piter=iter):
    """
    Feature orientation diversity.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe of linear features.
    dxy : float
        Raster cell size
    wsize : int
        Window size (must be odd)
    piter : iter
        Progressbar iterable.

    Returns
    -------
    dat : PyGMI Data
        Output raster data

    """
    transform, oshape = bounds_to_transform(gdf.total_bounds, dxy)
    gdf3 = segments_to_angles(gdf, piter)

    # Rasterize segments
    datall = {}
    linepix = 0
    for i in piter(range(180)):
        lines = gdf3.loc[gdf3.angle == i]
        if lines.size == 0:
            continue
        dat = rasterize(lines.geometry,
                        out_shape=oshape,
                        transform=transform,
                        all_touched=True)
        datall[i] = dat
        linepix += dat.sum()

    # Filter
    fmat = np.ones((wsize, wsize))
    E = np.zeros(oshape)
    for i in datall:
        pi = correlate(datall[i], fmat, 'same', 'direct')
        pi = 100*pi/linepix
        E += pi*np.log(pi, where=pi != 0)
    E = -E

    dat = Data()
    dat.data = np.ma.array(E)
    dat.crs = gdf.crs
    dat.set_transform(transform=transform)

    return dat


def feature_circular_stats(gdf, dxy, wsize=3, piter=iter):
    """
    Feature circular variance.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe of linear features.
    dxy : float
        Raster cell size
    wsize : int
        Window size (must be odd)
    piter : iter
        Progressbar iterable.

    Returns
    -------
    dat : PyGMI Data
        Output raster data

    """
    transform, oshape = bounds_to_transform(gdf.total_bounds, dxy)
    gdf3 = segments_to_angles(gdf, piter)

    # Rasterize segments
    datall = {}
    for i in piter(range(180)):
        lines = gdf3.loc[gdf3.angle == i]
        if lines.size == 0:
            continue
        dat = rasterize(lines.geometry,
                        out_shape=oshape,
                        transform=transform,
                        all_touched=True)
        datall[i] = dat

    # Filter
    fmat = np.ones((wsize, wsize))
    c1 = np.zeros(oshape)
    s1 = np.zeros(oshape)
    c2 = np.zeros(oshape)
    s2 = np.zeros(oshape)
    numall = np.zeros(oshape)
    for ai in datall:
        num = correlate(datall[ai], fmat, 'same', 'direct')
        c1 += num*np.cos(np.deg2rad(ai))
        s1 += num*np.sin(np.deg2rad(ai))

        c2 += num*np.cos(2*np.deg2rad(ai))
        s2 += num*np.sin(2*np.deg2rad(ai))

        numall += num

    numall[numall == 0] = 1
    r1 = np.sqrt(c1**2+s1**2)/numall
    v1 = 1-r1
    v1[v1 == 1] = 0.
    c2 = c2/numall
    s2 = s2/numall

    t2 = np.arctan2(s2, c2)
    # t2[c2 < 0] += np.pi
    # t2[np.logical_and(c2 > 0, s2 < 0)] += 2*np.pi

    d1 = (1-t2)/(2*r1**2)

    vdat = Data()
    vdat.data = np.ma.array(v1)
    vdat.crs = gdf.crs
    vdat.set_transform(transform=transform)

    ddat = Data()
    ddat.data = np.ma.masked_invalid(d1)
    ddat.crs = gdf.crs
    ddat.set_transform(transform=transform)

    return vdat, ddat


def feature_fracdim(gdf, dxy, wsize=21, piter=iter):
    """
    Feature fractal dimension.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe of linear features.
    dxy : float
        Raster cell size
    wsize : int
        Window size (must be odd)
    piter : iter
        Progressbar iterable.

    Returns
    -------
    dat : PyGMI Data
        Output raster data

    """
    transform, oshape = bounds_to_transform(gdf.total_bounds, dxy)

    dat = rasterize(gdf.geometry,
                    out_shape=oshape,
                    transform=transform,
                    all_touched=True)

    d1 = np.zeros(oshape)+np.nan
    w = wsize // 2
    rows, cols = oshape

    for i in piter(range(w, rows-w)):
        for j in range(w, cols-w):
            wdat = dat[i-w:i+w, j-w:j+w]
            d1[i, j] = fractal_dimension(wdat)

    # num = correlate(dat, fmat, 'same', 'direct')

    fdat = Data()
    fdat.data = np.ma.masked_invalid(d1)
    fdat.crs = gdf.crs
    fdat.set_transform(transform=transform)

    return fdat


def fractal_dimension(array, max_box_size=None, min_box_size=1,
                      n_samples=20, n_offsets=0):
    """
    Calculate the fractal dimension of a 3D numpy array.

    From: https://github.com/ChatzigeorgiouGroup/FractalDimension

    Parameters
    ----------
    array : np.array
        The array to calculate the fractal dimension of.
    max_box_size : int, optional
        The largest box size, given as the power of 2 so that 2**max_box_size
        gives the sidelength of the largest box. The default is None.
    min_box_size : int, optional
        The smallest box size, given as the power of 2 so that 2**min_box_size
        gives the sidelength of the smallest box. The default is 1.
    n_samples : int, optional
        number of scales to measure over. The default is 20.
    n_offsets : int, optional
        number of offsets to search over to find the smallest set N(s) to
        cover all voxels>0. The default is 0.

    Returns
    -------
    coeffs[0] : float
        Fractal dimension

    """
    if array.ndim == 2:
        array = np.expand_dims(array, 0)

    # determine the scales to measure on
    if max_box_size is None:
        # default max size is the largest power of 2 that fits in the
        # smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples,
                                  base=2))
    # remove duplicates that could occur as a result of the floor
    scales = np.unique(scales)

    # get the locations of all non-zero pixels
    locs = np.where(array > 0)
    voxels = np.array(list(zip(*locs)))

    # count the minimum amount of boxes touched
    Ns = []
    # loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        # search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in array.shape]
            bin_edges = [np.hstack([0-offset, x+offset]) for x in bin_edges]
            H1, _ = np.histogramdd(voxels, bins=bin_edges)
            touched.append(np.sum(H1 > 0))
        Ns.append(touched)
    Ns = np.array(Ns)

    # From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)

    # Only keep scales at which Ns changed
    scales = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])

    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]

    # perform fit
    coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)

    return coeffs[0]


def linesplit(curve):
    """Split LineString into segments."""
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


def segments_to_angles(gdf, piter=iter):
    """
    Get line segment angles.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with line segments.
    piter : iter
        Progressbar iterable.

    Returns
    -------
    gdf2 : GeoDataFrame
        GeoDataFrame with angles added.

    """
    segments = []
    for i, row in piter(gdf.iterrows()):
        segments += linesplit(row.geometry)
    gdf2 = gpd.GeoDataFrame(geometry=segments)
    gdf2['angle'] = np.nan

    for i, row in piter(gdf2.iterrows()):
        line = np.array(row.geometry.coords)
        p2, p1 = line[:2]

        if np.all(p1 == p2):
            continue

        theta = np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
        theta = np.rad2deg(theta)
        if theta < 0:
            theta += 180
        theta = int(theta)

        gdf2.loc[i, 'angle'] = theta

    gdf2 = gdf2.dropna()

    return gdf2


def _testfn():
    """Calculate structural complexity."""
    sfile = r"D:\Workdata\PyGMI Test Data\Vector\Rose\2329AC_lin_wgs84sutm35.shp"
    import sys
    from pygmi.vector.iodefs import ImportVector

    app = QtWidgets.QApplication(sys.argv)

    IO = ImportVector()
    IO.ifile = sfile
    IO.settings(True)

    SC = StructComp()
    SC.indata = IO.outdata
    SC.ifile = sfile
    SC.settings()




if __name__ == "__main__":
    _testfn()
