# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
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
"""Class for data types."""

import numpy as np
from pygmi.raster.datatypes import Data


class LithModel():
    """
    Lithological Model Data.

    This is the main data structure for the modelling program

    Attributes
    ----------
    mlut : dictionary
        color table for lithologies
    numx : int
        number of columns per layer in model
    numy : int):
        number of rows per layer in model
    numz : int
        number of layers in model
    dxy : float
        dimension of cubes in the x and y directions
    d_z : float
        dimension of cubes in the z direction
    lith_index : numpy array
        3D array of lithological indices.
    curlayer : int
        Current layer
    xrange : list
        minimum and maximum x coordinates
    yrange : list
        minimum and maximum y coordinates
    zrange : list
        minimum and maximum z coordinates
    curprof : int
        current profile (in x or y direction)
    griddata : dictionary
        dictionary of Data classes with raster data
    custprofx : dictionary
        custom profile x coordinates
    custprofy : dictionary
        custom profile y coordinates
    profpics : dictionary
        profile pictures
    lith_list : dictionary
        list of lithologies
    lith_list_reverse : dictionary
        reverse lookup for lith_list
    mht : float
        height of magnetic sensor
    ght : float
        height of gravity sensor
    gregional : float
        gravity regional correction
    name : str
        name of the model

    """

    def __init__(self):
        self.mlut = {0: [170, 125, 90], 1: [255, 255, 0]}
        self.numx = None
        self.numy = None
        self.numz = None
        self.dxy = None
        self.d_z = None
        self.lith_index = None
        self.lith_index_grv_old = None
        self.lith_index_mag_old = None
        self.xrange = [None, None]
        self.yrange = [None, None]
        self.zrange = [None, None]
        self.griddata = {}
        self.custprofx = {}
        self.custprofy = {}
        self.profpics = {}
        self.lith_list = {}
        self.lith_list_reverse = {}
        self.mht = None
        self.ght = None
        self.gregional = 0.
        self.dataid = '3D Model'
        self.tmpfiles = None
        # Next line calls a function to update the variables above.
        self.update(50, 40, 5, 0., 0., 0., 100., 100., 100., 0.)

        self.olith_index = None
        self.odxy = None
        self.od_z = None
        self.oxrng = None
        self.oyrng = None
        self.ozrng = None
        self.onumx = None
        self.onumy = None
        self.onumz = None

        # Obsolete
#        self.curlayer = None
#        self.is_ew = True
#        self.curprof = None

    def lithold_to_lith(self, nodtm=False, pbar=None):
        """
        Transfers an old lithology to the new one, using update parameters.

        Parameters
        ----------
        nodtm : bool, optional
            Flag for a DTM. The default is False.
        pbar : pygmi.misc.ProgressBar, optional
            Progressbar. The default is None.

        Returns
        -------
        None.

        """
        if self.olith_index is None:
            return

        if pbar is not None:
            piter = pbar.iter
        else:
            piter = iter


        xvals = np.arange(self.xrange[0], self.xrange[1], self.dxy)
        yvals = np.arange(self.yrange[0], self.yrange[1], self.dxy)
        zvals = np.arange(self.zrange[0], self.zrange[1], self.d_z)

        if xvals[-1] == self.xrange[1]:
            xvals = xvals[:-1]
        if yvals[-1] == self.yrange[1]:
            yvals = yvals[:-1]
        if zvals[-1] == self.zrange[1]:
            yvals = yvals[:-1]

        xvals += 0.5 * self.dxy
        yvals += 0.5 * self.dxy
        zvals += 0.5 * self.d_z

        xvals = xvals[self.oxrng[0] < xvals]
        xvals = xvals[xvals < self.oxrng[1]]
        yvals = yvals[self.oyrng[0] < yvals]
        yvals = yvals[yvals < self.oyrng[1]]
        zvals = zvals[self.ozrng[0] < zvals]
        zvals = zvals[zvals < self.ozrng[1]]

        for x_i in piter(xvals):
            o_i = int((x_i - self.oxrng[0]) / self.odxy)
            i = int((x_i - self.xrange[0]) / self.dxy)
            for x_j in yvals:
                o_j = int((x_j - self.oyrng[0]) / self.odxy)
                j = int((x_j - self.yrange[0]) / self.dxy)
                for x_k in zvals:
                    o_k = int((self.ozrng[1] - x_k) / self.od_z)
                    k = int((self.zrange[1] - x_k) / self.d_z)

                    if (self.lith_index[i, j, k] != -1 and
                            self.olith_index[o_i, o_j, o_k] != -1) or nodtm:
                        self.lith_index[i, j, k] = \
                            self.olith_index[o_i, o_j, o_k]

    def dtm_to_lith(self, pbar=None):
        """
        Assign the DTM to the model.

        This means creating nodata values in areas above the DTM. These values
        are assigned a lithology of -1.

        Parameters
        ----------
        pbar : pygmi.misc.ProgressBar, optional
            Progressbar. The default is None.

        Returns
        -------
        None.

        """
        if 'DTM Dataset' not in self.griddata:
            return

        if pbar is not None:
            piter = pbar.iter
        else:
            piter = iter


        self.lith_index = np.zeros([self.numx, self.numy, self.numz],
                                   dtype=int)

        curgrid = self.griddata['DTM Dataset']

        d_x = curgrid.xdim
        d_y = curgrid.ydim
        gxmin = curgrid.extent[0]
        gymax = curgrid.extent[-1]
        grows, gcols = curgrid.data.shape

        utlz = curgrid.data.max()

        self.lith_index[:, :, :] = 0

        for i in piter(range(self.numx)):
            xcrd = self.xrange[0] + (i + .5) * self.dxy
            xcrd2 = int((xcrd - gxmin) / d_x)
            for j in range(self.numy):
                ycrd = self.yrange[1] - (j + .5) * self.dxy
                ycrd2 = grows - int((gymax - ycrd) / d_y)
                if ycrd2 == grows:
                    ycrd2 = grows-1

#                if (ycrd2 >= 0 and xcrd2 >= 0 and ycrd2 < grows and
#                        xcrd2 < gcols):
                if (0 <= ycrd2 < grows and 0 <= xcrd2 < gcols):
                    alt = curgrid.data.data[ycrd2, xcrd2]
                    if (curgrid.data.mask[ycrd2, xcrd2] or
                            np.isnan(alt) or alt == curgrid.nullvalue):
                        alt = curgrid.data.mean()
                    k_2 = int((utlz - alt) / self.d_z)
                    self.lith_index[i, j, :k_2] = -1

    def init_grid(self, data):
        """
        Initializes raster variables in the Data class.

        Parameters
        ----------
        data : numpy array
             Masked array containing raster data.

        Returns
        -------
        grid : PyGMI Data
            PyGMI raster dataset.

        """
        grid = Data()
        grid.data = data
        grid.xdim = self.dxy
        grid.ydim = self.dxy
        grid.extent = [self.xrange[0], self.xrange[1],
                       self.yrange[0], self.yrange[1]]

        return grid

    def init_calc_grids(self):
        """
        Initializes mag and gravity from the model.

        Returns
        -------
        None.

        """
        tmp = np.ma.zeros([self.numy, self.numx])
        self.griddata['Calculated Magnetics'] = self.init_grid(tmp.copy())
        self.griddata['Calculated Magnetics'].dataid = 'Calculated Magnetics'
        self.griddata['Calculated Magnetics'].units = 'nT'
        self.griddata['Calculated Gravity'] = self.init_grid(tmp.copy())
        self.griddata['Calculated Gravity'].dataid = 'Calculated Gravity'
        self.griddata['Calculated Gravity'].units = 'mGal'

    def is_modified(self, modified=True):
        """
        Updates modified flag.

        Parameters
        ----------
        modified : bool, optional
            Flag for whether the lithology has been modified. The default is True.

        Returns
        -------
        None.

        """
        for i in self.lith_list:
            self.lith_list[i].modified = modified

    def update(self, cols, rows, layers, utlx, utly, utlz, dxy, d_z, mht=-1,
               ght=-1, usedtm=True, pbar=None):
        """
        Updates the local variables for the LithModel class.

        Parameters
        ----------
        cols : int
            Number of columns per layer in model.
        rows : int
            Number of rows per layer in model.
        layers : int
            Number of layers in model.
        utlx : float
            Upper top left (NW) x coordinate.
        utly : float
            Upper top left (NW) y coordinate.
        utlz : float
            Upper top left (NW) z coordinate.
        dxy : float
            Dimension of cubes in the x and y directions.
        d_z : float
            Dimension of cubes in the z direction.
        mht : float, optional
            Height of magnetic sensor. The default is -1.
        ght : float, optional
            Height of gravity sensor. The default is -1.
        usedtm : bool, optional
            Flag to use a DTM. The default is True.
        pbar : pygmi.misc.ProgressBar, optional
            Progressbar. The default is None.

        Returns
        -------
        None.

        """
        if mht != -1:
            self.mht = mht
        if ght != -1:
            self.ght = ght

        self.olith_index = self.lith_index
        self.odxy = self.dxy
        self.od_z = self.d_z
        self.oxrng = np.copy(self.xrange)
        self.oyrng = np.copy(self.yrange)
        self.ozrng = np.copy(self.zrange)
        self.onumx = self.numx
        self.onumy = self.numy
        self.onumz = self.numz

        xextent = cols * dxy
        yextent = rows * dxy
        zextent = layers * d_z

        self.numx = cols
        self.numy = rows
        self.numz = layers
        self.xrange = [utlx, utlx + xextent]
        self.yrange = [utly - yextent, utly]
        self.zrange = [utlz - zextent, utlz]

        self.dxy = dxy
        self.d_z = d_z
        self.lith_index = np.zeros([self.numx, self.numy, self.numz],
                                   dtype=int)
        self.lith_index_mag_old = np.zeros([self.numx, self.numy, self.numz],
                                           dtype=int)
        self.lith_index_mag_old[:] = -1

        self.lith_index_grv_old = np.zeros([self.numx, self.numy, self.numz],
                                           dtype=int)
        self.lith_index_grv_old[:] = -1

        self.init_calc_grids()
        if usedtm:
            self.dtm_to_lith(pbar)
        self.lithold_to_lith(not usedtm, pbar)
        self.update_lithlist()
        self.is_modified()

    def update_lithlist(self):
        """
        Updates lith_list from local variables.

        Returns
        -------
        None.

        """
        for i in self.lith_list:
            self.lith_list[i].set_xyz(self.numx, self.numy, self.numz,
                                      self.dxy, self.mht, self.ght, self.d_z,
                                      modified=False)

    def update_lith_list_reverse(self):
        """
        Update the lith_list reverse lookup.

        It must be run at least once before using lith_list_reverse.

        Returns
        -------
        None.

        """
        keys = list(self.lith_list.keys())
        values = list(self.lith_list.values())

        if not keys:
            return

        self.lith_list_reverse = {}
        for i in range(len(keys)):
            self.lith_list_reverse[list(values)[i].lith_index] = list(keys)[i]
