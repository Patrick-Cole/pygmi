# -----------------------------------------------------------------------------
# Name:        gridding.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2017 Council for Geoscience
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
""" These are helper routines for gridding up data. """

import pprint
import numpy as np
from pygmi.raster import iodefs as pio
from pygmi.raster.datatypes import Data
from pygmi.pfmod import iodefs as pio3d
import matplotlib.pyplot as plt
import scipy.interpolate as si


def grid():
    """ First 2 columns must be x and y """

    filename = r'C:\Work\Programming\pygmi\data\sue\filt_magdata.csv'
    ofile = r'C:\Work\Programming\pygmi\data\magdata.tif'
    srows = 0
    dlim = None
    xcol = 0
    ycol = 1
    zcol = 2
    dxy = 15

    # This bit reads in the first line to see if it is a header
    with open(filename) as pntfile:
        ltmp = pntfile.readline()

    ltmp = ltmp.lower()
    isheader = any(c.isalpha() for c in ltmp)

    # Check for comma delimiting
    if ',' in ltmp:
        dlim = ','

    # Set skip rows
    if isheader:
        srows = 1

    # Now read in data

    datatmp = np.genfromtxt(filename, unpack=True, delimiter=dlim,
                            skip_header=srows, usemask=False)

    # Now we interpolate
    xdata = datatmp[xcol]
    ydata = datatmp[ycol]
    zdata = datatmp[zcol]

    points = datatmp[:2].T

    newxdata = np.arange(xdata.min(), xdata.max(), dxy)
    newydata = np.arange(ydata.min(), ydata.max(), dxy)

    newpoints = np.meshgrid(newxdata, newydata)
    newpoints = (newpoints[0].flatten(), newpoints[1].flatten())

    grid = si.griddata(points, zdata, newpoints, method='cubic')

    grid.shape = (newydata.shape[0], newxdata.shape[0])

    grid = grid[::-1]

    # export data
    odat = Data()
    odat.dataid = ''
    odat.xdim = dxy
    odat.ydim = dxy
    odat.nullvalue = 1e+20
    odat.data = np.ma.masked_invalid(grid)
    odat.extent = [newxdata.min(), newxdata.max(),
                   newydata.min(), newydata.max()]

    tmp = pio.ExportData(None)
    tmp.ifile = ofile
#    tmp.export_ascii_xyz([odat])
#    tmp.export_gdal([odat], 'ENVI')
    tmp.export_gdal([odat], 'GTiff')

    # Plotting section

#    dataex = (newxdata.min(), newxdata.max(), newydata.min(), newydata.max())
#    plt.imshow(grid, cmap = plt.cm.jet, extent=dataex, origin='upper')

    plt.tricontourf(xdata, ydata, zdata, 40, cmap=plt.cm.jet)

#    plt.plot(xdata, ydata, '.')
    plt.colorbar()
    plt.show()

    breakpoint()


def model_to_grid_thickness():
    """ loads in a model """

    tmp = pio3d.ImportMod3D(None)
    tmp.ifile = r'C:\Work\Programming\pygmi\data\7-BC_57km_StagChamOnly_NEW.npz'
    ofile = r'C:\Work\Programming\pygmi\data\7-BC_57km_StagChamOnly_NEW.tif'

    # Reset Variables
    tmp.lmod.griddata.clear()
    tmp.lmod.lith_list.clear()

    # load model
    indict = np.load(tmp.ifile, allow_pickle=True)
    tmp.dict2lmod(indict)

    lith_index = tmp.lmod.lith_index

    lith_index[lith_index == -1] = 0
    lith_index[lith_index > 0] = 1

    dz = tmp.lmod.d_z
    out = lith_index.sum(2) * dz

    gout = tmp.lmod.griddata['Calculated Gravity']
    gout.data = out.T
    gout.data = gout.data[::-1]
    gout.nullvalue = 0.
    gout.data = np.ma.masked_equal(gout.data, 0.)

    tmp = pio.ExportData(None)
    tmp.ifile = ofile
    tmp.export_gdal([gout], 'GTiff')

    breakpoint()


def model_to_lith_depth():
    """ loads in a model """

    tmp = pio3d.ImportMod3D(None)
    tmp.ifile = r'C:\Work\Programming\pygmi\data\StagCham_Youssof_ALTComplexMantleLC_ds_extended.npz'
    ofile = r'C:\Work\Programming\pygmi\data\hope.tif'

    # Reset Variables
    tmp.lmod.griddata.clear()
    tmp.lmod.lith_list.clear()

    # load model
    indict = np.load(tmp.ifile, allow_pickle=True)
    tmp.dict2lmod(indict)

    tmp.lmod.update_lith_list_reverse()

    print('These are the lithologies and their codes:')
    pprint.pprint(tmp.lmod.lith_list_reverse)
    print('')
    lithcode = int(input("what lithology code do you wish? "))

    lith_index = tmp.lmod.lith_index

    dz = tmp.lmod.d_z
    dtm = (lith_index == -1).sum(2)*dz

    lith = (lith_index == lithcode)

    xxx, yyy, zzz = lith.shape

    out = np.zeros((xxx, yyy))-1.

    for i in range(xxx):
        for j in range(yyy):
            if True in lith[i, j]:
                out[i, j] = np.nonzero(lith[i, j])[0][0]*dz - dtm[i, j]

    gout = tmp.lmod.griddata['Calculated Gravity']
    gout.data = out.T
    gout.data = gout.data[::-1]
    gout.nullvalue = -1.
    gout.data = np.ma.masked_equal(gout.data, -1.)

    tmp = pio.ExportData(None)
    tmp.ifile = ofile
    tmp.export_gdal([gout], 'GTiff')


if __name__ == "__main__":
    model_to_lith_depth()
#    model_to_grid_thickness()
#    grid()
