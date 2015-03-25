# -----------------------------------------------------------------------------
# Name:        iodefs.py (part of PyGMI)
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
""" Import Data """

from PyQt4 import QtGui
from pygmi.raster.datatypes import Data
from pygmi.clust.datatypes import Clust
import numpy as np
from osgeo import gdal, osr
import struct
from pygmi.raster.dataprep import merge
import os


class ImportData(object):
    """
    Import Data - Interfaces with GDAL routines

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    ext : str
        filename extension
    """
    def __init__(self, parent=None):
        self.ifile = ""
        self.name = "Import Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        ext = \
            "ERMapper (*.ers);;" + \
            "ENVI (*.hdr);;" + \
            "GeoTiff (*.tif);;" + \
            "Geosoft UNCOMPRESSED grid (*.grd);;" + \
            "Geosoft (*.gxf);;" + \
            "Surfer grid (v.6) (*.grd);;" + \
            "GeoPak grid (*.grd);;" + \
            "ASCII with .hdr header (*.asc);;" + \
            "ASCII XYZ (*.xyz);;" + \
            "ArcGIS BIL (*.bil)"

        filename, filt = QtGui.QFileDialog.getOpenFileNameAndFilter(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)
        self.ext = filename[-3:]
        self.ext = self.ext.lower()

        if filt == 'GeoPak grid (*.grd)':
            dat = get_geopak(self.ifile)
        elif filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
            dat = get_geosoft(self.ifile)
        elif filt == 'ASCII with .hdr header (*.asc)':
            dat = get_ascii(self.ifile)
        else:
            dat = get_raster(self.ifile)

        if dat is None:
            if filt == 'Surfer grid (v.6) (*.grd)':
                QtGui.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import the surfer 6 '
                                          'grid. Please make sure it not '
                                          'another format, such as geosoft.',
                                          QtGui.QMessageBox.Ok,
                                          QtGui.QMessageBox.Ok)
            elif filt == 'Geosoft UNCOMPRESSED grid (*.grd)':
                QtGui.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import the grid. '
                                          'Please make sure it is a Geosoft '
                                          'FLOAT grid, and not a compressed '
                                          'grid. You can export your grid to '
                                          'this format using the Geosoft '
                                          'Viewer.',
                                          QtGui.QMessageBox.Ok,
                                          QtGui.QMessageBox.Ok)
            else:
                QtGui.QMessageBox.warning(self.parent, 'Error',
                                          'Could not import the grid.',
                                          QtGui.QMessageBox.Ok,
                                          QtGui.QMessageBox.Ok)
            return False

        output_type = 'Raster'
        if 'Cluster' in dat[0].dataid:
            output_type = 'Cluster'

        self.outdata[output_type] = dat
        return True


def get_ascii(ifile):
    """
    Import ascii raster dataset

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    afile = open(ifile, 'r')
    adata = afile.read()

    adata = adata.split()
    adata = np.array(adata, dtype=float)

    hfile = open(ifile[:-3]+'hdr', 'r')
    tmp = hfile.readlines()

    xdim = float(tmp[0].split()[-1])
    ydim = float(tmp[1].split()[-1])
    ncols = int(tmp[2].split()[-1])
    nrows = int(tmp[3].split()[-1])
    nbands = int(tmp[4].split()[-1])
    ulxmap = float(tmp[5].split()[-1])
    ulymap = float(tmp[6].split()[-1])
    bandid = ifile[:-4].rsplit('/')[-1]

    adata.shape = (nrows, ncols)

    if nbands > 1:
        print('PyGMI only supports single band ASCII files')

    dat = [Data()]
    i = 0

    dat[i].data = adata

    nval = -9999.0

    dat[i].data = np.ma.masked_equal(dat[i].data, nval)
    if dat[i].data.mask.size == 1:
        dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                            dat[i].data.mask)

    dat[i].nrofbands = nbands
    dat[i].tlx = ulxmap
    dat[i].tly = ulymap
    dat[i].dataid = bandid
    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = xdim
    dat[i].ydim = ydim

    return dat


def get_raster(ifile):
    """
    This function loads a raster dataset off the disk using the GDAL
    libraries. It returns the data in a PyGMI data object.

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """
    dat = []
    bname = ifile.split('/')[-1].rpartition('.')[0]+': '
    ifile = ifile[:]
    ext = ifile[-3:]
    if ext == 'hdr':
        ifile = ifile[:-4]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)

    if dataset is None:
        return None

    gtr = dataset.GetGeoTransform()
#    output_type = 'Raster'

    for i in range(dataset.RasterCount):
        rtmp = dataset.GetRasterBand(i+1)
        bandid = rtmp.GetDescription()
        nval = rtmp.GetNoDataValue()

        if 'Cluster' in bandid:
            # output_type = 'Cluster'
            dat.append(Clust())
        else:
            dat.append(Data())
        dat[i].data = rtmp.ReadAsArray()
        if dat[i].data.dtype.kind == 'i':
            if nval is None:
                nval = 999999
            nval = int(nval)
        else:
            if nval is None:
                nval = 1e+20
            nval = float(nval)
#            dtype = dat[i].data.dtype
#            if dtype != np.float64 and dtype != np.float32:
#                dat[i].data = dat[i].data.astype(np.float32)
#            if dtype == np.float64 or dtype == np.float32:
#                dat[i].data[dat[i].data == nval] = np.nan
        if ext == 'ers' and nval == -1.0e+32:
            dat[i].data[np.ma.less_equal(dat[i].data, nval)] = -1.0e+32
#                dat[i].data[np.ma.less_equal(dat[i].data, nval)] = np.nan

#            dat[i].data = np.ma.masked_invalid(dat[i].data)
# Note that because the data is stored in a masked array, the array ends up
# being double the size that it was on the disk.
        dat[i].data = np.ma.masked_invalid(dat[i].data)
        dat[i].data.mask = dat[i].data.mask | (dat[i].data == nval)
#        dat[i].data = np.ma.masked_equal(dat[i].data, nval)
        if dat[i].data.mask.size == 1:
            dat[i].data.mask = (np.ma.make_mask_none(dat[i].data.shape) +
                                dat[i].data.mask)

        dat[i].nrofbands = dataset.RasterCount
        dat[i].tlx = gtr[0]
        dat[i].tly = gtr[3]
        if bandid == '':
            bandid = bname+str(i+1)
        dat[i].dataid = bandid
        if bandid[-1] == ')':
            dat[i].units = bandid[bandid.rfind('(')+1:-1]

        dat[i].nullvalue = nval
        dat[i].rows = dataset.RasterYSize
        dat[i].cols = dataset.RasterXSize
        dat[i].xdim = abs(gtr[1])
        dat[i].ydim = abs(gtr[5])
        dat[i].gtr = gtr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs.AutoIdentifyEPSG()

        dat[i].wkt = srs.ExportToWkt()

        if 'Cluster' in bandid:
            dat[i].no_clusters = int(dat[i].data.max()+1)
#                dat[i].no_clusters = np.unique(dat[i].data).count()

    return dat


class ExportData(object):
    """
    Export Data

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    ext : str
        filename extension
    """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Export Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
#        self.dirname = ""

    def run(self):
        """ Show Info """
        if 'Cluster' in self.indata:
            data = self.indata['Cluster']
        elif 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            self.parent.showprocesslog('No raster data')
            return

        ext = \
            "ENVI (*.hdr);;" + \
            "ERMapper (*.ers);;" + \
            "GeoTiff (*.tif);;" + \
            "Geosoft (*.gxf);;" + \
            "Surfer grid (v.6) (*.grd);;" + \
            "ArcInfo ASCII (*.asc);;" + \
            "ASCII XYZ (*.xyz);;" + \
            "ArcGIS BIL (*.bil)"

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        self.parent.showprocesslog('Export Data Busy...')

    # Pop up save dialog box
        if self.ext == 'ers':
            self.export_gdal(data, 'ERS')
        if self.ext == 'gxf':
            self.export_gxf(data)
        if self.ext == 'grd':
            self.export_surfer(data)
        if self.ext == 'asc':
            self.export_ascii(data)
        if self.ext == 'xyz':
            self.export_ascii_xyz(data)
        if self.ext == 'tif':
            self.export_gdal(data, 'GTiff')
        if self.ext == 'hdr':
            self.export_gdal(data, 'ENVI')
        if self.ext == 'bil':
            self.export_gdal(data, 'EHdr')

        self.parent.showprocesslog('Export Data Finished!')

    def export_gdal(self, dat, drv):
        """
        Export to GDAL format

        Parameters
        ----------
        dat : PyGMI raster Data
            dataset to export
        drv : str
            name of the GDAL driver to use
        """

        data = merge(dat)
        xmin = data[0].tlx
        ymax = data[0].tly

        driver = gdal.GetDriverByName(drv)
        dtype = data[0].data.dtype

        if dtype == np.uint8:
            fmt = gdal.GDT_Byte
        elif dtype == np.int32:
            fmt = gdal.GDT_Int32
        elif dtype == np.float64:
            fmt = gdal.GDT_Float64
        else:
            fmt = gdal.GDT_Float32

        tmp = self.ifile.rpartition('.')

        if drv == 'GTiff':
            tmpfile = tmp[0] + '.tif'
        elif drv == 'EHdr':
            fmt = gdal.GDT_Float32
            dtype = np.float32
            tmpfile = tmp[0] + '.bil'
#        elif drv == 'VRT':
#            tmpfile = tmp[0] + '.vrt'
        else:
            tmpfile = tmp[0]

        if drv == 'GTiff' and dtype == np.uint8:
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt, options=['COMPRESS=NONE',
                                                         'TFW=YES'])
        else:
            out = driver.Create(tmpfile, int(data[0].cols), int(data[0].rows),
                                len(data), fmt)
        out.SetGeoTransform([xmin, data[0].xdim, 0, ymax, 0, -data[0].ydim])
#        orig = osr.SpatialReference()
#        orig.SetWellKnownGeogCS('WGS84')
#        orig.ImportFromEPSG(4222)  # Cape
#        orig.SetTM(0.0, 31.0, 1.0, 0.0, 0.0)
#        out.SetProjection(orig.ExportToWkt())

        out.SetProjection(data[0].wkt)

        for i in range(len(data)):
            rtmp = out.GetRasterBand(i+1)
            rtmp.SetDescription(data[i].dataid)

            dtmp = np.ma.array(data[i].data).astype(dtype)

            # This section tries to overcome null values with round off error
            # in 32-bit numbers.
            if dtype == np.float32:
                data[i].nullvalue = np.float64(np.float32(data[i].nullvalue))
                if data[i].data.min() > -1e+10:
                    data[i].nullvalue = np.float64(np.float32(-1e+10))
                elif data[i].data.max() < 1e+10:
                    data[i].nullvalue = np.float64(np.float32(1e+10))

            elif dtype == np.float or dtype == np.float64:
                data[i].nullvalue = np.float64(dtmp.fill_value)

            dtmp.set_fill_value(data[i].nullvalue)
            dtmp = dtmp.filled()

            if drv != 'GTiff':
                rtmp.SetNoDataValue(data[i].nullvalue)
            elif len(data) == 1:
                rtmp.SetNoDataValue(data[i].nullvalue)
            rtmp.WriteArray(dtmp)

        out = None  # Close File
        if drv == 'ENVI':
            with open(tmpfile+'.hdr', 'a') as myfile:
                myfile.write('data ignore value = ' + str(data[0].nullvalue))

    def export_gxf(self, data):
        """
        Export GXF data

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'gxf')
            fno = open(file_out, 'w')

            xmin = k.tlx
#            xmax = k.tlx + k.cols*k.xdim
            ymin = k.tly - k.rows*k.ydim
#            ymax = k.tly

            fno.write("#TITLE\n")
            fno.write(self.name)
            fno.write("\n#POINTS\n")
            fno.write(str(k.cols))
            fno.write("\n#ROWS\n")
            fno.write(str(k.rows))
            fno.write("\n#PTSEPARATION\n")
            fno.write(str(k.xdim))
            fno.write("\n#RWSEPARATION\n")
            fno.write(str(k.ydim))
            fno.write("\n#XORIGIN\n")
            fno.write(str(xmin))
            fno.write("\n#YORIGIN\n")
            fno.write(str(ymin))
            fno.write("\n#SENSE\n")
            fno.write("1")
            fno.write("\n#DUMMY\n")
            fno.write(str(k.nullvalue))
            fno.write("\n#GRID\n")
            tmp = k.data.filled(k.nullvalue)

            for i in range(k.data.shape[0]-1, -1, -1):
                kkk = 0
# write only 5 numbers in a row
                for j in range(k.data.shape[1]):
                    if kkk == 5:
                        kkk = 0
                    if kkk == 0:
                        fno.write("\n")

                    fno.write(str(tmp[i, j]) + "  ")
                    kkk += 1

            fno.close()

    def export_surfer(self, data):
        """
        Export a surfer binary grid

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'grd')
            fno = open(file_out, 'wb')

            xmin = k.tlx
            xmax = k.tlx + k.cols*k.xdim
            ymin = k.tly - k.rows*k.ydim
            ymax = k.tly

            bintmp = struct.pack('cccchhdddddd', b'D', b'S', b'B', b'B',
                                 k.cols, k.rows,
                                 xmin, xmax,
                                 ymin, ymax,
                                 np.min(k.data),
                                 np.max(k.data))
            fno.write(bintmp)

            ntmp = 1.701410009187828e+38
            tmp = (k.data.filled(ntmp)).astype('f')
            tmp = tmp[::-1]
            fno.write(tmp.tostring())

            fno.close()

    def export_ascii(self, data):
        """
        Export Ascii file

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'asc')
            fno = open(file_out, 'w')

            xmin = k.tlx
#            xmax = k.tlx + k.cols*k.xdim
            ymin = k.tly - k.rows*k.ydim
#            ymax = k.tly

            fno.write("ncols \t\t\t" + str(k.cols))
            fno.write("\nnrows \t\t\t" + str(k.rows))
            fno.write("\nxllcorner \t\t\t" + str(xmin))
            fno.write("\nyllcorner \t\t\t" + str(ymin))
            fno.write("\ncellsize \t\t\t" + str(k.xdim))
            fno.write("\nnodata_value \t\t" + str(k.nullvalue))

            tmp = k.data.filled(k.nullvalue)

            for j in range(k.rows):
                fno.write("\n")
                for i in range(k.cols):
                    fno.write(str(tmp[j, i]) + " ")
                    # fno.write(str(data[0].data[j].data[i]) + " ")

            fno.close()

    def export_ascii_xyz(self, data):
        """
        Export and xyz file

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to export
        """
        for k in data:
            file_out = self.get_filename(k, 'xyz')
            fno = open(file_out, 'w')

            tmp = k.data.filled(k.nullvalue)

            xmin = k.tlx
#            xmax = k.tlx + k.cols*k.xdim
#            ymin = k.tly - k.rows*k.ydim
            ymax = k.tly

            for j in range(k.rows):
                for i in range(k.cols):
                    fno.write(str(xmin+i*k.xdim) + " " +
                              str(ymax-j*k.ydim) + " " +
                              str(tmp[j, i]) + "\n")
            fno.close()

    def get_filename(self, data, ext):
        """
        Gets a valid filename

        Parameters
        ----------
        data : PyGMI raster Data
            dataset to get filename from
        ext : str
            filename extension to use
        """
        file_band = data.dataid.split('_')[0].strip('"')
        file_band = file_band.replace('/', '')
        file_band = file_band.replace(':', '')
        file_out = self.ifile.rpartition(".")[0]+"_"+file_band+'.'+ext

        return file_out


def get_geopak(hfile):
    """ GeoPak Import """

    fin = open(hfile, 'rb')
    fall = fin.read()
    fin.close()

    # bof = np.frombuffer(fall, dtype=np.uint8, count=1, offset=0)
    off = 0
#    fnew = b''
    fnew = []
    while off < len(fall):
        off += 1
        breclen = np.frombuffer(fall, dtype=np.uint8, count=1, offset=off)[0]

        if breclen == 130:
            break

        reclen = breclen

        if breclen == 129:
            reclen = 128

        off += 1

#        fnew += fall[off:off+reclen]
        fnew.append(fall[off:off+reclen])
        off += reclen
#        ereclen = np.frombuffer(fall, dtype=np.uint8, count=1, offset=off)[0]
#        print(breclen, ereclen, reclen)

    fnew = b''.join(fnew)
    header = np.frombuffer(fnew, dtype=np.float32, count=32, offset=0)

#     Lines in grid      1
#     Points per line    2
#     Grid factor        3
#     Grid base value    4
#     Grid X origin      5
#     Grid Y origin      6
#     Grid rotation      7
#     Grid dummy value   8
#     Map scale          9
#     Cell size (X)     10
#     Cell size (Y)     11
#     Inches/unit       12
#     Grid X offset     13
#     Grid Y offset     14
#     Grid hdr version  15
#
#     Lines in grid     17
#     Points per line   18
#     Grid factor       21
#     Grid base value   22
#     Z maximum         23
#     Z minimum         24
#
#     Grid dummy value  26

    nrows = int(header[0])
    ncols = int(header[1])
    gfactor = header[2]
    gbase = header[3]
    x0 = header[4]
    y0 = header[5]
#    rotation = header[6]
    nval = header[7]
#    mapscale = header[8]
    dx = header[9]
    dy = header[10]
#    inches_per_unit = header[11]
#    xoffset = header[12]
#    yoffset = header[13]
#    hver = header[14]
#    zmax = header[22]
#    zmin = header[23]

    data = np.frombuffer(fnew, dtype=np.int16, count=(nrows*ncols), offset=128)

    data = np.ma.masked_equal(data, nval)
    data = data/gfactor+gbase
    data.shape = (nrows, ncols)
    data = data[::-1]

    dat = []
    dat.append(Data())
    i = 0

    dat[i].data = data
    dat[i].nrofbands = 1
    dat[i].tlx = x0
    dat[i].tly = y0+dy*nrows
    dat[i].dataid = hfile[:-4]

    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = dx
    dat[i].ydim = dy

    return dat


def get_geosoft(hfile):
    """ Get geosoft file """
    f = open(hfile, mode='rb')

    es = np.fromfile(f, dtype=np.int32, count=1)[0]  # 4
    sf = np.fromfile(f, dtype=np.int32, count=1)[0]  # signf
    ncols = np.fromfile(f, dtype=np.int32, count=1)[0]  # ncol
    nrows = np.fromfile(f, dtype=np.int32, count=1)[0]  # nrow
    kx = np.fromfile(f, dtype=np.int32, count=1)[0]  # 1

    dx = np.fromfile(f, dtype=np.float64, count=1)[0]  # dx
    dy = np.fromfile(f, dtype=np.float64, count=1)[0]  # dy
    x0 = np.fromfile(f, dtype=np.float64, count=1)[0]  # xllcor
    y0 = np.fromfile(f, dtype=np.float64, count=1)[0]  # yllcor
    rot = np.fromfile(f, dtype=np.float64, count=1)[0]  # rot
    zbase = np.fromfile(f, dtype=np.float64, count=1)[0]  # zbase
    zmult = np.fromfile(f, dtype=np.float64, count=1)[0]  # zmult

    label = np.fromfile(f, dtype='a48', count=1)[0]
    mapno = np.fromfile(f, dtype='a16', count=1)[0]

    proj = np.fromfile(f, dtype=np.int32, count=1)[0]
    unitx = np.fromfile(f, dtype=np.int32, count=1)[0]
    unity = np.fromfile(f, dtype=np.int32, count=1)[0]
    unitz = np.fromfile(f, dtype=np.int32, count=1)[0]
    nvpts = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmin = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmax = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmed = np.fromfile(f, dtype=np.int32, count=1)[0]
    izmea = np.fromfile(f, dtype=np.int32, count=1)[0]

    zvar = np.fromfile(f, dtype=np.float64, count=1)[0]

    prcs = np.fromfile(f, dtype=np.int32, count=1)[0]

    temspc = np.fromfile(f, dtype='a324', count=1)[0]

#    pdb.set_trace()

    if es == 2:
        nval = -32767
        data = np.fromfile(f, dtype=np.int16, count=nrows*ncols)

    elif es == 4:
        data = np.fromfile(f, dtype=np.float32, count=nrows*ncols)
        nval = -1.0E+32
    else:
        return None

    data = np.ma.masked_equal(data, nval)

    data = data/zmult + zbase
    data.shape = (nrows, ncols)
    data = data[::-1]

    f.close()

    dat = []
    dat.append(Data())
    i = 0

    dat[i].data = data
    dat[i].nrofbands = 1
    dat[i].tlx = x0
    dat[i].tly = y0+dy*nrows
    dat[i].dataid = hfile[:-4]

    dat[i].nullvalue = nval
    dat[i].rows = nrows
    dat[i].cols = ncols
    dat[i].xdim = dx
    dat[i].ydim = dy

    return dat
