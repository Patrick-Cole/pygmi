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
"""Import Data."""

import sys
import os
import re
import zipfile
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import osr, ogr
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.crs import CRS

from pygmi.pfmod.datatypes import LithModel
import pygmi.pfmod.grvmag3d as grvmag3d
import pygmi.pfmod.cubes as mvis3d
import pygmi.menu_default as menu_default
import pygmi.raster.dataprep as dp
from pygmi.misc import ProgressBarText
# This is necessary for loading npz files, since I moved the location of
# datatypes.
from pygmi.pfmod import datatypes

sys.modules['datatypes'] = datatypes


class ImportMod3D():
    """Import Data."""

    def __init__(self, parent=None):
        self.parent = parent
        self.lmod = LithModel()

        self.ifile = ''
        self.filt = ''
        self.indata = {}
        self.outdata = {}

        if parent is not None:
            self.piter = parent.pbar.iter
            self.showprocesslog = parent.showprocesslog
        else:
            self.piter = ProgressBarText().iter
            self.showprocesslog = print

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
            ext = ('npz (*.npz);;'
                   'Leapfrog Block Model (*.csv);;'
                   'x, y, z, label (*.csv);;'
                   'x, y, z, label (*.txt)')

            self.ifile, self.filt = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)

            if self.ifile == '':
                return False
        os.chdir(os.path.dirname(self.ifile))
        # self.parent.modelfilename = self.ifile.rpartition('.')[0]

# Reset Variables
        self.lmod.griddata.clear()
        self.lmod.lith_list.clear()

        if self.filt == 'Leapfrog Block Model (*.csv)':
            self.import_leapfrog_csv(self.ifile)
        elif self.filt in ('x, y, z, label (*.csv)', 'x, y, z, label (*.txt)'):
            self.import_ascii_xyz_model(self.ifile)
        else:
            indict = np.load(self.ifile, allow_pickle=True)
            self.dict2lmod(indict)

        self.outdata['Model3D'] = [self.lmod]
        self.lmod.name = os.path.basename(self.ifile)

        for i in self.lmod.griddata:
            if self.lmod.griddata[i].dataid == '':
                self.lmod.griddata[i].dataid = i
            if hasattr(self.lmod.griddata[i], 'isrgb') is False:
                self.lmod.griddata[i].isrgb = False

        tmp = list(set(self.lmod.griddata.values()))
        self.outdata['Raster'] = tmp

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        self.ifile = projdata['ifile']
        self.filt = projdata['filt']

        chk = self.settings(True)

        return chk

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['ifile'] = self.ifile
        projdata['filt'] = self.filt

        return projdata

    def import_leapfrog_csv(self, filename):
        """
        Import leapfrog csv block models.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        None.

        """
        with open(filename) as fno:
            tmp = fno.readlines()

        while tmp[0][0] == '#':
            tmp.pop(0)

        if not tmp:
            return

        header = tmp.pop(0).split(',')
        header = header[7:]

        mtmp = MessageCombo(header)
        mtmp.exec_()
        datindx = mtmp.master.currentIndex()

        x = []
        y = []
        z = []
        label = []
        xcell = float(tmp[0].split(',')[3])
        ycell = float(tmp[0].split(',')[4])
        zcell = float(tmp[0].split(',')[5])

        for i in self.piter(tmp):
            i2 = i.split(',')
            x.append(float(i2[0]))
            y.append(float(i2[1]))
            z.append(float(i2[2]))
            label.append(i2[7+datindx])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        x_u = np.unique(x)
        y_u = np.unique(y)
        z_u = np.unique(z)
        labelu = np.unique(label)
        labelu[labelu == 'blank'] = 'Background'

        lmod = self.lmod

        lmod.numx = x_u.shape[0]
        lmod.numy = y_u.shape[0]
        lmod.numz = z_u.shape[0]
        lmod.dxy = max(xcell, ycell)
        lmod.d_z = zcell
        lmod.xrange = [x_u.min()-lmod.dxy/2., x_u.max()+lmod.dxy/2.]
        lmod.yrange = [y_u.min()-lmod.dxy/2., y_u.max()+lmod.dxy/2.]
        lmod.zrange = [z_u.min()-lmod.d_z/2., z_u.max()+lmod.d_z/2.]

        lindx = 0
        for itxt in labelu:
            lindx += 1
            if itxt == 'Background':
                lmod.lith_list[itxt] = grvmag3d.GeoData(
                    self.parent, ncols=lmod.numx, nrows=lmod.numy,
                    numz=lmod.numz, dxy=lmod.dxy, d_z=lmod.d_z)
                lmod.lith_list[itxt].lith_index = 0
                lmod.mlut[0] = [np.random.randint(0, 255),
                                np.random.randint(0, 255),
                                np.random.randint(0, 255)]
            else:
                lmod.lith_list[itxt] = grvmag3d.GeoData(
                    self.parent, ncols=lmod.numx, nrows=lmod.numy,
                    numz=lmod.numz, dxy=lmod.dxy, d_z=lmod.d_z)
                lmod.lith_list[itxt].lith_index = lindx
                lmod.mlut[lindx] = [np.random.randint(0, 255),
                                    np.random.randint(0, 255),
                                    np.random.randint(0, 255)]

            lmod.lith_list[itxt].modified = True
            lmod.lith_list[itxt].set_xyz12()

        lmod.lith_index = None
        lmod.update(lmod.numx, lmod.numy, lmod.numz, lmod.xrange[0],
                    lmod.yrange[1], lmod.zrange[1], lmod.dxy, lmod.d_z,
                    usedtm=True)
        lmod.update_lith_list_reverse()

        for i in self.piter(range(len(x))):
            xi = x[i]
            col = int((xi-lmod.xrange[0])/lmod.dxy)
            row = int((lmod.yrange[1]-y[i])/lmod.dxy)
            layer = int((lmod.zrange[1]-z[i])/lmod.d_z)
            if label[i] == 'blank':
                lmod.lith_index[col, row, layer] = \
                    lmod.lith_list['Background'].lith_index
            else:
                lmod.lith_index[col, row, layer] = \
                    lmod.lith_list[label[i]].lith_index

    def import_ascii_xyz_model(self, filename):
        """
        Use to import ASCII XYZ Models of the form x,y,z,label.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        None.

        """
        names = ['x', 'y', 'z', 'label']
        try:
            if filename.find('.csv') > -1:
                df1 = pd.read_csv(filename, delimiter=',', names=names)
            else:
                df1 = pd.read_csv(filename, delimeter=' ', names=names)
        except:
            self.showprocesslog('Unable to import file')
            return

        x = df1.x.to_numpy(float)
        y = df1.y.to_numpy(float)
        z = df1.z.to_numpy(float)
        label = df1.label.to_numpy(str)

        x_u = df1.x.unique()
        y_u = df1.y.unique()
        z_u = df1.z.unique()
        labelu = df1.label.unique()

        x_u.sort()
        y_u.sort()
        z_u.sort()

        dx_u = np.diff(x_u)
        dy_u = np.diff(y_u)
        dz_u = np.diff(z_u)

        xcell = np.min(dx_u)
        ycell = np.min(dy_u)
        zcell = np.min(dz_u)

        lmod = self.lmod
        lmod.lith_list['Background'] = grvmag3d.GeoData(self.parent)

        lmod.dxy = min(xcell, ycell)
        lmod.d_z = zcell
        lmod.xrange = [x_u.min()-lmod.dxy/2., x_u.max()+lmod.dxy/2.]
        lmod.yrange = [y_u.min()-lmod.dxy/2., y_u.max()+lmod.dxy/2.]
        lmod.zrange = [z_u.min()-lmod.d_z/2., z_u.max()+lmod.d_z/2.]
        lmod.numx = int(np.ptp(lmod.xrange)/lmod.dxy+1)
        lmod.numy = int(np.ptp(lmod.yrange)/lmod.dxy+1)
        lmod.numz = int(np.ptp(lmod.zrange)/lmod.d_z+1)

    # Section to load lithologies.
        lindx = 0
        for itxt in labelu:
            lindx += 1
            lmod.mlut[lindx] = [np.random.randint(0, 255),
                                np.random.randint(0, 255),
                                np.random.randint(0, 255)]
            lmod.lith_list[itxt] = grvmag3d.GeoData(
                self.parent, ncols=lmod.numx, nrows=lmod.numy, numz=lmod.numz,
                dxy=lmod.dxy, d_z=lmod.d_z)

            lmod.lith_list[itxt].lith_index = lindx
            lmod.lith_list[itxt].modified = True
            lmod.lith_list[itxt].set_xyz12()

        lmod.lith_index = None
        lmod.update(lmod.numx, lmod.numy, lmod.numz, lmod.xrange[0],
                    lmod.yrange[1], lmod.zrange[1], lmod.dxy, lmod.d_z)
        lmod.update_lith_list_reverse()

        for i, xi in enumerate(x):
            col = int((xi-lmod.xrange[0])/lmod.dxy)
            row = int((y[i]-lmod.yrange[0])/lmod.dxy)
            # row = int((lmod.yrange[1]-y[i])/lmod.dxy)
            layer = int((lmod.zrange[1]-z[i])/lmod.d_z)
            lmod.lith_index[col, row, layer] = \
                lmod.lith_list[label[i]].lith_index

    def dict2lmod(self, indict, pre=''):
        """
        Convert a dictionary to a LithModel.

        Parameters
        ----------
        indict : dictionary
            Imported dictionary.
        pre : str, optional
            Text. The default is ''.

        Returns
        -------
        None.

        """
        lithkeys = indict[pre+'lithkeys']

        lmod = self.lmod

        lmod.gregional = indict[pre+'gregional']
        lmod.ght = indict[pre+'ght']
        lmod.mht = indict[pre+'mht']
        lmod.numx = indict[pre+'numx']
        lmod.numy = indict[pre+'numy']
        lmod.numz = indict[pre+'numz']
        lmod.dxy = indict[pre+'dxy']
        lmod.d_z = indict[pre+'d_z']
        lmod.lith_index = indict[pre+'lith_index']
        lmod.xrange = np.array(indict[pre+'xrange']).tolist()
        lmod.yrange = np.array(indict[pre+'yrange']).tolist()
        lmod.zrange = np.array(indict[pre+'zrange']).tolist()
        if pre+'custprofx' in indict:
            lmod.custprofx = indict[pre+'custprofx'].item()
        else:
            lmod.custprofx = {0: (lmod.xrange[0], lmod.xrange[1])}
        if pre+'custprofy' in indict:
            lmod.custprofy = indict[pre+'custprofy'].item()
        else:
            lmod.custprofy = {0: (lmod.yrange[0], lmod.yrange[0])}

        for i in lmod.custprofx:
            if len(lmod.custprofx[i]) == 2:
                lmod.custprofx[i] += lmod.custprofx[i]
        for i in lmod.custprofy:
            if len(lmod.custprofy[i]) == 2:
                lmod.custprofy[i] += lmod.custprofy[i]

        lmod.mlut = indict[pre+'mlut'].item()
        lmod.init_calc_grids()

        lmod.griddata = indict[pre+'griddata'].item()

        for i in lmod.griddata:
            lmod.griddata[i].data = np.ma.array(lmod.griddata[i].data)

        if pre+'profpics' in indict:
            lmod.profpics = indict[pre+'profpics'].item()

            for i in lmod.profpics:
                lmod.profpics[i].data = np.ma.array(lmod.profpics[i].data)

        # This gets rid of a legacy variable names and updates to new ones
        for i in lmod.griddata:
            if not hasattr(lmod.griddata[i], 'units'):
                lmod.griddata[i].units = ''
            if not hasattr(lmod.griddata[i], 'isrgb'):
                lmod.griddata[i].isrgb = False
            if not hasattr(lmod.griddata[i], 'metadata'):
                lmod.griddata[i].metadata = {'Cluster': {}, 'Raster': {}}
            if not hasattr(lmod.griddata[i], 'filename'):
                lmod.griddata[i].filename = ''

            if not hasattr(lmod.griddata[i], 'nodata'):
                lmod.griddata[i].nodata = lmod.griddata[i].nullvalue
            if not hasattr(lmod.griddata[i], 'crs'):
                wkt = lmod.griddata[i].wkt
                if wkt == '' or wkt is None:
                    lmod.griddata[i].crs = None
                else:
                    lmod.griddata[i].crs = CRS.from_wkt(wkt)
            if not hasattr(lmod.griddata[i], 'dataid'):
                lmod.griddata[i].dataid = ''
            if hasattr(lmod.griddata[i], 'bandid'):
                if lmod.griddata[i].dataid == '':
                    lmod.griddata[i].dataid = lmod.griddata[i].bandid
                del lmod.griddata[i].bandid
            if not hasattr(lmod.griddata[i], 'extent'):
                if lmod.griddata[i].data.ndim == 2:
                    rows, cols = lmod.griddata[i].data.shape
                elif lmod.griddata[i].data.ndim == 3:
                    rows, cols, _ = lmod.griddata[i].data.shape

                xmin = lmod.griddata[i].tlx
                ymax = lmod.griddata[i].tly

                ydim = lmod.griddata[i].ydim
                xdim = lmod.griddata[i].xdim

                lmod.griddata[i].set_transform(xdim, xmin, ydim, ymax)
                del lmod.griddata[i].tlx
                del lmod.griddata[i].tly
            elif not hasattr(lmod.griddata[i], 'transform'):
                ydim = lmod.griddata[i].ydim
                xdim = lmod.griddata[i].xdim
                xmin, _, _, ymax = lmod.griddata[i].extent
                lmod.griddata[i].set_transform(xdim, xmin, ydim, ymax)

        crsfin = None
        for i in lmod.griddata:
            if lmod.griddata[i].crs is not None:
                crsfin = lmod.griddata[i].crs

        if crsfin is not None:
            for i in lmod.griddata:
                if lmod.griddata[i].crs is None:
                    lmod.griddata[i].crs = crsfin

# Section to load lithologies.
        lmod.lith_list['Background'] = grvmag3d.GeoData(self.parent)

        for itxt in lithkeys:
            if itxt != 'Background':
                lmod.lith_list[itxt] = grvmag3d.GeoData(self.parent)

            lmod.lith_list[itxt].hintn = indict[pre+itxt+'_hintn'].item()
            lmod.lith_list[itxt].finc = indict[pre+itxt+'_finc'].item()
            lmod.lith_list[itxt].fdec = indict[pre+itxt+'_fdec'].item()
            lmod.lith_list[itxt].zobsm = indict[pre+itxt+'_zobsm'].item()
            lmod.lith_list[itxt].susc = indict[pre+itxt+'_susc'].item()
            lmod.lith_list[itxt].mstrength = indict[pre+itxt+'_mstrength'].item()
            lmod.lith_list[itxt].qratio = indict[pre+itxt+'_qratio'].item()
            lmod.lith_list[itxt].minc = indict[pre+itxt+'_minc'].item()
            lmod.lith_list[itxt].mdec = indict[pre+itxt+'_mdec'].item()
            lmod.lith_list[itxt].density = indict[pre+itxt+'_density'].item()
            lmod.lith_list[itxt].bdensity = indict[pre+itxt+'_bdensity'].item()
            lmod.lith_list[itxt].lith_index = indict[pre+itxt+'_lith_index'].item()
            lmod.lith_list[itxt].g_cols = indict[pre+itxt+'_numx'].item()
            lmod.lith_list[itxt].g_rows = indict[pre+itxt+'_numy'].item()
            lmod.lith_list[itxt].numz = indict[pre+itxt+'_numz'].item()
            lmod.lith_list[itxt].g_dxy = indict[pre+itxt+'_dxy'].item()
            lmod.lith_list[itxt].dxy = indict[pre+itxt+'_dxy'].item()
            lmod.lith_list[itxt].d_z = indict[pre+itxt+'_d_z'].item()
            lmod.lith_list[itxt].zobsm = indict[pre+itxt+'_zobsm'].item()
            lmod.lith_list[itxt].zobsg = indict[pre+itxt+'_zobsg'].item()
            if pre+itxt+'_lithcode' in indict:
                lmod.lith_list[itxt].lithcode = indict[pre+itxt+'_lithcode'].item()
            if pre+itxt+'_lithnotes' in indict:
                lmod.lith_list[itxt].lithnotes = indict[pre+itxt+'_lithnotes'].item()

            lmod.lith_list[itxt].modified = True
            lmod.lith_list[itxt].set_xyz12()


class ExportMod3D():
    """Export Data."""

    def __init__(self, parent=None):
        self.ifile = ''
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        if 'Model3D' not in self.indata:
            self.showprocesslog('Error: You need to have a model first!')
            return

        for self.lmod in self.indata['Model3D']:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.parent, 'Save File', '.',
                'npz (*.npz);;shapefile (*.shp);;kmz (*.kmz);;csv (*.csv)')

            if filename == '':
                return

            os.chdir(os.path.dirname(filename))
            self.ifile = str(filename)
            ext = filename[-3:]

            self.showprocesslog('Saving '+self.ifile+'...')

        # Pop up save dialog box
            if ext == 'npz':
                self.savemodel()
            if ext == 'kmz':
                self.mod3dtokmz()
            if ext == 'shp':
                self.mod3dtoshp()
            if ext == 'csv':
                self.mod3dtocsv()

    def savemodel(self):
        """
        Save model.

        Returns
        -------
        None.

        """
# Open file
        filename = self.ifile

# Construct output dictionary
        outdict = {}
        outdict = self.lmod2dict(outdict)

# Save data
        try:
            np.savez_compressed(filename, **outdict)
            self.showprocesslog('Model save complete!')
        except:
            self.showprocesslog('ERROR! Model save failed!')

    def lmod2dict(self, outdict, pre=''):
        """
        Convert LithModel to a dictionary.

        Parameters
        ----------
        outdict : dictionary
            Output dictionary.
        pre : str, optional
            Text. The default is ''.

        Returns
        -------
        outdict : dictionary
            Output dictionary.

        """
        outdict[pre+'gregional'] = self.lmod.gregional
        outdict[pre+'ght'] = self.lmod.ght
        outdict[pre+'mht'] = self.lmod.mht
        outdict[pre+'numx'] = self.lmod.numx
        outdict[pre+'numy'] = self.lmod.numy
        outdict[pre+'numz'] = self.lmod.numz
        outdict[pre+'dxy'] = self.lmod.dxy
        outdict[pre+'d_z'] = self.lmod.d_z
        outdict[pre+'lith_index'] = self.lmod.lith_index
        outdict[pre+'xrange'] = self.lmod.xrange
        outdict[pre+'yrange'] = self.lmod.yrange
        outdict[pre+'zrange'] = self.lmod.zrange
        outdict[pre+'mlut'] = self.lmod.mlut
        outdict[pre+'griddata'] = self.lmod.griddata
        outdict[pre+'profpics'] = self.lmod.profpics
        outdict[pre+'custprofx'] = self.lmod.custprofx
        outdict[pre+'custprofy'] = self.lmod.custprofy

# Section to save lithologies.
        outdict[pre+'lithkeys'] = list(self.lmod.lith_list.keys())

        for i in self.lmod.lith_list.items():
            curkey = i[0]
            outdict[pre+curkey+'_hintn'] = i[1].hintn
            outdict[pre+curkey+'_finc'] = i[1].finc
            outdict[pre+curkey+'_fdec'] = i[1].fdec
            outdict[pre+curkey+'_zobsm'] = i[1].zobsm
            outdict[pre+curkey+'_susc'] = i[1].susc
            outdict[pre+curkey+'_mstrength'] = i[1].mstrength
            outdict[pre+curkey+'_qratio'] = i[1].qratio
            outdict[pre+curkey+'_minc'] = i[1].minc
            outdict[pre+curkey+'_mdec'] = i[1].mdec
            outdict[pre+curkey+'_density'] = i[1].density
            outdict[pre+curkey+'_bdensity'] = i[1].bdensity
            outdict[pre+curkey+'_lith_index'] = i[1].lith_index
            outdict[pre+curkey+'_numx'] = i[1].g_cols
            outdict[pre+curkey+'_numy'] = i[1].g_rows
            outdict[pre+curkey+'_numz'] = i[1].numz
            outdict[pre+curkey+'_dxy'] = i[1].g_dxy
            outdict[pre+curkey+'_d_z'] = i[1].d_z
            outdict[pre+curkey+'_zobsm'] = i[1].zobsm
            outdict[pre+curkey+'_zobsg'] = i[1].zobsg
            outdict[pre+curkey+'_x12'] = i[1].x12
            outdict[pre+curkey+'_y12'] = i[1].y12
            outdict[pre+curkey+'_z12'] = i[1].z12
            outdict[pre+curkey+'_lithcode'] = i[1].lithcode
            outdict[pre+curkey+'_lithnotes'] = i[1].lithnotes

        return outdict

    def mod3dtocsv(self):
        """
        Save the 3D model in a csv file.

        Returns
        -------
        None.

        """
        self.showprocesslog('csv export starting...')

        self.lmod.update_lith_list_reverse()
        lithname = self.lmod.lith_list_reverse.copy()
        lithlist = self.lmod.lith_list.copy()

        tmp = []
        ltmp = []
        for i in range(self.lmod.numx):
            x = self.lmod.xrange[0]+i*self.lmod.dxy
            for j in range(self.lmod.numy):
                y = self.lmod.yrange[0]+j*self.lmod.dxy
                for k in range(self.lmod.numz):
                    z = self.lmod.zrange[1]-k*self.lmod.d_z
                    lith = self.lmod.lith_index[i, j, k]
                    if lith > -1:
                        name = lithname[lith]
                        dens = lithlist[name].density
                        susc = lithlist[name].susc
                        tmp.append([x, y, z, dens, susc, lith])
                        ltmp.append(lithname[lith])

        tmp = np.array(tmp)
        ltmp = np.array(ltmp)
        stmp = np.zeros(len(tmp), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('dens', 'f4'), ('susc', 'f4'),
                                         ('lith', 'i4'), ('lithname', 'a24')])

        stmp['x'] = tmp[:, 0]
        stmp['y'] = tmp[:, 1]
        stmp['z'] = tmp[:, 2]
        stmp['dens'] = tmp[:, 3]
        stmp['susc'] = tmp[:, 4]
        stmp['lith'] = tmp[:, 5]
        stmp['lithname'] = ltmp

        head = 'X, Y, Z, Density, Susceptibility, Lithology Code, Lithology'
        np.savetxt(self.ifile, stmp, fmt="%f, %f, %f, %f, %f, %i, %s",
                   header=head)

        self.showprocesslog('csv export complete!')

    def mod3dtokmz(self):
        """
        Save the 3D model and grids in a kmz file.

        Only the boundary of the area is in degrees. The actual coordinates
        are still in meters.

        Returns
        -------
        None.

        """
        mvis_3d = mvis3d.Mod3dDisplay()
        mvis_3d.lmod1 = self.lmod

        rev = 1  # should be 1 normally

        xrng = np.array(self.lmod.xrange, dtype=float)
        yrng = np.array(self.lmod.yrange, dtype=float)
        zrng = np.array(self.lmod.zrange, dtype=float)

        if 'Raster' in self.indata:
            wkt = self.indata['Raster'][0].wkt
        else:
            wkt = ''
        prjkmz = Exportkmz(wkt)
        tmp = prjkmz.exec_()

        if tmp == 0:
            return

        if prjkmz.proj.wkt == '':
            QtWidgets.QMessageBox.warning(self.parent, 'Warning',
                                          ' You need a projection!',
                                          QtWidgets.QMessageBox.Ok)
            return

        smooth = prjkmz.checkbox_smooth.isChecked()

        orig_wkt = prjkmz.proj.wkt
        orig = osr.SpatialReference()
        orig.ImportFromWkt(orig_wkt)
        orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')
        targ.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        prj = osr.CoordinateTransformation(orig, targ)

        res = prj.TransformPoint(xrng[0], yrng[0])
        lonwest, latsouth = res[0], res[1]
        res = prj.TransformPoint(xrng[1], yrng[1])
        loneast, latnorth = res[0], res[1]

# Get Save Name
        filename = self.ifile

        self.showprocesslog('kmz export starting...')

# Move to 3d model tab to update the model stuff
        self.showprocesslog('updating 3d model...')

        mvis_3d.spacing = [self.lmod.dxy, self.lmod.dxy, self.lmod.d_z]
        mvis_3d.origin = [xrng[0], yrng[0], zrng[0]]
        mvis_3d.gdata = self.lmod.lith_index[::1, ::1, ::-1]
        itmp = np.sort(np.unique(self.lmod.lith_index))
        itmp = itmp[itmp > 0]
        tmp = np.ones((255, 4))*255
        for i in itmp:
            tmp[i, :3] = self.lmod.mlut[i]
        mvis_3d.lut = tmp
        mvis_3d.update_model(smooth)

        self.showprocesslog('creating kmz file')
        heading = str(0.)
        tilt = str(45.)  # angle from vertical
        lat = str(np.mean([latsouth, latnorth]))  # coord of object
        lon = str(np.mean([lonwest, loneast]))  # coord of object
        rng = str(max(xrng.ptp(), yrng.ptp(), zrng.ptp()))  # range to object
        alt = str(0)  # alt of object eye is looking at (meters)
        lato = str(latsouth)
        lono = str(lonwest)

# update colors
        self.lmod.update_lith_list_reverse()

        dockml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\r\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:gx="http://www.google.com/kml/ext/2.2">\r\n'
            '\r\n'
            '  <Folder>\r\n'
            '    <name>Lithological Model</name>\r\n'
            '    <description>Created with PyGMI</description>\r\n'
            '    <visibility>1</visibility>\r\n'
            '    <LookAt>\r\n'
            '      <heading>' + heading + '</heading>\r\n'
            '      <tilt>' + tilt + '</tilt>\r\n'
            '      <latitude>' + lat + '</latitude>\r\n'
            '      <longitude>' + lon + '</longitude>\r\n'
            '      <range>' + rng + '</range>\r\n'
            '      <altitude>' + alt + '</altitude>\r\n'
            '    </LookAt>\r\n')

        mvis_3d.update_for_kmz()

        modeldae = []
        lkey = list(mvis_3d.faces.keys())
        lkey.pop(lkey.index(0))
        lithcnt = -1

        alt = str(0)
        for lith in lkey:
            faces = np.array(mvis_3d.gfaces[lith])
            # Google wants the model to have origin (0,0)

            points = mvis_3d.gpoints[lith]

            if points == []:
                continue

            points -= mvis_3d.origin

            x = points[:, 0]
            y = points[:, 1]
            earthrad = 6378137.
            z = earthrad-np.sqrt(earthrad**2-(x**2+y**2))
            points[:, 2] -= z

            if rev == -1:
                points += [xrng.ptp(), yrng.ptp(), 0]

            norm = np.abs(mvis_3d.gnorms[lith])
            clrtmp = np.array(self.lmod.mlut[lith])/255.
            curmod = self.lmod.lith_list_reverse[lith]

            if len(points) > 60000:
                self.showprocesslog(curmod + ' has too many points (' +
                                    str(len(points))+'). Not exported')
                points = points[:60000]
                norm = norm[:60000]
                faces = faces[faces.max(1) < 60000]

            lithcnt += 1

            dockml += (
                '    <Placemark>\r\n'
                '      <name>' + curmod + '</name>\r\n'
                '      <description></description>\r\n'
                '      <Style id="default"/>\r\n'
                '      <Model>\r\n'
                '        <altitudeMode>absolute</altitudeMode>\r\n'
                '        <Location>\r\n'
                '          <latitude>' + lato + '</latitude>\r\n'
                '          <longitude>' + lono + '</longitude>\r\n'
                '          <altitude>' + str(alt) + '</altitude>\r\n'
                '        </Location>\r\n'
                '        <Orientation>\r\n'
                '          <heading>0</heading>\r\n'
                '          <tilt>0</tilt>\r\n'
                '          <roll>0</roll>\r\n'
                '        </Orientation>\r\n'
                '        <Scale>\r\n'
                '          <x>1</x>\r\n'
                '          <y>1</y>\r\n'
                '          <z>1</z>\r\n'
                '        </Scale>\r\n'
                '        <Link>\r\n'
                '          <href>models/mod3d' + str(lithcnt) +
                '.dae</href>\r\n'
                '        </Link>\r\n'
                '      </Model>\r\n'
                '    </Placemark>\r\n')

            position = str(points.flatten().tolist())
            position = position.replace('[', '')
            position = position.replace(']', '')
            position = position.replace(',', '')
            vertex = str(faces.flatten().tolist())
            vertex = vertex.replace('[', '')
            vertex = vertex.replace(']', '')
            vertex = vertex.replace(',', '')
            normal = str(norm.flatten().tolist())
            normal = normal.replace('[', '')
            normal = normal.replace(']', '')
            normal = normal.replace(',', '')
            color = str(clrtmp.flatten().tolist())
            color = color.replace('[', '')
            color = color.replace(']', '')
            color = color.replace(',', '')

            modeldae.append(
                '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\r\n'
                '<COLLADA xmlns="http://www.collada.org/2005'
                '/11/COLLADASchema" '
                'version="1.4.1">\r\n'
                '  <asset>\r\n'
                '    <contributor>\r\n'
                '      <authoring_tool>PyGMI</authoring_tool>\r\n'
                '    </contributor>\r\n'
                '    <created>2012-03-01T10:36:38Z</created>\r\n'
                '    <modified>2012-03-01T10:36:38Z</modified>\r\n'
                '    <up_axis>Z_UP</up_axis>\r\n'
                '  </asset>\r\n'
                '  <library_visual_scenes>\r\n'
                '    <visual_scene id="ID1">\r\n'
                '      <node name="SketchUp">\r\n'
                '        <node id="ID2" name="instance_0">\r\n'
                '          <matrix>    1 0 0 0 \r\n'
                '                      0 1 0 0 \r\n'
                '                      0 0 1 0 \r\n'
                '                      0 0 0 1 \r\n'
                '          </matrix>\r\n'
                '          <instance_node url="#ID3" />\r\n'
                '        </node>\r\n'
                '      </node>\r\n'
                '    </visual_scene>\r\n'
                '  </library_visual_scenes>\r\n'
                '  <library_nodes>\r\n'
                '    <node id="ID3" name="skp489E">\r\n'
                '      <instance_geometry url="#ID4">\r\n'
                '        <bind_material>\r\n'
                '          <technique_common>\r\n'
                '            <instance_material symbol="Material2"'
                ' target="#ID5">\r\n'
                '              <bind_vertex_input semantic="UVSET0" '
                'input_semantic="TEXCOORD" input_set="0" />\r\n'
                '            </instance_material>\r\n'
                '          </technique_common>\r\n'
                '        </bind_material>\r\n'
                '      </instance_geometry>\r\n'
                '    </node>\r\n'
                '  </library_nodes>\r\n'
                '  <library_geometries>\r\n'
                '    <geometry id="ID4">\r\n'
                '      <mesh>\r\n'
                '        <source id="ID7">\r\n'
                '          <float_array id="ID10" count="' +
                str(points.size) + '">' + position +
                '          </float_array>\r\n'
                '          <technique_common>\r\n'
                '            <accessor count="' + str(points.shape[0]) +
                '" source="#ID10" stride="3">\r\n'
                '              <param name="X" type="float" />\r\n'
                '              <param name="Y" type="float" />\r\n'
                '              <param name="Z" type="float" />\r\n'
                '            </accessor>\r\n'
                '          </technique_common>\r\n'
                '        </source>\r\n'
                '        <source id="ID8">\r\n'
                '          <float_array id="ID11" count="' + str(norm.size) +
                '">' + normal +
                '          </float_array>\r\n'
                '          <technique_common>\r\n'
                '            <accessor count="' + str(norm.shape[0]) +
                '" source="#ID11" stride="3">\r\n'
                '              <param name="X" type="float" />\r\n'
                '              <param name="Y" type="float" />\r\n'
                '              <param name="Z" type="float" />\r\n'
                '            </accessor>\r\n'
                '          </technique_common>\r\n'
                '        </source>\r\n'
                '        <vertices id="ID9">\r\n'
                '          <input semantic="POSITION" source="#ID7" />\r\n'
                '          <input semantic="NORMAL" source="#ID8" />\r\n'
                '        </vertices>\r\n'
                '        <triangles count="' + str(faces.shape[0]) +
                '" material="Material2">\r\n'
                '          <input offset="0" semantic="VERTEX" '
                'source="#ID9" />\r\n'
                '          <p>' + vertex + '</p>\r\n'
                '        </triangles>\r\n'
                '      </mesh>\r\n'
                '    </geometry>\r\n'
                '  </library_geometries>\r\n'
                '  <library_materials>\r\n'
                '    <material id="ID5" name="__auto_">\r\n'
                '      <instance_effect url="#ID6" />\r\n'
                '    </material>\r\n'
                '  </library_materials>\r\n'
                '  <library_effects>\r\n'
                '    <effect id="ID6">\r\n'
                '      <profile_COMMON>\r\n'
                '        <technique sid="COMMON">\r\n'
                '          <lambert>\r\n'
                '            <diffuse>\r\n'
                '              <color>' + color + '</color>\r\n'
                '            </diffuse>\r\n'
                '          </lambert>\r\n'
                '        </technique>\r\n'
                '        <extra> />\r\n'
                '          <technique profile="GOOGLEEARTH"> />\r\n'
                '            <double_sided>1</double_sided> />\r\n'
                '          </technique> />\r\n'
                '        </extra> />\r\n'
                '      </profile_COMMON>\r\n'
                '    </effect>\r\n'
                '  </library_effects>\r\n'
                '  <scene>\r\n'
                '    <instance_visual_scene url="#ID1" />\r\n'
                '  </scene>\r\n'
                '</COLLADA>')

        zfile = zipfile.ZipFile(filename, 'w')
        for i, _ in enumerate(modeldae):
            zfile.writestr('models\\mod3d'+str(i)+'.dae', modeldae[i])

        for i in self.lmod.griddata:
            x_1, x_2, y_1, y_2 = self.lmod.griddata[i].extent

            res = prj.TransformPoint(x_1, y_1)
            lonwest, latsouth = res[0], res[1]
            res = prj.TransformPoint(x_2, y_2)
            loneast, latnorth = res[0], res[1]

            dockml += (
                '    <GroundOverlay>\r\n'
                '        <name>' + i + '</name>\r\n'
                '        <description></description>\r\n'
                '        <Icon>\r\n'
                '            <href>models/' + i + '.png</href>\r\n'
                '        </Icon>\r\n'
                '        <LatLonBox>\r\n'
                '            <north>' + str(latnorth) + '</north>\r\n'
                '            <south>' + str(latsouth) + '</south>\r\n'
                '            <east>' + str(loneast) + '</east>\r\n'
                '            <west>' + str(lonwest) + '</west>\r\n'
                '            <rotation>0.0</rotation>\r\n'
                '        </LatLonBox>\r\n'
                '    </GroundOverlay>\r\n')

            fig = plt.figure('tmp930', frameon=False)
            ax1 = plt.Axes(fig, [0., 0., 1., 1.])
            ax1.set_axis_off()
            fig.add_axes(ax1)

            plt.imshow(self.lmod.griddata[i].data,
                       extent=(lonwest, loneast, latsouth, latnorth),
                       aspect='auto',
                       interpolation='nearest')
            plt.savefig('tmp930.png')

            zfile.write('tmp930.png', 'models\\'+i+'.png')
            os.remove('tmp930.png')

        dockml += (
            '  </Folder>\r\n'
            '  \r\n'
            '  </kml>')

        zfile.writestr('doc.kml', dockml)

        zfile.close()
        self.showprocesslog('kmz export complete!')

    def mod3dtoshp(self):
        """
        Save the 3D model and grids in a shapefile file.

        Only the boundary of the area is in degrees. The actual coordinates
        are still in meters.

        Returns
        -------
        None.

        """
        mvis_3d = mvis3d.Mod3dDisplay()
        mvis_3d.lmod1 = self.lmod

        xrng = np.array(self.lmod.xrange, dtype=float)
        yrng = np.array(self.lmod.yrange, dtype=float)
        zrng = np.array(self.lmod.zrange, dtype=float)

        if 'Raster' in self.indata:
            wkt = self.indata['Raster'][0].wkt
        else:
            wkt = ''
        prjkmz = Exportkmz(wkt)
        tmp = prjkmz.exec_()

        if tmp == 0:
            return

        smooth = prjkmz.checkbox_smooth.isChecked()

        self.showprocesslog('shapefile export starting...')

# Move to 3d model tab to update the model stuff
        if smooth is True:
            self.showprocesslog('updating and smoothing 3d model...')
        else:
            self.showprocesslog('updating 3d model...')

        mvis_3d.spacing = [self.lmod.dxy, self.lmod.dxy, self.lmod.d_z]
        mvis_3d.origin = [xrng[0], yrng[0], zrng[0]]
        mvis_3d.gdata = self.lmod.lith_index[::1, ::1, ::-1]
        itmp = np.sort(np.unique(self.lmod.lith_index))
        itmp = itmp[itmp > 0]
        tmp = np.ones((255, 4))*255
        for i in itmp:
            tmp[i, :3] = self.lmod.mlut[i]
        mvis_3d.lut = tmp
        mvis_3d.update_model(smooth)

        self.showprocesslog('creating shapefile file')

        driver = ogr.GetDriverByName('ESRI Shapefile')

# update colors
        self.lmod.update_lith_list_reverse()

        mvis_3d.update_for_kmz()

        lkey = list(mvis_3d.faces.keys())
        lkey.pop(lkey.index(0))

        for lith in lkey:
            lithtext = mvis_3d.lmod1.lith_list_reverse[lith]
            lithsusc = self.lmod.lith_list[lithtext].susc
            lithdens = self.lmod.lith_list[lithtext].density
            self.showprocesslog(' '+lithtext)

            faces = np.array(mvis_3d.gfaces[lith])

            if faces.size == 0:
                continue

            ifile = self.ifile[:-4]+'_'+re.sub(r'[^A-Za-z]+', '_',
                                               lithtext)+'.shp'
            datasource = driver.CreateDataSource(ifile)
            layer = datasource.CreateLayer('Model',
                                           geom_type=ogr.wkbMultiPolygon25D)

            layer.CreateField(ogr.FieldDefn('Lithology', ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('Susc', ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn('Density', ogr.OFTReal))

            points = mvis_3d.gpoints[lith]

            for f in faces:
                multipolygon = ogr.Geometry(ogr.wkbMultiPolygon25D)
                tmp = points[f]

                ring1 = ogr.Geometry(ogr.wkbLinearRing)
                ring1.AddPoint(tmp[0, 0], tmp[0, 1], tmp[0, 2])
                ring1.AddPoint(tmp[1, 0], tmp[1, 1], tmp[1, 2])
                ring1.AddPoint(tmp[2, 0], tmp[2, 1], tmp[2, 2])
                ring1.AddPoint(tmp[0, 0], tmp[0, 1], tmp[0, 2])

            # Create polygon #1
                poly1 = ogr.Geometry(ogr.wkbPolygon25D)
                poly1.AddGeometry(ring1)
                multipolygon.AddGeometry(poly1)

                ring1 = None
                poly1 = None

                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(multipolygon)
                feature.SetField('Lithology', lithtext)
                feature.SetField('Susc', lithsusc)
                feature.SetField('Density', lithdens)
                layer.CreateFeature(feature)

                multipolygon = None

            # flush memory
            layer = None
            feature = None
            datasource = None

        self.showprocesslog('shapefile export complete!')


class Exportkmz(QtWidgets.QDialog):
    """Export kmz dialog."""

    def __init__(self, wkt, parent=None):
        super().__init__(parent)

        self.checkbox_smooth = QtWidgets.QCheckBox()
        self.proj = dp.GroupProj('Confirm Model Projection')
        self.proj.set_current(wkt)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.pfmod.iodefs.exportkmz')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gridlayout.addWidget(self.proj, 0, 0, 1, 2)
        gridlayout.addWidget(self.checkbox_smooth, 1, 0, 1, 2)
        gridlayout.addWidget(helpdocs, 2, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 2, 1, 1, 1)

        self.setWindowTitle('Google Earth kmz Export')
        self.checkbox_smooth.setText('Smooth Model')

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)


class MessageCombo(QtWidgets.QDialog):
    """
    Message combo box.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, combotext, parent=None):
        super().__init__(parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.master = QtWidgets.QComboBox()
        self.master.addItems(combotext)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        label_master = QtWidgets.QLabel()

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Ok)

        self.setWindowTitle('Model Choice')
        label_master.setText('Choose Model:')

        gridlayout_main.addWidget(label_master, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.master, 0, 1, 1, 1)
        gridlayout_main.addWidget(buttonbox, 3, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)

    def acceptall(self):
        """
        Accept option.

        Returns
        -------
        str
            Returns current text.

        """
        return self.master.currentText()


def _testfn():
    """Test."""
    import matplotlib.pyplot as plt
    from pygmi.misc import getinfo
    from pygmi.raster.dataprep import lstack
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    print('Starting')
    pbar = ProgressBarText()

    ifile = r"E:\Workdata\modelling\small_upper.npz"
    ofile = r"C:\Workdata\hope.tif"


    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    DM = ImportMod3D()
    DM.ifile = ifile
    DM.settings(nodialog=True)

    odata = DM.outdata['Raster']

    for dat in odata:
        plt.figure(dpi=150)
        plt.title(dat.dataid)
        plt.imshow(dat.data, extent=dat.extent)
        plt.colorbar()
        plt.show()

    odata2 = lstack(odata)

    for dat in odata:
        plt.figure(dpi=150)
        plt.title(dat.dataid)
        plt.imshow(dat.data, extent=dat.extent)
        plt.colorbar()
        plt.show()


    breakpoint()


if __name__ == "__main__":
    _testfn()
