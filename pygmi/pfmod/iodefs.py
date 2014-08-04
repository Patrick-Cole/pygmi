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
# pylint: disable=E1101

""" Import Data """
from PyQt4 import QtGui, QtCore
from osgeo import osr, gdal
import os
import numpy as np
import sys
from .datatypes import Data, LithModel
import matplotlib.pyplot as plt
import zipfile
import pygmi.pfmod.grvmag3d as grvmag3d
import pygmi.pfmod.mvis3d as mvis3d

# This is necessary for loading npz files, since I moved the location of
# datatypes.
import pygmi.pfmod.datatypes as datatypes
sys.modules['datatypes'] = datatypes


class ImportMod3D(object):
    """ Import Data """
    def __init__(self, parent):
        self.parent = parent
        self.lmod = LithModel()

        self.ifile = ""
        self.name = "Import 3D Model: "
        self.ext = ""
        self.pbar = None
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """
        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.',
            'npz (*.npz);;csv (*.csv);; txt (*.txt)')
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)
        self.parent.modelfilename = filename.rpartition('.')[0]

# Reset Variables
        self.lmod.griddata.clear()
        self.lmod.lith_list.clear()

        if filename.find('.csv') > -1:
            self.import_ascii_xyz_model(filename)
        elif filename.find('.txt') > -1:
            self.import_ascii_xyz_model(filename)
        else:
            indict = np.load(filename)
            self.dict2lmod(indict)

        self.outdata['Model3D'] = self.lmod

        for i in self.lmod.griddata.keys():
            if self.lmod.griddata[i].bandid == '':
                self.lmod.griddata[i].bandid = i

        self.outdata['Raster'] = list(self.lmod.griddata.values())

        return True

    def import_ascii_xyz_model(self, filename):
        """ Used to import ASCII XYZ Models of the form x,y,z,label"""

        if filename.find('.csv') > -1:
            tmp = np.genfromtxt(filename, delimiter=',', dtype=np.str)
        else:
            tmp = np.genfromtxt(filename, dtype=np.str)

        x = tmp[:, 0].astype(np.float)
        y = tmp[:, 1].astype(np.float)
        z = tmp[:, 2].astype(np.float)
        label = tmp[:, 3]

        x_u = np.array(np.unique(x))
        y_u = np.array(np.unique(y))
        z_u = np.array(np.unique(z))
        labelu = np.unique(label)
        xcell = x.ptp()/float(x_u.shape[0]-1)
        ycell = x.ptp()/float(y_u.shape[0]-1)
        zcell = x.ptp()/float(z_u.shape[0]-1)

        lmod = self.lmod

        lmod.numx = x_u.shape[0]
        lmod.numy = y_u.shape[0]
        lmod.numz = z_u.shape[0]
        lmod.dxy = max(xcell, ycell)
        lmod.d_z = zcell
#        lmod.lith_index = indict[pre+'lith_index']
        lmod.curprof = 0
        lmod.curlayer = 0
        lmod.xrange = [x_u.min()-lmod.dxy/2., x_u.max()+lmod.dxy/2.]
        lmod.yrange = [y_u.min()-lmod.dxy/2., y_u.max()+lmod.dxy/2.]
        lmod.zrange = [z_u.min()-lmod.d_z/2., z_u.max()+lmod.d_z/2.]

# Section to load lithologies.
        lmod.lith_list.pop('Generic 1')

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

        lmod.update(lmod.numx, lmod.numy, lmod.numz, lmod.xrange[0],
                    lmod.yrange[1], lmod.zrange[1], lmod.dxy, lmod.d_z)
        lmod.update_lith_list_reverse()

        for i in range(len(x)):
            col = int((x[i]-lmod.xrange[0])/lmod.dxy)
            row = int((lmod.yrange[1]-y[i])/lmod.dxy)
            layer = int((lmod.zrange[1]-z[i])/lmod.d_z)
            lmod.lith_index[col, row, layer] = lmod.lith_list[label[i]
                                                              ].lith_index

    def dict2lmod(self, indict, pre=''):
        """ routine to convert a dictionary to an lmod """
        lithkeys = indict[pre+'lithkeys']
#        self.pbars.resetall(maximum=lithkeys.size+1)

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
        lmod.curprof = 0
        lmod.curlayer = 0
        lmod.xrange = np.array(indict[pre+'xrange']).tolist()
        lmod.yrange = np.array(indict[pre+'yrange']).tolist()
        lmod.zrange = np.array(indict[pre+'zrange']).tolist()
        if pre+'custprofx' in indict:
            lmod.custprofx = np.asscalar(indict[pre+'custprofx'])
        else:
            lmod.custprofx = {0: (lmod.xrange[0], lmod.xrange[1])}
        if pre+'custprofy' in indict:
            lmod.custprofy = np.asscalar(indict[pre+'custprofy'])
        else:
            lmod.custprofy = {0: (lmod.yrange[0], lmod.yrange[0])}

        lmod.mlut = np.asscalar(indict[pre+'mlut'])
        lmod.init_calc_grids()
        lmod.griddata = np.asscalar(indict[pre+'griddata'])
#        self.pbars.incr()

# Section to load lithologies.
        lmod.lith_list['Background'] = grvmag3d.GeoData(self.parent)

        for itxt in lithkeys:
            if itxt != 'Background':
                lmod.lith_list[itxt] = grvmag3d.GeoData(self.parent)

            lmod.lith_list[itxt].hintn = np.asscalar(indict[pre+itxt+'_hintn'])
            lmod.lith_list[itxt].finc = np.asscalar(indict[pre+itxt+'_finc'])
            lmod.lith_list[itxt].fdec = np.asscalar(indict[pre+itxt+'_fdec'])
            lmod.lith_list[itxt].zobsm = np.asscalar(indict[pre+itxt+'_zobsm'])
            lmod.lith_list[itxt].susc = np.asscalar(indict[pre+itxt+'_susc'])
            lmod.lith_list[itxt].mstrength = np.asscalar(
                indict[pre+itxt+'_mstrength'])
            lmod.lith_list[itxt].qratio = np.asscalar(
                indict[pre+itxt+'_qratio'])
            lmod.lith_list[itxt].minc = np.asscalar(indict[pre+itxt+'_minc'])
            lmod.lith_list[itxt].mdec = np.asscalar(indict[pre+itxt+'_mdec'])
            lmod.lith_list[itxt].density = np.asscalar(
                indict[pre+itxt+'_density'])
            lmod.lith_list[itxt].bdensity = np.asscalar(
                indict[pre+itxt+'_bdensity'])
            lmod.lith_list[itxt].lith_index = np.asscalar(
                indict[pre+itxt+'_lith_index'])
            lmod.lith_list[itxt].g_cols = np.asscalar(indict[pre+itxt+'_numx'])
            lmod.lith_list[itxt].g_rows = np.asscalar(indict[pre+itxt+'_numy'])
            lmod.lith_list[itxt].numz = np.asscalar(indict[pre+itxt+'_numz'])
            lmod.lith_list[itxt].g_dxy = np.asscalar(indict[pre+itxt+'_dxy'])
            lmod.lith_list[itxt].dxy = np.asscalar(indict[pre+itxt+'_dxy'])
            lmod.lith_list[itxt].d_z = np.asscalar(indict[pre+itxt+'_d_z'])
            lmod.lith_list[itxt].zobsm = np.asscalar(indict[pre+itxt+'_zobsm'])
            lmod.lith_list[itxt].zobsg = np.asscalar(indict[pre+itxt+'_zobsg'])
            lmod.lith_list[itxt].modified = True
            lmod.lith_list[itxt].set_xyz12()
#            self.pbars.incr()


class ExportMod3D(object):
    """ Export Data """
    def __init__(self, parent):
        self.ifile = ""
        self.name = "Import Data: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.lmod = None
#        self.dirname = ""
        self.showtext = self.parent.showprocesslog

    def run(self):
        """ Show Info """
        if 'Model3D' not in self.indata:
            self.parent.showprocesslog(
                'Error: You need to have a model first!')
            return

        self.lmod = self.indata['Model3D']

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.',
            'npz (*.npz);;kmz (*.kmz);;xyz (*.xyz)')

        if filename == '':
            return

        os.chdir(filename.rpartition('/')[0])
        self.ifile = str(filename)
        self.ext = filename[-3:]

    # Pop up save dialog box
        if self.ext == 'npz':
            self.savemodel()
        if self.ext == 'kmz':
            self.mod3dtokmz()
        if self.ext == 'xyz':
            self.mod3dtoxyz()

    def savemodel(self):
        """ Save model """
# Open file
#        self.pbars.resetall(maximum = 2)
        filename = self.ifile

# Construct output dictionary
        outdict = {}
        outdict = self.lmod2dict(outdict)
#        self.pbars.incr()

# Save data
        np.savez_compressed(filename, **outdict)
#        self.pbars.incr()
        self.showtext('Model save complete')

    def lmod2dict(self, outdict, pre=''):
        """ place lmod in dictionary """

        outdict[pre+"gregional"] = self.lmod.gregional
        outdict[pre+"ght"] = self.lmod.ght
        outdict[pre+"mht"] = self.lmod.mht
        outdict[pre+"numx"] = self.lmod.numx
        outdict[pre+"numy"] = self.lmod.numy
        outdict[pre+"numz"] = self.lmod.numz
        outdict[pre+"dxy"] = self.lmod.dxy
        outdict[pre+"d_z"] = self.lmod.d_z
        outdict[pre+"lith_index"] = self.lmod.lith_index
        outdict[pre+"xrange"] = self.lmod.xrange
        outdict[pre+"yrange"] = self.lmod.yrange
        outdict[pre+"zrange"] = self.lmod.zrange
        outdict[pre+"mlut"] = self.lmod.mlut
        outdict[pre+"griddata"] = self.lmod.griddata
        outdict[pre+"custprofx"] = self.lmod.custprofx
        outdict[pre+"custprofy"] = self.lmod.custprofy

# Section to save lithologies.
        outdict[pre+"lithkeys"] = list(self.lmod.lith_list.keys())

        for i in self.lmod.lith_list.items():
            curkey = i[0]
            outdict[pre+curkey+"_hintn"] = i[1].hintn
            outdict[pre+curkey+"_finc"] = i[1].finc
            outdict[pre+curkey+"_fdec"] = i[1].fdec
            outdict[pre+curkey+"_zobsm"] = i[1].zobsm
            outdict[pre+curkey+"_susc"] = i[1].susc
            outdict[pre+curkey+"_mstrength"] = i[1].mstrength
            outdict[pre+curkey+"_qratio"] = i[1].qratio
            outdict[pre+curkey+"_minc"] = i[1].minc
            outdict[pre+curkey+"_mdec"] = i[1].mdec
            outdict[pre+curkey+"_density"] = i[1].density
            outdict[pre+curkey+"_bdensity"] = i[1].bdensity
            outdict[pre+curkey+"_lith_index"] = i[1].lith_index
            outdict[pre+curkey+"_numx"] = i[1].g_cols
            outdict[pre+curkey+"_numy"] = i[1].g_rows
            outdict[pre+curkey+"_numz"] = i[1].numz
            outdict[pre+curkey+"_dxy"] = i[1].g_dxy
            outdict[pre+curkey+"_d_z"] = i[1].d_z
            outdict[pre+curkey+"_zobsm"] = i[1].zobsm
            outdict[pre+curkey+"_zobsg"] = i[1].zobsg
            outdict[pre+curkey+"_x12"] = i[1].x12
            outdict[pre+curkey+"_y12"] = i[1].y12
            outdict[pre+curkey+"_z12"] = i[1].z12

        return outdict

    def mod3dtoxyz(self):
        """ Saves the 3D model in an xyz file. """
# Get Save Name
#        self.pbars.resetall(maximum = self.lmod1.numx, mmax = 2)

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'xyz Files (*.xyz)')
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])
        self.showtext('xyz export starting...')

        self.lmod.update_lith_list_reverse()
        lithname = self.lmod.lith_list_reverse.copy()
        lithlist = self.lmod.lith_list.copy()

        tmp = []
        for i in range(self.lmod.numx):
#            self.pbars.incr()
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
                        tmp.append([x, y, z, lith, dens, susc])

#        self.pbars.resetsub(1)
        tmp = np.array(tmp)
        np.savetxt(filename, tmp, delimiter=',')
        tmp = np.array(lithname.items())
        np.savetxt(filename[:filename.rindex('.xyz')]+'_key.csv', tmp, '%s',
                   delimiter=',')
#        self.pbars.incr()

        self.showtext('xyz export complete...')

    def mod3dtokmz(self):
        """ Saves the 3D model and grids in a kmz file.
        Note:
        Only the boundary of the area is in degrees. The actual coordinates
        are still in meters.
        """

        mvis_3d = mvis3d.Mod3dDisplay()

        rev = 1  # should be 1 normally

        xrng = np.array(self.lmod.xrange, dtype=float)
        yrng = np.array(self.lmod.yrange, dtype=float)
        zrng = np.array(self.lmod.zrange, dtype=float)

        prjkmz = Exportkmz()
        tmp = prjkmz.exec_()

        if prjkmz.checkbox_rev_local.isChecked() is True:
            rev = -1

        if tmp == 0:
            return

        if prjkmz.ctrans is None:
            lonwest = prjkmz.dsb_kmz_longwest.value()
            loneast = prjkmz.dsb_kmz_longeast.value()
            latsouth = prjkmz.dsb_kmz_latsouth.value()
            latnorth = prjkmz.dsb_kmz_latnorth.value()

            if latsouth >= latnorth:
                self.showtext('Error: Southern Latitude must less '
                              'than northern latitude.')
                return

            if lonwest >= loneast:
                self.showtext('Error: Western Longitude must be less '
                              'than eastern longitude.')
                return
        else:
            prj = prjkmz.ctrans

            res = prj.TransformPoint(xrng[0], yrng[0])
            lonwest, latsouth = res[0], res[1]
            res = prj.TransformPoint(xrng[1], yrng[1])
            loneast, latnorth = res[0], res[1]

# Get Save Name
        filename = self.ifile

        self.showtext('kmz export starting...')

# Move to 3d model tab to update the model stuff
        self.showtext('updating 3d model...')

        mvis_3d.spacing = [self.lmod.dxy, self.lmod.dxy, self.lmod.d_z]
        mvis_3d.origin = [xrng[0], yrng[0], zrng[0]]
        mvis_3d.gdata = self.lmod.lith_index[::1, ::1, ::-1]
        itmp = np.sort(np.unique(self.lmod.lith_index))
        itmp = itmp[itmp > 0]
        tmp = np.ones((255, 4))*255
        for i in itmp:
            tmp[i, :3] = self.lmod.mlut[i]
        mvis_3d.lut = tmp
#        mvis_3d.update_plot(fullcalc = True)
        mvis_3d.update_model()

        self.showtext('creating kmz file')
        heading = str(0.)
        tilt = str(45.)  # angle from vertical
        lat = str(np.mean([latsouth, latnorth]))  # coord of object
        lon = str(np.mean([lonwest, loneast]))  # coord of object
        rng = str(10000.)  # range to object (meters)
        alt = str(50.)  # alt of object eye is looking at (meters)
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
#        pmax = len(mvis_3d.gfaces)
#        self.pbars.resetsub(maximum=pmax)
        lkey = list(mvis_3d.faces.keys())
        lkey.pop(lkey.index(0))
        lithcnt = -1

        for lith in lkey:
#            self.pbars.incr()
#            if lith not in self.lmod.lith_list_reverse:
#                print('problem')
#                continue

            faces = np.array(mvis_3d.gfaces[lith])
            points = [rev, rev, 1]*(mvis_3d.gpoints[lith] - mvis_3d.origin)
            if rev == -1:
                points += [xrng.ptp(), yrng.ptp(), 0]

            norm = np.abs(mvis_3d.gnorms[lith])
            clrtmp = np.array(self.lmod.mlut[lith])/255.
            curmod = self.lmod.lith_list_reverse[lith]

            if len(points) > 60000:
                self.showtext(curmod + ' has too many points (' +
                              str(len(points))+'). Not exported')
                points = points[:60000]
                norm = norm[:60000]
                faces = faces[faces.max(1) < 60000]
#                continue

            lithcnt += 1

#            print len(points), curmod
            dockml += (
                '    <Placemark>\r\n'
                '      <name>' + curmod + '</name>\r\n'
                '      <description></description>\r\n'
                '      <Style id="default"/>\r\n'
                '      <Model>\r\n'
                '        <altitudeMode>relativeToGround</altitudeMode>\r\n'
                '        <Location>\r\n'
                '          <latitude>' + lato + '</latitude>\r\n'
                '          <longitude>' + lono + '</longitude>\r\n'
                '          <altitude>' + alt + '</altitude>\r\n'
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
        for i in range(len(modeldae)):
            zfile.writestr('models\\mod3d'+str(i)+'.dae', modeldae[i])

        for i in self.lmod.griddata.keys():
            prj = prjkmz.ctrans
            x_1 = self.lmod.griddata[i].tlx
            x_2 = x_1 + self.lmod.griddata[i].xdim*self.lmod.griddata[i].cols
            y_2 = self.lmod.griddata[i].tly
            y_1 = y_2 - self.lmod.griddata[i].ydim*self.lmod.griddata[i].rows

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

            plt.imshow(self.lmod.griddata[i].data[::-rev, ::rev],
                       extent=(lonwest, loneast, latsouth, latnorth),
                       aspect='normal',
                       interpolation='nearest')
            plt.savefig('tmp930.png')
#            plt.close('tmp930')

            zfile.write('tmp930.png', 'models\\'+i+'.png')
            os.remove('tmp930.png')

        dockml += (
            '  </Folder>\r\n'
            '  \r\n'
            '  </kml>')

        zfile.writestr('doc.kml', dockml)

        zfile.close()
#        self.pbars.resetsub(maximum=1)
#        self.pbars.incr()
        self.showtext('kmz export complete')


class Exportkmz(QtGui.QDialog):
    """ Class to call up a dialog """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.gridlayout_2 = QtGui.QGridLayout(self)
        self.groupbox = QtGui.QGroupBox(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.groupbox_2 = QtGui.QGroupBox(self)
        self.gridlayout = QtGui.QGridLayout(self.groupbox)
        self.label_3 = QtGui.QLabel(self.groupbox)
        self.label_4 = QtGui.QLabel(self.groupbox)
        self.combobox_kmz_datum = QtGui.QComboBox(self.groupbox)
        self.combobox_kmz_proj = QtGui.QComboBox(self.groupbox)
        self.dsb_kmz_latorigin = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_cm = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_scalefactor = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_fnorthing = QtGui.QDoubleSpinBox(self.groupbox)
        self.label_6 = QtGui.QLabel(self.groupbox)
        self.label_7 = QtGui.QLabel(self.groupbox)
        self.label_8 = QtGui.QLabel(self.groupbox)
        self.label_9 = QtGui.QLabel(self.groupbox)
        self.label_10 = QtGui.QLabel(self.groupbox)
        self.label_11 = QtGui.QLabel(self.groupbox)
        self.dsb_kmz_feasting = QtGui.QDoubleSpinBox(self.groupbox)
        self.sb_kmz_zone = QtGui.QSpinBox(self.groupbox)
        self.gridlayout_3 = QtGui.QGridLayout(self.groupbox_2)
        self.label = QtGui.QLabel(self.groupbox_2)
        self.dsb_kmz_latnorth = QtGui.QDoubleSpinBox(self.groupbox_2)
        self.dsb_kmz_longwest = QtGui.QDoubleSpinBox(self.groupbox_2)
        self.dsb_kmz_longeast = QtGui.QDoubleSpinBox(self.groupbox_2)
        self.dsb_kmz_latsouth = QtGui.QDoubleSpinBox(self.groupbox_2)
        self.checkbox_rev_local = QtGui.QCheckBox(self.groupbox_2)
        self.label_2 = QtGui.QLabel(self.groupbox_2)
        self.label_5 = QtGui.QLabel(self.groupbox_2)
        self.label_12 = QtGui.QLabel(self.groupbox_2)

        self.setupui()

        self.combobox_kmz_datum.addItem('WGS84')
        self.combobox_kmz_datum.addItem('Cape (Clarke1880)')
        self.combobox_kmz_proj.addItem('UTM (South)')
        self.combobox_kmz_proj.addItem('UTM (North)')
        self.combobox_kmz_proj.addItem('Transverse Mercator')
        self.combobox_kmz_proj.addItem('Local')
        self.zone(35)

        self.combobox_kmz_proj.currentIndexChanged.connect(self.proj)
        self.sb_kmz_zone.valueChanged.connect(self.zone)
        self.buttonbox.accepted.connect(self.acceptall)

        self.datum = {}
        self.datum['Cape'] = (
            'GEOGCS["Cape",'
            'DATUM["D_Cape",'
            'SPHEROID["Clarke_1880_Arc",6378249.145,293.4663077]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Degree",0.017453292519943295]]')
        self.datum['Hartebeesthoek94'] = (
            'GEOGCS["Hartebeesthoek94",'
            'DATUM["D_Hartebeesthoek_1994",'
            'SPHEROID["WGS_1984",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Degree",0.017453292519943295]]')

        self.ctrans = None

    def acceptall(self):
        """ accept """
        orig = osr.SpatialReference()
        orig.SetWellKnownGeogCS('WGS84')
        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')

        indx = self.combobox_kmz_datum.currentIndex()
        txt = self.combobox_kmz_datum.itemText(indx)

        if 'Cape' in txt:
            orig.ImportFromWkt(self.datum['Cape'])

        indx = self.combobox_kmz_proj.currentIndex()
        txt = self.combobox_kmz_proj.itemText(indx)

        if 'Local' in txt:
            self.ctrans = None

        if 'UTM' in txt:
            utmzone = self.sb_kmz_zone.value()
            if 'North' in txt:
                orig.SetUTM(utmzone, True)
            else:
                orig.SetUTM(utmzone, False)

        if 'Transverse Mercator' in txt:
            clat = self.dsb_kmz_latorigin.value()
            clong = self.dsb_kmz_cm.value()
            scale = self.dsb_kmz_scalefactor.value()
            f_e = self.dsb_kmz_feasting.value()
            f_n = self.dsb_kmz_fnorthing.value()
            orig.SetTM(clat, clong, scale, f_e, f_n)

        self.ctrans = osr.CoordinateTransformation(orig, targ)

        if 'Local' in txt:
            self.ctrans = None

        self.accept()

    def proj(self, indx):
        """ used for choosing the projection """
        txt = self.combobox_kmz_proj.itemText(indx)
        if 'UTM' in txt:
            self.sb_kmz_zone.setEnabled(True)
            self.zone(self.sb_kmz_zone.value())
        else:
            self.sb_kmz_zone.setEnabled(False)

        if txt == 'Transverse Mercator':
            self.dsb_kmz_feasting.setValue(0.)
            self.dsb_kmz_fnorthing.setValue(0.)
            self.dsb_kmz_scalefactor.setValue(1.0)

        if txt == 'Local':
            self.dsb_kmz_latnorth.setEnabled(True)
            self.dsb_kmz_latsouth.setEnabled(True)
            self.dsb_kmz_longwest.setEnabled(True)
            self.dsb_kmz_longeast.setEnabled(True)
            self.dsb_kmz_cm.setEnabled(False)
            self.dsb_kmz_latorigin.setEnabled(False)
            self.dsb_kmz_feasting.setEnabled(False)
            self.dsb_kmz_fnorthing.setEnabled(False)
            self.dsb_kmz_scalefactor.setEnabled(False)
        else:
            self.dsb_kmz_latnorth.setEnabled(False)
            self.dsb_kmz_latsouth.setEnabled(False)
            self.dsb_kmz_longwest.setEnabled(False)
            self.dsb_kmz_longeast.setEnabled(False)
            self.dsb_kmz_cm.setEnabled(True)
            self.dsb_kmz_latorigin.setEnabled(True)
            self.dsb_kmz_feasting.setEnabled(True)
            self.dsb_kmz_fnorthing.setEnabled(True)
            self.dsb_kmz_scalefactor.setEnabled(True)

    def zone(self, val):
        """ used for changing UTM zone """
        c_m = -180.+(val-1)*6+3
        self.dsb_kmz_cm.setValue(c_m)
        self.dsb_kmz_latorigin.setValue(0.)
        self.dsb_kmz_feasting.setValue(500000.)
        self.dsb_kmz_fnorthing.setValue(0.)
        self.dsb_kmz_scalefactor.setValue(0.9996)

        indx = self.combobox_kmz_proj.currentIndex()
        txt = self.combobox_kmz_proj.itemText(indx)

        if txt == 'UTM (South)':
            self.dsb_kmz_fnorthing.setValue(10000000.)

    def setupui(self):
        """ Setup UI """
        self.gridlayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.combobox_kmz_datum, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.combobox_kmz_proj, 1, 1, 1, 1)
        self.dsb_kmz_latorigin.setMinimum(-90.0)
        self.dsb_kmz_latorigin.setMaximum(90.0)
        self.gridlayout.addWidget(self.dsb_kmz_latorigin, 3, 1, 1, 1)
        self.dsb_kmz_cm.setMinimum(-180.0)
        self.dsb_kmz_cm.setMaximum(180.0)
        self.dsb_kmz_cm.setProperty("value", 27.0)
        self.gridlayout.addWidget(self.dsb_kmz_cm, 4, 1, 1, 1)
        self.dsb_kmz_scalefactor.setDecimals(4)
        self.dsb_kmz_scalefactor.setProperty("value", 0.9996)
        self.gridlayout.addWidget(self.dsb_kmz_scalefactor, 5, 1, 1, 1)
        self.dsb_kmz_fnorthing.setMaximum(1000000000.0)
        self.dsb_kmz_fnorthing.setProperty("value", 10000000.0)
        self.gridlayout.addWidget(self.dsb_kmz_fnorthing, 7, 1, 1, 1)
        self.gridlayout.addWidget(self.label_6, 3, 0, 1, 1)
        self.gridlayout.addWidget(self.label_7, 4, 0, 1, 1)
        self.gridlayout.addWidget(self.label_8, 5, 0, 1, 1)
        self.gridlayout.addWidget(self.label_9, 6, 0, 1, 1)
        self.gridlayout.addWidget(self.label_10, 7, 0, 1, 1)
        self.gridlayout.addWidget(self.label_11, 2, 0, 1, 1)
        self.dsb_kmz_feasting.setMaximum(1000000000.0)
        self.dsb_kmz_feasting.setProperty("value", 500000.0)
        self.gridlayout.addWidget(self.dsb_kmz_feasting, 6, 1, 1, 1)
        self.sb_kmz_zone.setMaximum(60)
        self.sb_kmz_zone.setProperty("value", 35)
        self.gridlayout.addWidget(self.sb_kmz_zone, 2, 1, 1, 1)
        self.gridlayout_2.addWidget(self.groupbox, 0, 0, 1, 2)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout_2.addWidget(self.buttonbox, 2, 0, 1, 2)
        self.gridlayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.gridlayout_3.addWidget(self.label_2, 3, 0, 1, 1)
        self.gridlayout_3.addWidget(self.label_5, 4, 0, 1, 1)
        self.gridlayout_3.addWidget(self.label_12, 5, 0, 1, 1)
        self.dsb_kmz_longwest.setEnabled(False)
        self.dsb_kmz_longwest.setDecimals(6)
        self.dsb_kmz_longwest.setMinimum(-180.0)
        self.dsb_kmz_longwest.setMaximum(180.0)
        self.dsb_kmz_longwest.setProperty("value", 0.0)
        self.gridlayout_3.addWidget(self.dsb_kmz_longwest, 1, 1, 1, 1)
        self.dsb_kmz_longeast.setEnabled(False)
        self.dsb_kmz_longeast.setDecimals(6)
        self.dsb_kmz_longeast.setMinimum(-180.0)
        self.dsb_kmz_longeast.setMaximum(180.0)
        self.dsb_kmz_longeast.setProperty("value", 0.0)
        self.gridlayout_3.addWidget(self.dsb_kmz_longeast, 3, 1, 1, 1)
        self.dsb_kmz_latsouth.setEnabled(False)
        self.dsb_kmz_latsouth.setDecimals(6)
        self.dsb_kmz_latsouth.setMinimum(-90.0)
        self.dsb_kmz_latsouth.setMaximum(90.0)
        self.dsb_kmz_latsouth.setProperty("value", 0.0)
        self.gridlayout_3.addWidget(self.dsb_kmz_latsouth, 4, 1, 1, 1)
        self.dsb_kmz_latnorth.setEnabled(False)
        self.dsb_kmz_latnorth.setDecimals(6)
        self.dsb_kmz_latnorth.setMinimum(-90.0)
        self.dsb_kmz_latnorth.setMaximum(90.0)
        self.dsb_kmz_latnorth.setProperty("value", 0.0)
        self.gridlayout_3.addWidget(self.dsb_kmz_latnorth, 5, 1, 1, 1)
        self.gridlayout_3.addWidget(self.checkbox_rev_local, 0, 0, 1, 2)
        self.gridlayout_2.addWidget(self.groupbox_2, 1, 0, 1, 2)

        self.setWindowTitle("Google Earth kmz Export")
        self.groupbox.setTitle("Input Projection")
        self.label_3.setText("Datum")
        self.label_4.setText("Projection")
        self.label_6.setText("Latitude of Origin")
        self.label_7.setText("Central Meridian")
        self.label_8.setText("Scale Factor")
        self.label_9.setText("False Easting")
        self.label_10.setText("False Northing")
        self.label_11.setText("UTM Zone")
        self.groupbox_2.setTitle("Model Boundary (Local Projection Only)")
        self.label.setText("Longitude West")
        self.label_2.setText("Longitude East")
        self.label_5.setText("Latitude South")
        self.label_12.setText("Latitude North")
        self.checkbox_rev_local.setText(
            "Reverse sign of X and Y coordinates (i.e X is now -X)")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)


class ImportPicture(QtGui.QDialog):
    """ Class to call up a dialog """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.lmod = LithModel()

        self.ifile = ""
        self.name = "Import 3D Model: "
        self.ext = ""
        self.pbar = None
        self.indata = {}
        self.outdata = {}
        self.grid = None

        self.gridlayout_2 = QtGui.QGridLayout(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.groupbox = QtGui.QGroupBox(self)
        self.label = QtGui.QLabel(self.groupbox)
        self.label_2 = QtGui.QLabel(self.groupbox)
        self.label_3 = QtGui.QLabel(self.groupbox)
        self.label_5 = QtGui.QLabel(self)
        self.dsb_picimp_west = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_picimp_east = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_picimp_depth = QtGui.QDoubleSpinBox(self.groupbox)
        self.rb_picimp_westeast = QtGui.QRadioButton(self.groupbox)
        self.rb_picimp_southnorth = QtGui.QRadioButton(self.groupbox)
        self.dsb_picimp_maxalt = QtGui.QDoubleSpinBox(self.groupbox)
        self.label_4 = QtGui.QLabel(self.groupbox)
        self.gridlayout_3 = QtGui.QGridLayout(self.groupbox)

        self.setupui()

        self.min_coord = None
        self.max_coord = None
        self.max_alt = None
        self.min_alt = None
        self.is_eastwest = None

        self.rb_picimp_westeast.clicked.connect(self.lmod2var)
        self.rb_picimp_southnorth.clicked.connect(self.lmod2var)

        self.lmod2var()

    def setupui(self):
        """ Setup UI """
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout_2.addWidget(self.buttonbox, 2, 0, 1, 2)
        self.gridlayout_3.addWidget(self.label, 2, 0, 1, 1)
        self.gridlayout_3.addWidget(self.label_2, 4, 0, 1, 1)
        self.gridlayout_3.addWidget(self.label_3, 6, 0, 1, 1)
        self.dsb_picimp_west.setDecimals(6)
        self.dsb_picimp_west.setMinimum(-999999999.0)
        self.dsb_picimp_west.setMaximum(999999999.0)
        self.dsb_picimp_west.setProperty("value", 0.0)
        self.gridlayout_3.addWidget(self.dsb_picimp_west, 2, 1, 1, 1)
        self.dsb_picimp_east.setDecimals(6)
        self.dsb_picimp_east.setMinimum(-999999999.0)
        self.dsb_picimp_east.setMaximum(999999999.0)
        self.dsb_picimp_east.setProperty("value", 1000.0)
        self.gridlayout_3.addWidget(self.dsb_picimp_east, 4, 1, 1, 1)
        self.dsb_picimp_depth.setDecimals(6)
        self.dsb_picimp_depth.setMinimum(0.0)
        self.dsb_picimp_depth.setMaximum(999999999.0)
        self.dsb_picimp_depth.setProperty("value", 1000.0)
        self.gridlayout_3.addWidget(self.dsb_picimp_depth, 6, 1, 1, 1)
        self.rb_picimp_westeast.setChecked(True)
        self.gridlayout_3.addWidget(self.rb_picimp_westeast, 0, 0, 1, 1)
        self.gridlayout_3.addWidget(self.rb_picimp_southnorth, 1, 0, 1, 1)
        self.dsb_picimp_maxalt.setDecimals(6)
        self.dsb_picimp_maxalt.setMinimum(-999999999.0)
        self.dsb_picimp_maxalt.setMaximum(999999999.0)
        self.dsb_picimp_maxalt.setProperty("value", 1000.0)
        self.gridlayout_3.addWidget(self.dsb_picimp_maxalt, 5, 1, 1, 1)
        self.gridlayout_3.addWidget(self.label_4, 5, 0, 1, 1)
        self.gridlayout_2.addWidget(self.groupbox, 0, 0, 1, 2)
        self.gridlayout_2.addWidget(self.label_5, 1, 0, 1, 2)

        self.setWindowTitle("Profile Picture Importer")
        self.groupbox.setTitle("Profile Coordinates")
        self.label.setText("West/South Coordinate")
        self.label_2.setText("East/North Coordinate")
        self.label_3.setText("Depth")
        self.rb_picimp_westeast.setText("Profile is from West to East")
        self.rb_picimp_southnorth.setText("Profile is from South to North")
        self.label_4.setText("Maximum Altitude")
        self.label_5.setText('Press Cancel if you wish to connect profile '
                             'information from a 3D model')

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)

    def lmod2var(self):
        """ lmod 2 var """
        self.is_eastwest = self.rb_picimp_westeast.isChecked()
        self.min_alt, self.max_alt = self.lmod.zrange

        if self.is_eastwest:
            self.min_coord, self.max_coord = self.lmod.xrange
        else:
            self.min_coord, self.max_coord = self.lmod.yrange

        self.update_win()

    def update_var(self):
        """ Updates the values """
        self.min_coord = self.dsb_picimp_west.value()
        self.max_coord = self.dsb_picimp_east.value()
        self.max_alt = self.dsb_picimp_maxalt.value()
        self.min_alt = self.max_alt - self.dsb_picimp_depth.value()
        self.is_eastwest = self.rb_picimp_westeast.isChecked()

        if self.is_eastwest:
            self.grid.bandid = r'West to East'
        else:
            self.grid.bandid = r'South to North'

        self.grid.xdim = (self.max_coord-self.min_coord)/self.grid.cols
        self.grid.ydim = (self.max_alt-self.min_alt)/self.grid.rows
        self.grid.tlx = self.min_coord
        self.grid.tly = self.max_alt

#        filename = self.ifile
#        if filename.rfind('/') != -1:
#            datatext = filename.rpartition(r'/')[-1]
#        else:
#            datatext = filename
#        self.lmod.profpics[datatext] = grid

    def update_win(self):
        """ Updates the window values """
        self.dsb_picimp_west.setValue(self.min_coord)
        self.dsb_picimp_east.setValue(self.max_coord)
        self.dsb_picimp_maxalt.setValue(self.max_alt)
        self.dsb_picimp_depth.setValue(self.max_alt-self.min_alt)

    def settings(self):
        """ Load GeoTiff """
        if 'Model3D' in self.indata.keys():
            self.lmod = self.indata['Model3D']
            self.lmod2var()

        temp = self.exec_()
        if temp == 0:
            return

        filename = QtGui.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', '*.jpg *.tif *.bmp')

        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = filename

        data = gtiff(filename)
        self.grid = data[0]

        if (self.dsb_picimp_west.value() >=
                self.dsb_picimp_east.value()):
            return
        if self.dsb_picimp_depth.value() == 0.0:
            return

        self.update_var()
        self.outdata['ProfPic'] = [self.grid]

        return True


def gtiff(filename):
    """ Utility to import geotiffs """

    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    nred = dataset.GetRasterBand(1).ReadAsArray()
    ngreen = dataset.GetRasterBand(2).ReadAsArray()
    nblue = dataset.GetRasterBand(3).ReadAsArray()
    itmp = np.uint32(nred*65536+ngreen*256+nblue+int('FF000000', 16))

    gtr = dataset.GetGeoTransform()
    dat = [Data()]

    dat[0].tlx = gtr[0]
    dat[0].tly = gtr[3]
    dat[0].bandid = "Image"
    dat[0].rows = dataset.RasterYSize
    dat[0].cols = dataset.RasterXSize
    dat[0].xdim = abs(gtr[1])
    dat[0].ydim = abs(gtr[5])
    dat[0].data = np.ma.array(itmp)
    dat[0].nullvalue = np.nan  # This was erread.nullvalue, is changed above

    return dat
