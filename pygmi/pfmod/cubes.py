# -----------------------------------------------------------------------------
# Name:        cubes.py (part of PyGMI)
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
""" This is code for the 3d model creation. """

from __future__ import print_function

import os
import sys
import numpy as np
from PyQt4 import QtCore, QtGui, QtOpenGL
# import OpenGL
# OpenGL.ERROR_CHECKING = False  # Note This!!!!
from OpenGL import GL
from OpenGL import GLU
from OpenGL.arrays import vbo
from scipy.ndimage.interpolation import zoom
import scipy.ndimage.filters as sf
from numba import jit
from PIL import Image
import pygmi.pfmod.misc as misc


class Mod3dDisplay(QtGui.QDialog):
    """ Widget class to call the main interface """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.parent = parent
        self.lmod1 = None
        self.indata = {}
        self.outdata = self.indata

        if hasattr(parent, 'showtext'):
            self.showtext = parent.showtext
        else:
            self.showtext = print

        if hasattr(parent, 'showprocesslog'):
            self.showprocesslog = parent.showprocesslog
        else:
            self.showprocesslog = print

        self.corners = []
        self.faces = {}
        self.norms = []
        self.gdata = np.zeros([4, 3, 2])
        self.gdata[0, 0, 0] = -1
        self.sliths = np.array([])  # selected lithologies
        self.origin = [0., 0., 0.]
        self.spacing = [10., 10., 10.]
        self.zmult = 1.
        self.lut = np.ones((255, 4))*255
        self.lut[0] = [255, 0, 0, 255]
        self.lut[1] = [0, 255, 0, 255]
        self.gfaces = []
        self.gpoints = []
        self.gnorms = []
        self.glutlith = []
        self.demsurf = None
        self.qdiv = 0
        self.mesh = {}
        self.opac = 0.0
        self.cust_z = None

# Back to normal stuff
        self.lw_3dmod_defs = QtGui.QListWidget()
        self.label = QtGui.QLabel()
        self.label2 = QtGui.QLabel()
        self.pb_save = QtGui.QPushButton()
        self.pb_refresh = QtGui.QPushButton()
        self.checkbox_smooth = QtGui.QCheckBox()
        self.pbar = QtGui.QProgressBar()
        self.glwidget = GLWidget()
        self.vslider_3dmodel = QtGui.QSlider()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        horizontallayout = QtGui.QHBoxLayout(self)
        vbox_cmodel = QtGui.QVBoxLayout()
        verticallayout = QtGui.QVBoxLayout()

        self.vslider_3dmodel.setMinimum(1)
        self.vslider_3dmodel.setMaximum(1000)
        self.vslider_3dmodel.setOrientation(QtCore.Qt.Vertical)
        vbox_cmodel.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                                       QtGui.QSizePolicy.Fixed)

        sizepolicy_pb = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum,
                                          QtGui.QSizePolicy.Maximum)

        self.lw_3dmod_defs.setSizePolicy(sizepolicy)
        self.lw_3dmod_defs.setSelectionMode(
            QtGui.QAbstractItemView.MultiSelection)
        self.lw_3dmod_defs.setFixedWidth(220)
        self.checkbox_smooth.setSizePolicy(sizepolicy)
        self.pb_save.setSizePolicy(sizepolicy_pb)
        self.pb_refresh.setSizePolicy(sizepolicy_pb)
        self.pbar.setOrientation(QtCore.Qt.Vertical)

        self.checkbox_smooth.setText("Smooth Model")
        self.pb_save.setText("Save to Image File (JPG or PNG)")
        self.pb_refresh.setText('Refresh Model')

        verticallayout.addWidget(self.lw_3dmod_defs)
        verticallayout.addWidget(self.checkbox_smooth)
        verticallayout.addWidget(self.pb_save)
        verticallayout.addWidget(self.pb_refresh)
        vbox_cmodel.addWidget(self.glwidget)
        horizontallayout.addWidget(self.vslider_3dmodel)
        horizontallayout.addLayout(vbox_cmodel)
        horizontallayout.addLayout(verticallayout)
        horizontallayout.addWidget(self.pbar)

        self.lw_3dmod_defs.clicked.connect(self.change_defs)
        self.vslider_3dmodel.sliderReleased.connect(self.mod3d_vs)
        self.pb_save.clicked.connect(self.save)
        self.pb_refresh.clicked.connect(self.run)
        self.checkbox_smooth.stateChanged.connect(self.update_plot)

    def save(self):
        """ This saves a jpg """
        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'JPG (*.jpg);;PNG (*.png)')
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])

        ftype = 'JPEG'

        if 'PNG' in filename:
            ftype = 'PNG'

        width = self.glwidget.width()
        height = self.glwidget.height()
        tmp = self.glwidget.readPixels()
        image = Image.frombytes('RGB', (width, height), tmp)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filename, ftype)

    def update_for_kmz(self):
        """ Updates for the kmz file """

        self.gpoints = self.corners
        self.gnorms = self.norms
        self.gfaces = {}
#        pdb.set_trace()

# (Pdb) self.corners[0].max(0)
# array([ 505000.,  320000.,   60000.])
# (Pdb) self.corners[0].min(0)
# array([     0.,      0.,  31000.])


# (Pdb) self.corners[0].max(0)
# array([ 509386.36363636,  324386.36363636,      -0.        ])
# (Pdb) self.corners[0].min(0)
# array([     0.        ,      0.        , -60877.27272727])

        if list(self.faces.values())[0].shape[1] == 4:
            for i in self.faces:
                self.gfaces[i] = np.append(self.faces[i][:, :-1],
                                           self.faces[i][:, [0, 2, 3]])
                self.gfaces[i].shape = (self.gfaces[i].shape[0]/3, 3)
        else:
            self.gfaces = self.faces.copy()

        self.glutlith = range(1, len(self.gfaces)+1)

    def change_defs(self):
        """ List box routine """
        if len(self.lmod1.lith_list) == 0:
            return
        self.set_selected_liths()
        self.update_color()

    def data_init(self):
        """ Data initialisation routine """
        self.outdata = self.indata

    def set_selected_liths(self):
        """ Sets the selected lithologies """
        i = self.lw_3dmod_defs.selectedItems()

        itxt = [j.text() for j in i]
        lith = [self.lmod1.lith_list[j] for j in itxt]
        lith3d = [j.lith_index for j in lith]

        self.sliths = np.intersect1d(self.gdata, lith3d)

    def mod3d_vs(self):
        """ Vertical slider used to scale 3d view """
        perc = (float(self.vslider_3dmodel.value()) /
                float(self.vslider_3dmodel.maximum()))

        zdist = self.lmod1.numz*self.lmod1.d_z
        if self.lmod1.numx > self.lmod1.numy:
            xydist = self.lmod1.numx*self.lmod1.dxy
        else:
            xydist = self.lmod1.numy*self.lmod1.dxy
        xy_z_ratio = xydist/zdist

        self.zmult = 1.0 + perc*xy_z_ratio
        self.update_model2()

    def update_color(self):
        """ Update color only """
        liths = np.unique(self.gdata)
        liths = liths[liths < 900]

        if liths.max() == -1:
            return
        if liths[0] == -1:
            liths = liths[1:]
        if liths[0] == 0:
            liths = liths[1:]

        lut = self.lut[:, [0, 1, 2]]/255.

        clr = np.array([])
        lcheck = np.unique(self.lmod1.lith_index)

        for lno in liths:
            if lno not in lcheck:
                continue
            if self.corners[lno] == []:
                continue
            if lno in self.sliths:
                clrtmp = lut[lno].tolist()+[1.]
            else:
                clrtmp = lut[lno].tolist()+[self.opac]

            clr = np.append(clr,
                            np.zeros([self.corners[lno].shape[0], 4])+clrtmp)

        clr.shape = (clr.shape[0]/4, 4)

        self.glwidget.cubeClrArray = clr

        self.glwidget.init_object()
        self.glwidget.updateGL()

    def run(self):
        """ Process data """
        if 'Model3D' not in self.indata:
            self.showprocesslog('No 3D model. You may need to execute' +
                                ' that module first')
            return False

        self.lmod1 = self.indata['Model3D'][0]

#        self.vslider_3dmodel.setValue(1)

        liths = np.unique(self.lmod1.lith_index[::1, ::1, ::-1])
        liths = np.array(liths).astype(int)  # needed for use in faces array
        if liths[0] == -1:
            liths = liths[1:]
        if liths[0] == 0:
            liths = liths[1:]
        if liths.size == 0:
            self.showprocesslog('No 3D model. You need to draw in at' +
                                ' least part of a lithology first.')
            return False

        self.show()
        misc.update_lith_lw(self.lmod1, self.lw_3dmod_defs)
        for i in range(self.lw_3dmod_defs.count()-1, -1, -1):
            if self.lw_3dmod_defs.item(i).text() == 'Background':
                self.lw_3dmod_defs.takeItem(i)

        for i in range(self.lw_3dmod_defs.count()):
            self.lw_3dmod_defs.item(i).setSelected(True)
        self.update_plot()
        return True

    def update_plot(self):
        """ Update 3D Model """
        QtGui.QApplication.processEvents()
    # Update 3D model
        self.spacing = [self.lmod1.dxy, self.lmod1.dxy, self.lmod1.d_z]
        self.origin = [self.lmod1.xrange[0], self.lmod1.yrange[0],
                       self.lmod1.zrange[0]]
        self.gdata = self.lmod1.lith_index[::1, ::1, ::-1]

#     update colors
        i = self.lw_3dmod_defs.findItems("*", QtCore.Qt.MatchWildcard)
        itxt = [j.text() for j in i]
        itmp = []
        for i in itxt:
            itmp.append(self.lmod1.lith_list[i].lith_index)

        itmp = np.sort(itmp)
        tmp = np.ones((255, 4))*255

        for i in itmp:
            tmp[i, :3] = self.lmod1.mlut[i]

        self.lut = tmp

        if len(self.lmod1.lith_list) > 0:
            self.set_selected_liths()
            self.update_model()
            self.update_model2()

        self.glwidget.xRot = 0*16
        self.glwidget.zRot = 0*16

        self.glwidget.init_object()
        self.glwidget.updateGL()

    def update_model(self, issmooth=None):
        """ Update the 3d model. Faces, nodes and face normals are calculated
        here, from the voxel model. """
        QtGui.QApplication.processEvents()

        if issmooth is None:
            issmooth = self.checkbox_smooth.isChecked()

        self.faces = {}
        self.norms = {}
        self.corners = {}

        liths = np.unique(self.gdata)
        liths = np.array(liths).astype(int)  # needed for use in faces array
        lcheck = np.unique(self.lmod1.lith_index)

        if liths.max() == -1:
            return
        if liths[0] == -1:
            liths = liths[1:]

        self.pbar.setMaximum(liths.size)
        self.pbar.setValue(0)

        if not issmooth:
            igd, jgd, kgd = self.gdata.shape
            cloc = np.indices(((kgd+1), (jgd+1), (igd+1))).T.reshape(
                (igd+1)*(jgd+1)*(kgd+1), 3).T[::-1].T
            cloc = cloc * self.spacing + self.origin
            cindx = np.arange(cloc.size/3, dtype=int)
            cindx.shape = (igd+1, jgd+1, kgd+1)

            tmpdat = np.zeros([igd+2, jgd+2, kgd+2])-1
            tmpdat[1:-1, 1:-1, 1:-1] = self.gdata

        else:
            # Setup stuff for triangle calcs
            nshape = np.array(self.lmod1.lith_index.shape)+[2, 2, 2]
            x = np.arange(nshape[1]) * self.spacing[1]
            y = np.arange(nshape[0]) * self.spacing[0]
            z = np.arange(nshape[2]) * self.spacing[2]
#            if self.cust_z is None:
#                z = np.arange(nshape[2]) * self.spacing[2]
#            else:
#                z = ([self.cust_z[0] - self.cust_z[1]] + self.cust_z.tolist()
#                     + [2*self.cust_z[-1] - self.cust_z[-2]])
            xx, yy, zz = np.meshgrid(x, y, z)

    # Set up gaussian smoothing filter
            ix, iy, iz = np.mgrid[-1:2, -1:2, -1:2]
            sigma = 2
            cci = np.exp(-(ix**2+iy**2+iz**2)/(3*sigma**2))

        tmppval = 0
        for lno in liths:
            tmppval += 1
            self.pbar.setValue(tmppval)
            if lno not in lcheck:
                continue
            if not issmooth:
                gdat2 = tmpdat.copy()
                gdat2[gdat2 != lno] = -0.5
                gdat2[gdat2 == lno] = 0.5

                newfaces = []

                ndiff = np.diff(gdat2, 1, 2).astype(int)
                nd1 = ndiff[1:, 1:]
                nd2 = ndiff[:-1, 1:]
                nd3 = ndiff[:-1, :-1]
                nd4 = ndiff[1:, :-1]

                c_1 = cindx[nd1 == 1]
                c_2 = cindx[nd2 == 1]
                c_3 = cindx[nd3 == 1]
                c_4 = cindx[nd4 == 1]
                ccc = np.transpose([c_1, c_4, c_3, c_2])
                newfaces = np.append(newfaces, ccc)

                c_1 = cindx[nd1 == -1]
                c_2 = cindx[nd2 == -1]
                c_3 = cindx[nd3 == -1]
                c_4 = cindx[nd4 == -1]
                ccc = np.transpose([c_1, c_2, c_3, c_4])
                newfaces = np.append(newfaces, ccc)

                ndiff = np.diff(gdat2, 1, 1).astype(int)
                nd1 = ndiff[1:, :, 1:]
                nd2 = ndiff[:-1, :, 1:]
                nd3 = ndiff[:-1, :, :-1]
                nd4 = ndiff[1:, :, :-1]

                c_1 = cindx[nd1 == 1]
                c_2 = cindx[nd2 == 1]
                c_3 = cindx[nd3 == 1]
                c_4 = cindx[nd4 == 1]
                ccc = np.transpose([c_1, c_2, c_3, c_4])
                newfaces = np.append(newfaces, ccc)

                c_1 = cindx[nd1 == -1]
                c_2 = cindx[nd2 == -1]
                c_3 = cindx[nd3 == -1]
                c_4 = cindx[nd4 == -1]
                ccc = np.transpose([c_1, c_4, c_3, c_2])
                newfaces = np.append(newfaces, ccc)

                ndiff = np.diff(gdat2, 1, 0).astype(int)
                nd1 = ndiff[:, 1:, 1:]
                nd2 = ndiff[:, 1:, :-1]
                nd3 = ndiff[:, :-1, :-1]
                nd4 = ndiff[:, :-1, 1:]

                c_1 = cindx[nd1 == 1]
                c_2 = cindx[nd2 == 1]
                c_3 = cindx[nd3 == 1]
                c_4 = cindx[nd4 == 1]
                ccc = np.transpose([c_1, c_2, c_3, c_4])
                newfaces = np.append(newfaces, ccc)

                c_1 = cindx[nd1 == -1]
                c_2 = cindx[nd2 == -1]
                c_3 = cindx[nd3 == -1]
                c_4 = cindx[nd4 == -1]
                ccc = np.transpose([c_1, c_4, c_3, c_2])
                newfaces = np.append(newfaces, ccc)

                uuu, i = np.unique(newfaces, return_inverse=True)
                uuu = uuu.astype(int)
                n_f = np.arange(uuu.size)
                newfaces = n_f[i]
                newcorners = cloc[uuu]
                newfaces.shape = (newfaces.size/4, 4)

                self.faces[lno] = newfaces
                self.corners[lno] = newcorners

            else:
                c = np.zeros(nshape)

                cc = self.lmod1.lith_index.copy()
                cc[cc != lno] = 0
                cc[cc == lno] = 1

                cc = sf.convolve(cc, cci)/cci.size

# shrink cc to match only visible lithology? Origin offset would need to be
# checked.

                c[1:-1, 1:-1, 1:-1] = cc

                faces, vtx = MarchingCubes(xx, yy, zz, c, .1)

                if vtx == []:
                    self.lmod1.update_lith_list_reverse()
                    lithtext = self.lmod1.lith_list_reverse[lno]
                    print(lithtext)

                    self.faces[lno] = []
                    self.corners[lno] = []
                    self.norms[lno] = []
                    continue

                self.faces[lno] = faces

                vtx[:, 2] *= -1
                vtx[:, 2] += zz.max()

                self.corners[lno] = vtx[:, [1, 0, 2]] + self.origin

            self.norms[lno] = calc_norms(self.faces[lno], self.corners[lno])

#            ftmp = np.transpose(self.faces[lno])
#            I = np.lexsort(ftmp)
#            self.faces[lno] = self.faces[lno][I]

    def update_model2(self):
        """ Update the 3d model. Faces, nodes and face normals are calculated
        here, from the voxel model. """
        liths = np.unique(self.gdata)
        liths = np.array(liths).astype(int)  # needed for use in faces array
        liths = liths[liths < 900]

        if liths.max() == -1:
            return
        if liths[0] == -1:
            liths = liths[1:]
        if liths[0] == 0:
            liths = liths[1:]

        lut = self.lut[:, [0, 1, 2]]/255.

        vtx = np.array([])
        clr = np.array([])
        nrm = np.array([])
        idx = np.array([])
        idxmax = 0
        lcheck = np.unique(self.lmod1.lith_index)

        self.pbar.setMaximum(liths.size)
        self.pbar.setValue(0)
        tmppval = 0

        for lno in liths:
            tmppval += 1
            self.pbar.setValue(tmppval)

            if lno not in lcheck:
                continue
            if self.corners[lno] == []:
                continue
            if lno in self.sliths:
                clrtmp = lut[lno].tolist()+[1.]
            else:
                clrtmp = lut[lno].tolist()+[self.opac]

            vtx = np.append(vtx, self.corners[lno])
            clr = np.append(clr,
                            np.zeros([self.corners[lno].shape[0], 4])+clrtmp)

            nrm2 = calc_norms(self.faces[lno], self.corners[lno] *
                              [1, 1, self.zmult])

            nrm = np.append(nrm, nrm2)
            idx = np.append(idx, self.faces[lno].flatten()+idxmax)
            idxmax = idx.max()+1

        vtx.shape = (vtx.shape[0]/3, 3)
        clr.shape = (clr.shape[0]/4, 4)

        vtx[:, -1] = (vtx[:, -1]-self.origin[-1])*self.zmult + self.origin[-1]

        cptp = vtx.ptp(0).max()/100.
        cmin = vtx.min(0)
        cptpd2 = vtx.ptp(0)/2.
        vtx = (vtx-cmin-cptpd2)/cptp

        self.glwidget.hastriangles = self.checkbox_smooth.isChecked()
        self.glwidget.cubeVtxArray = vtx
        self.glwidget.cubeClrArray = clr
        self.glwidget.cubeNrmArray = nrm
        self.glwidget.cubeIdxArray = idx.astype(np.uint32)

        self.glwidget.init_object()
        self.glwidget.updateGL()


class GLWidget(QtOpenGL.QGLWidget):
    """ OpenGL Widget """
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.data = None
        self.idx = None
        self.data_buffer = None
        self.indx_buffer = None
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.zoomfactor = 1.0
        self.aspect = 1.
        self.glist = None
        self.hastriangles = False

        self.cubeVtxArray = np.array([[0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0],
                                      [1.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0],
                                      [1.0, 0.0, 1.0],
                                      [1.0, 1.0, 1.0],
                                      [0.0, 1.0, 1.0]])

        self.cubeIdxArray = np.array([0, 1, 2, 3,
                                      3, 2, 6, 7,
                                      1, 0, 4, 5,
                                      2, 1, 5, 6,
                                      0, 3, 7, 4,
                                      7, 6, 5, 4])

        self.cubeClrArray = np.array([[0.0, 0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0],
                                      [1.0, 1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [1.0, 0.0, 1.0, 0.0],
                                      [1.0, 1.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0]])

        self.cubeNrmArray = np.array([[0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0],
                                      [0.0, 0.0, 1.0],
                                      [1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0]])

        self.lastPos = QtCore.QPoint()

    def minimumSizeHint(self):
        """ minimum size hint """
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        """ size hint """
        return QtCore.QSize(400, 400)

    def setXRotation(self, angle):
        """ set X rotation """
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle

    def setYRotation(self, angle):
        """ set Y rotation """
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle

    def setZRotation(self, angle):
        """ set Z rotation """
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle

    def initializeGL(self):
        """ initialize OpenGL """
        ctmp = QtGui.QColor.fromCmykF(0., 0., 0., 0.0)
        self.qglClearColor(ctmp)
        self.initGeometry()

#        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
#        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
#        GL.glShadeModel(GL.GL_SMOOTH)

#############
#        GL.glEnable(GL.GL_NORMALIZE)
# Blend allows transparency

        GL.glEnable(GL.GL_ALPHA_TEST)
        GL.glAlphaFunc(GL.GL_GREATER, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)

        GL.glEnable(GL.GL_CULL_FACE)

        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1., 1., 1., 0.])

##################
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.,1.,1.,1.])
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.,1.,1.,1.])
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.,0.,0.,1.])
#
#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_EMISSION, [0., 1., 0., 1.0])
#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [1., 0., 0., 1.0])
#
#        shininess = 64.
#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, [1., 1., 1., 1.0])
#        GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, shininess);

    def initGeometry(self):
        """ Initialize Geometry """
        self.init_object()

    def init_object(self):
        """ Initialise VBO """
        self.cubeNrmArray.shape = self.cubeVtxArray.shape

        data = np.hstack((self.cubeVtxArray,
                          self.cubeClrArray,
                          self.cubeNrmArray))

        data = data.astype(np.float32)
        idx = self.cubeIdxArray.astype(np.uint32)

        if self.data_buffer is None:
            self.data_buffer = vbo.VBO(data)
            self.indx_buffer = vbo.VBO(idx, target='GL_ELEMENT_ARRAY_BUFFER')
        else:
            self.data_buffer.set_array(data)
            self.indx_buffer.set_array(idx)

    def paintGL(self):
        """ Paint OpenGL """
        float_size = 4
        voff = 0 * float_size
        coff = 3 * float_size
        noff = 7 * float_size
        record_len = 10 * float_size

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.data_buffer.bind()
        self.indx_buffer.bind()

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, record_len, self.data_buffer + voff)
        GL.glColorPointer(4, GL.GL_FLOAT, record_len, self.data_buffer + coff)
        GL.glNormalPointer(GL.GL_FLOAT, record_len, self.data_buffer + noff)

        GL.glLoadIdentity()
        GL.glTranslated(0.0, 0.0, -100.0)
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        if self.hastriangles:
            GL.glDrawElements(GL.GL_TRIANGLES, self.cubeIdxArray.size,
                              GL.GL_UNSIGNED_INT, self.indx_buffer)
        else:
            GL.glDrawElements(GL.GL_QUADS, self.cubeIdxArray.size,
                              GL.GL_UNSIGNED_INT, self.indx_buffer)

        self.data_buffer.unbind()
        self.indx_buffer.unbind()

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)

    def resizeGL(self, width, height):
        """ Resize OpenGL """
        side = min(width, height)
        if side < 0:
            return

        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        self.aspect = width / float(height)
#        GL.glFrustum(-1.0, +1.0, -1.0, 1.0, 1.0, 201.)

        GLU.gluPerspective(70.0*self.zoomfactor, self.aspect, 1.0, 201.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)

    def readPixels(self):
        """ Reads pixels from the window """
        data = GL.glReadPixels(0, 0, self.width(), self.height(), GL.GL_RGB,
                               GL.GL_UNSIGNED_BYTE)
        return data

    def mousePressEvent(self, event):
        """ Mouse Press Event """
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        """ Mouse Move Event """
        dxx = event.x() - self.lastPos.x()
        dyy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dyy)
            self.setYRotation(self.yRot + 8 * dxx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dyy)
            self.setZRotation(self.zRot + 8 * dxx)

        self.updateGL()
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        """ Mouse wheel event """
        self.zoomfactor -= event.delta()/1000.

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(70.0*self.zoomfactor, self.aspect, 1.0, 201.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        self.updateGL()

    def normalizeAngle(self, angle):
        """ Normalize angle """
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle


def calc_norms(faces, vtx):
    """ Calculates normals """

    nrm = np.zeros(vtx.shape, dtype=vtx.dtype)
    tris = vtx[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] -
                 tris[::, 0])
    normalize_v3(n)

    nrm[faces[:, 0]] += n
    nrm[faces[:, 1]] += n
    nrm[faces[:, 2]] += n

    normalize_v3(nrm)

    return nrm


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    lens[lens == 0] = 1  # Get rid of divide by zero.

    arr /= lens[:, np.newaxis]
#    arr[:, 0] /= lens
#    arr[:, 1] /= lens
#    arr[:, 2] /= lens
    return arr


def MarchingCubes(x, y, z, c, iso):
    """
    # function [F,V,col] = MarchingCubes(x,y,z,c,iso,colors)

    # [F,V] = MarchingCubes(X,Y,Z,C,ISO)
    # [F,V,COL] = MarchingCubes(X,Y,Z,C,ISO,COLORS)
    #
    # Use marching cubes algorithm to compute a triangulated mesh of the
    # isosurface within the 3D matrix of scalar values C at isosurface value
    # ISO. The 3D matrices (X,Y,Z) represent a Cartesian, axis-aligned grid
    # specifying the points at which the data C is given. These coordinate
    # arrays must be in the format produced by Matlab's meshgrid function.
    # Output arguments F and V are the face list and vertex list
    # of the resulting triangulated mesh. The orientation of the triangles is
    # chosen such that the normals point from the higher values to the lower
    # values. Optional arguments COLORS ans COLS can be used to produce
    # interpolated mesh face colors. For usage, see Matlab's isosurface.m.
    # To avoid Out of Memory errors when matrix C is large, convert matrices
    # X,Y,Z and C from doubles (Matlab default) to singles (32-bit floats).
    #
    # Adapted for Matlab by Peter Hammer in 2011 based on an
    # Octave function written by Martin Helm <martin@mhelm.de> in 2009
    # http://www.mhelm.de/octave/m/marching_cube.m
    #
    # Revised 30 September, 2011 to add code by Oliver Woodford for removing
    # duplicate vertices.

    #    error('x, y, z, c must be matrices of dim 3')
    #    error('x, y, z, c must be the same size')
    #    error('grid size must be at least 2x2x2')
    #    error('iso needs to be scalar value')
    #    error( 'color must be matrix of same size as c')
    """
    lindex = 4

    [edgeTable, triTable] = GetTables()

    n = np.array(c.shape) - 1  # number of cubes along each direction of image

    # for each cube, assign which edges are intersected by the isosurface
    # 3d array of 8-bit vertex codes
    cc = np.zeros(n, dtype=np.uint16)

    n1 = np.arange(n[0])
    n2 = np.arange(n[1])
    n3 = np.arange(n[2])

    vertex_idx = np.array([[n1, n2, n3],
                           [n1+1, n2, n3],
                           [n1+1, n2+1, n3],
                           [n1, n2+1, n3],
                           [n1, n2, n3+1],
                           [n1+1, n2, n3+1],
                           [n1+1, n2+1, n3+1],
                           [n1, n2+1, n3+1]])

    # loop thru vertices of all cubes

    out = np.zeros(n)
    for ii in range(8):
        # which cubes have vtx ii > iso
        tmp2 = fancyindex(out, c, vertex_idx[ii, 0], vertex_idx[ii, 1],
                          vertex_idx[ii, 2])
        idx = tmp2 > iso
        cc[idx] = bitset(cc[idx], ii)     # for those cubes, turn bit ii on

    # intersected edges for each cube ([n1 x n2 x n3] mtx)
    cedge = edgeTable[cc]
    # voxels which are intersected (col of indcs into cedge)
    iden = np.nonzero(cedge.flatten(order='F'))[0]

    if iden.size == 0:          # all voxels are above or below iso
        print('No such lithology, or all voxels are above or below iso')
        F = []
        V = []
        return F, V

    # calculate the list of intersection points
    xyz_off = np.array([[1, 1, 1],
                        [2, 1, 1],
                        [2, 2, 1],
                        [1, 2, 1],
                        [1, 1, 2],
                        [2, 1, 2],
                        [2, 2, 2],
                        [1, 2, 2]])-1
    edges = np.array([[1, 2], [2, 3], [3, 4], [4, 1],
                      [5, 6], [6, 7], [7, 8], [8, 5],
                      [1, 5], [2, 6], [3, 7], [4, 8]])-1

    offset = sub2ind(c.shape, xyz_off[:, 0], xyz_off[:, 1], xyz_off[:, 2])
    pp = np.zeros([iden.size, lindex, 12])
    ccedge = np.array([cedge.flatten(order='F')[iden], iden])  # uses vec
    ccedge = np.transpose(ccedge)
    ix_offset = 0

    x = x.flatten(order='F')
    y = y.flatten(order='F')
    z = z.flatten(order='F')
    cp = c.flatten(order='F')

    for jj in range(12):
        id__ = bitget(ccedge[:, 0], jj)  # used for logical indexing
        id_ = ccedge[id__, 1]
        ix, iy, iz = ind2sub(cc.shape, id_)
        id_c = sub2ind(c.shape, ix, iy, iz)
        id1 = id_c + offset[edges[jj, 0]]
        id2 = id_c + offset[edges[jj, 1]]

        pp[id__, :3, jj] = InterpolateVertices(iso,
                                               x[id1], y[id1], z[id1],
                                               x[id2], y[id2], z[id2],
                                               cp[id1], cp[id2])
        pp[id__, 3, jj] = np.arange(1, id_.shape[0]+1) + ix_offset

        ix_offset = ix_offset + id_.shape[0]

    pp2 = pp.astype(int)
    # calculate the triangulation from the point list
    F1 = np.array([], dtype=np.int32)
    F2 = np.array([], dtype=np.int32)
    F3 = np.array([], dtype=np.int32)
    tri = triTable[cc.flatten(order='F')[iden]]

    pp2f = pp2.flatten(order='F')

    for jj in range(0, 15, 3):
        id_ = np.nonzero(tri[:, jj] > 0)[0]
        if id_.size > 0:
            V = np.zeros([id_.size, 5], dtype=int)
            V[:, 0] = id_
            V[:, 1] = (lindex-1)*np.ones(id_.shape[0])
            V[:, 2] = tri[id_, jj] - 1
            V[:, 3] = tri[id_, jj+1] - 1
            V[:, 4] = tri[id_, jj+2] - 1

            p1 = sub2ind(pp.shape, V[:, 0], V[:, 1], V[:, 2])
            p2 = sub2ind(pp.shape, V[:, 0], V[:, 1], V[:, 3])
            p3 = sub2ind(pp.shape, V[:, 0], V[:, 1], V[:, 4])

            F1 = np.hstack((F1, pp2f[p1]))
            F2 = np.hstack((F2, pp2f[p2]))
            F3 = np.hstack((F3, pp2f[p3]))

    F = np.transpose([F1, F2, F3])-1
    V = np.zeros([pp2.max(), 3])

    for jj in range(12):
        idp = pp[:, lindex-1, jj] > 0
        if any(idp):
            V[pp2[idp, lindex-1, jj]-1, :3] = pp[idp, :3, jj]

    # Remove duplicate vertices (by Oliver Woodford)
    I = np.lexsort(V.T)
    V = V[I]

    M = np.hstack(([True], np.sum(np.abs(V[1:]-V[:-1]), 1).astype(bool)))

    V = V[M]
    newI = np.zeros_like(I)
    newI[I] = np.cumsum(M)-1
    F = newI[F]

    # Eliminate duplicate faces
#    F.sort(0)
#    F = np.vstack({tuple(row) for row in F})

    return F, V
# ============================================================
# ==================  SUBFUNCTIONS ===========================
# ============================================================


def InterpolateVertices(isolevel, p1x, p1y, p1z, p2x, p2y, p2z, valp1, valp2):
    """Interpolate vertices """
    p = np.zeros([len(p1x), 3])

    eps = np.spacing(1)
    mu = np.zeros(len(p1x))
    iden = abs(valp1-valp2) < (10*eps) * (abs(valp1) + abs(valp2))
    if any(iden):
        p[iden, 0] = p1x[iden]
        p[iden, 1] = p1y[iden]
        p[iden, 2] = p1z[iden]

    nid = np.logical_not(iden)

    if any(nid):
        mu[nid] = (isolevel - valp1[nid]) / (valp2[nid] - valp1[nid])
        p[nid, 0] = p1x[nid] + mu[nid] * (p2x[nid] - p1x[nid])
        p[nid, 1] = p1y[nid] + mu[nid] * (p2y[nid] - p1y[nid])
        p[nid, 2] = p1z[nid] + mu[nid] * (p2z[nid] - p1z[nid])
    return p


@jit
def fancyindex(out, var1, ii, jj, kk):
    """ fancy """

    i1 = -1
    for i in ii:
        i1 += 1
        j1 = -1
        for j in jj:
            j1 += 1
            k1 = -1
            for k in kk:
                k1 += 1
                out[i1, j1, k1] = var1[i, j, k]
    return out


def bitget(byteval, idx):
    """ bitget """
    return (byteval & (1 << idx)) != 0


def bitset(byteval, idx):
    """ bitset """
    return byteval | (1 << idx)


def sub2ind(msize, row, col, layer):
    """ Sub2ind """
    nrows, ncols, _ = msize
#    tmp = layer*ncols*nrows+row*ncols+col
    tmp = layer*ncols*nrows+nrows*col+row
    return tmp.astype(int)


def ind2sub(msize, idx):
    """ Sub2ind """
    nrows, ncols, _ = msize
    layer = idx/(nrows*ncols)
    layer = layer.astype(int)
    idx = idx - layer*nrows*ncols
    col = idx/nrows
    col = col.astype(int)
    row = idx - col*nrows

    return row, col, layer


def GetTables():
    """ Get Tables """
    edgeTable = np.array([0, 265, 515, 778, 1030, 1295, 1541, 1804,
                          2060, 2309, 2575, 2822, 3082, 3331, 3593, 3840,
                          400, 153, 915, 666, 1430, 1183, 1941, 1692,
                          2460, 2197, 2975, 2710, 3482, 3219, 3993, 3728,
                          560, 825, 51, 314, 1590, 1855, 1077, 1340,
                          2620, 2869, 2111, 2358, 3642, 3891, 3129, 3376,
                          928, 681, 419, 170, 1958, 1711, 1445, 1196,
                          2988, 2725, 2479, 2214, 4010, 3747, 3497, 3232,
                          1120, 1385, 1635, 1898, 102, 367, 613, 876,
                          3180, 3429, 3695, 3942, 2154, 2403, 2665, 2912,
                          1520, 1273, 2035, 1786, 502, 255, 1013, 764,
                          3580, 3317, 4095, 3830, 2554, 2291, 3065, 2800,
                          1616, 1881, 1107, 1370, 598, 863, 85, 348,
                          3676, 3925, 3167, 3414, 2650, 2899, 2137, 2384,
                          1984, 1737, 1475, 1226, 966, 719, 453, 204,
                          4044, 3781, 3535, 3270, 3018, 2755, 2505, 2240,
                          2240, 2505, 2755, 3018, 3270, 3535, 3781, 4044,
                          204, 453, 719, 966, 1226, 1475, 1737, 1984,
                          2384, 2137, 2899, 2650, 3414, 3167, 3925, 3676,
                          348, 85, 863, 598, 1370, 1107, 1881, 1616,
                          2800, 3065, 2291, 2554, 3830, 4095, 3317, 3580,
                          764, 1013, 255, 502, 1786, 2035, 1273, 1520,
                          2912, 2665, 2403, 2154, 3942, 3695, 3429, 3180,
                          876, 613, 367, 102, 1898, 1635, 1385, 1120,
                          3232, 3497, 3747, 4010, 2214, 2479, 2725, 2988,
                          1196, 1445, 1711, 1958, 170, 419, 681, 928,
                          3376, 3129, 3891, 3642, 2358, 2111, 2869, 2620,
                          1340, 1077, 1855, 1590, 314, 51, 825, 560,
                          3728, 3993, 3219, 3482, 2710, 2975, 2197, 2460,
                          1692, 1941, 1183, 1430, 666, 915, 153, 400,
                          3840, 3593, 3331, 3082, 2822, 2575, 2309, 2060,
                          1804, 1541, 1295, 1030, 778, 515, 265, 0])

    triTable = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]) + 1

    return [edgeTable, triTable]


def main():
    """ Main routine """

# Create model

#    lmod = quick_model(numx=3, numy=3, numz=3, dxy=100, d_z=100,
#                       tlx=0, tly=0, tlz=0, mht=100, ght=45, finc=90, fdec=0,
#                       inputliths=['Generic'], susc=[0.01], dens=[3.0])
#
#    lmod.lith_index[2, 2, 2] = 1


#    faces, norms, corners = trimain(lmod.lith_index, 0.1)
#

    c = np.zeros([5, 5, 5])
    c[1:4, 1:4, 1:4] = 1
    c = zoom(c, 1, order=1)

    x = np.arange(c.shape[1])
    y = np.arange(c.shape[0])
    z = np.arange(c.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    faces, vtx = MarchingCubes(xx, yy, zz, c, .5)

#    x = np.linspace(0, 2, 20)
#    y = np.linspace(0, 2, 20)
#    z = np.linspace(0, 2, 20)
#    xx, yy, zz = np.meshgrid(x, y, z)
#    c = (xx-.5)**2 + (yy-.5)**2 + (zz-.5)**2
#    faces, vtx = MarchingCubes(xx, yy, zz, c, .5)

    app = QtGui.QApplication(sys.argv)
    wid = Mod3dDisplay()
    wid.setWindowState(wid.windowState() & ~QtCore.Qt.WindowMinimized |
                       QtCore.Qt.WindowActive)

    # this will activate the window

###############################
    faces = np.array(faces)
    vtx = np.array(vtx)
# Create a zeroed array with the same type and shape as our vertices i.e.,
# per vertex normal
    norm = np.zeros(vtx.shape, dtype=vtx.dtype)
# Create an indexed view into the vertex array using the array of three indices
# for triangles
    tris = vtx[faces]
# Calculate the normal for all the triangles, by taking the cross product of
# the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
# n is now an array of normals per triangle. The length of each normal is
# dependent the vertices, we need to normalize these, so that our next step
# weights each normal equally.
    normalize_v3(n)
#    n[2] *= -1
# now we have a normalized array of normals, one per triangle, i.e., per
# triangle normals. But instead of one per triangle (i.e., flat shading), we
# add to each vertex in that triangle, the triangles' normal. Multiple
# triangles would then contribute to every vertex, so we need to normalize
# again afterwards. The cool part, we can actually add the normals through an
# indexed view of our (zeroed) per vertex normal array.

    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

# Now we have a vertex array, vertices, a normal array, norm, and the index
# array, faces, and we are ready to pass it on to our rendering algorithm.
# To render without the index list, we create a flattened array where
# the triangle indices are replaced with the actual vertices.

    cptp = vtx.ptp(0).max()/100
    cmin = vtx.min(0)
    cptpd2 = vtx.ptp(0)/2.
    vtx = (vtx-cmin-cptpd2)/cptp

    wid.glwidget.hastriangles = True
    wid.glwidget.cubeVtxArray = vtx
    wid.glwidget.cubeClrArray = (np.zeros([vtx.shape[0], 4]) +
                                 np.array([0.9, 0.4, 0.0, 0.5]))
    wid.glwidget.cubeNrmArray = norm

#    ftmp = np.transpose(faces)
#    I = np.lexsort(ftmp)
#    faces = faces[I]

    wid.glwidget.cubeIdxArray = faces.flatten().astype(np.uint32)

# This activates the opengl stuff

#    wid.glwidget.init_object()
    print('widshow')
    wid.show()
#    wid.activateWindow()

#    wid.glwidget.updateGL()
#
#    wid.run()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
