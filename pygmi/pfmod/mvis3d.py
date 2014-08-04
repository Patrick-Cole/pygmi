# -----------------------------------------------------------------------------
# Name:        mvis3d.py (part of PyGMI)
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

# pylint: disable=E1101, C0103
import numpy as np
from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL
from OpenGL import GLU
import pygmi.pfmod.misc as misc
import os
# import pygmi.ptimer


class Mod3dDisplay(QtGui.QDialog):
    """ Widget class to call the main interface """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.parent = parent
#        self.lmod1 = parent.lmod1
        self.lmod1 = None
        self.indata = {}
        self.outdata = self.indata
#        self.outdata = {}

# mayavi vars
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
        self.pbars = None
        self.gfaces = []
        self.gpoints = []
        self.gnorms = []
        self.glutlith = []
        self.demsurf = None
        self.qdiv = 0
        self.mesh = {}
        self.opac = 0.5

# Back to normal stuff
        self.userint = self
        self.gridlayout = QtGui.QGridLayout(self)
        self.dial_3dmod = QtGui.QDial(self)
        self.lw_3dmod_defs = QtGui.QListWidget(self)
        self.label = QtGui.QLabel(self)
        self.label2 = QtGui.QLabel(self)
        self.vbox_cmodel = QtGui.QVBoxLayout()
        self.vslider_3dmodel = QtGui.QSlider(self)
        self.pb_save = QtGui.QPushButton(self)
        self.pb_refresh = QtGui.QPushButton(self)
        self.checkbox_bg = QtGui.QCheckBox(self)
        self.setupui()

    # GL Widget
        self.glwidget = GLWidget()
        self.vbox_cmodel.addWidget(self.glwidget)

    # Buttons
        self.lw_3dmod_defs.clicked.connect(self.change_defs)
        self.vslider_3dmodel.sliderReleased.connect(self.mod3d_vs)
        self.dial_3dmod.sliderMoved.connect(self.opacity)
        self.pb_save.clicked.connect(self.save)
        self.pb_refresh.clicked.connect(self.run)
        self.checkbox_bg.stateChanged.connect(self.change_defs)

    def setupui(self):
        """ Setup UI """
# Column 0
        self.vslider_3dmodel.setMinimum(1)
        self.vslider_3dmodel.setMaximum(1000)
        self.vslider_3dmodel.setOrientation(QtCore.Qt.Vertical)
        self.gridlayout.addWidget(self.vslider_3dmodel, 0, 0, 5, 1)

# Column 1
        self.gridlayout.addLayout(self.vbox_cmodel, 0, 1, 5, 1)

# Column 4
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Fixed)
        self.lw_3dmod_defs.setSizePolicy(sizepolicy)
        self.lw_3dmod_defs.setSelectionMode(
            QtGui.QAbstractItemView.MultiSelection)
        self.gridlayout.addWidget(self.lw_3dmod_defs, 0, 4, 1, 1)

        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setText("Background Model Opacity")
        self.gridlayout.addWidget(self.label, 1, 4, 1, 1)

        self.checkbox_bg.setText("Include Background")
        self.gridlayout.addWidget(self.checkbox_bg, 3, 4, 1, 1)

        self.dial_3dmod.setMaximum(255)
        self.dial_3dmod.setProperty("value", 127)
        self.dial_3dmod.setNotchesVisible(True)
        self.gridlayout.addWidget(self.dial_3dmod, 2, 4, 1, 1)
        self.gridlayout.addWidget(self.pb_save, 4, 4, 1, 1)
        self.gridlayout.addWidget(self.pb_refresh, 5, 4, 1, 1)
        self.pb_save.setText("Save to Image File (JPG or PNG)")
        self.pb_refresh.setText("Refresh Model")

    def save(self):
        """ This saves a jpg """
        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'JPG (*.jpg);;PNG (*.png)')
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])

        ftype = 'JPG'

        if 'PNG' in filename:
            ftype = 'PNG'

        text, ok = QtGui.QInputDialog.getText(self, "3D Model",
                                              "Enter pixmap size:",
                                              QtGui.QLineEdit.Normal,
                                              "%d x %d" %
                                              (self.glwidget.width(),
                                               self.glwidget.height()))

        if not ok:
            return
        size = QtCore.QSize()

        regExp = QtCore.QRegExp("([0-9]+) *x *([0-9]+)")

        if regExp.exactMatch(text):
            width = int(regExp.cap(1))
            height = int(regExp.cap(2))
            size = QtCore.QSize(width, height)

        if size.isValid():
            tmp = self.glwidget.renderPixmap(size.width(), size.height())
            if ftype == 'JPG':
                tmp.save(filename, ftype, 100)
            else:
                tmp.save(filename, ftype)

    def update_for_kmz(self):
        """ Updates for the kmz file """

        self.gpoints = self.corners
        self.gnorms = self.norms
        self.gfaces = {}
#        self.gfaces = self.faces
        for i in list(self.faces.keys()):
            self.gfaces[i] = np.append(self.faces[i][:, :-1],
                                       self.faces[i][:, [0, 2, 3]])
            self.gfaces[i].shape = (self.gfaces[i].shape[0]/3, 3)

#        self.gpoints = []
#        self.gnorms = []
#        self.gfaces = []
#        self.glutlith = []
#
#
#        for i in list(self.faces.keys()):
#            self.gfaces = np.append(self.gfaces, self.faces[i])
#            self.gpoints = np.append(self.gpoints, self.corners[i])
#            self.gnorms = np.append(self.gnorms, self.norms[i])
#
#        self.gfaces.shape = (self.gfaces.shape[0]/4, 4)
#        self.gpoints.shape = (self.gpoints.shape[0]/3, 3)
#        self.gnorms.shape = (self.gnorms.shape[0]/3, 3)

#        facestmp = np.array([i[1] for i in self.faces.items() if i[0] != 0])
#        normstmp = self.norms
#
#        nrms = []
#        crns = []
#        fcs = []
#        for i in facestmp:
#            fmin = np.min(i)
#            fmax = np.max(i)+1
#            nrms.append(normstmp[fmin:fmax])
#            crns.append(self.corners[fmin:fmax])
#            fnew = i-fmin
#            fcs.append(fnew)
#
#        self.gpoints = crns
#        self.gfaces = fcs
#        self.gnorms = nrms

        self.glutlith = range(1, len(list(self.gfaces.keys()))+1)

    def change_defs(self):
        """ List box routine """
        self.defs()

    def data_init(self):
        """ Data initialisation routine """
        self.outdata = self.indata

    def defs(self, fcalc=False):
        """ List box in layer tab for definitions """
        if len(self.lmod1.lith_list.keys()) == 0:
            return

        itxt = self.get_selected()
        lith = [self.lmod1.lith_list[j] for j in itxt]
        lith3d = [j.lith_index for j in lith]

        self.sliths = np.intersect1d(self.gdata, lith3d)
        self.update_plot2(fullcalc=fcalc)

    def get_selected(self):
        """ gets selected items on the list widget """
        i = self.lw_3dmod_defs.selectedItems()

        itxt = [j.text() for j in i]
        return itxt

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
        self.defs()
#        self.update_plot(fullcalc=False)

    def opacity(self):
        """ Dial to change opacity of background """
        perc = (float(self.dial_3dmod.value()) /
                float(self.dial_3dmod.maximum()))

        self.opac = perc
        self.update_model2()

#        self.glwidget.cubeClrArray[:, -1] = perc
#        self.glwidget.updateGL()

    def run(self):
        """ Process data """
        if 'Model3D' not in self.indata:
            self.parent.showprocesslog('No 3D model. You may need to execute' +
                                       ' that module first')
            return False

        self.lmod1 = self.indata['Model3D']
        liths = np.unique(self.lmod1.lith_index[::1, ::1, ::-1])
        liths = np.array(liths).astype(int)  # needed for use in faces array
        if liths[0] == -1:
            liths = liths[1:]
        if liths[0] == 0:
            liths = liths[1:]
        if liths.size == 0:
            self.parent.showprocesslog('No 3D model. You need to draw in at' +
                                       ' least part of a lithology first.')
            return False

        self.show()
        misc.update_lith_lw(self.lmod1, self.lw_3dmod_defs)
        self.update_plot()
        return True

    def update_plot(self):
        """ Update 3D Model """

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

        self.defs(fcalc=True)

    def update_plot2(self, fullcalc=True):
        """ Update plot """

# Update the model if necessary
        if fullcalc is True:
            self.update_model()

        self.update_model2()

    def update_model(self):
        """ Update the 3d model. Faces, nodes and face normals are calculated
        here, from the voxel model. """
        QtGui.QApplication.processEvents()

        self.faces = {}
        self.norms = {}
        self.corners = {}

        liths = np.unique(self.gdata)
        liths = np.array(liths).astype(int)  # needed for use in faces array
        if liths.max() == -1:
            return
        if liths[0] == -1:
            liths = liths[1:]

        if self.pbars is not None:
#            self.pbars.resetsub(maximum=liths.size)
            self.pbars.resetall(maximum=liths.size, mmax=2)

#        for lno in liths:
#            self.faces[lno], self.norms[lno], self.corners[lno] = main(
#                self.gdata, lno)
#            if self.pbars is not None:
#                self.pbars.incr()

        igd, jgd, kgd = self.gdata.shape
        cloc = np.indices(((kgd+1), (jgd+1), (igd+1))).T.reshape(
            (igd+1)*(jgd+1)*(kgd+1), 3).T[::-1].T
        cloc = cloc * self.spacing + self.origin
        cindx = np.arange(cloc.size/3, dtype=int)
        cindx.shape = (igd+1, jgd+1, kgd+1)

        tmpdat = np.zeros([igd+2, jgd+2, kgd+2])-1
        tmpdat[1:-1, 1:-1, 1:-1] = self.gdata

        for lno in liths:
            gdat2 = tmpdat.copy()
            gdat2[gdat2 != lno] = -0.5
            gdat2[gdat2 == lno] = 0.5

            newfaces = []
            cnrm = np.zeros_like(cloc)

# Face order may have to be reversed if normal is negative.

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
            cnrm[ccc] += [0, 0, -1]
            newfaces = np.append(newfaces, ccc)

            c_1 = cindx[nd1 == -1]
            c_2 = cindx[nd2 == -1]
            c_3 = cindx[nd3 == -1]
            c_4 = cindx[nd4 == -1]
            ccc = np.transpose([c_1, c_2, c_3, c_4])
            cnrm[ccc] += [0, 0, 1]
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
            cnrm[ccc] += [0, -1, 0]
            newfaces = np.append(newfaces, ccc)

            c_1 = cindx[nd1 == -1]
            c_2 = cindx[nd2 == -1]
            c_3 = cindx[nd3 == -1]
            c_4 = cindx[nd4 == -1]
            ccc = np.transpose([c_1, c_4, c_3, c_2])
            cnrm[ccc] += [0, 1, 0]
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
            cnrm[ccc] += [-1, 0, 0]
            newfaces = np.append(newfaces, ccc)

            c_1 = cindx[nd1 == -1]
            c_2 = cindx[nd2 == -1]
            c_3 = cindx[nd3 == -1]
            c_4 = cindx[nd4 == -1]
            ccc = np.transpose([c_1, c_4, c_3, c_2])
            cnrm[ccc] += [1, 0, 0]
            newfaces = np.append(newfaces, ccc)

            uuu, i = np.unique(newfaces, return_inverse=True)
            uuu = uuu.astype(int)
            n_f = np.arange(uuu.size)
            newfaces = n_f[i]
            newcorners = cloc[uuu]
            newnorms = cnrm[uuu]
            newfaces.shape = (newfaces.size/4, 4)

            self.faces[lno] = newfaces
            self.corners[lno] = newcorners

            aaa = np.sqrt(np.sum(np.square(newnorms), 1))
            aaa[aaa == 0] = 1.
            newnorms /= aaa[:, np.newaxis]
            self.norms[lno] = newnorms

            if self.pbars is not None:
                self.pbars.incr()

    def update_model2(self):
        """ Update the 3d model. Faces, nodes and face normals are calculated
        here, from the voxel model. """

        liths = np.unique(self.gdata)
        liths = np.array(liths).astype(int)  # needed for use in faces array

        if liths.max() == -1:
            return
        if liths[0] == -1:
            liths = liths[1:]
        if not self.checkbox_bg.isChecked() and liths[0] == 0:
            liths = liths[1:]

#        if self.sliths.size > 0:
#            liths = self.sliths.tolist()

        lut = self.lut[:, [0, 1, 2]]/255.

        vtx = np.array([])
        clr = np.array([])
        nrm = np.array([])
        idx = np.array([])
        idxmax = 0

        for lno in self.sliths:
            vtx = np.append(vtx, self.corners[lno])
            clrtmp = lut[lno].tolist()+[1.0]
            clr = np.append(clr,
                            np.zeros([self.corners[lno].shape[0], 4])+clrtmp)
            nrm = np.append(nrm, self.norms[lno])
            idx = np.append(idx, self.faces[lno].flatten()+idxmax)
            idxmax = idx.max()+1

        for lno in liths:
            if lno in self.sliths:
                continue
            vtx = np.append(vtx, self.corners[lno])
            clrtmp = lut[lno].tolist()+[self.opac]
            clr = np.append(clr,
                            np.zeros([self.corners[lno].shape[0], 4])+clrtmp)
            nrm = np.append(nrm, self.norms[lno])
            idx = np.append(idx, self.faces[lno].flatten()+idxmax)
            idxmax = idx.max()+1

        vtx.shape = (vtx.shape[0]/3, 3)
        clr.shape = (clr.shape[0]/4, 4)
        vtx[:, -1] = (vtx[:, -1]-self.origin[-1])*self.zmult + self.origin[-1]

        cptp = vtx.ptp(0).max()/100.
        cmin = vtx.min(0)
        cptpd2 = vtx.ptp(0)/2.
        vtx = (vtx-cmin-cptpd2)/cptp

        self.glwidget.cubeVtxArray = vtx
        self.glwidget.cubeClrArray = clr
        self.glwidget.cubeNrmArray = nrm
        self.glwidget.cubeIdxArray = idx

        self.glwidget.updatelist()
        self.glwidget.updateGL()


class GLWidget(QtOpenGL.QGLWidget):
    """ OpenGL Widget """
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.zoomfactor = 1.0
        self.aspect = 1.
        self.glist = None

        self.cubeVtxArray = [[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [0.0, 1.0, 1.0]]

        self.cubeIdxArray = [0, 1, 2, 3,
                             3, 2, 6, 7,
                             1, 0, 4, 5,
                             2, 1, 5, 6,
                             0, 3, 7, 4,
                             7, 6, 5, 4]

        self.cubeClrArray = [[0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0, 0.0],
                             [1.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0]]

        self.cubeNrmArray = [[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]]

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
            self.updateGL()

    def setYRotation(self, angle):
        """ set Y rotation """
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.updateGL()

    def setZRotation(self, angle):
        """ set Z rotation """
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.updateGL()

    def initializeGL(self):
        """ initialize OpenGL """
        ctmp = QtGui.QColor.fromCmykF(0., 0., 0., 0.0)
        self.qglClearColor(ctmp)
        self.initGeometry()

#        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
#        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
#        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glAlphaFunc(GL.GL_GREATER, 0.1)
        GL.glEnable(GL.GL_ALPHA_TEST)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)

        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1., 1., 1., 0.])

#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.,1.,1.,1.])
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.,1.,1.,1.])
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.,0.,0.,1.])

#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_EMISSION, [0., 1., 0., 1.0])
#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, [1., 0., 0., 1.0])

#        shininess = 64.
#        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, [1., 1., 1., 1.0])
#        GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, shininess);

    def initGeometry(self):
        """ Initialize Geometry """
        self.updatelist()

    def updatelist(self):
        """ Updates the list """
        if self.glist is not None:
            GL.glDeleteLists(self.glist, 1)

        self.glist = GL.glGenLists(1)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointerf(self.cubeVtxArray)
        GL.glNormalPointerf(self.cubeNrmArray)
        GL.glColorPointerf(self.cubeClrArray)

        GL.glNewList(self.glist, GL.GL_COMPILE)

        GL.glDrawElementsui(GL.GL_QUADS, self.cubeIdxArray)

        GL.glEndList()
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)

    def paintGL(self):
        """ Paint OpenGL """
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslated(0.0, 0.0, -100.0)
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        GL.glCallList(self.glist)

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


class Vector(object):  # struct XYZ
    """ Vector Class """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.x)+" "+str(self.y)+" "+str(self.z)


class Gridcell(object):  # struct GRIDCELL
    """ Gridcell class """
    def __init__(self, p, n, val):
        self.p = p   # p=[8]
        self.n = n   # n=[8]
        self.val = val  # val=[8]


class Triangle(object):  # struct TRIANGLE
    """ Triangle Class """
    def __init__(self, p1, p2, p3):
        self.p = [p1, p2, p3]  # vertices


def main(data, isolevel):
    """ Main class for marching """
#    tt = ptimer.PTime()
# print(data)
    faces = []
    norms = []
    corners = []

    triangles = []
    for i in range(len(data)-1):
        print(str(i)+' of  '+str(len(data)-1))
#        tt.since_last_call()
        for j in range(len(data[i])-1):
            for k in range(len(data[i][j])-1):
                p = [None]*8
                val = [None]*8

                p[0] = Vector(i, j, k)
                val[0] = data[i][j][k]
                p[1] = Vector(i+1, j, k)
                val[1] = data[i+1][j][k]
                p[2] = Vector(i+1, j+1, k)
                val[2] = data[i+1][j+1][k]
                p[3] = Vector(i, j+1, k)
                val[3] = data[i][j+1][k]
                p[4] = Vector(i, j, k+1)
                val[4] = data[i][j][k+1]
                p[5] = Vector(i+1, j, k+1)
                val[5] = data[i+1][j][k+1]
                p[6] = Vector(i+1, j+1, k+1)
                val[6] = data[i+1][j+1][k+1]
                p[7] = Vector(i, j+1, k+1)
                val[7] = data[i][j+1][k+1]
                grid = Gridcell(p, [], val)
                triangles.extend(PolygoniseTri(grid, isolevel, 0, 2, 3, 7))
                triangles.extend(PolygoniseTri(grid, isolevel, 0, 2, 6, 7))
                triangles.extend(PolygoniseTri(grid, isolevel, 0, 4, 6, 7))
                triangles.extend(PolygoniseTri(grid, isolevel, 0, 6, 1, 2))
                triangles.extend(PolygoniseTri(grid, isolevel, 0, 6, 1, 4))
                triangles.extend(PolygoniseTri(grid, isolevel, 5, 6, 1, 4))

    j = 0
    for i in triangles:
        faces.append([j+2, j+1, j])
        corners.append([i.p[0].x, i.p[0].y, i.p[0].z])
        corners.append([i.p[1].x, i.p[1].y, i.p[1].z])
        corners.append([i.p[2].x, i.p[2].y, i.p[2].z])
        v1 = np.subtract(corners[-1], corners[-2])
        v2 = np.subtract(corners[-2], corners[-3])
        n1 = np.cross(v1, v2)
        n1 /= np.sqrt(np.dot(n1, n1))

        norms.append(n1)
        norms.append(n1)
        norms.append(n1)
        j += 3

    faces = np.array(faces)
    norms = np.array(norms)
    corners = np.array(corners)

    return faces, norms, corners


def t000F(g, iso, v0, v1, v2, v3):
    """ t000F """
    return []


def t0E01(g, iso, v0, v1, v2, v3):
    """ t0E01 """
    return [Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                     VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                     VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]))
            ]


def t0D02(g, iso, v0, v1, v2, v3):
    """ t0D02 """
    return [Triangle(VertexInterp(iso, g.p[v1], g.p[v0], g.val[v1], g.val[v0]),
                     VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]),
                     VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]))
            ]


def t0C03(g, iso, v0, v1, v2, v3):
    """ t0C03 """
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]),
                   VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                   VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]))
    return [tri,
            Triangle(tri.p[2],
                     VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]),
                     tri.p[1])
            ]


def t0B04(g, iso, v0, v1, v2, v3):
    """ t0B04 """
    return [Triangle(VertexInterp(iso, g.p[v2], g.p[v0], g.val[v2], g.val[v0]),
                     VertexInterp(iso, g.p[v2], g.p[v1], g.val[v2], g.val[v1]),
                     VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]))
            ]


def t0A05(g, iso, v0, v1, v2, v3):
    """ t0A05 """
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                   VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]),
                   VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]))
    return [tri,
            Triangle(tri.p[1],
                     VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]),
                     tri.p[0])
            ]


def t0906(g, iso, v0, v1, v2, v3):
    """ t0906 """
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                   VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]),
                   VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]))
    return [tri,
            Triangle(tri.p[0],
                     VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                     tri.p[2])
            ]


def t0708(g, iso, v0, v1, v2, v3):
    """ t0708 """
    return [Triangle(VertexInterp(iso, g.p[v3], g.p[v0], g.val[v3], g.val[v0]),
                     VertexInterp(iso, g.p[v3], g.p[v2], g.val[v3], g.val[v2]),
                     VertexInterp(iso, g.p[v3], g.p[v1], g.val[v3], g.val[v1]))
            ]

trianglefs = {7: t0708, 8: t0708, 9: t0906, 6: t0906, 10: t0A05, 5: t0A05,
              11: t0B04, 4: t0B04, 12: t0C03, 3: t0C03, 13: t0D02, 2: t0D02,
              14: t0E01, 1: t0E01, 0: t000F, 15: t000F}


def PolygoniseTri(g, iso, v0, v1, v2, v3):
    """ PolygoniseTri """
#    triangles = []

#   Determine which of the 16 cases we have given which vertices
#   are above or below the isosurface

    triindex = 0
    if g.val[v0] == iso:
        triindex |= 1
    if g.val[v1] == iso:
        triindex |= 2
    if g.val[v2] == iso:
        triindex |= 4
    if g.val[v3] == iso:
        triindex |= 8

    return trianglefs[triindex](g, iso, v0, v1, v2, v3)


def VertexInterp(isolevel, p1, p2, valp1, valp2):
    """ VertexInterp """
    if isolevel == valp1:
        return p1
    if isolevel == valp2:
        return p2
    if valp1 == valp2:
        return p1
    mu = (isolevel - valp1) / (valp2 - valp1)
    return Vector(p1.x + mu * (p2.x - p1.x), p1.y + mu * (p2.y - p1.y),
                  p1.z + mu * (p2.z - p1.z))
