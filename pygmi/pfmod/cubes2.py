""" From http://michelanders.blogspot.com/2012/02/
               marching-tetrahedrons-in-python.html, accessed on 2014/11/13"""

# pylint: disable=C0103

import numpy as np
from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL
from OpenGL import GLU
import misc
import os
from math import cos, exp, atan2
import sys
from pygmi.pfmod.grvmag3d import quick_model
from pygmi.pfmod.grvmag3d import calc_field


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

        if hasattr(parent, 'showtext'):
            self.showtext = parent.showtext
        else:
            self.showtext = print

        if hasattr(parent, 'showprocesslog'):
            self.showprocesslog = parent.showprocesslog
        else:
            self.showprocesslog = print

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
            self.showprocesslog('No 3D model. You may need to execute' +
                                ' that module first')
            return False

        self.lmod1 = self.indata['Model3D'][0]
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
            # self.pbars.resetsub(maximum=liths.size)
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
        liths = liths[liths < 900]

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

#############
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glAlphaFunc(GL.GL_GREATER, 0.1)
        GL.glEnable(GL.GL_ALPHA_TEST)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
#        GL.glEnable(GL.GL_CULL_FACE)

        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_LIGHTING)
#        GL.glEnable(GL.GL_LIGHT0)
#        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1., 1., 1., 0.])

##################
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

        GL.glDrawElementsui(GL.GL_TRIANGLES, self.cubeIdxArray)

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


class Vector:  # struct XYZ
    """ Vector Class """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.x)+" "+str(self.y)+" "+str(self.z)


class Gridcell:  # struct GRIDCELL
    """ Gridcell """
    def __init__(self, p, n, val):
        self.p = p   # p=[8]
        self.n = n   # n=[8]
        self.val = val  # val=[8]


class Triangle:  # struct TRIANGLE
    """ Triangle """
    def __init__(self, p1, p2, p3):
        self.p = [p1, p2, p3]  # vertices

# return triangle as an ascii STL facet
    def __str__(self):
        return """facet normal 0 0 0
outer loop
vertex %s
vertex %s
vertex %s
endloop
endfacet""" % (self.p[0], self.p[1], self.p[2])


# return a 3d list of values
def readdata(f=lambda x, y, z: x*x+y*y+z*z, size=5.0, steps=11):
    """ Readdata """
    m = int(steps/2)
    ki = []
    for i in range(steps):
        kj = []
        for j in range(steps):
            kd = []
            for k in range(steps):
                kd.append(f(size*(i-m)/m, size*(j-m)/m, size*(k-m)/m))
            kj.append(kd)
        ki.append(kj)
    return ki


def lobes(x, y, z):
    """ Lobes """
    try:
        theta = atan2(x, y)         # sin t = o
    except:
        theta = 0
    try:
        phi = atan2(z, y)
    except:
        phi = 0
    r = x*x+y*y+z*z
    ct = cos(theta)
    cp = cos(phi)
    return ct*ct*cp*cp*exp(-r/10)


def trimain(data=None, isolevel=0.1):
    """ Tri Main """

    if data is None:
        data = readdata(lobes, 5, 41)

    # print(data)

    triangles = []
    for i in range(len(data)-1):
        for j in range(len(data[i])-1):
            for k in range(len(data[i][j])-1):
                p = [None]*8
                val = [None]*8
                # print(i,j,k)
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

    faces = []
    norms = []
    corners = []

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
    """t000F"""
    return []


def t0E01(g, iso, v0, v1, v2, v3):
    """t0E01"""
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                   VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                   VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]))
    return [tri]


def t0D02(g, iso, v0, v1, v2, v3):
    """t0D02"""
    tri = Triangle(VertexInterp(iso, g.p[v1], g.p[v0], g.val[v1], g.val[v0]),
                   VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]),
                   VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]))
    return [tri]


def t0C03(g, iso, v0, v1, v2, v3):
    """t0C03"""
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]),
                   VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                   VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]))
    return [tri,
            Triangle(tri.p[2],
                     VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]),
                     tri.p[1])]


def t0B04(g, iso, v0, v1, v2, v3):
    """t0B04"""
    tri = Triangle(VertexInterp(iso, g.p[v2], g.p[v0], g.val[v2], g.val[v0]),
                   VertexInterp(iso, g.p[v2], g.p[v1], g.val[v2], g.val[v1]),
                   VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]))
    return [tri]


def t0A05(g, iso, v0, v1, v2, v3):
    """t0A05"""
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                   VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]),
                   VertexInterp(iso, g.p[v0], g.p[v3], g.val[v0], g.val[v3]))
    tri2 = Triangle(tri.p[0],
                    VertexInterp(iso, g.p[v1], g.p[v2], g.val[v1], g.val[v2]),
                    tri.p[1])
    return [tri, tri2]


def t0906(g, iso, v0, v1, v2, v3):
    """t0906"""
    tri = Triangle(VertexInterp(iso, g.p[v0], g.p[v1], g.val[v0], g.val[v1]),
                   VertexInterp(iso, g.p[v1], g.p[v3], g.val[v1], g.val[v3]),
                   VertexInterp(iso, g.p[v2], g.p[v3], g.val[v2], g.val[v3]))
    return [tri,
            Triangle(tri.p[0],
                     VertexInterp(iso, g.p[v0], g.p[v2], g.val[v0], g.val[v2]),
                     tri.p[2])]


def t0708(g, iso, v0, v1, v2, v3):
    """t0708"""
    tri = Triangle(VertexInterp(iso, g.p[v3], g.p[v0], g.val[v3], g.val[v0]),
                   VertexInterp(iso, g.p[v3], g.p[v2], g.val[v3], g.val[v2]),
                   VertexInterp(iso, g.p[v3], g.p[v1], g.val[v3], g.val[v1]))
    return [tri]

trianglefs = {7: t0708, 8: t0708, 9: t0906, 6: t0906, 10: t0A05, 5: t0A05,
              11: t0B04, 4: t0B04, 12: t0C03, 3: t0C03, 13: t0D02, 2: t0D02,
              14: t0E01, 1: t0E01, 0: t000F, 15: t000F}


def PolygoniseTri(g, iso, v0, v1, v2, v3):
    """Polygonise Tri"""
#    triangles = []

    #   Determine which of the 16 cases we have given which vertices
    #   are above or below the isosurface

    triindex = 0
    if g.val[v0] < iso:
        triindex |= 1
    if g.val[v1] < iso:
        triindex |= 2
    if g.val[v2] < iso:
        triindex |= 4
    if g.val[v3] < iso:
        triindex |= 8

    return trianglefs[triindex](g, iso, v0, v1, v2, v3)


def VertexInterp(isolevel, p1, p2, valp1, valp2):
    """ Vertex Interp """
    if abs(isolevel-valp1) < 0.00001:
        return p1
    if abs(isolevel-valp2) < 0.00001:
        return p2
    if abs(valp1-valp2) < 0.00001:
        return p1
    mu = (isolevel - valp1) / (valp2 - valp1)
    return Vector(p1.x + mu * (p2.x - p1.x),
                  p1.y + mu * (p2.y - p1.y),
                  p1.z + mu * (p2.z - p1.z))


def MarchingCubes(x, y, z, c, iso, colors=None):
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

    calc_cols = False
    lindex = 4

    if colors is not None:
        calc_cols = True
        lindex = 5

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
    for ii in range(8):
        # which cubes have vtx ii > iso
        idx = c[np.ix_(vertex_idx[ii, 0], vertex_idx[ii, 1],
                       vertex_idx[ii, 2])] > iso
        cc[idx] = bitset(cc[idx], ii)     # for those cubes, turn bit ii on
#        cc[idx] = np.bitwise_or(cc[idx], int(str(10**ii), 2))

    # intersected edges for each cube ([n1 x n2 x n3] mtx)
    cedge = edgeTable[cc+1]
    # voxels which are intersected (col of indcs into cedge)
    iden = np.nonzero(cedge)
    if iden[0].size == 0:          # all voxels are above or below iso
        F = []
        V = []
        col = []
        return F, V, col

    # calculate the list of intersection points
    xyz_off = [[1, 1, 1],
               [2, 1, 1],
               [2, 2, 1],
               [1, 2, 1],
               [1, 1, 2],
               [2, 1, 2],
               [2, 2, 2],
               [1, 2, 2]]
    edges = [[1, 2], [2, 3], [3, 4], [4, 1],
             [5, 6], [6, 7], [7, 8], [8, 5],
             [1, 5], [2, 6], [3, 7], [4, 8]]

    offset = sub2ind(c.shape, xyz_off[:, 1], xyz_off[:, 2], xyz_off[:, 3]) - 1
    pp = np.zeros(iden.size, lindex, 12)
    ccedge = np.array([cedge(iden), iden])
    ix_offset = 0
    for jj in range(12):
        id__ = bool(bitget(ccedge[:, 1], jj))  # used for logical indexing
        id_ = ccedge[id__, 2]
        [ix, iy, iz] = ind2sub(cc.shape, id_)
        id_c = sub2ind(c.shape, ix, iy, iz)
        id1 = id_c + offset(edges(jj, 1))
        id2 = id_c + offset(edges(jj, 2))
        if calc_cols:
            pp[id__, 0:5, jj] = [InterpolateVertices(iso, x(id1), y(id1),
                                 z(id1), x(id2), y(id2), z(id2), c(id1),
                                 c(id2), colors(id1), colors(id2)),
                                 np.arange(id_.shape[0]).T + ix_offset]
        else:
            pp[id__, 0:4, jj] = [InterpolateVertices(iso, x(id1), y(id1),
                                 z(id1), x(id2), y(id2), z(id2), c(id1),
                                 c(id2)), np.arange(id_.shape[9]).T +
                                 ix_offset]
            ix_offset = ix_offset + np.shape(id_, 1)

    # calculate the triangulation from the point list
    F = []
    tri = triTable[cc[iden]+1, :]
    for jj in range(0, 15, 3):
        id_ = np.nonzero(tri[:, jj] > 0)
        V = [id_, lindex*np.ones(np.shape(id_, 1), 1), tri[id_, jj:jj+2]]
        if V.size > 0:
            p1 = sub2ind(pp.shape, V[:, 1], V[:, 2], V[:, 3])
            p2 = sub2ind(pp.shape, V[:, 1], V[:, 2], V[:, 4])
            p3 = sub2ind(pp.shape, V[:, 1], V[:, 2], V[:, 5])
            F = [[F], [pp[p1], pp[p2], pp[p3]]]

    V = []
    col = []
    for jj in range(12):
        idp = pp[:, lindex, jj] > 0
        if any(idp):
            V[pp[idp, lindex, jj], 0:3] = pp[idp, 1:3, jj]
            if calc_cols:
                col[pp[idp, lindex, jj], 1] = pp[idp, 4, jj]

    # Remove duplicate vertices (by Oliver Woodford)
    [V, I] = V.sort(1)
    M = [[True], [any(np.diff(V), 2)]]
    V = V[M, :]
    I[I] = np.cumsum(M)
    F = I[F]

    return F, V, col
# ============================================================
# ==================  SUBFUNCTIONS ===========================
# ============================================================


def InterpolateVertices(isolevel, p1x, p1y, p1z, p2x, p2y, p2z, valp1, valp2,
                        col1=None, col2=None):
    """Interpolate vertices """
    if col1 is None:
        p = np.zeros(len(p1x), 3)
    else:
        p = np.zeros(len(p1x), 4)

    eps = np.spacing(1)
    mu = np.zeros(len(p1x), 1)
    iden = abs(valp1-valp2) < (10*eps) * (abs(valp1) + abs(valp2))
    if any(iden):
        p[iden, 0:3] = [p1x(iden), p1y(iden), p1z(iden)]
        if col1 is not None:
            p[iden, ] = col1[iden]

    nid = not iden
    if any(nid):
        mu[nid] = (isolevel - valp1[nid]) / (valp2[nid] - valp1[nid])
        p[nid, 0:3] = ([p1x[nid] + mu[nid] * (p2x[nid] - p1x[nid]),
                        p1y[nid] + mu[nid] * (p2y[nid] - p1y[nid]),
                        p1z[nid] + mu[nid] * (p2z[nid] - p1z[nid])])
        if col1 is not None:
            p[nid, 4] = col1[nid] + mu[nid] * (col2[nid] - col1[nid])
    return p


def bitget(byteval, idx):
    """ bitget """
    return np.bitwise_and(byteval, (1 << idx))


def bitset(byteval, idx):
    """ bitset """
    return np.bitwise_or(byteval, (1 << idx))


def sub2ind(msize, row, col, layer):
    """ Sub2ind """
    nrows, ncols, nlayers = msize
    tmp = row*ncols*nlayers+col*nlayers+layer
    return tmp


def ind2sub(msize, idx):
    """ Sub2ind """
    nrows, ncols, nlayers = msize
    layer = int(idx/(nrows*ncols))
    idx = idx - layer*nrows*ncols
    row = int(idx/ncols)
    col = idx - row*ncols

    return row, col, layer


def GetTables():
    edgeTable = np.array([0,     265,  515,  778, 1030, 1295, 1541, 1804,
                          2060, 2309, 2575, 2822, 3082, 3331, 3593, 3840,
                          400,   153,  915,  666, 1430, 1183, 1941, 1692,
                          2460, 2197, 2975, 2710, 3482, 3219, 3993, 3728,
                          560,   825,   51,  314, 1590, 1855, 1077, 1340,
                          2620, 2869, 2111, 2358, 3642, 3891, 3129, 3376,
                          928,   681,  419,  170, 1958, 1711, 1445, 1196,
                          2988, 2725, 2479, 2214, 4010, 3747, 3497, 3232,
                          1120, 1385, 1635, 1898,  102,  367,  613,  876,
                          3180, 3429, 3695, 3942, 2154, 2403, 2665, 2912,
                          1520, 1273, 2035, 1786,  502,  255, 1013,  764,
                          3580, 3317, 4095, 3830, 2554, 2291, 3065, 2800,
                          1616, 1881, 1107, 1370,  598,  863,   85,  348,
                          3676, 3925, 3167, 3414, 2650, 2899, 2137, 2384,
                          1984, 1737, 1475, 1226,  966,  719,  453,  204,
                          4044, 3781, 3535, 3270, 3018, 2755, 2505, 2240,
                          2240, 2505, 2755, 3018, 3270, 3535, 3781, 4044,
                          204,   453,  719,  966, 1226, 1475, 1737, 1984,
                          2384, 2137, 2899, 2650, 3414, 3167, 3925, 3676,
                          348,    85,  863,  598, 1370, 1107, 1881, 1616,
                          2800, 3065, 2291, 2554, 3830, 4095, 3317, 3580,
                          764,  1013,  255,  502, 1786, 2035, 1273, 1520,
                          2912, 2665, 2403, 2154, 3942, 3695, 3429, 3180,
                          876,   613,  367,  102, 1898, 1635, 1385, 1120,
                          3232, 3497, 3747, 4010, 2214, 2479, 2725, 2988,
                          1196, 1445, 1711, 1958,  170,  419,  681,  928,
                          3376, 3129, 3891, 3642, 2358, 2111, 2869, 2620,
                          1340, 1077, 1855, 1590,  314,   51,  825,  560,
                          3728, 3993, 3219, 3482, 2710, 2975, 2197, 2460,
                          1692, 1941, 1183, 1430,  666,  915,  153,  400,
                          3840, 3593, 3331, 3082, 2822, 2575, 2309, 2060,
                          1804, 1541, 1295, 1030,  778,  515,  265,    0])

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
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])+1

    return [edgeTable, triTable]


if __name__ == "__main__":

    lmod = quick_model(numx=5, numy=5, numz=5, dxy=100, d_z=100,
                       tlx=0, tly=0, tlz=0, mht=100, ght=45, finc=90, fdec=0,
                       inputliths=['Generic'], susc=[0.01], dens=[3.0])

    lmod.lith_index[1:4, 1:4, 1:4] = 1

    app = QtGui.QApplication(sys.argv)
    wid = Mod3dDisplay()

    wid.setWindowState(wid.windowState() & ~QtCore.Qt.WindowMinimized |
                       QtCore.Qt.WindowActive)

    faces, norms, corners = trimain(lmod.lith_index, 0.1)

    vtx = corners
    cptp = vtx.ptp(0).max()/100.
    cmin = vtx.min(0)
    cptpd2 = vtx.ptp(0)/2.
    vtx = (vtx-cmin-cptpd2)/cptp

    wid.glwidget.cubeVtxArray = vtx
    wid.glwidget.cubeClrArray = (np.zeros([corners.shape[0], 4]) +
                                 np.array([0.9, 0.4, 0.0, 1.0]))
    wid.glwidget.cubeNrmArray = np.ones([corners.shape[0], 4])
    wid.glwidget.cubeIdxArray = faces.flatten()

    # this will activate the window
    wid.show()
    wid.activateWindow()

    wid.glwidget.updatelist()
    wid.glwidget.updateGL()

#    wid.run()

    sys.exit(app.exec_())
