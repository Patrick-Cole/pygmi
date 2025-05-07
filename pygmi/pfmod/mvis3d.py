# -----------------------------------------------------------------------------
# Name:        mvid3d.py (part of PyGMI)
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
"""Code for the 3d model creation."""

import os
import sys
import numpy as np

from PyQt6 import QtCore, QtWidgets

from scipy.ndimage import convolve
from numba import jit
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor

from pygmi.pfmod import misc
from pygmi.misc import ContextModule


class Mod3dDisplay(ContextModule):
    """
    Widget class to call the main interface.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lmod1 = None
        self.outdata = self.indata

        if hasattr(parent, 'showtext'):
            self.showtext = parent.showtext
        else:
            self.showtext = sys.stdout

        self.setWindowTitle('3D Model Display')

        self.corners = []
        self.faces = {}
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
        self.pvmesh = None
        self.light = pv.Light()

        # Back to normal stuff
        self.lw_3dmod_defs = QtWidgets.QListWidget()
        self.pb_save = QtWidgets.QPushButton('Save to Image File (JPG or PNG)')
        self.pb_resetlight = QtWidgets.QPushButton('Reset Light')
        self.pb_refresh = QtWidgets.QPushButton('Refresh Model')
        self.cb_smooth = QtWidgets.QCheckBox('Smooth Model')
        self.cb_ortho = QtWidgets.QCheckBox('Orthographic Projection')
        self.cb_axis = QtWidgets.QCheckBox('Display Axis')
        self.pbar = QtWidgets.QProgressBar()
        self.plotter = QtInteractor(self)  # , lighting='none')

        self.vslider_3dmodel = QtWidgets.QSlider()
        self.msc = MySunCanvas(self)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.buttonbox.buttonbox.hide()
        self.buttonbox.htmlfile = 'pfmod.cm.show3dmodel'
        hbl = QtWidgets.QHBoxLayout(self)
        vbl_cmodel = QtWidgets.QVBoxLayout()
        vbl = QtWidgets.QVBoxLayout()

        self.vslider_3dmodel.setMinimum(1)
        self.vslider_3dmodel.setMaximum(20)
        self.vslider_3dmodel.setTickInterval(1)
        self.vslider_3dmodel.setOrientation(QtCore.Qt.Orientation.Vertical)
        vbl_cmodel.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                           QtWidgets.QSizePolicy.Policy.Fixed)

        sizepolicy_pb = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Maximum)

        self.lw_3dmod_defs.setSizePolicy(sizepolicy)
        self.lw_3dmod_defs.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.lw_3dmod_defs.setFixedWidth(220)
        self.cb_smooth.setSizePolicy(sizepolicy)
        self.pb_save.setSizePolicy(sizepolicy_pb)
        self.pb_refresh.setSizePolicy(sizepolicy_pb)
        self.pbar.setOrientation(QtCore.Qt.Orientation.Vertical)

        self.cb_ortho.setChecked(True)
        self.cb_axis.setChecked(True)

        vbl.addWidget(self.lw_3dmod_defs)
        vbl.addWidget(QtWidgets.QLabel('Light Position:'))
        vbl.addWidget(self.msc)
        vbl.addWidget(self.pb_resetlight)
        vbl.addWidget(self.cb_smooth)
        # vbl.addWidget(self.cb_ortho)
        # vbl.addWidget(self.cb_axis)
        vbl.addWidget(self.pb_save)
        vbl.addWidget(self.pb_refresh)
        vbl.addWidget(self.buttonbox)
        vbl_cmodel.addWidget(self.plotter)
        hbl.addWidget(self.vslider_3dmodel)
        hbl.addLayout(vbl_cmodel)
        hbl.addLayout(vbl)
        hbl.addWidget(self.pbar)

        self.lw_3dmod_defs.clicked.connect(self.change_defs)
        self.vslider_3dmodel.valueChanged.connect(self.mod3d_vs)
        self.pb_save.clicked.connect(self.save)
        self.pb_refresh.clicked.connect(self.run)
        self.pb_resetlight.clicked.connect(self.resetlight)
        self.cb_smooth.stateChanged.connect(self.update_plot)
        self.cb_ortho.stateChanged.connect(self.update_model2)
        self.cb_axis.stateChanged.connect(self.update_model2)
        self.msc.figure.canvas.mpl_connect('button_press_event', self.sunclick)

    def closeEvent(self, QCloseEvent):
        """
        Close event.

        Parameters
        ----------
        QCloseEvent : TYPE
            Close event.

        Returns
        -------
        None.

        """
        super().closeEvent(QCloseEvent)
        self.plotter.close()

    def save(self):
        """
        Save a jpg.

        Returns
        -------
        None.

        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'JPG (*.jpg);;PNG (*.png)')
        if filename == '':
            return
        os.chdir(os.path.dirname(filename))

        self.plotter.screenshot(filename)

    def update_for_kmz(self):
        """
        Update for the kmz file.

        Returns
        -------
        None.

        """
        self.gpoints = self.corners
        self.gnorms = self.norms
        self.gfaces = {}

        if list(self.faces.values())[0].shape[1] == 4:
            for i in self.faces:
                self.gfaces[i] = np.append(self.faces[i][:, :-1],
                                           self.faces[i][:, [0, 2, 3]])
                self.gfaces[i].shape = (int(self.gfaces[i].shape[0]/3), 3)
        else:
            self.gfaces = self.faces.copy()

        self.glutlith = range(1, len(self.gfaces)+1)

    def change_defs(self):
        """
        List widget routine.

        Returns
        -------
        None.

        """
        if not self.lmod1.lith_list:
            return
        self.set_selected_liths()
        self.update_color()
        QtWidgets.QApplication.processEvents()

    def data_init(self):
        """
        Initialise data.

        Returns
        -------
        None.

        """
        self.outdata = self.indata

    def set_selected_liths(self):
        """
        Set the selected lithologies.

        Returns
        -------
        None.

        """
        item = self.lw_3dmod_defs.currentItem()

        if item.text()[0] == ' ':
            item.setText('\u2713' + item.text()[1:])
        else:
            item.setText(' ' + item.text()[1:])

        itxt = []
        for i in range(self.lw_3dmod_defs.count()):
            item = self.lw_3dmod_defs.item(i)
            if '\u2713' in item.text():
                itxt.append(item.text()[2:])

        lith = [self.lmod1.lith_list[j] for j in itxt]
        lith3d = [j.lith_index for j in lith]

        self.sliths = np.intersect1d(self.gdata, lith3d)

    def mod3d_vs(self):
        """Vertical slider used to scale 3d view."""
        self.plotter.set_scale(zscale=self.vslider_3dmodel.value())

    def resetlight(self):
        """
        Reset light to the current model position.

        Returns
        -------
        None.

        """
        self.msc.init_graph()

        elev = 45
        azim = 45
        self.light.set_direction_angle(elev, azim)

    def sunclick(self, event):
        """
        Sunclick event is used to track changes to the sunshading.

        Parameters
        ----------
        event - matplotlib button press event
             event returned by matplotlib when a button is pressed
        """
        if event.inaxes == self.msc.axes:
            self.msc.sun.set_xdata([event.xdata])
            self.msc.sun.set_ydata([event.ydata])
            self.msc.figure.canvas.draw()

            azim = np.rad2deg(event.xdata)
            elev = -np.rad2deg(event.ydata)

            self.light.set_direction_angle(elev, azim)

    def update_color(self):
        """
        Update colour only.

        Returns
        -------
        None.

        """
        liths = np.unique(self.gdata)
        liths = liths[liths > 0]

        if liths.size == 0:
            return

        lut = self.lut[:, :3].astype(np.uint8)

        lcheck = np.unique(self.lmod1.lith_index)

        clr = []

        for lno in liths:
            if lno not in lcheck:
                continue
            if len(self.corners[lno]) == 0:
                continue
            if lno in self.sliths:
                clrtmp = lut[lno].tolist()+[255]
            else:
                clrtmp = lut[lno].tolist()+[0]

            clr = np.append(clr, self.faces[lno].shape[0]*[clrtmp])

        clr.shape = (clr.shape[0]//4, 4)
        clr = clr.astype(np.uint8)

        self.pvmesh['clr'] = clr
        # breakpoint()

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Model3D' not in self.indata:
            self.showlog('No 3D model. You may need to execute that '
                         'module first')
            return False

        self.lmod1 = self.indata['Model3D'][0]

        liths = np.unique(self.lmod1.lith_index[::1, ::1, ::-1])
        liths = np.array(liths).astype(int)  # needed for use in faces array
        liths = liths[liths > 0]

        if liths.size == 0:
            self.showlog('No 3D model. You need to draw in at least '
                         'part of a lithology first.')
            return False

        misc.update_lith_lw(self.lmod1, self.lw_3dmod_defs)
        for i in range(self.lw_3dmod_defs.count()-1, -1, -1):
            if self.lw_3dmod_defs.item(i).text() == 'Background':
                self.lw_3dmod_defs.takeItem(i)

        for i in range(self.lw_3dmod_defs.count()):
            item = self.lw_3dmod_defs.item(i)
            # item.setSelected(True)
            item.setText('\u2713 ' + item.text())

        self.show()
        self.lw_3dmod_defs.setFocus()
        self.update_plot()

        return True

    def update_plot(self):
        """
        Update 3D model.

        Returns
        -------
        None.

        """
        QtWidgets.QApplication.processEvents()

        # Update 3D model
        self.spacing = [self.lmod1.dxy, self.lmod1.dxy, self.lmod1.d_z]
        self.origin = [self.lmod1.xrange[0], self.lmod1.yrange[0],
                       self.lmod1.zrange[0]]
        self.gdata = self.lmod1.lith_index[::1, ::1, ::-1]

        # update colors
        i = self.lw_3dmod_defs.findItems('*',
                                         QtCore.Qt.MatchFlag.MatchWildcard)
        itxt = [j.text()[2:] for j in i]
        itmp = []
        for i in itxt:
            itmp.append(self.lmod1.lith_list[i].lith_index)

        itmp = np.sort(itmp)
        tmp = np.ones((255, 4))*255

        for i in itmp:
            tmp[i, :3] = self.lmod1.mlut[i]

        self.lut = tmp

        self.set_selected_liths()

        if self.lmod1.lith_list:
            self.set_selected_liths()
            self.update_model()
            points, cells, clr = self.update_model2()
            if points is None:
                return

            if self.cb_smooth.isChecked():
                psides = 3
                ctype = pv.CellType.TRIANGLE
            else:
                psides = 4
                ctype = pv.CellType.QUAD
            numcells = cells.size//psides
            cells.shape = (numcells, psides)

            tmp = np.full(numcells, psides).reshape(-1, 1)
            cells = np.hstack([tmp, cells]).ravel()

            celltypes = np.full(numcells, ctype, dtype=np.uint8)

            self.pvmesh = pv.UnstructuredGrid(cells, celltypes, points)
            self.pvmesh['clr'] = clr

            self.plotter.clear()
            self.plotter.enable_depth_peeling()
            self.plotter.add_mesh(self.pvmesh, scalars='clr', rgb=True,
                                  name='mymesh')

            self.plotter.set_scale(zscale=self.vslider_3dmodel.value())
            self.plotter.add_axes()

            elev = 45
            azim = 45
            self.light.set_direction_angle(elev, azim)
            self.plotter.add_light(self.light)
            # self.plotter.show_grid(use_2d=True)

    def update_model(self, issmooth=None):
        """
        Update the 3d model.

        Faces, nodes and face normals are calculated here, from the voxel
        model.

        Parameters
        ----------
        issmooth : bool, optional
            Flag to indicate a smooth model. The default is None.

        Returns
        -------
        None.

        """
        QtWidgets.QApplication.processEvents()

        if issmooth is None:
            issmooth = self.cb_smooth.isChecked()

        self.faces = {}
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
            xx, yy, zz = np.meshgrid(x, y, z)

            # Set up Gaussian smoothing filter
            ix, iy, iz = np.mgrid[-1:2, -1:2, -1:2]
            sigma = 2
            cci = np.exp(-(ix**2+iy**2+iz**2)/(3*sigma**2))

        tmppval = 0
        for lno in liths:
            tmppval += 1
            self.pbar.setValue(tmppval)
            QtWidgets.QApplication.processEvents()

            if lno not in lcheck:
                continue
            if not issmooth:
                gdat2 = tmpdat.copy()
                gdat2[gdat2 != lno] = -0.5
                gdat2[gdat2 == lno] = 0.5
                newcorners, newfaces = updatemod(gdat2, cindx, cloc)

                self.faces[lno] = newfaces
                self.corners[lno] = newcorners

            else:
                c = np.zeros(nshape)

                cc = self.lmod1.lith_index.copy()
                cc[cc != lno] = 0
                cc[cc == lno] = 1

                cc = convolve(cc, cci)/cci.size

# shrink cc to match only visible lithology? Origin offset would need to be
# checked.

                c[1:-1, 1:-1, 1:-1] = cc

                faces, vtx = MarchingCubes(xx, yy, zz, c, .1,
                                           showlog=self.showlog)

                if vtx.size == 0:
                    self.lmod1.update_lith_list_reverse()

                    self.faces[lno] = []
                    self.corners[lno] = []

                    continue

                self.faces[lno] = faces

                vtx[:, 2] *= -1
                vtx[:, 2] += zz.max()

                self.corners[lno] = vtx[:, [1, 0, 2]] + self.origin

    def update_model2(self):
        """
        Update the 3d model part 2.

        Returns
        -------
        None.

        """
        liths = np.unique(self.gdata)
        liths = np.array(liths).astype(int)  # needed for use in faces array
        liths = liths[liths < 900]

        if liths.max() == -1:
            return None, None, None
        if liths[0] == -1:
            liths = liths[1:]
        if liths[0] == 0:
            liths = liths[1:]

        lut = (self.lut[:, [0, 1, 2]]).astype(np.uint8)

        vtx = np.array([])
        clr = np.array([])
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
            if len(self.corners[lno]) == 0:
                continue
            if lno in self.sliths:
                clrtmp = lut[lno].tolist()+[255]
            else:
                clrtmp = lut[lno].tolist()+[0]

            vtx = np.append(vtx, self.corners[lno])
            clr = np.append(clr, self.faces[lno].shape[0]*[clrtmp])

            idx = np.append(idx, self.faces[lno].flatten()+idxmax)
            idxmax = idx.max()+1

        vtx.shape = (vtx.shape[0]//3, 3)
        clr.shape = (clr.shape[0]//4, 4)
        clr = clr.astype(np.uint8)

        vtx[:, -1] = (vtx[:, -1]-self.origin[-1])*self.zmult + self.origin[-1]

        cptp = np.ptp(vtx, 0).max()/100.
        cmin = vtx.min(0)
        cptpd2 = np.ptp(vtx, 0)/2.
        vtx = (vtx-cmin-cptpd2)/cptp

        idx = idx.astype(np.uint32)

        return vtx, idx, clr


class MySunCanvas(FigureCanvasQTAgg):
    """
    Canvas for the sunshading tool.

    Parameters
    ----------
    parent : parent, optional
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    sun: matplotlib plot instance
        plot of a circle 'o' showing where the sun is
    axes: matplotlib axes instance
        axes on which the sun is drawn
    """

    def __init__(self, parent=None):
        fig = Figure(layout='constrained')
        super().__init__(fig)

        self.sun = None
        self.axes = fig.add_subplot(111, polar=True)
        fig.set_facecolor('None')

        self.setMaximumSize(200, 200)
        self.setMinimumSize(120, 120)

        self.init_graph()

    def init_graph(self):
        """
        Initialise graph.

        Returns
        -------
        None.

        """
        self.axes.clear()
        self.axes.xaxis.set_tick_params(labelsize=6)
        self.axes.tick_params(labelleft=False, labelright=False)

        self.axes.set_autoscaley_on(False)
        self.axes.set_rmax(np.pi/2.)
        self.axes.set_rmin(-np.pi/2.)
        self.axes.set_yticks([-np.pi/4, 0, np.pi/4])
        self.axes.tick_params(axis='x', pad=0)

        self.sun, = self.axes.plot(np.pi/4., -np.pi/4., 'yo')
        self.figure.canvas.draw()


def updatemod(gdat2, cindx, cloc):
    """
    Update model without smooothing.

    Parameters
    ----------
    gdat2 : numpy array
        Model values.
    cindx : numpy array
        Corner index.
    cloc : numpy array
        Corner location.

    Returns
    -------
    newcorners : numpy array
        New corner coordinates.
    newfaces : numpy array
        New face indices.

    """
    newfaces = []

    ndiff = gdat2[:, :, 1:]-gdat2[:, :, :-1]

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

    ndiff = gdat2[:, 1:, :]-gdat2[:, :-1, :]
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

    ndiff = gdat2[1:, :, :]-gdat2[:-1, :, :]
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
    newfaces.shape = (newfaces.size//4, 4)

    return newcorners, newfaces


def MarchingCubes(x, y, z, c, iso, *, showlog=print):
    """
    Marching cubes.

    Use marching cubes algorithm to compute a triangulated mesh of the
    isosurface within the 3D matrix of scalar values C at isosurface value
    ISO. The 3D matrices (X,Y,Z) represent a Cartesian, axis-aligned grid
    specifying the points at which the data C is given. These coordinate
    arrays must be in the format produced by Matlab's meshgrid function.
    Output arguments F and V are the face list and vertex list
    of the resulting triangulated mesh. The orientation of the triangles is
    chosen such that the normals point from the higher values to the lower
    values. Optional arguments COLORS ans COLS can be used to produce
    interpolated mesh face colours. For usage, see Matlab's isosurface.m.
    To avoid Out of Memory errors when matrix C is large, convert matrices
    X,Y,Z and C from doubles (Matlab default) to singles (32-bit floats).

    Originally Adapted for Matlab by Peter Hammer in 2011 based on an
    Octave function written by Martin Helm <martin@mhelm.de> in 2009
    http://www.mhelm.de/octave/m/marching_cube.m

    Revised 30 September, 2011 to add code by Oliver Woodford for removing
    duplicate vertices.

    Parameters
    ----------
    x : numpy array
        X coordinates.
    y : numpy array
        Y coordinates.
    z : numpy array
        Z coordinates.
    c : numpy array
        Data.
    iso : float
        Isosurface level.
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    F : numpy array
        Face list.
    V : numpy array
        Vertex list.

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
                           [n1, n2+1, n3+1]], dtype=object)

    # loop through vertices of all cubes

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
        showlog('Warning: No such lithology, or all voxels are above '
                'or below iso')
        F = np.array([])
        V = np.array([])
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

    return F, V


def InterpolateVertices(isolevel, p1x, p1y, p1z, p2x, p2y, p2z, valp1, valp2):
    """
    Interpolate vertices.

    Parameters
    ----------
    isolevel : float
        ISO level.
    p1x : numpy array
        p1 x coordinate.
    p1y : numpy array
        p1 y coordinate.
    p1z : numpy array
        p1 z coordinate.
    p2x : numpy array
        p2 x coordinate.
    p2y : numpy array
        p2 y coordinate.
    p2z : numpy array
        p2 z coordinate.
    valp1 : numpy array
        p1 value.
    valp2 : numpy array
        p2 value.

    Returns
    -------
    p : numpy array
        Interpolated vertices.

    """
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


@jit(nopython=True)
def fancyindex(out, var1, ii, jj, kk):
    """
    Fancy index.

    Parameters
    ----------
    out : numpy array
        Input data.
    var1 : numpy array
        Input data.
    ii : numpy array
        i indices.
    jj : numpy array
        j indices.
    kk : numpy array
        k indices.

    Returns
    -------
    out : numpy array
        Output data with new values.

    """
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
    """
    Bit get.

    Parameters
    ----------
    byteval : int
        Input value to get bit from.
    idx : int
        Position of bit to get.

    Returns
    -------
    bool
        True if not 0, False otherwise.

    """
    return (byteval & (1 << idx)) != 0


def bitset(byteval, idx):
    """
    Bit set.

    Parameters
    ----------
    byteval : int
        Input value to get bit from.
    idx : int
        Position of bit to get.

    Returns
    -------
    int
        Output value with bit set.

    """
    return byteval | (1 << idx)


def sub2ind(msize, row, col, layer):
    """
    Sub to index.

    Parameters
    ----------
    msize : tuple
        Tuple with number of rows and columns as first two elements.
    row : int
        Row.
    col : int
        Column.
    layer : numpy array
        Layer.

    Returns
    -------
    tmp : numpy array
        Index returned.

    """
    nrows, ncols, _ = msize
    tmp = layer*ncols*nrows+nrows*col+row
    return tmp.astype(int)


def ind2sub(msize, idx):
    """
    Index to sub.

    Parameters
    ----------
    msize : tuple
        Tuple with number of rows and columns as first two elements.
    idx : numpy array
        Array of indices.

    Returns
    -------
    row : int
        Row.
    col : int
        Column.
    layer : numpy array
        Layer.

    """
    nrows, ncols, _ = msize
    layer = idx/(nrows*ncols)
    layer = layer.astype(int)
    idx = idx - layer*nrows*ncols
    col = idx/nrows
    col = col.astype(int)
    row = idx - col*nrows

    return row, col, layer


def GetTables():
    """
    Get tables.

    Returns
    -------
    list
        A list with edgetable and tritable.

    """
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


def _testfn():
    """Test function."""
    from pygmi.pfmod.iodefs import ImportMod3D

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    ifile = r'C:/Workdata/modelling/Magmodel_Upper22km_AveAll_diapir_withDeepDens_newdens.npz'

    IM = ImportMod3D()
    IM.ifile = ifile
    IM.settings(True)

    print('Model loaded')

    M3D = Mod3dDisplay()
    M3D.indata = IM.outdata
    M3D.data_init()
    M3D.run()
    M3D.exec()


def _testfn2():
    """Test function."""
    from pygmi.pfmod.iodefs import ImportMod3D

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    ifile = r'C:/Workdata/modelling/Magmodel_Upper22km_AveAll_diapir_withDeepDens_newdens.npz'

    IM = ImportMod3D()
    IM.ifile = ifile
    IM.settings(True)

    print('Model loaded')

    lmod1 = IM.outdata['Model3D'][0]

    values = lmod1.lith_index[::1, ::1, ::-1]

    opac = values.copy()
    opac[opac>0] = 1.
    opac[opac<1] = 0.

    # opac= opac.flatten()

    # Create the spatial reference
    grid = pv.ImageData()
    grid.dimensions = values.shape

    grid.origin = (lmod1.xrange[0], lmod1.yrange[0], lmod1.zrange[0])
    grid.spacing = (lmod1.dxy, lmod1.dxy, lmod1.d_z)


    grid.point_data['values'] = values.flatten(order='F')  # Flatten the array
    grid.point_data['opac'] = opac.flatten(order='F')  # Flatten the array


    # breakpoint()
    # Now plot the grid
    # grid.plot(show_edges=False)
    vol = grid
    p = pv.Plotter()
    # p.add_mesh_slice(vol)
    p.add_mesh_clip_plane(grid, scalars='values', opacity='opac')


    p.show()


if __name__ == "__main__":
    _testfn2()
