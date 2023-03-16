# -----------------------------------------------------------------------------
# Name:        super_class.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
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
"""Supervised Classification tool."""

import os
import sys
import copy
import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.patches import Polygon as mPolygon
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from pygmi.raster.datatypes import Data
from pygmi.misc import frm, BasicModule
from pygmi.raster.modest_image import imshow


class GraphMap(FigureCanvasQTAgg):
    """
    Graph Map.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        self.figure = Figure()

        super().__init__(self.figure)
        self.setParent(parent)

        self.parent = parent
        self.polyi = None
        self.data = []
        self.cdata = []
        self.mindx = [0, 0]
        self.csp = None
        self.subplot = None

    def init_graph(self):
        """
        Initialise the graph.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        self.figure.clf()
        self.subplot = self.figure.add_subplot(111)

        self.csp = imshow(self.subplot, dat.data, extent=dat.extent,
                          cmap=colormaps['jet'])
        axes = self.figure.gca()

        if dat.crs is not None:
            crs = dat.crs.to_dict()
        else:
            crs = {}

        if 'proj' in crs and crs['proj'] == 'longlat':
            axes.set_xlabel('Longitude')
            axes.set_ylabel('Latitude')
        else:
            axes.set_xlabel('Eastings')
            axes.set_ylabel('Northings')

        axes.xaxis.set_major_formatter(frm)
        axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()

    def polyint(self):
        """
        Polygon integrator.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]].data

        xtmp = np.arange(dat.shape[1])
        ytmp = np.arange(dat.shape[0])
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        xmesh = np.ma.array(xmesh, dtype=float, mask=dat.mask)
        ymesh = np.ma.array(ymesh, dtype=float, mask=dat.mask)
        xmesh = xmesh.flatten()
        ymesh = ymesh.flatten()
        xmesh = xmesh.filled(np.nan)
        ymesh = ymesh.filled(np.nan)
        pntxy = np.transpose([xmesh, ymesh])
        self.polyi = PolygonInteractor(self.subplot, pntxy)

    def update_graph(self):
        """
        Update graph.

        Returns
        -------
        None.

        """
        mtmp = self.mindx
        dat = self.data[mtmp[0]]

        if mtmp[1] > 0:
            cdat = self.cdata[mtmp[1] - 1].data
            self.csp.set_data(cdat)
            self.csp.set_clim(cdat.min(), cdat.max())
            self.csp.set_clim(cdat.min(), cdat.max())
        else:
            self.csp.set_data(dat.data)
            self.csp.set_clim(dat.data.min(), dat.data.max())
            self.csp.set_clim(dat.data.min(), dat.data.max())

        self.csp.changed()
        self.figure.canvas.draw()


class PolygonInteractor(QtCore.QObject):
    """
    Polygon Interactor for the supervised classification tool.

    Parameters
    ----------
        showverts : bool
        epsilon : int
        polyi_changed : signal

    """

    showverts = True
    epsilon = 5
    polyi_changed = QtCore.pyqtSignal(list)  #: polygon changed signal.

    def __init__(self, axtmp, pntxy):
        super().__init__()
        self.ax = axtmp
        self.poly = mPolygon([(1, 1)], animated=True)
        self.ax.add_patch(self.poly)
        self.canvas = self.poly.figure.canvas
        self.poly.set_alpha(0.5)
        self.pntxy = pntxy
        self.background = None
        self.isactive = False

        xtmp, ytmp = zip(*self.poly.xy)

        self.line = Line2D(xtmp, ytmp, marker='o', markerfacecolor='r',
                           color='y', animated=True)
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self.motion_notify_callback)

    def draw_callback(self, event=None):
        """
        Draw callback.

        Parameters
        ----------
        event : TYPE, optional
            Unused. The default is None.

        Returns
        -------
        None.

        """
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        if self.isactive is False:
            return

        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)

    def new_poly(self, npoly=None):
        """
        Create new polygon.

        Parameters
        ----------
        npoly : list or None, optional
            New polygon coordinates.

        Returns
        -------
        None.

        """
        if npoly is None:
            npoly = [[1, 1]]
        self.poly.set_xy(npoly)
        self.line.set_data(zip(*self.poly.xy))

        self.update_plots()
        self.canvas.draw()

    def poly_changed(self, poly):
        """
        Polygon changed.

        Parameters
        ----------
        poly : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # this method is called whenever the polygon object is called
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Get the index of vertex under point if within epsilon tolerance.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        ind : int or None
            Index of vertex under point.

        """
        # display coords
        xytmp = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xytmp)
        xtt, ytt = xyt[:, 0], xyt[:, 1]
        dtt = np.sqrt((xtt - event.x) ** 2 + (ytt - event.y) ** 2)
        indseq = np.nonzero(np.equal(dtt, np.amin(dtt)))[0]
        ind = indseq[0]

        if dtt[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        """
        Button press callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.isactive is False:
            return

        if self.ax.get_navigate_mode() is not None:
            return

        self._ind = self.get_ind_under_point(event)

        if self._ind is None:
            xys = self.poly.get_transform().transform(self.poly.xy)
            ptmp = self.poly.get_transform().transform([event.xdata,
                                                        event.ydata])

            if len(xys) == 1:
                self.poly.xy = np.array(
                    [(event.xdata, event.ydata)] +
                    [(event.xdata, event.ydata)])
                self.line.set_data(zip(*self.poly.xy))

                self.ax.draw_artist(self.poly)
                self.ax.draw_artist(self.line)
                self.canvas.update()
                return
            dmin = -1
            imin = -1
            for i in range(len(xys) - 1):
                s0tmp = xys[i]
                s1tmp = xys[i + 1]
                dtmp = dist_point_to_segment(ptmp, s0tmp, s1tmp)

                if dmin == -1:
                    dmin = dtmp
                    imin = i
                elif dtmp < dmin:
                    dmin = dtmp
                    imin = i
            i = imin

            if np.array_equal(self.poly.xy, np.ones((2, 2))):
                self.poly.set_xy([[event.xdata, event.ydata]])
            else:
                self.poly.xy = np.array(list(self.poly.xy[:i + 1]) +
                                        [(event.xdata, event.ydata)] +
                                        list(self.poly.xy[i + 1:]))

            self.line.set_data(list(zip(*self.poly.xy)))

            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.poly)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)

    def button_release_callback(self, event):
        """
        Button release callback.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if event.button != 1:
            return
        if self.isactive is False:
            return
        self._ind = None
        self.update_plots()

    def update_plots(self):
        """
        Update plots.

        Returns
        -------
        None.

        """
        if self.poly.xy.size < 8:
            return
        self.polyi_changed.emit(self.poly.xy.tolist())

    def motion_notify_callback(self, event):
        """
        Motion notify on mouse movement.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        xtmp, ytmp = event.xdata, event.ydata

        self.poly.xy[self._ind] = xtmp, ytmp
        if self._ind == 0:
            self.poly.xy[-1] = xtmp, ytmp

        self.line.set_data(list(zip(*self.poly.xy)))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


class SuperClass(BasicModule):
    """Main Supervised Classification Tool Routine."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m1 = 0
        self.c = [0, 1, 0]
        self.m = [0, 0]
        self.df = None

        self.map = GraphMap(self)
        self.dpoly = QtWidgets.QPushButton('Delete Polygon')
        self.apoly = QtWidgets.QPushButton('Add Polygon')
        self.combo = QtWidgets.QComboBox()
        self.combo_class = QtWidgets.QComboBox()
        self.tablewidget = QtWidgets.QTableWidget()
        self.KNalgorithm = QtWidgets.QComboBox()
        self.SVCkernel = QtWidgets.QComboBox()
        self.DTcriterion = QtWidgets.QComboBox()
        self.RFcriterion = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel()

        self.mpl_toolbar = NavigationToolbar2QT(self.map, self.parent)

        self.setupui()

        self.map.mindx = self.m

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        grid_main = QtWidgets.QGridLayout(self)
        group_map = QtWidgets.QGroupBox('Class Edit')
        grid_right = QtWidgets.QGridLayout(group_map)

        group_class = QtWidgets.QGroupBox('Supervised Classification')
        grid_class = QtWidgets.QGridLayout(group_class)

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        loadshape = QtWidgets.QPushButton('Load Class Shapefile')
        saveshape = QtWidgets.QPushButton('Save Class Shapefile')
        calcmetrics = QtWidgets.QPushButton('Calculate and Display Metrics')

        self.setWindowTitle('Supervised Classification')
        self.tablewidget.setRowCount(0)
        self.tablewidget.setColumnCount(1)
        self.tablewidget.setHorizontalHeaderLabels(['Class Names'])

        self.apoly.setAutoDefault(False)
        self.dpoly.setAutoDefault(False)

        choices = ['K Neighbors Classifier',
                   'Decision Tree Classifier',
                   'Random Forest Classifier',
                   'Support Vector Classifier']

        self.combo_class.clear()
        self.combo_class.addItems(choices)

        lbl_combo = QtWidgets.QLabel('Data Band:')
        lbl_class = QtWidgets.QLabel('Classifier:')
        self.label1.setText('Algorithm:')

        self.KNalgorithm.addItems(['auto', 'ball_tree', 'kd_tree', 'brute'])
        self.DTcriterion.addItems(['gini', 'entropy'])
        self.RFcriterion.addItems(['gini', 'entropy'])
        self.SVCkernel.addItems(['rbf', 'linear', 'poly'])

        self.SVCkernel.setHidden(True)
        self.DTcriterion.setHidden(True)
        self.RFcriterion.setHidden(True)

        grid_right.addWidget(lbl_combo, 0, 0, 1, 1)
        grid_right.addWidget(self.combo, 0, 1, 1, 2)

        grid_right.addWidget(self.tablewidget, 1, 0, 3, 2)
        grid_right.addWidget(self.apoly, 1, 2, 1, 1)
        grid_right.addWidget(self.dpoly, 2, 2, 1, 1)
        grid_right.addWidget(calcmetrics, 3, 2, 1, 1)
        grid_right.addWidget(loadshape, 4, 0, 1, 1)
        grid_right.addWidget(saveshape, 4, 1, 1, 1)

        grid_class.addWidget(lbl_class, 0, 0, 1, 1)
        grid_class.addWidget(self.combo_class, 0, 1, 1, 1)
        grid_class.addWidget(self.label1, 1, 0, 1, 1)
        grid_class.addWidget(self.KNalgorithm, 1, 1, 1, 1)
        grid_class.addWidget(self.DTcriterion, 1, 1, 1, 1)
        grid_class.addWidget(self.RFcriterion, 1, 1, 1, 1)
        grid_class.addWidget(self.SVCkernel, 1, 1, 1, 1)

        grid_main.addWidget(self.map, 0, 0, 2, 1)
        grid_main.addWidget(self.mpl_toolbar, 2, 0, 1, 1)

        grid_main.addWidget(group_map, 0, 1, 1, 1)
        grid_main.addWidget(group_class, 1, 1, 1, 1)
        grid_main.addWidget(buttonbox, 2, 1, 1, 1)

        self.apoly.clicked.connect(self.on_apoly)
        self.dpoly.clicked.connect(self.on_dpoly)
        loadshape.clicked.connect(self.load_shape)
        saveshape.clicked.connect(self.save_shape)
        calcmetrics.clicked.connect(self.calc_metrics)

        self.tablewidget.currentItemChanged.connect(self.onrowchange)
        self.tablewidget.cellChanged.connect(self.oncellchange)
        self.combo_class.currentIndexChanged.connect(self.class_change)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def class_change(self):
        """
        Routine called when current classification choice changes.

        Returns
        -------
        None.

        """
        ctext = self.combo_class.currentText()

        self.SVCkernel.setHidden(True)
        self.DTcriterion.setHidden(True)
        self.RFcriterion.setHidden(True)
        self.KNalgorithm.setHidden(True)

        if ctext == 'K Neighbors Classifier':
            self.KNalgorithm.setHidden(False)
            self.label1.setText('Algorithm:')
        elif ctext == 'Decision Tree Classifier':
            self.DTcriterion.setHidden(False)
            self.label1.setText('Criterion:')
        elif ctext == 'Random Forest Classifier':
            self.RFcriterion.setHidden(False)
            self.label1.setText('Criterion:')
        elif ctext == 'Support Vector Classifier':
            self.SVCkernel.setHidden(False)
            self.label1.setText('Kernel:')

    def calc_metrics(self):
        """
        Calculate metrics.

        Returns
        -------
        None.

        """
        if self.df is None:
            return

        classifier, _, _, X_test, y_test, tlbls = self.init_classifier()

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        cmat = skm.confusion_matrix(y_test, y_pred)
        accuracy = skm.accuracy_score(y_test, y_pred)
        kappa = skm.cohen_kappa_score(y_pred, y_test)

        message = '<p>Confusion Matrix:</p>'
        message += pd.DataFrame(cmat, columns=tlbls, index=tlbls).to_html()
        message += '<p>Accuracy: '+str(accuracy)+'</p>'
        message += '<p>Kappa:\t  '+str(kappa)+'</p>'

        qsave = QtWidgets.QMessageBox.Save
        qokay = QtWidgets.QMessageBox.Ok
        ret = QtWidgets.QMessageBox.information(self.parent, 'Metrics',
                                                message,
                                                buttons=qsave | qokay)
        if ret == qsave:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.parent, 'Save File', '.', 'Excel spreadsheet (*.xlsx)')

            if filename != '':
                df = pd.DataFrame(cmat, columns=tlbls, index=tlbls)
                df.loc['Accuracy'] = np.nan
                df.loc['Accuracy', tlbls[0]] = accuracy
                df.loc['Kappa'] = np.nan
                df.loc['Kappa', tlbls[0]] = kappa

                df.to_excel(filename)

    def updatepoly(self, xycoords=None):
        """
        Update polygon.

        Parameters
        ----------
        xycoords : numpy array, optional
            x, y coordinates. The default is None.

        Returns
        -------
        None.

        """
        row = self.tablewidget.currentRow()
        if row == -1:
            return

        self.df.loc[row] = pd.Series(dtype='object')
        self.df.loc[row, 'class'] = self.tablewidget.item(row, 0).text()

        xycoords = self.map.polyi.poly.xy
        if xycoords.size < 8:
            self.df.loc[row, 'geometry'] = Polygon([])
        else:
            self.df.loc[row, 'geometry'] = Polygon(xycoords)

    def oncellchange(self, row, col):
        """
        Routine activated whenever a cell is changed.

        Parameters
        ----------
        row : int
            Current row.
        col : int
            Current column.

        Returns
        -------
        None.

        """
        if self.tablewidget.currentItem() is None or col != 0:
            return

        if row not in self.df.index:
            self.df.loc[row] = pd.Series(dtype='object')
        self.df.loc[row, 'class'] = self.tablewidget.item(row, 0).text()

    def onrowchange(self, current, previous):
        """
        Routine activated whenever a row is changed.

        Parameters
        ----------
        current : QTableWidgetItem
            current item.
        previous : QTableWidgetItem
            previous item.

        Returns
        -------
        None.

        """
        if previous is None or current is None:
            return
        if current.row() == previous.row():
            return
        row = current.row()

        if self.df.loc[row, 'geometry'] == Polygon([]):
            return
        coords = list(self.df.loc[row, 'geometry'].exterior.coords)

        self.update_class_polys()

        self.map.polyi.new_poly(coords)

    def on_apoly(self):
        """
        On add polygon.

        Returns
        -------
        None.

        """
        if self.df is None:
            self.df = gpd.GeoDataFrame(columns=['class', 'geometry'])
            self.df.set_geometry('geometry')

        row = self.tablewidget.rowCount()
        self.tablewidget.insertRow(row)
        item = QtWidgets.QTableWidgetItem('Class '+str(row+1))
        self.tablewidget.setItem(row, 0, item)

        item = QtWidgets.QTableWidgetItem('1')
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.tablewidget.setItem(row, 1, item)

        self.update_class_polys()
        self.map.polyi.new_poly([[1, 1]])

        self.df.loc[row] = pd.Series(dtype='object')
        self.df.loc[row, 'class'] = self.tablewidget.item(row, 0).text()
        self.df.loc[row, 'geometry'] = Polygon([])

        self.tablewidget.selectRow(row)
        self.map.polyi.isactive = True

    def on_dpoly(self):
        """
        On delete polygon.

        Returns
        -------
        None.

        """
        row = self.tablewidget.currentRow()
        self.tablewidget.removeRow(self.tablewidget.currentRow())
        self.df = self.df.drop(row)
        self.df = self.df.reset_index(drop=True)

        self.update_class_polys()
        if self.tablewidget.rowCount() == 0:
            self.map.polyi.new_poly()
            self.map.polyi.isactive = False

    def on_combo(self):
        """
        On combo.

        Returns
        -------
        None.

        """
        self.m[0] = self.combo.currentIndex()
        self.map.update_graph()

    def load_shape(self):
        """
        Load shapefile.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'Shapefile (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent,
                                                            'Open File',
                                                            '.', ext)
        if filename == '':
            return False

        df = gpd.read_file(filename)
        df.columns = df.columns.str.lower()
        if 'id' in df:
            df = df.drop('id', axis='columns')

        if 'class' not in df or 'geometry' not in df:
            return False

        self.df = df.dropna()
        self.tablewidget.setRowCount(0)
        for index, _ in self.df.iterrows():
            self.tablewidget.insertRow(index)
            item = QtWidgets.QTableWidgetItem(self.df['class'].iloc[index])
            self.tablewidget.setItem(index, 0, item)

        self.map.polyi.isactive = True
        self.tablewidget.selectRow(0)
        coords = list(self.df.loc[0, 'geometry'].exterior.coords)
        self.map.polyi.new_poly(coords)

        self.update_class_polys()

        return True

    def save_shape(self):
        """
        Save shapefile.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Shapefile (*.shp)')

        if filename == '':
            return False

        self.df.to_file(filename)
        return True

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
        if 'Raster' not in self.indata:
            self.showprocesslog('Error: You must have a multi-band raster '
                                'dataset in addition to your cluster '
                                'analysis results')
            return False

        self.map.data = self.indata['Raster']

        bands = [i.dataid for i in self.indata['Raster']]

        self.combo.clear()
        self.combo.addItems(bands)
        self.combo.currentIndexChanged.connect(self.on_combo)

        self.map.init_graph()

        self.map.polyint()
        self.map.polyi.polyi_changed.connect(self.updatepoly)
        self.map.update_graph()

        tmp = self.exec_()

        if tmp == 0:
            return False

        classifier, lbls, datall, _, _, _ = self.init_classifier()

        mask = self.map.data[0].data.mask
        yout = np.zeros_like(datall[:, :, 0], dtype=int)
        datall = datall[~mask]

        yout1 = classifier.predict(datall)
        yout[~mask] = yout1

        data = copy.deepcopy(self.indata['Raster'])
        dat_out = [Data()]

        dat_out[-1].metadata['Cluster']['input_type'] = []
        for k in data:
            dat_out[-1].metadata['Cluster']['input_type'].append(k.dataid)

        zonal = np.ma.array(yout, mask=self.map.data[0].data.mask)

        if self.parent is None:
            plt.imshow(zonal)
            plt.show()

        i = len(lbls)

        dat_out[-1].data = zonal
        dat_out[-1].nodata = zonal.fill_value
        dat_out[-1].metadata['Cluster']['no_clusters'] = i
        dat_out[-1].metadata['Cluster']['center'] = np.zeros([i, len(data)])
        dat_out[-1].metadata['Cluster']['center_std'] = np.zeros([i, len(data)])

        m = []
        s = []
        for i2 in lbls:
            m.append(datall[yout1 == i2].mean(0))
            s.append(datall[yout1 == i2].std(0))

        dat_out[-1].metadata['Cluster']['center'] = np.array(m)
        dat_out[-1].metadata['Cluster']['center_std'] = np.array(s)

        dat_out[-1].crs = data[0].crs
        dat_out[-1].dataid = 'Clusters: '+str(dat_out[-1].metadata['Cluster']['no_clusters'])
        dat_out[-1].nodata = data[0].nodata
        dat_out[-1].set_transform(transform=data[0].transform)

        for i in dat_out:
            i.data += 1
            i.data = i.data.astype(np.uint8)
            i.nodata = 0
            i.data.data[i.data.mask] = 0

        self.showprocesslog('Cluster complete')

        self.outdata['Cluster'] = dat_out
        self.outdata['Raster'] = self.indata['Raster']

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
        self.combo_class.setCurrentText(projdata['combo_class'])
        self.KNalgorithm.setCurrentText(projdata['KNalgorithm'])
        self.DTcriterion.setCurrentText(projdata['DTcriterion'])
        self.RFcriterion.setCurrentText(projdata['RFcriterion'])
        self.SVCkernel.setCurrentText(projdata['SVCkernel'])

        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        projdata['combo_class'] = self.combo_class.currentText()
        projdata['KNalgorithm'] = self.KNalgorithm.currentText()
        projdata['DTcriterion'] = self.DTcriterion.currentText()
        projdata['RFcriterion'] = self.RFcriterion.currentText()
        projdata['SVCkernel'] = self.SVCkernel.currentText()

        return projdata

    def init_classifier(self):
        """
        Initialise classifier.

        Returns
        -------
        classifier : object
            Scikit learn classification object.
        lbls : numpy array
            Class labels.
        datall : numpy array
            Dataset.
        X_test : numpy array
            X test dataset.
        y_test : numpy array
            Y test dataset.
        tlbls : numpy array
            Class labels.

        """
        ctext = self.combo_class.currentText()

        if ctext == 'K Neighbors Classifier':
            alg = self.KNalgorithm.currentText()
            classifier = KNeighborsClassifier(algorithm=alg)
        elif ctext == 'Decision Tree Classifier':
            crit = self.DTcriterion.currentText()
            classifier = DecisionTreeClassifier(criterion=crit)
        elif ctext == 'Random Forest Classifier':
            crit = self.RFcriterion.currentText()
            classifier = RandomForestClassifier(criterion=crit)
        elif ctext == 'Support Vector Classifier':
            ker = self.SVCkernel.currentText()
            classifier = SVC(gamma='scale', kernel=ker)

        rows, cols = self.map.data[0].data.shape
        masks = {}
        for _, row in self.df.iterrows():
            pixels = np.array(row['geometry'].exterior.coords)
            pixels[:, 0] = pixels[:, 0]-self.map.data[0].extent[0]
            pixels[:, 0] /= self.map.data[0].xdim
            pixels[:, 1] = self.map.data[0].extent[3]-pixels[:, 1]
            pixels[:, 1] /= self.map.data[0].ydim

            pixels = tuple(map(tuple, pixels))

            cname = row['class']
            if cname not in masks:
                masks[cname] = np.zeros((rows, cols), dtype=bool)

            rasterPoly = Image.new("L", (cols, rows), 1)
            rasterize = ImageDraw.Draw(rasterPoly)
            rasterize.polygon(pixels, 0)
            mask = np.array(rasterPoly, dtype=bool)

            masks[cname] = np.logical_or(~mask, masks[cname])

        datall = []
        for i in self.map.data:
            datall.append(i.data)
        datall = np.array(datall)
        datall = np.moveaxis(datall, 0, -1)

        y = []
        x = []
        tlbls = []
        for i, lbl in enumerate(masks):
            y += [i]*masks[lbl].sum()
            x.append(datall[masks[lbl]])
            tlbls.append(lbl)

        y = np.array(y)
        x = np.vstack(x)
        lbls = np.unique(y)

        if len(lbls) < 2:
            self.showprocesslog('Error: You need at least two classes')

        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y)

        classifier.fit(X_train, y_train)

        return classifier, lbls, datall, X_test, y_test, tlbls

    def update_map(self, polymask):
        """
        Update map.

        Parameters
        ----------
        polymask : numpy array
            Polygon mask.

        Returns
        -------
        None.

        """
        if max(polymask) is False:
            return

        mtmp = self.combo.currentIndex()
        mask = self.indata['Raster'][mtmp].data.mask

        polymask = np.array(polymask)
        polymask.shape = mask.shape
        polymask = np.logical_or(polymask, mask)

        dattmp = self.map.csp.get_array()
        dattmp.mask = polymask
        self.map.csp.changed()
        self.map.figure.canvas.draw()

    def update_class_polys(self):
        """Update class poly summaries."""
        axes = self.map.figure.gca()

        [p.remove() for p in reversed(axes.patches)]

        for _, row in self.df.iterrows():
            if row['geometry'] is None:
                return
            crds = np.array(row['geometry'].exterior.coords)

            poly = mPolygon(crds, ec='k', fill=False)
            axes.add_patch(poly)

        self.map.figure.canvas.draw()


def dist_point_to_segment(p, s0, s1):
    """
    Dist point to segment.

    Reimplementation of Matplotlib's dist_point_to_segment, after it was
    depreciated. Follows http://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    p : numpy array
        Point.
    s0 : numpy array
        Start of segment.
    s1 : numpy array
        End of segment.

    Returns
    -------
    numpy array
        Distance of point to segment.

    """
    p = np.array(p)
    s0 = np.array(s0)
    s1 = np.array(s1)

    v = s1 - s0
    w = p - s0

    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s0)

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s1)

    b = c1/c2
    pb = s0 + b*v

    return np.linalg.norm(p - pb)


def _testfn():
    """Test."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..//..')))
    from pygmi.raster import iodefs

    app = QtWidgets.QApplication(sys.argv)

    ifile = r"D:\Workdata\PyGMI Test Data\Classification\Cut_K_Th_U.ers"

    data = iodefs.get_raster(ifile)
    # os.chdir(r'd:\Workdata\people\janinetest2')

    tmp = SuperClass(None)
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":

    _testfn()
