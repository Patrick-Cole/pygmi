# -----------------------------------------------------------------------------
# Name:        seis/graphs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""
AI Seismology Routines.
"""

import os
import sys

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import fiona
from shapely.geometry import shape, Point, LineString
import pyproj
from osgeo import gdal
from PIL import Image, ImageTk
from geopandas import GeoDataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.utils import shuffle
from sklearn.cluster import DBSCAN
import sklearn.neighbors
from PyQt5 import QtWidgets
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.pyplot as plt

from pygmi.misc import ProgressBarText

import tkinter as tk
import tkinter.font
import tkinter.ttk
import tkinter.filedialog
import tensorflow.compat.v1 as tf
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

tf.disable_v2_behavior()
register_matplotlib_converters()
gdal.UseExceptions()

LARGE_FONT = ("Verdana", 12)


class MyMplCanvas(FigureCanvasQTAgg):
    """Canvas for the actual plot."""

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        # figure stuff
        self.axes = fig.add_subplot(111)

        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def init_graph(self):
        """
        Initialize the graph.

        Returns
        -------
        None.

        """
        self.axes.clear()
        self.axes.set_aspect('equal')

        maxdiam = self.pwidth*self.data[:, -1].max()
        xmin = self.data[:, 0].min()-maxdiam
        xmax = self.data[:, 0].max()+maxdiam
        ymin = self.data[:, 1].min()-maxdiam
        ymax = self.data[:, 1].max()+maxdiam

        self.axes.set_xlim((xmin, xmax))
        self.axes.set_ylim((ymin, ymax))

        self.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        # for idat in self.data:
        #     pxy = idat[:2]
        #     np1 = idat[3:-1]
        #     pwidth = self.pwidth*idat[-1]
        #     xxx, yyy, xxx2, yyy2 = beachball(np1, pxy[0], pxy[1], pwidth,
        #                                      self.isgeog)

        #     pvert1 = np.transpose([yyy, xxx])
        #     pvert0 = np.transpose([xxx2, yyy2])

        #     self.axes.add_patch(patches.Polygon(pvert1,
        #                                         edgecolor=(0.0, 0.0, 0.0)))
        #     self.axes.add_patch(patches.Polygon(pvert0, facecolor='none',
        #                                         edgecolor=(0.0, 0.0, 0.0)))

        self.figure.canvas.draw()


    def t2_linegraph(self, ifile, efile, title, ylabel, ycol, magmin, magmax):
        """
        Common routine to plot a line graph of X vs earthquakes.

        Parameters
        ----------
        ifile : str
            input filename.
        title : str
            graph title.
        ylabel: str
            y axis label.

        Returns
        -------
        None.

        """
        self.figure.clf()

        headers = ['date', ycol]

        dp = pd.read_excel(ifile, usecols=headers)
        dp['date'] = pd.to_datetime(dp['date'], infer_datetime_format=True)
        dp['year'], dp['month'], dp['day'] = (dp['date'].dt.year,
                                              dp['date'].dt.month,
                                              dp['date'].dt.day)

        headers = ['lat', 'long', 'depth', 'date', 'time', 'mag']
        dm = pd.read_excel(efile, usecols=headers)

        dm = dm[(magmin <= dm.mag) & (dm.mag <= magmax)]
        dm['date'] = pd.to_datetime(dm['date'], infer_datetime_format=True)
        dm['year'], dm['month'], dm['day'] = (dm['date'].dt.year,
                                              dm['date'].dt.month,
                                              dm['date'].dt.day)

        ds = dm.groupby(['year', 'month'])['month'].count().to_frame('count').reset_index()
        ds["date"] = pd.to_datetime(ds['year'].map(str) + ' ' +
                                    ds["month"].map(str), format='%Y/%m')

        ax1 = self.figure.add_subplot(111, label='2D')
        ax1.plot(dp['date'], dp[ycol], color='b', label=ylabel)
        ax2 = ax1.twinx()
        ax2.plot(ds['date'], ds['count'], color='r',  label='Earthquakes')

        ax1.set_title(title)
        ax1.set_xlabel('Date')
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel('Number of Earthquakes')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        ax1.legend(h1+h2, l1+l2, loc=2)
        self.figure.tight_layout()
        self.figure.canvas.draw()



class AI_Seis(QtWidgets.QDialog):
    """AI Sesimology routines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.ifiles = {'edata': '',
                       'rain': '',
                       'lineaments': '',
                       'streamflow': '',
                       'streamshp': '',
                       'lineamentshp': ''}
        self.parent = parent
        self.indata = {}
        self.outdata = {}

        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab5 = QtWidgets.QWidget()

        # Tab 1
        self.qfile = {}

        # Tab 2
        self.mt2 = MyMplCanvas(self)
        self.t2_combobox1 = QtWidgets.QComboBox()
        self.t2_maxmag = QtWidgets.QDoubleSpinBox()
        self.t2_minmag = QtWidgets.QDoubleSpinBox()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(r'AI in Seismology')
        self.resize(640, 480)

        # Initialize tab screen

# Add tabs
        self.tabs.addTab(self.tab1, 'Import Data')
        self.tabs.addTab(self.tab2, 'View Data')
        self.tabs.addTab(self.tab3, 'Cluster Determination')
        self.tabs.addTab(self.tab4, 'Completeness and b-value')
        self.tabs.addTab(self.tab5, 'AI')

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)

# Create first tab
        pb_edata = QtWidgets.QPushButton('Earthquake Data (.xlsx)')
        pb_rain = QtWidgets.QPushButton('Monthly Rainfall Data (.xlsx)')
        pb_lineaments = QtWidgets.QPushButton('Geological Lineament Data (.xlsx)')
        pb_streamflow = QtWidgets.QPushButton('Monthly Stream Flow Data (.xlsx)')
        pb_streamshp = QtWidgets.QPushButton(r'Stream/River Vector Data (.shp)')
        pb_lineamentshp = QtWidgets.QPushButton('Geology Lineament Vector Data (.shp)')

        self.qfile['edata'] = QtWidgets.QLineEdit('')
        self.qfile['rain'] = QtWidgets.QLineEdit('')
        self.qfile['lineaments'] = QtWidgets.QLineEdit('')
        self.qfile['streamflow'] = QtWidgets.QLineEdit('')
        self.qfile['streamshp'] = QtWidgets.QLineEdit('')
        self.qfile['lineamentshp'] = QtWidgets.QLineEdit('')

        tab1_layout = QtWidgets.QGridLayout(self)

        tab1_layout.addWidget(pb_edata, 0, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['edata'], 0, 1, 1, 1)
        tab1_layout.addWidget(pb_rain, 1, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['rain'], 1, 1, 1, 1)
        tab1_layout.addWidget(pb_lineaments, 2, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['lineaments'], 2, 1, 1, 1)
        tab1_layout.addWidget(pb_streamflow, 3, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['streamflow'], 3, 1, 1, 1)
        tab1_layout.addWidget(pb_streamshp, 4, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['streamshp'], 4, 1, 1, 1)
        tab1_layout.addWidget(pb_lineamentshp, 5, 0, 1, 1)
        tab1_layout.addWidget(self.qfile['lineamentshp'], 5, 1, 1, 1)

        self.tab1.setLayout(tab1_layout)

        pb_edata.clicked.connect(lambda: self.load_data('edata', 'xlsx'))
        pb_rain.clicked.connect(lambda: self.load_data('rain', 'xlsx'))
        pb_lineaments.clicked.connect(lambda: self.load_data('lineaments',
                                                             'xlsx'))
        pb_streamflow.clicked.connect(lambda: self.load_data('streamflow',
                                                             'xlsx'))
        pb_streamshp.clicked.connect(lambda: self.load_data('streamshp', 'shp'))
        pb_lineamentshp.clicked.connect(lambda: self.load_data('lineamentshp',
                                                               'shp'))


# Create Second Tab
        mpl_toolbar_t2 = NavigationToolbar2QT(self.mt2, self.parent)
        t2_label1 = QtWidgets.QLabel('Calculation:')
        t2_label2 = QtWidgets.QLabel('Minimum Magnitude:')
        t2_label3 = QtWidgets.QLabel('Maximum Magnitude:')
        self.t2_maxmag.setValue(5.)
        self.t2_minmag.setValue(0.)

        self.t2_combobox1.addItems(['Patterns in Seismicity',
                                    'Correlations with rainfall',
                                    'Correlations with stream flow',
                                    'Correlations with geological lineaments'])

        tab2_layout = QtWidgets.QGridLayout(self)
        tab2_layout.addWidget(self.mt2, 0, 0, 1, 2)
        tab2_layout.addWidget(mpl_toolbar_t2, 1, 0, 1, 2)
        tab2_layout.addWidget(t2_label2, 2, 0, 1, 1)
        tab2_layout.addWidget(self.t2_minmag, 2, 1, 1, 1)
        tab2_layout.addWidget(t2_label3, 3, 0, 1, 1)
        tab2_layout.addWidget(self.t2_maxmag, 3, 1, 1, 1)

        tab2_layout.addWidget(t2_label1, 4, 0, 1, 1)
        tab2_layout.addWidget(self.t2_combobox1, 4, 1, 1, 1)

        self.tab2.setLayout(tab2_layout)

        self.t2_combobox1.currentIndexChanged.connect(self.t2_change_graph)
        self.t2_minmag.valueChanged.connect(self.t2_change_graph)
        self.t2_maxmag.valueChanged.connect(self.t2_change_graph)

        # General

        self.tabs.currentChanged.connect(self.change_tab)


# Add tabs to widget
        layout.addWidget(self.tabs)

    def change_tab(self, index):
        """
        Change tab

        Parameters
        ----------
        index : int
            Tab index.

        Returns
        -------
        None.

        """
        if index == 1:
            self.t2_change_graph()

    def t2_change_graph(self):
        """
        Change the graph type on tab 2.

        Returns
        -------
        None.

        """
        text = self.t2_combobox1.currentText()
        minmag = self.t2_minmag.value()
        maxmag = self.t2_maxmag.value()
        efile = self.qfile['edata'].text()

        if text == 'Patterns in Seismicity':
            self.mt2.figure.clf()

            headers = ['lat', 'long', 'depth', 'date', 'time', 'mag']
            de = pd.read_excel(efile, usecols=headers)
            de = de[(minmag <= de.mag) & (de.mag <= maxmag)]

            ax = self.mt2.figure.add_subplot(111, projection="3d")
            ax.set_title('Patterns in seismicity')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Magnitude')
            ax.set_facecolor('xkcd:white')
            self.mt2.figure.patch.set_facecolor('xkcd:white')

            ax.scatter(de['lat'], de['long'], de['mag'], label='Earthquakes')
            ax.legend()
            self.mt2.figure.tight_layout()
            self.mt2.figure.canvas.draw()

        elif text == 'Correlations with geological lineaments':
            self.mt2.figure.clf()

            headers = ['lat', 'long', 'properties']
            degeo = pd.read_excel(self.qfile['lineaments'].text(),
                                  usecols=headers)
            headers = ['lat', 'long', 'depth', 'date', 'time', 'mag']
            de = pd.read_excel(efile, usecols=headers)
            de = de[(minmag <= de.mag) & (de.mag <= maxmag)]

            ax = self.mt2.figure.add_subplot(111, projection="3d",
                                                label='3D')
            ax.set_title('Correlation between seismicity and geological '
                            'structures')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Magnitude')
            ax.set_facecolor('xkcd:white')
            self.mt2.figure.patch.set_facecolor('xkcd:white')

            ax.scatter(de['lat'], de['long'], de['mag'], label='Earthquakes')
            ax.scatter(degeo['lat'], degeo['long'], label='Lineaments')

            ax.legend()

            self.mt2.figure.tight_layout()
            self.mt2.figure.canvas.draw()

        elif text == 'Correlations with rainfall':
            ifile = self.qfile['rain'].text()
            title = 'Rainfall and number of earthquakes per month'
            ylabel = 'Rainfall'
            ycol = 'rain'

            self.mt2.t2_linegraph(ifile, efile, title, ylabel, ycol,
                                  minmag, maxmag)

        elif text == 'Correlations with stream flow':
            ifile = self.qfile['streamflow'].text()
            title = 'Streamflow and number of earthquakes per month'
            ylabel = 'Volume of water'

            self.mt2.t2_linegraph(ifile, efile, title, ylabel, 'metre',
                                  minmag, maxmag)

    def load_data(self, datatype, ext):
        """
        Load data.

        Returns
        -------
        None.

        """

        if ext == 'xlsx':
            ext = 'Excel file (*.xlsx)'
        else:
            ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        self.qfile[datatype].setText(filename)


        test = [self.qfile[i].text() for i in self.qfile]

        if '' not in test:
            self.tabs.setTabEnabled(1, True)

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
            tmp = self.exec_()

        if tmp != 1:
            return False

        # self.show()
        # QtWidgets.QApplication.processEvents()

        # self.mmc.init_graph()
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

#        projdata['ftype'] = '2D Mean'

        return projdata


def aihelp():
    """
    Help.

    Returns
    -------
    None.

    """
    os.startfile('GRS-MN-004_rev0_AI_SEIS3.pdf')


def click_about():
    """
    Click About.

    Returns
    -------
    None.

    """
    about_text = """About

    The program is developed to understand seismicity by identifying patterns.
    Coded by M Grobbelaar and E Sakala"""

    software_version = """SOFTWARE VERSION 3.0"""
    copy_right = """(c) Council for Geoscience 2021"""
    toplevel = tk.Toplevel()
    label1 = tk.Label(toplevel, text=about_text, height=0, width=100)
    label1.pack()
    label1 = tk.Label(toplevel, text=software_version, height=0, width=100)
    label1.pack()
    label1 = tk.Label(toplevel, text=copy_right, height=0, width=100)
    label1.pack()


def select_csv(ent):
    """
    Select Test File.

    Parameters
    ----------
    ent : tk.Entry
        Tkinter Entry Widget.

    Returns
    -------
    str
        Filename of csv.

    """
    testname = tk.filedialog.askopenfilename(title="Select csv file",
                                             filetypes=(("csv files", "*.csv"),
                                                        ("", "")))
    ent.delete('0', tk.END)
    ent.insert('end', testname)
    return str(testname)


def select_xlsx(ent):
    """
    Select Excel File.

    Parameters
    ----------
    ent : tk.Entry
        Tkinter Entry Widget.

    Returns
    -------
    str
        Filename of xlsx file.

    """
    filename = tk.filedialog.askopenfilename(title="Select Excel file",
                                             filetype=(("xlsx files",
                                                        "*.xlsx"), ("", "")))
    ent.delete('0', tk.END)
    ent.insert('end', filename)
    return str(filename)


def select_shp(ent):
    """
    Select Shapefile.

    Parameters
    ----------
    ent : tk.Entry
        Tkinter Entry Widget.

    Returns
    -------
    str
        Filename of shp file.

    """
    filename = tk.filedialog.askopenfilename(title=("Select Esri "
                                                    "Shapefile file"),
                                             filetypes=(("Esri Shapefile",
                                                         "*.shp"),
                                                        ("", "")))
    ent.delete('0', tk.END)
    ent.insert('end', filename)
    return str(filename)


def output_folder(ent):
    """
    Select output folder.

    Parameters
    ----------
    ent : tk.Entry
        Tkinter Entry Widget.

    Returns
    -------
    str
        Output directory

    """
    out_dir = tk.filedialog.askdirectory()
    ent.delete('0', tk.END)
    ent.insert('end', out_dir)
    return str(out_dir)


def one_hot_encode(labels):
    """
    Encoder.

    Parameters
    ----------
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    one_hot_encode1 : TYPE
        DESCRIPTION.

    """
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode1 = np.zeros((n_labels, n_unique_labels))
    one_hot_encode1[np.arange(n_labels), labels] = 1
    return one_hot_encode1


def multilayer_perceptron(x, weights, biases):
    """
    Multilayer perceptron.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.
    biases : TYPE
        DESCRIPTION.

    Returns
    -------
    out_layer : TYPE
        DESCRIPTION.

    """
    # hidden layer1 with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.tanh(layer_1)
    # hidden layer2 with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),
                     biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.nn.tanh(layer_2)
    # hidden layer3 with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']),
                     biases['b3'])
    # layer_3 = tf.nn.relu(layer_3)
    # layer_3 = tf.nn.sigmoid(layer_3)
    layer_3 = tf.nn.tanh(layer_3)
    # hidden layer4 with sigmoid activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']),
                     biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.tanh(layer_4)
    # hidden layer4 with sigmoid activation
    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']),
    #                  biases['b5'])
    # layer_5 = tf.nn.sigmoid(layer_5)
    # layer_5 = tf.nn.relu(layer_5)
    # layer_5 = tf.nn.tanh(layer_5)
    # output layer with linear activation
    out_layer = tf.matmul(layer_4,
                          weights['out'] + biases['out'])
    return out_layer


def get_distances(ifile, geo_df, df_mg, lbl):
    """
    Get distances between

    Parameters
    ----------
    ifile : str
        File of lines.
    geo_df : GeoDataFrame
        Dataframe of points.

    Returns
    -------
    pd1 : DataFrame
        Output distances.

    """
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:32735",
                                              always_xy=True)
    line = fiona.open(ifile)

    points = [(feat.xy[0][0], feat.xy[1][0]) for feat in geo_df.geometry]

    lines = [zip(shape(feat["geometry"]).coords.xy[0],
                 shape(feat["geometry"]).coords.xy[1])
             for feat in line]

    proj_lines = [[] for i in range(len(lines))]
    for i, item in enumerate(lines):
        for element in item:
            x = element[0]
            y = element[1]
            # x, y = pyproj.transform(srcProj, dstProj, x, y)
            x, y = transformer.transform(x, y)
            proj_lines[i].append((x, y))

    proj_points = []
    for point in points:
        x = point[0]
        y = point[1]
        # x, y = pyproj.transform(srcProj, dstProj, x, y)
        x, y = transformer.transform(x, y)
        proj_points.append(Point(x, y))

    distances = [[] for i in range(len(lines))]

    for i, line in enumerate(proj_lines):
        for point in proj_points:
            distances[i].append(LineString(line).distance(point))

    file1 = pd.DataFrame(min(distances))
    file2 = pd.DataFrame(points)

    results1 = pd.concat([file2, file1], axis=1)
    results1.columns = ["long", "lat", lbl]

    pd1 = pd.merge(results1, df_mg, how='left', on=['long', 'lat'])
    pd1['mag'].replace('', np.nan, inplace=True)
    pd1.dropna(subset=['mag'], inplace=True)

    return pd1


class PageCluster(tk.Frame):
    """PageCluster."""
    counter = 0

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        self.filename1 = self.controller.shared_data["filename1"]
        self.filename3 = self.controller.shared_data["filename3"]
        self.filename4 = self.controller.shared_data["filename4"]
        self.filename5 = self.controller.shared_data["filename5"]
        self.filename6 = self.controller.shared_data["filename6"]
        self.out_dir = self.controller.shared_data["out_dir"]
        self.clusterdat = None

        tk.Frame.config(self, bg='white')

        if os.path.exists("pages.jpg"):
            load = Image.open(r"pages.jpg")
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            img.place(x=0, y=0, relwidth=1, relheight=1)

        label = tk.Label(self, text="Determining Clusters within the data",
                         bg="white", fg="brown",
                         font=controller.title_font)
        label.pack()
        label.place(x=350, y=50)

        label3 = tk.Label(self, text="Please input the minimum number of "
                          "events to be used for DBSCAN", bg="white",
                          fg="brown", font=("Times", 15))
        label3.pack()
        label3.place(x=170, y=120)

        label4 = tk.Label(self, text="Please input the maximum distance for "
                          "two events to be grouped (eps)",
                          bg="white", fg="brown", font=("Times", 15))
        label4.pack()
        label4.place(x=170, y=140)

        self.dbsVariable = tk.IntVar()
        dbs = tk.Entry(self, textvariable=self.dbsVariable, fg="white",
                       bg="brown")
        self.dbsVariable.set(30)
        dbs.pack()
        dbs.place(x=750, y=120)

        self.epsVariable = tk.DoubleVar()
        eps = tk.Entry(self, textvariable=self.epsVariable, fg="white",
                       bg="brown")
        self.epsVariable.set(0.01)
        eps.pack()
        eps.place(x=750, y=140)

        buttondb = tk.Button(self, text='  Run DBSCAN  ',
                             font=("Times", 15), fg="white", bg="brown",
                             state=tk.NORMAL, command=self.calculate_eps)
        buttondb.pack()
        buttondb.place(x=900, y=120)

        ll = tk.Label(self, text='Results from DBSCAN:', bg="white",
                      fg="brown", font=("Times", 15))
        ll.pack()

        ll.place(x=50, y=240)

        button = tk.Button(self, text="Return to the home page",
                           font=("Times", 14),
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        button.place(x=30, y=650)

        button = tk.Button(self, text="Return to STEP 3", font=("Times", 14),
                           command=lambda: controller.show_frame("PageThree"))
        button.pack()
        button.place(x=400, y=650)

        self.button = tk.Button(self, text="Proceed To Completeness",
                                font=("Times", 14),
                                command=lambda:
                                controller.show_frame("PageComplete"))
        self.button.pack()
        self.button.place(x=800, y=650)
        self.button['state'] = 'disabled'

        self.lbl = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.lbl.pack()
        self.lbl.place(x=50, y=285)

        self.m = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.m.pack()
        self.m.place(x=50, y=310)

        self.n = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.n.pack()
        self.n.place(x=50, y=400)

        self.fig = Figure(figsize=(6, 4))
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.get_tk_widget().place(x=580, y=180)

        toolbarFrame = tk.Frame(self)
        toolbarFrame.pack()
        toolbarFrame.place(x=710, y=600)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

        OPTIONS = [r"N/A"]
        self.variable = tk.StringVar(self)
        self.variable.set(OPTIONS[0])

        self.om = tk.OptionMenu(self, self.variable, *OPTIONS)
        self.om.config(width=20, bg='brown', fg='white', font=("Times", 15))
        self.om.pack()
        self.om.place(x=50, y=340)

    def calculate_eps(self):
        """
        Calculate eps

        Returns
        -------
        None

        """

        headers = ['lat', 'long', 'depth', 'date', 'time', 'mag']
        df = pd.read_excel(self.filename1.get(), usecols=headers)

        # lat = df['lat']
        # lon = df['long']

        # dq = pd.DataFrame(list(zip(lat, lon)))

        # To determine eps:
        # neigh = NearestNeighbors(n_neighbors=2)
        # nbrs = neigh.fit(dq)
        # distances, _ = nbrs.kneighbors(dq)

        # distances = np.sort(distances, axis=0)
        # distances = distances[:, 1]

        # dq = pd.DataFrame({'B': distances})
        # epsilon = dq['B'].max()
        epsilon = self.epsVariable.get()

        # dq['B_dif'] = dq['B'].diff()
        # dq = dq.dropna()

        self.lbl['text'] = f'Epsilon value used: {epsilon}'
        ##########################################################
        # first the eps should be determined, then the dbscan can run
        # after this file is the sorting clusters file
        # this works below
        dbs = self.dbsVariable.get()
        # perform dbscan

        coords = df[['lat', 'long']].to_numpy()

        db = DBSCAN(eps=epsilon, min_samples=dbs,
                    algorithm='ball_tree').fit(coords)

        cluster_labels = db.labels_
        self.clusterdat = np.transpose(np.unique(db.labels_,
                                                 return_counts=True))

        num_clusters = cluster_labels.max()+1

        menu = self.om["menu"]
        menu.delete(0, "end")

        options = ['Noise cluster']
        for i in range(num_clusters):
            options.append('Cluster '+str(i))

        for string in options:
            menu.add_command(label=string, command=lambda value=string:
                             self.change_option(value))
        self.variable.set(options[0])

        num_clusters = self.clusterdat[0, 1]

        self.n['text'] = f'Number of events: {num_clusters}'

        # save as extra column in dataframe
        df["cluster"] = pd.DataFrame({'cluster': cluster_labels})

        self.controller.shared_data['cluster'] = df

        self.fig.clf()
        ax = self.fig.add_subplot(111)

        ax.set_title('Clusters identified')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_facecolor('xkcd:white')
        self.fig.patch.set_facecolor('xkcd:white')

        ax.scatter(df['long'], df['lat'], c=cluster_labels, cmap='Paired')
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.button['state'] = 'normal'

    def change_option(self, option):
        """
        Change option menu.

        Returns
        -------
        None.

        """
        self.variable.set(option)

        if option == "Noise cluster":
            num_clusters = self.clusterdat[0, 1]
        else:
            cnum = int(option.split()[1])
            num_clusters = self.clusterdat[cnum+1, 1]

        self.n['text'] = f'Number of events: {num_clusters}'


class PageComplete(tk.Frame):
    """PageComplete."""
    counter = 0

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        self.filename1 = self.controller.shared_data["filename1"]
        self.filename3 = self.controller.shared_data["filename3"]
        self.filename4 = self.controller.shared_data["filename4"]
        self.filename5 = self.controller.shared_data["filename5"]
        self.filename6 = self.controller.shared_data["filename6"]
        self.out_dir = self.controller.shared_data["out_dir"]

        tk.Frame.config(self, bg='white')
        self.controller = controller

        if os.path.exists("pages.jpg"):
            load = Image.open(r"pages.jpg")
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            img.place(x=0, y=0, relwidth=1, relheight=1)

        label = tk.Label(self, text="Determining Magnitude of Completeness "
                         "of the data", bg="white", fg="brown",
                         font=controller.title_font)
        label.pack()
        label.place(x=180, y=50)

        label3 = tk.Label(self, text="Please select the cluster for which "
                          "you would like to \ndetermine the magnitude of "
                          "completeness:", bg="white", fg="brown",
                          font=("Times", 15))
        label3.pack()
        label3.place(x=50, y=280)

        OPTIONS = ["All data"]

        self.variable = tk.StringVar(self)
        self.variable.set(OPTIONS[0])

        self.om = tk.OptionMenu(self, self.variable, *OPTIONS)
        self.om.config(width=20, bg='brown', fg='white', font=("Times", 15))
        self.om.pack()
        self.om.place(x=140, y=370)

        button = tk.Button(self, text="Return to the home page",
                           font=("Times", 14),
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        button.place(x=30, y=650)

        button = tk.Button(self, text="Return to STEP 3", font=("Times", 14),
                           command=lambda:
                               controller.show_frame("PageThree"))
        button.pack()
        button.place(x=400, y=650)

        buttonbv = tk.Button(self, text="Proceed to b-value",
                             font=("Times", 14),
                             command=lambda:
                                 controller.show_frame("Pagebvalue"))
        buttonbv.pack()
        buttonbv.place(x=800, y=650)

        self.fig = Figure(figsize=(6, 4))
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.get_tk_widget().place(x=580, y=180)

        toolbarFrame = tk.Frame(self)
        toolbarFrame.pack()
        toolbarFrame.place(x=710, y=600)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

        self.mlbl = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.mlbl.pack()
        self.mlbl.place(x=50, y=520)

    def entrypnt(self):
        """
        Entrypoint into frame.

        Returns
        -------
        None.

        """
        menu = self.om["menu"]
        menu.delete(0, "end")

        dr = self.controller.shared_data['cluster']
        clist = np.unique(dr['cluster'])
        clist = clist[clist >= 0]

        options = ['All Data']
        for i in clist:
            options.append('Cluster '+str(i))

        for string in options:
            menu.add_command(label=string, command=lambda
                             value=string: self.change_option(value))

        self.change_option(options[0])

    def change_option(self, option):
        """
        Press Okay Button and display class stuff.

        Returns
        -------
        None
        """
        self.variable.set(option)

        dr = self.controller.shared_data['cluster']

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_title('Number of earthquakes per magnitude')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Number of earthquakes')
        ax.set_facecolor('xkcd:white')
        self.fig.patch.set_facecolor('xkcd:white')

        if option == "All Data":
            df = dr['mag'].value_counts().to_frame('count').reset_index()
            mag = df['index'].iloc[0]

            ax.scatter(x=df['index'], y=df['count'], color='red')
            ax.set_title('Number of earthquakes per magnitude for data set')
            self.mlbl['text'] = ('Magnitude at which data is '
                                 f'complete in data set: {mag}')
        else:
            cnum = int(self.variable.get().split()[1])
            df = dr.loc[dr['cluster'] == cnum]
            df = df['mag'].value_counts().to_frame('count').reset_index()
            mag = df['index'].iloc[0]

            ax.scatter(x=df['index'], y=df['count'], color='blue')
            ax.set_title('Number of earthquakes per magnitude for cluster '
                         f'{cnum}')
            self.mlbl['text'] = ('Magnitude at which data is complete '
                                 f'in cluster {cnum}: {mag}')
        self.fig.tight_layout()
        self.fig.canvas.draw()


class Pagebvalue(tk.Frame):
    """Pagebvalue."""
    counter = 0

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        self.filename1 = self.controller.shared_data["filename1"]
        self.filename3 = self.controller.shared_data["filename3"]
        self.filename4 = self.controller.shared_data["filename4"]
        self.filename5 = self.controller.shared_data["filename5"]
        self.filename6 = self.controller.shared_data["filename6"]
        self.out_dir = self.controller.shared_data["out_dir"]

        tk.Frame.config(self, bg='white')

        if os.path.exists("pages.jpg"):
            load = Image.open(r"pages.jpg")
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            img.place(x=0, y=0, relwidth=1, relheight=1)

        label = tk.Label(self, text="Determining b-value for the data",
                         bg="white", fg="brown",
                         font=controller.title_font)
        label.pack()
        label.place(x=250, y=50)

        label3 = tk.Label(self, text="Please select the cluster for which you "
                          "\nwould like to determine the b-value:",
                          bg="white", fg="brown", font=("Times", 15))
        label3.pack()
        label3.place(x=50, y=220)

        OPTIONS = ["All data",
                   "Cluster zero",
                   "Cluster one",
                   "Cluster two",
                   "Cluster three"]

        self.variable = tk.StringVar(self)
        self.variable.set(OPTIONS[0])

        self.om = tk.OptionMenu(self, self.variable, *OPTIONS)
        self.om.config(width=20, bg='brown', fg='white', font=("Times", 15))
        self.om.pack()
        self.om.place(x=100, y=280)

        button = tk.Button(self, text="Return to the home page",
                           font=("Times", 14),
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        button.place(x=30, y=650)

        button = tk.Button(self, text="Return to STEP 3", font=("Times", 14),
                           command=lambda: controller.show_frame("PageThree"))
        button.pack()
        button.place(x=400, y=650)

        buttonbv = tk.Button(self, text="Go to STEP 4", font=("Times", 14),
                             command=lambda: controller.show_frame("PageFour"))
        buttonbv.pack()
        buttonbv.place(x=800, y=650)

        self.mima = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.mima.pack()
        self.mima.place(x=50, y=390)

        self.tote = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.tote.pack()
        self.tote.place(x=50, y=420)

        self.anea = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.anea.pack()
        self.anea.place(x=50, y=450)

        self.mama = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.mama.pack()
        self.mama.place(x=50, y=480)

        self.mamam = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.mamam.pack()
        self.mamam.place(x=50, y=510)

        self.leas1 = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.leas1.pack()
        self.leas1.place(x=50, y=540)

        self.leas2 = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.leas2.pack()
        self.leas2.place(x=50, y=570)

        self.maxli = tk.Label(self, bg="white", fg="brown", font=("Times", 15))
        self.maxli.pack()
        self.maxli.place(x=50, y=600)

        self.fig = Figure(figsize=(6, 4))
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        canvas.get_tk_widget().place(x=580, y=140)

        toolbarFrame = tk.Frame(self)
        toolbarFrame.pack()
        toolbarFrame.place(x=710, y=600)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    def entrypnt(self):
        """
        Entrypoint into frame.

        Returns
        -------
        None.

        """
        menu = self.om["menu"]
        menu.delete(0, "end")

        dr = self.controller.shared_data['cluster']
        clist = np.unique(dr['cluster'])
        clist = clist[clist >= 0]

        options = ['All Data']
        for i in clist:
            options.append('Cluster '+str(i))

        for string in options:
            menu.add_command(label=string,
                             command=lambda
                             value=string: self.change_option(value))

        self.change_option(options[0])

    def change_option(self, option):
        """
        Okay B Value.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        self.variable.set(option)

        dr = self.controller.shared_data['cluster']

        if option == "All Data":
            df = dr
            dfa = dr['mag'].value_counts().to_frame('count').reset_index()
        else:
            cnum = int(self.variable.get().split()[1])
            df = dr.loc[dr['cluster'] == cnum]
            dfa = df['mag'].value_counts().to_frame('count').reset_index()

        mag = dfa['index'].iloc[0]
        df = df[df['mag'] >= mag]

        magnitudes = df['mag']
        years = df['date'].dt.year
        # This should be the magnitude of completeness from the script
        min_mag = min(magnitudes)
        max_mag = max(magnitudes) + 0.1

        num_eq = len(magnitudes)

        num_years = max(years)-min(years)
        annual_num_eq = num_eq/num_years

        max_mag_bin = max(magnitudes) + 0.15
        self.mima['text'] = f'Magnitude of completeness (MC): {min_mag}'
        self.tote['text'] = f'Total number of earthquakes: {num_eq}'
        self.anea['text'] = ('Annual number of earthquakes greater '
                             f'than MC {annual_num_eq}')
        self.mama['text'] = f'Maximum catalog magnitude: {max(magnitudes)}'
        self.mamam['text'] = f'Mmax = {max_mag}'
        # Magnitude bins
        bins = np.arange(min_mag, max_mag_bin, 0.05)
        # Magnitude bins for plotting - we will re-arrange bins later
        plot_bins = np.arange(min_mag, max_mag, 0.05)

        # #####################################################################
        # Generate distribution
        # #####################################################################
        # Generate histogram
        hist = np.histogram(magnitudes, bins=bins)

        # # Reverse array order
        hist = hist[0][::-1]
        bins = bins[::-1]

        # Calculate cumulative sum
        cum_hist = hist.cumsum()
        # Ensure bins have the same length has the cumulative histogram.
        # Remove the upper bound for the highest interval.
        bins = bins[1:]

        # Get annual rate
        cum_annual_rate = cum_hist/num_years

        new_cum_annual_rate = []
        for i in cum_annual_rate:
            new_cum_annual_rate.append(i+1e-20)

        # Take logarithm
        log_cum_sum = np.log10(new_cum_annual_rate)

        # #####################################################################
        # Fit a and b parameters using a varity of methods
        # #####################################################################

        # Fit a least squares curve
        b, a = np.polyfit(bins, log_cum_sum, 1)
        self.leas1['text'] = f'Least Squares: b value {-b}'
        self.leas2['text'] = f'Least Squares: a value {a}'

        # alpha = np.log(10) * a
        beta = -1.0 * np.log(10) * b
        # Maximum Likelihood Estimator fitting
        # b value
        b_mle = np.log10(np.exp(1)) / (np.mean(magnitudes) - min_mag)
        beta_mle = np.log(10) * b_mle
        self.maxli['text'] = f'Maximum Likelihood: b value {b_mle}'

        # #####################################################################
        # Generate data to plot results
        # #####################################################################
        # Generate data to plot least squares linear curve
        # Calculate y-intercept for least squares solution
        yintercept = log_cum_sum[-1] - b * min_mag
        ls_fit = b * plot_bins + yintercept
        log_ls_fit = []
        for value in ls_fit:
            log_ls_fit.append(np.power(10, value))
        # Generate data to plot bounded Gutenberg-Richter for LS solution
        numer = (np.exp(-1. * beta * (plot_bins - min_mag)) -
                 np.exp(-1. * beta * (max_mag - min_mag)))
        denom = 1. - np.exp(-1. * beta * (max_mag - min_mag))
        ls_bounded = annual_num_eq * (numer / denom)

        # Generate data to plot maximum likelihood linear curve
        mle_fit = (-1.0 * b_mle * plot_bins + 1.0 * b_mle * min_mag +
                   np.log10(annual_num_eq))
        log_mle_fit = []
        for value in mle_fit:
            log_mle_fit.append(np.power(10, value))
        # Generate data to plot bounded Gutenberg-Richter for MLE solution
        numer = (np.exp(-1. * beta_mle * (plot_bins - min_mag)) -
                 np.exp(-1. * beta_mle * (max_mag - min_mag)))
        denom = 1. - np.exp(-1. * beta_mle * (max_mag - min_mag))
        mle_bounded = annual_num_eq * (numer / denom)
        # Compare b-value of 1
        fit_data = -1.0 * plot_bins + min_mag + np.log10(annual_num_eq)
        log_fit_data = []
        for value in fit_data:
            log_fit_data.append(np.power(10, value))
        # #####################################################################
        # Plot the results
        # #####################################################################

        self.fig.clf()
        ax = self.fig.add_subplot(111)

        ax.set_facecolor('xkcd:white')
        self.fig.patch.set_facecolor('xkcd:white')

        ax.scatter(bins, new_cum_annual_rate, label='Catalogue')
        ax.plot(plot_bins, log_ls_fit, c='r', label='Least Squares')
        ax.plot(plot_bins, ls_bounded, c='r', linestyle='--',
                label='Least Squares Bounded')
        ax.plot(plot_bins, log_mle_fit, c='g', label='Maximum Likelihood')
        ax.plot(plot_bins, mle_bounded, c='g', linestyle='--',
                label='Maximum Likelihood Bounded')
        ax.plot(plot_bins, log_fit_data, c='b', label='b = 1')

        ax.set_yscale('log')
        ax.legend()
        ax.set_ylim([min(new_cum_annual_rate) * 0.1,
                     max(new_cum_annual_rate) * 10.])
        ax.set_xlim([min_mag - 0.5, max_mag + 0.5])
        ax.set_ylabel('Annual probability')
        ax.set_xlabel('Magnitude')
        ax.set_title('B-value for the data')
        self.fig.tight_layout()
        self.fig.canvas.draw()


class PageFour(tk.Frame):
    """PageFour."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Frame.config(self, bg='light blue')
        self.controller = controller
        self.filename1 = self.controller.shared_data["filename1"]
        self.filename3 = self.controller.shared_data["filename3"]
        self.filename4 = self.controller.shared_data["filename4"]
        self.out_dir = self.controller.shared_data["out_dir"]
        self.file_line = self.controller.shared_data["file_line"]
        self.file_stream = self.controller.shared_data["file_stream"]

        if os.path.exists("pages.jpg"):
            load = Image.open(r'pages.jpg')
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            img.place(x=0, y=0, relwidth=1, relheight=1)

        self.e11latVariable = tk.DoubleVar()
        self.e11lat = tk.Entry(self, textvariable=self.e11latVariable,
                               bg='brown', fg='white')
        self.e21lonVariable = tk.DoubleVar()
        self.e21lon = tk.Entry(self, textvariable=self.e21lonVariable,
                               bg='brown', fg='white')

        self.e17Variable = tk.DoubleVar()
        self.e17 = tk.Entry(self, textvariable=self.e17Variable,
                            bg='brown', fg='white')

        # self.e11lat.pack()
        # self.e11lat.place(x=300, y=313)
        # self.e21lon.pack()
        # self.e21lon.place(x=300, y=343)

        label = tk.Label(self, text="STEP 4: Parameters associated with the "
                         "point of interest",
                         bg="white", fg="brown",
                         font=controller.title_font)
        label.pack()
        label.place(x=90, y=15)
        label2A = tk.Label(self, text=("The algorithm uses a combination of "
                                       "parameters. Please enter the "
                                       "parameters for testing:"),
                           bg="white", fg="brown", font=("Times", 15))
        label2A.pack()
        label2A.place(x=80, y=100)
        label11 = tk.Label(self, text=("Distance to closest geological "
                                       "lineament (metres)"), bg="white",
                           fg="brown", font=("Times", 13))
        label11.pack()
        label11.place(x=80, y=150)
        self.e11Variable = tk.DoubleVar()
        self.e11Variable.set(2000.0)
        self.e11 = tk.Entry(self, textvariable=self.e11Variable, bg='brown',
                            fg='white')
        self.e11.pack()
        self.e11.place(x=450, y=153)

        label14 = tk.Label(self, text=("Distance to closest stream/ "
                                       "river (metres)"), bg="white",
                           fg="brown", font=("Times", 13))
        label14.pack()
        label14.place(x=80, y=180)
        self.e14Variable = tk.DoubleVar()
        self.e14Variable.set(2000.0)
        self.e14 = tk.Entry(self, textvariable=self.e14Variable,
                            bg='brown', fg='white')
        self.e14.pack()
        self.e14.place(x=450, y=183)
        label15 = tk.Label(self, text=("Water volume of stream flow "
                                       "(cubic metres)"),
                           bg="white", fg="brown", font=("Times", 13))
        label15.pack()
        label15.place(x=80, y=210)
        self.e15Variable = tk.DoubleVar()
        self.e15Variable.set(100.0)
        self.e15 = tk.Entry(self, textvariable=self.e15Variable,
                            bg='brown', fg='white')
        self.e15.pack()
        self.e15.place(x=450, y=213)
        label16 = tk.Label(self, text="Average monthly rainfall (mm)",
                           bg="white", fg="brown", font=("Times", 13))
        label16.pack()
        label16.place(x=80, y=240)
        self.e16Variable = tk.DoubleVar()
        self.e16Variable.set(10.0)
        self.e16 = tk.Entry(self, textvariable=self.e16Variable,
                            bg='brown', fg='white')
        self.e16.pack()
        self.e16.place(x=450, y=243)

        butnext = tk.Button(self, text='More details', font=("Times", 12),
                            bg='brown', fg='white',
                            state=tk.NORMAL, command=self.netstep)
        butnext.pack()
        butnext.place(x=650, y=243)

        label18 = tk.Label(self, text="Process Log", bg="white", fg="brown",
                           font=("Times", 10))
        label18.pack()
        label18.place(x=1000, y=180)

        self.progress_bar = tk.ttk.Progressbar(self, orient='horizontal',
                                               length=300, mode='determinate')
        self.progress_bar.grid(column=0, row=3, pady=3)
        self.progress_bar.place(x=900, y=480)

        label18 = tk.Label(self, text=("For batch testing, please select "
                                       "file (.csv) to test:"),
                           bg="white", fg="brown", font=("Times", 15))
        label18.pack()
        label18.place(x=50, y=600)
        self.ent_bp18 = tk.Entry(self, state=tk.DISABLED, width=50)
        self.ent_bp18.pack()
        self.ent_bp18.place(x=450, y=603)

        button18 = tk.Button(self, text='Browse', font=("Times", 12),
                             bg='brown', fg='white', state=tk.NORMAL,
                             command=self.file_test)
        button18.pack()
        button18.place(x=800, y=600)

        self.button = tk.Button(self, text="RUN PROGRAM",
                                font=('Times', 15, "bold"), bg='brown',
                                fg='white', command=self.ann_test)
        self.button.pack()
        self.button.place(x=800, y=650)

        button = tk.Button(self, text="Return to STEP 3", font=("Times", 14),
                           command=lambda: controller.show_frame("PageThree"))
        button.pack()
        button.place(x=30, y=650)

        self.results_txt = tk.Text(self, width=35, height=15)
        self.results_txt.pack()
        self.results_txt.place(x=900, y=200)
        self.results_txt.bind("<Key>", self.update_size)

    def netstep(self):
        """
        Net Step.

        Returns
        -------
        None.

        """
        df = self.controller.shared_data['cluster']

        if df['cluster'].max() > -1:
            tmp = df[df['cluster'] == 0]
            lonn = tmp['long'].iloc[0]
            latn = tmp['lat'].iloc[0]
        else:
            lonn = df['long'].mean()
            latn = df['lat'].mean()

        label2 = tk.Label(self, text=("Please enter the "
                                      "latitude and longitude of the location "
                                      "of interest:"),
                          bg="white", fg="brown", font=("Times", 13))
        label2.pack()
        label2.place(x=50, y=280)
        label11 = tk.Label(self, text=("Latitude (decimal degrees):"),
                           bg="white", fg="brown", font=("Times", 12))
        label11.pack()
        label11.place(x=50, y=310)

        label14 = tk.Label(self, text=("Longitude (decimal degrees):"),
                           bg="white", fg="brown", font=("Times", 12))
        label14.pack()
        label14.place(x=50, y=340)

        self.e11latVariable.set(latn)
        self.e21lonVariable.set(lonn)
        self.e11lat.pack()
        self.e11lat.place(x=300, y=313)
        self.e21lon.pack()
        self.e21lon.place(x=300, y=343)

        button18 = tk.Button(self, text='OK', font=("Times", 12), bg='brown',
                             fg='white', state=tk.NORMAL,
                             command=self.closest_eq)
        button18.pack()
        button18.place(x=450, y=310)

        label3a = tk.Label(self, text="Details of closest earthquake:",
                           bg="white", fg="brown", font=("Times", 13))
        label3a.pack()
        label3a.place(x=50, y=370)

    def closest_eq(self):
        """
        Closest eq

        Returns
        -------
        None.

        """

        # latn = "{: .3f}".format(float(self.e11latVariable.get()))
        # lonn = "{: .3f}".format(float(self.e21lonVariable.get()))

        latn = float(self.e11latVariable.get())
        lonn = float(self.e21lonVariable.get())

        locations_A = self.controller.shared_data['cluster']
        # locations_A = pd.read_csv(filein)
        # locations_new = pd.read_csv(filenew)
        locations_new =  pd.DataFrame({"lat_new": [latn], "lon_new": [lonn]})

        # add columns with radians for latitude and longitude
        locations_A[['lat_radians_A', 'long_radians_A']] = (
            np.radians(locations_A.loc[:, ['lat', 'long']]))

        locations_new[['lat_radians_B', 'long_radians_B']] = (
            np.radians(locations_new.loc[:, ['lat_new', 'lon_new']]))

        dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
        dist_matrix = (dist.pairwise(
            locations_A[['lat_radians_A', 'long_radians_A']],
            locations_new[['lat_radians_B', 'long_radians_B']])*6371.0)
        # Note that 6371.0 is the radius of the earth in kilometres
        # dz = pd.DataFrame(dist_matrix)

        # data = dz['0']
        # ftColz = dz.iloc[0:, 1].values

        # locations_A["distance to new point"] = pd.DataFrame(list(zip(*[ftColz])))

        # dx = locations_A

        locations_A["distance to new point"] = dist_matrix.flatten()

        # locations_A[locations_A['class']>0]
        # locations_A = locations_A[locations_A['cluster']>-1]

        close = locations_A[locations_A["distance to new point"] ==
                            locations_A["distance to new point"].min()]
        close = close.iloc[0]

        label3 = tk.Label(self, text=f"Distance to closest earthquake "
                          f"(metres): {close['distance to new point']}",
                          bg="white", fg="brown", font=("Times", 12))
        label3.pack()
        label3.place(x=80, y=400)

        label4 = tk.Label(self, text="Date and time of closest earthquake: "
                          f"{close['date']}", bg="white", fg="brown",
                          font=("Times", 12))
        label4.pack()
        label4.place(x=80, y=430)

        label5 = tk.Label(self, text="Magnitude of closest earthquake (ML): "
                          f"{close['mag']}",
                          bg="white", fg="brown", font=("Times", 12))
        label5.pack()
        label5.place(x=80, y=460)

        label6 = tk.Label(self, text="Cluster in which closest earthquake "
                          f"resides: {close['cluster']}",
                          bg="white", fg="brown", font=("Times", 12))
        label6.pack()
        label6.place(x=80, y=490)

        if close['cluster'].max() == -1:
            minclose = -1
        else:
            minclose = close['cluster'][close['cluster'] > -1].min()

        if minclose == -1:
            label17 = tk.Label(self, text="Magnitude of completeness of "
                               "cluster: outside the clusters",
                               bg="white", fg="brown", font=("Times", 12))
            label17.pack()
            label17.place(x=80, y=520)
        else:
            dr = self.controller.shared_data['cluster']
            df = dr.loc[dr['cluster'] == minclose]
            df = df['mag'].value_counts().to_frame('count').reset_index()
            # minmag = min(df['index'])
            minmag = df.loc[0, 'index']

            label17 = tk.Label(self, text=f"Magnitude of completeness of "
                               f"cluster: {minmag}",
                               bg="white", fg="brown", font=("Times", 12))
            label17.pack()
            label17.place(x=80, y=520)
            self.e17Variable = tk.DoubleVar()
            self.e17 = tk.Entry(self, textvariable=self.e17Variable,
                                bg='brown', fg='white')
            self.e17Variable.set(minmag)
            self.e17.pack()
            self.e17.place(x=450, y=5233)

        if minclose == -1:
            label17bv = tk.Label(self, text="B-value of cluster: outside "
                                 "the clusters", bg="white", fg="brown",
                                 font=("Times", 12))
            label17bv.pack()
            label17bv.place(x=80, y=550)
        else:
            # cluster = fr'C:\AI_SEISMOLOGY\cluster{minclose}_short.csv'
            # infilebv = pd.read_csv(cluster)

            # magnitudes = dr['mag']
            df = dr.loc[dr['cluster'] == minclose]
            # magnitudes = df['mag']
            magnitudes = df.loc[df.mag > minmag].mag


            # mag = dfa['index'].iloc[0]
            # df = df[df['index'] >= minmag]

            # years = dr.date.dt.year
            # this should be the magnitude of completeness from the script

            min_mag = magnitudes.min()

            # max_mag = max(magnitudes) + 0.1

            # num_eq = len(magnitudes)

            # num_years = max(years)-min(years)
            # annual_num_eq = num_eq/num_years

            # max_mag_bin = max(magnitudes) + 0.15

            # Magnitude bins
            # bins = np.arange(min_mag, max_mag_bin, 0.05)
            # Magnitude bins for plotting - we will re-arrange bins later
            # plot_bins = np.arange(min_mag, max_mag, 0.05)

            # #################################################################
            # Generate distribution
            # #################################################################
            # Generate histogram
            # hist = np.histogram(magnitudes, bins=bins)

            # Reverse array order
            # hist = hist[0][::-1]
            # bins = bins[::-1]

            # Calculate cumulative sum
            # cum_hist = hist.cumsum()
            # Ensure bins have the same length has the cumulative histogram.
            # Remove the upper bound for the highest interval.
            # bins = bins[1:]

            # Get annual rate
            # cum_annual_rate = cum_hist/num_years

            # new_cum_annual_rate = []
            # for i in cum_annual_rate:
            #     new_cum_annual_rate.append(i+1e-20)

            # Take logarithm
            # log_cum_sum = np.log10(new_cum_annual_rate)

            # #################################################################
            # Fit a and b parameters using a varity of methods
            # #################################################################

            # Fit a least squares curve
            # b, a = np.polyfit(bins, log_cum_sum, 1)

            # print('Least Squares: b value', -1. * b, 'a value', a)
            # alpha = np.log(10) * a
            # beta = -1.0 * np.log(10) * b
            # Maximum Likelihood Estimator fitting
            # b value
            b_mle = np.log10(np.exp(1)) / (np.mean(magnitudes) - min_mag)

            # beta_mle = np.log(10) * b_mle
            maxli = tk.Label(self, text=f'B-value of cluster: {b_mle}',
                             bg="white", fg="brown", font=("Times", 12))
            maxli.pack()
            maxli.place(x=80, y=550)

    def ann_test(self):
        """
        ANN test.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        Final_Result_L = self.out_dir.get()+'/'+'Output_Result_All.csv'

        cut_off_MG = float(self.e17.get())

        # ###################### Merge files ##################################

        self.progress_bar["maximum"] = 100
        self.progress_bar.start()
        self.progress_bar.update()
        self.progress_bar["value"] = 10

        in_rain = self.filename3.get()
        in_stream = self.filename4.get()
        in_earth = self.filename1.get()
        stream_file = self.file_stream.get()
        lineament_file = self.file_line.get()

        if len(self.ent_bp18.get()) == 0:
            my_dict = {'Rainfall': [float(self.e16.get())],
                       'Stream_flow': [float(self.e15.get())],
                       'Dis_lineament': [float(self.e11.get())],
                       'Dis_Stream': [float(self.e14.get())]}
            f_pred = pd.DataFrame(my_dict)
        else:
            f_pred = pd.read_csv(self.ent_bp18.get())

        f1 = pd.read_excel(in_rain)
        f2 = pd.read_excel(in_stream)

        file_out_var = pd.merge(f1, f2, on='date')
        # #############Convert datetime to date and time ######################

        headers = ['lat', 'long', 'depth', 'date', 'time', 'mag']
        df_mg = pd.read_excel(in_earth, usecols=headers)

        # ################Seting the lowest cutoff ############################
        mg1 = df_mg['mag'].values.reshape(-1, 1)
        z = pd.DataFrame(mg1)
        transformer = Binarizer(threshold=1).fit(z)
        output = transformer.transform(z)

        df_mg['Binary'] = output
        data = df_mg[df_mg.Binary != 0]

        points = [Point(row['long'], row['lat'])
                  for key, row in data.iterrows()]

        geo_df = GeoDataFrame(data, geometry=points)

        self.results_txt.insert(tk.INSERT, 'Merging Input Files - DONE\n')
        self.progress_bar.update()
        self.progress_bar["value"] = 25
        self.results_txt.insert(tk.INSERT, ('Running Distance '
                                            'computations >>>>>\n'))

        # #####################################################################
        # Stream File
        output_stream_var = get_distances(stream_file, geo_df, df_mg,
                                          "Dis_Stream")

        self.progress_bar.update()
        self.progress_bar["value"] = 50
        # #####################################################################
        # Lineament File
        output_line_var = get_distances(lineament_file, geo_df, df_mg,
                                        "Dis_lineament")

        self.results_txt.insert(tk.INSERT,
                                'Compute distance to Stream  - DONE\n')
        self.progress_bar.update()
        self.progress_bar["value"] = 75

        # #####################################################################
        # Merge lineaments and stream distances
        out_line_stream_var = pd.merge(output_stream_var,
                                       output_line_var,
                                       how='left', on=['long', 'lat'])

        # #####################################################################
        df1 = out_line_stream_var
        df2 = file_out_var

        df1['Date'] = pd.to_datetime(df1['date_x'])
        df2['Date'] = pd.to_datetime(df2['date'])

        lefton = df1['Date'].apply(lambda x: (x.year, x.month))
        righton = df2['Date'].apply(lambda y: (y.year, y.month))

        df4 = pd.merge(df1, df2, left_on=lefton, right_on=righton, how='outer')
        # #####################################################################

        df4['MG_Reclass'] = np.where((df4['mag_x'].astype(float) >=
                                      cut_off_MG), int(1), int(0))

        df4.sort_values(by=['MG_Reclass'], inplace=True)

        df4 = df4[['rain', 'metre', 'Dis_lineament', 'Dis_Stream',
                   'mag_y']]
        df4.rename(columns={'rain': 'Rainfall', 'metre': 'Stream_flow',
                            'mag_y': 'Magnitude'}, inplace=True)

        self.results_txt.insert(tk.INSERT,
                                'Running ANN Algorithm >>>\n')
        self.progress_bar.update()
        self.progress_bar["value"] = 85
        # #####################################################################

        mg1 = df4['Magnitude'].values.reshape(-1, 1)
        z = pd.DataFrame(mg1)
        transformer = Binarizer(threshold=2).fit(z)
        output = transformer.transform(z)
        ser = pd.DataFrame(output)
        ser.columns = ["Magnitude"]

        df1 = df4.copy()
        df1['Magnitude'] = ser['Magnitude']

        dfaa = (df1-df1.min())/(df1.max()-df1.min())

        dfaa.dropna(inplace=True, axis=0, how='all')
        Xa = dfaa[['Rainfall', 'Stream_flow', 'Dis_lineament',
                   'Dis_Stream', 'Magnitude']]
        Xa.sort_values(by=['Magnitude'], inplace=True)

        RF = df1["Rainfall"]
        MF = df1["Stream_flow"]
        DL = df1["Dis_lineament"]
        DS = df1["Dis_Stream"]

        RF1 = f_pred["Rainfall"]
        MF1 = f_pred["Stream_flow"]
        DL1 = f_pred["Dis_lineament"]
        DS1 = f_pred["Dis_Stream"]

        X11 = ((RF1-RF.min())/(RF1.max()-RF.min()))
        X21 = ((MF1-MF.min())/(MF1.max()-MF.min()))
        X31 = ((DL1-DL.min())/(DL1.max()-DL.min()))
        X41 = ((DS1-DS.min())/(DS1.max()-DS.min()))

        # The variable is always 1 for parameters on interface.
        file_pred_out_var = pd.concat([X11, X21, X31, X41], axis=1)

        X1a = file_pred_out_var.to_numpy()

        tmp = Xa.to_numpy()
        X = tmp[:, :4]
        Y = tmp[:, 4:]
        Y = Y.astype(int)
        Y = one_hot_encode(Y)

        X, Y = shuffle(X, Y, random_state=1)

        # Convert the dataset into train and test parts
        train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=415)

        # Define the important parameters and variables to work with
        # the tensors
        learning_rate = 0.3
        training_epochs = 100
        cost_history = np.empty(shape=[1], dtype=float)
        n_dim = X.shape[1]
        n_class = 2

        # Define the number of hidden layers and number of neurons for
        # each layer
        n_hidden_1 = 30
        n_hidden_2 = 30
        n_hidden_3 = 30
        n_hidden_4 = 30

        x = tf.placeholder(tf.float32, [None, n_dim])
        y_ = tf.placeholder(tf.float32, [None, n_class])

        # Define the model

        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim,
                                                   n_hidden_1])),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1,
                                                   n_hidden_2])),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2,
                                                   n_hidden_3])),
            'h4': tf.Variable(tf.truncated_normal([n_hidden_3,
                                                   n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_hidden_4,
                                                    n_class]))
            }
        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
            'out': tf.Variable(tf.truncated_normal([n_class]))
            }
        init = tf.global_variables_initializer()

        y = multilayer_perceptron(x, weights, biases)
        cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

        sess = tf.Session()
        sess.run(init)

        mse_history = []
        accuracy_history = []

        for _ in range(training_epochs):
            sess.run(training_step,
                     feed_dict={x: train_x, y_: train_y})
            cost = sess.run(cost_function,
                            feed_dict={x: train_x, y_: train_y})
            cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(y, 1),
                                          tf.argmax(y_, 1))
            accuracy = abs(tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32)))*100

            pred_y = sess.run(y, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y-test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)
            accuracy = (sess.run(accuracy, feed_dict={x: train_x,
                                                      y_: train_y}))
            accuracy_history.append(accuracy)

        test_out_history = []
        df_ct = file_pred_out_var
        X1a = X1a.reshape(df_ct.shape[0], 4)
        test_out = sess.run(y, feed_dict={x: X1a})
        test_out_history.append(test_out)

        tmp = np.transpose(test_out_history).squeeze(axis=-1)
        df_final = pd.DataFrame(tmp)
        df_final.columns = ['Result']

        plt.figure('MSE plot', figsize=(8, 6), dpi=100)
        plt.plot(mse_history, 'r', linewidth=2)
        plt.ylabel('MSE Error', fontsize=9)
        plt.xlabel('Epochs (Iterations)', fontsize=9)
        plt.tick_params(labelsize=8)
        plt.show(block=False)

        plt.figure('Accuracy', figsize=(8, 6), dpi=100)
        plt.plot(accuracy_history, linewidth=2)
        plt.ylabel('Accuracy (%)', fontsize=9)
        plt.xlabel('Epochs (Iterations)', fontsize=9)
        plt.tick_params(labelsize=8)
        plt.show(block=False)

        correct_prediction = tf.equal(tf.argmax(y, 1),
                                      tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32))
        pred_y = sess.run(y, feed_dict={x: test_x})
        mse = tf.reduce_mean(tf.square(pred_y-test_y))
        self.results_txt.insert(tk.INSERT, 'Test Accuracy: ' +
                                str(round(sess.run(accuracy,
                                                   feed_dict={x: test_x,
                                                              y_: test_y}),
                                          0)*100)+' % ' + ' ' + 'MSE: ' +
                                str(round(sess.run(mse), 3)) + '\n')
        mg_tsh = self.e17.get()

        df1 = df_final
        x1 = abs(df1['Result'][0])

        self.progress_bar.stop()
        self.progress_bar["maximum"] = 100

        if x1 > 0.5:
            x = ('For these given parameters there is a higher '
                 'chance of an earthquake with magnitude '
                 'greater than '+str(mg_tsh))

        else:
            x = ('For these given parameters there is less chance of '
                 'an earthquake with magnitude '
                 'greater than '+str(mg_tsh))

        if len(self.ent_bp18.get()) == 0:
            popup = tk.Tk()
            popup.wm_title("Results from testing")
            popup.geometry("680x210+30+30")
            popup.config(bg='white')

            s = tk.ttk.Style()
            s.configure('my.TButton', font=('Times', 12, "bold"))
            sess.close()

            label2 = tk.Label(popup, text=x, bg="white", fg="black",
                              font=("Times", 12))
            label2.pack()
            label2.place(x=30, y=50)

            button1 = tk.ttk.Button(popup, text="OK",
                                    style='my.TButton',
                                    command=popup.destroy)
            button1.pack()
            button1.place(x=350, y=160)
            popup.mainloop()

        else:
            data1 = f_pred
            data2 = df_final

            data2['Result'] = data2['Result'][data2['Result'].index %
                                              2 != 1]
            data2 = data2[data2['Result'].notna()]
            data2['Result'] = np.where((data2['Result'].astype(float)
                                        >= 0), 'YES', 'NO')
            data2 = data2[data2['Result'].notna()]

            data3 = data2.copy()
            data_out = pd.concat([data1, data3], axis=1)
            data_out.to_csv(Final_Result_L, index=False)

    def file_test(self):
        """
        File test.

        Returns
        -------
        None.

        """
        self.ent_bp18.config(state='normal')
        select_csv(self.ent_bp18)
        self.ent_bp18.config(state='normal')

    def update_size(self, event):
        """
        Update size.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        widget_width = 0
        widget_height = float(event.widget.index(tk.END))
        for line in event.widget.get("1.0", tk.END).split("\n"):
            if len(line) > widget_width:
                widget_width = len(line)+1
        event.widget.config(width=widget_width, height=widget_height)


def _testfn():
    """Test routine."""
    pbar = ProgressBarText()

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    tmp = AI_Seis(None)

    tmp.qfile['edata'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\1_earthquake_data.xlsx')
    tmp.qfile['rain'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\2_monthly_rainfall_data.xlsx')
    tmp.qfile['lineaments'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\3_geological_lineament_data.xlsx')
    tmp.qfile['streamflow'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\4_monthly_stream_flow_data.xlsx')
    tmp.qfile['streamshp'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\Vaal_River.shp')
    tmp.qfile['lineamentshp'].setText(r'C:\Work\Programming\AI\AI_SEIS\data\geo_lineament.shp')
    tmp.tabs.setTabEnabled(1, True)

    tmp.settings()


if __name__ == "__main__":
    _testfn()

    # app = SampleApp()

    # app.shared_data['filename1'].set(r'C:\Work\Programming\AI\AI_SEIS\data\1_earthquake_data.xlsx')
    # app.shared_data['filename3'].set(r'C:\Work\Programming\AI\AI_SEIS\data\2_monthly_rainfall_data.xlsx')
    # app.shared_data['filename6'].set(r'C:\Work\Programming\AI\AI_SEIS\data\3_geological_lineament_data.xlsx')
    # app.shared_data['filename4'].set(r'C:\Work\Programming\AI\AI_SEIS\data\4_monthly_stream_flow_data.xlsx')
    # app.shared_data['file_stream'].set(r'C:\Work\Programming\AI\AI_SEIS\data\Vaal_River.shp')
    # app.shared_data['file_line'].set(r'C:\Work\Programming\AI\AI_SEIS\data\geo_lineament.shp')
    # app.shared_data['out_dir'].set(r'C:\Work\Programming\AI\AI_SEIS\output')

    # app.mainloop()
