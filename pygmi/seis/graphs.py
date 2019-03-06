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
Plot Seismology Data

This module provides a variety of methods to plot raster data via the context
menu.
"""

import os
import numpy as np
from osgeo import ogr, osr
from PyQt5 import QtWidgets, QtCore
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as \
    NavigationToolbar
from matplotlib.patches import Ellipse


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot

    Attributes
    ----------
    axes : matplotlib subplot
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        self.parent = parent

        self.ellipses = []

        FigureCanvas.__init__(self, fig)

    def update_ellipse(self, datd, nodepth=False):
        """ Update error ellipse plot """

        self.figure.clear()

        x = np.ma.masked_invalid(datd['1_longitude'])
        y = np.ma.masked_invalid(datd['1_latitude'])

        xmin = x.min()-0.5
        xmax = x.max()+0.5
        ymin = y.min()-0.5
        ymax = y.max()+0.5

        self.axes = self.figure.add_subplot(111)  # , projection=ccrs.PlateCarree())
        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(ymin, ymax)

#        extent = [xmin, xmax, ymax, ymin]
#        request = cimgt.GoogleTiles(style='satellite')
#        self.axes.set_extent(extent, crs=ccrs.PlateCarree())

#        self.axes.add_image(request, 6)
#        self.axes.gridlines(draw_labels=True)

        for dat in self.parent.indata['Seis']:
            if 'E' not in dat:
                continue

            lon = dat['1'].longitude
            lat = dat['1'].latitude

            erx = dat['E'].longitude_error
            ery = dat['E'].latitude_error
            erz = dat['E'].depth_error
            cvxy = dat['E'].cov_xy
            cvxz = dat['E'].cov_xz
            cvyz = dat['E'].cov_yz

            if nodepth is True:
                cvxz = 0
                cvyz = 0
                erz = 0

            cov = np.array([[erx*erx, cvxy, cvxz],
                            [cvxy, ery*ery, cvyz],
                            [cvxz, cvyz, erz*erz]])

            if True in np.isnan(cov):
                continue

            vals, vecs = eigsorted(cov)
            abc = (2*np.sqrt(abs(vals)) *
                   np.cos(np.arctan2(vecs[2, :],
                                     np.sqrt(vecs[0, :]**2+vecs[1, :]**2))))

            if abc[0] == abc.max():
                ang = np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0]))
            if abc[1] == abc.max():
                ang = np.rad2deg(np.arctan2(vecs[1, 1], vecs[0, 1]))
            if abc[2] == abc.max():
                ang = np.rad2deg(np.arctan2(vecs[1, 2], vecs[0, 2]))

            abc[::-1].sort()  # sort in reverse order
            emaj = abc[0]
            emin = abc[1]

            # approx conversion to degrees
            demin = emin/110.93  # lat
            demaj = emaj/(111.3*np.cos(np.deg2rad(lat+demin)))  # long from lat

            ell = Ellipse(xy=(lon, lat),
                          width=demaj, height=demin,
                          angle=ang, color='black')
            ell.set_facecolor('none')
            #    ell.set_edgecolor('black')

#            self.axes.add_artist(ell)
            self.ellipses.append(ell.get_verts())
            self.axes.add_artist(ell)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_hexbin(self, data1, data2, xlbl="Time", ylbl="ML",
                      xbin=None, xrng=None):
        """
        Update the hexbin plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        data2 : PyGMI raster Data
            raster dataset to be used
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        x = np.ma.masked_invalid(data1)
        y = np.ma.masked_invalid(data2)

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        if xmin == xmax:
            xmin = xmin-1
            xmax = xmax+1
        if ymin == ymax:
            ymin = ymin-1
            ymax = ymax+1

        bins = int(np.sqrt(x.size))
        bins = (bins, bins)
        rng = [[x.min()-0.5, x.max()+0.5], [y.min(), y.max()]]

        if xbin is not None:
            bins = (xbin, bins[1])
            rng[0] = xrng
        elif np.unique(x).size < bins[0]:
            bins = (np.unique(x).size, bins[1])
        if np.unique(y).size < bins[1]:
            bins = (bins[0], np.unique(y).size)

#        hbin = self.axes.hexbin(x, y, gridsize=50, mincnt=1)
        hbin = self.axes.hist2d(x, y, bins=bins, cmin=1, range=rng)
#        self.axes.axis([xmin, xmax, ymin, ymax])
        self.axes.set_xlabel(xlbl, fontsize=8)
        self.axes.set_ylabel(ylbl, fontsize=8)
        self.axes.set_xticks(np.arange(x.min(), x.max()+1))

        cbar = self.figure.colorbar(hbin[3])
        cbar.set_label('Number of Events')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_hist(self, data1, xlbl="Data Value",
                    ylbl="Number of Observations", bins='doane', rng=None):
        """
        Update the histogram plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        """

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        dattmp = np.array(data1)
        dattmp = dattmp[~np.isnan(dattmp)]

        if np.unique(dattmp).size == 1:
            bins = 5
            rng = (np.unique(dattmp)[0]-2.5, np.unique(dattmp)[0]+2.5)

        self.axes.hist(dattmp, bins, edgecolor='black', range=rng)
        self.axes.set_xlabel(xlbl, fontsize=8)
        self.axes.set_ylabel(ylbl, fontsize=8)

        if rng is not None:
            self.axes.set_xlim(rng[0], rng[1])
            self.axes.set_xticks(np.arange(int(rng[0]+1), int(rng[1])+1))

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_bvalue(self, data1a, bins='doane'):
        """
        Update the histogram plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        data1 = np.ma.masked_invalid(data1a)

        num, bins2 = np.histogram(data1, bins)
        bins2 = bins2[:-1]+(bins2[1]-bins2[0])/2
        num2 = np.cumsum(num[::-1])[::-1]
        num3 = np.log10(num2)

        nmax = np.percentile(num3, 75)
        nmin = np.percentile(num3, 25)

        i1 = np.nonzero(num3 < nmax)[0][0]
        i2 = np.nonzero(num3 < nmin)[0][0]

        xtmp = bins2[i1:i2+1]
        ytmp = num3[i1:i2+1]
        abvals = np.polyfit(xtmp, ytmp, 1)
        aval = np.around(abvals, 2)[1]
        bval = -np.around(abvals, 2)[0]

        dattmp = data1
        self.axes.hist(dattmp, bins, edgecolor='black', label='actual')
        self.axes.set_yscale('log')
        self.axes.plot(bins2, num2, '.')

        txt = 'a-value: '+str(aval)+'\nb-value: '+str(bval)
        self.axes.plot(xtmp, 10**np.poly1d(abvals)(xtmp), 'k', label=txt)

        self.axes.set_xlabel('ML', fontsize=8)
        self.axes.set_ylabel('Number of observations', fontsize=8)
        self.axes.legend()

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_pres(self, data1, phase='P', bins='doane'):
        """
        Update the histogram plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        pid = np.array(data1['4_phase_id'])
        tres = np.array(data1['4_travel_time_residual'])

        pid = pid[~np.isnan(tres)]
        tres = tres[~np.isnan(tres)].astype(float)

        if phase == 'P':
            ptres = tres[pid == 'P   ']
        else:
            ptres = tres[pid == 'S   ']

        txt = 'mean: '+str(np.around(ptres.mean(), 3))
        txt += '\nstd: '+str(np.around(ptres.std(), 3))

        weights = 100*np.ones_like(ptres)/ptres.size
        self.axes.text(0.75, 0.9, txt, transform=self.axes.transAxes)
        self.axes.hist(ptres, 40, weights=weights, edgecolor='black')
        self.axes.set_xlabel('Time Residual (seconds)')
        self.axes.set_ylabel('Frequency (%)')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_residual(self, dat, res='ML'):
        """
        Update the histogram plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        A = {}
        T = {}
        for event in dat:
            A1 = {}
            T1 = {}
            for rec in event['4']:
                if rec.phase_id == 'IAML':
                    if rec.amplitude is None or rec.epicentral_distance is None:
                        continue
                    ML = (np.log10(rec.amplitude) +
                          1.149*np.log10(rec.epicentral_distance) +
                          0.00063*rec.epicentral_distance-2.04)
                    A1[rec.station_name] = ML
                if rec.travel_time_residual is not None:
                    T1[rec.station_name] = rec.travel_time_residual

            if A1:
                A1mean = np.mean((list(A1.values())))
            for i in A1:
                if i not in A:
                    A[i] = []
                A[i].append(A1[i]-A1mean)

            for i in T1:
                if i not in T:
                    T[i] = []
                T[i].append(T1[i])

        sname_list = list(A.keys())

        dmean = {}
        dstd = {}

        if res != 'ML':
            A = T

        for j, i in enumerate(sname_list):
            if np.nonzero(~np.isnan(A[i]))[0].size == 0:
                continue
            dmean[i] = np.nanmean(A[i])
            dstd[i] = np.nanstd(A[i])

            self.axes.errorbar(j, dmean[i], yerr=dstd[i], fmt='k+', capsize=5)

        self.axes.set_xticks(range(len(sname_list)))
        self.axes.set_xticklabels(sname_list, rotation=90)
        self.axes.set_xlabel('Station Name')
        if res != 'ML':
            self.axes.set_ylabel('Travel Time Residual (Seconds)')
        else:
            self.axes.set_ylabel('ML-mean(ML)')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_wadati(self, dat, min_wad=5, min_vps=1.53,
                      max_vps=1.93):
        """
        Update the histogram plot

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        VPS = []
        for event in dat:
            P = {}
            S = {}
            for rec in event['4']:
                time = rec.hour*3600+rec.minutes*60+rec.seconds
                if rec.phase_id == 'P   ':
                    P[rec.station_name] = time
                if rec.phase_id == 'S   ':
                    S[rec.station_name] = time

            # Make sure P and S times are from same stations
            P2 = []
            S2 = []
            for i in P:
                if i not in S:
                    continue
                P2.append(P[i])
                S2.append(S[i])

            if len(P2) <= min_wad:
                continue

            P2 = np.transpose([P2])
            S2 = np.transpose([S2])

            Pdist = cdist(P2, P2)
            Sdist = cdist(S2, S2)

            filt = np.triu(np.ones_like(Pdist, dtype=bool), 1)

            Pdist = Pdist[filt]
            Sdist = Sdist[filt]

            PSfit = linregress(Pdist, Sdist)

            if PSfit.slope < min_vps or PSfit.slope > max_vps:
                continue

            VPS.append(PSfit)
            self.axes.plot(Pdist, Sdist, '.')

        slope = np.mean([i.slope for i in VPS])
        intercept = np.mean([i.intercept for i in VPS])

        x = self.axes.get_xlim()
        self.axes.plot(x, np.poly1d([slope, intercept])(x), 'k')

        txt = 'Vp/Vs (Ave)='+str(np.around(np.mean(slope), 4))
        self.axes.text(0.1, 0.9, txt, transform=self.axes.transAxes)
        self.axes.set_xlabel('P Time (seconds)')
        self.axes.set_ylabel('S-P Time (seconds)')

        self.figure.tight_layout()
        self.figure.canvas.draw()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window - The QDialog window which will contain our image

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        super(QtWidgets.QDialog, self).__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar(self.mmc, self.parent)

        self.btn_saveshp = QtWidgets.QPushButton('Save Shapefile')

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel()
        self.label2 = QtWidgets.QLabel()
        self.label1.setText('Bands:')
        self.label2.setText('Bands:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)
        self.hbl.addWidget(self.label2)
        self.hbl.addWidget(self.combobox2)

        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)
        vbl.addWidget(self.btn_saveshp)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)
        self.btn_saveshp.clicked.connect(self.save_shp)

    def change_band(self):
        """ Combo box to choose band """
        pass

    def save_shp(self):
        """ saver shapefile """
        pass


class PlotQC(GraphWindow):
    """
    Plot Hist Class

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.label2.hide()
        self.combobox2.hide()
        self.indata = {}
        self.parent = parent
        self.datd = None

    def change_band(self):
        """ Combo box to choose band """
        self.btn_saveshp.hide()

        i = self.combobox1.currentText()
        if i == 'Hour Histogram':
            self.mmc.update_hist(self.datd['1_hour'], 'Hour', bins=24,
                                 rng=(-0.5, 23.5))
        elif i == 'Month Histogram':
            self.mmc.update_hist(self.datd['1_month'], 'Month', bins=12,
                                 rng=(0.5, 12.5))
        elif i == 'Year Histogram':
            bins = np.unique(self.datd['1_year']).size
            bmin = np.nanmin(self.datd['1_year'])-0.5
            bmax = np.nanmax(self.datd['1_year'])+0.5
            self.mmc.update_hist(self.datd['1_year'], 'Year', bins=bins,
                                 rng=(bmin, bmax))
        elif i == 'Number of Stations':
            self.mmc.update_hist(self.datd['1_number_of_stations_used'], i)
        elif i == 'RMS of time residuals':
            rts = np.array(self.datd['1_rms_of_time_residuals'])
            self.mmc.update_hist(rts, i)
        elif i == 'ML vs Time':
            self.mmc.update_hexbin(self.datd['1_ML_time'], self.datd['1_ML'],
                                   'Time (Hours)', 'ML',
                                   xbin=25, xrng=(-0.5, 24.5))
        elif i == 'ML vs Year':
            self.mmc.update_hexbin(self.datd['1_ML_year'], self.datd['1_ML'],
                                   'Year', 'ML')
        elif i == 'Error Ellipse':
            self.btn_saveshp.show()
            self.mmc.update_ellipse(self.datd)
        elif i == 'Error Ellipse (No depth errors)':
            self.btn_saveshp.show()
            self.mmc.update_ellipse(self.datd, True)
        elif i == 'GAP':
            self.mmc.update_hist(self.datd['E_gap'], i)
        elif i == 'Longitude Error':
            self.mmc.update_hist(self.datd['E_longitude_error'], i)
        elif i == 'Latitude Error':
            self.mmc.update_hist(self.datd['E_latitude_error'], i)
        elif i == 'b-Value':
            self.mmc.update_bvalue(self.datd['1_ML'])
        elif i == 'P-Phase Residuals':
            self.mmc.update_pres(self.datd, 'P')
        elif i == 'S-Phase Residuals':
            self.mmc.update_pres(self.datd, 'S')
        elif i == 'ML Residual':
            self.mmc.update_residual(self.indata['Seis'], 'ML')
        elif i == 'Time Residual':
            self.mmc.update_residual(self.indata['Seis'], 'Time')
        elif i == 'Wadati':
            self.mmc.update_wadati(self.indata['Seis'])

    def run(self):
        """ Run """
        self.show()
        data = self.indata['Seis']
        self.datd = import_for_plots(data)

        if len(self.datd) == 0:
            self.parent.showprocesslog('There is no compatible '
                                       'data in the file')
            return

        products = ['Hour Histogram',
                    'Month Histogram',
                    'Year Histogram',
                    'Number of Stations',
                    'RMS of time residuals',
                    'ML vs Time',
                    'ML vs Year',
                    'b-Value']
        if 'E_gap' in self.datd:
            products += ['Error Ellipse',
                         'Error Ellipse (No depth errors)',
                         'GAP',
                         'Longitude Error',
                         'Latitude Error']
        if '4_phase_id' in self.datd:
            products += ['P-Phase Residuals',
                         'S-Phase Residuals',
                         'ML Residual',
                         'Time Residual',
                         'Wadati']

        for i in products:
            self.combobox1.addItem(i)

        self.label1.setText('Product:')
        self.combobox1.setCurrentIndex(0)
        self.change_band()

    def save_shp(self):
        """Save shapefile """

        ext = "Shape file (*.shp)"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        ifile = str(filename)

        if os.path.isfile(ifile):
            tmp = ifile[:-4]
            os.remove(tmp+'.shp')
            os.remove(tmp+'.shx')
            os.remove(tmp+'.prj')
            os.remove(tmp+'.dbf')

        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(ifile)

        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # create the layer
        layer = data_source.CreateLayer("Fault Plane Solution", srs,
                                        ogr.wkbPolygon)
#        layer.CreateField(ogr.FieldDefn("Strike", ogr.OFTReal))

        # Calculate BeachBall
        indata = self.mmc.ellipses
        for pvert in indata:
            # Create Geometry
            outring = ogr.Geometry(ogr.wkbLinearRing)
            for i in pvert:
                outring.AddPoint(i[0], i[1])

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(outring)

            feature = ogr.Feature(layer.GetLayerDefn())

#            feature.SetField("Strike", np1[0])

            feature.SetGeometry(poly)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Destroy the feature to free resources

            feature.Destroy()

        data_source.Destroy()

        return True


def import_for_plots(dat):
    """ imports data to plot """

    datd = {}

    for event in dat:
        if '1' not in event:
            continue

        for rectype in event:
            if rectype in('1', 'E'):
                tmp = vars(event[rectype])
                for j in tmp:
                    newkey = rectype+'_'+j
                    if newkey not in datd:
                        datd[newkey] = []
                    datd[newkey].append(tmp[j])

                    if 'type_of_magnitude' in j:
                        newkey = '1_M'+tmp[j]
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp[j.split('_of_')[1]])

                        time = (tmp['hour'] + tmp['minutes']/60. +
                                tmp['seconds']/3600.)
                        newkey = '1_M'+tmp[j]+'_time'
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(time)

                        newkey = '1_M'+tmp[j]+'_year'
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp['year'])

            if rectype == '4':
                for i in event[rectype]:
                    tmp = vars(i)
                    for j in tmp:
                        newkey = rectype+'_'+j
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp[j])

    return datd


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]
