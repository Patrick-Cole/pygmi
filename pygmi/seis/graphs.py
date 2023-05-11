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
Plot Seismology Data.

This module provides a variety of methods to plot raster data via the context
menu.
"""

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.patches import Ellipse
import contextily as ctx
import pyproj

from pygmi.misc import ContextModule


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    axes : matplotlib axes
        axes for matplotlib subplot
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):

        fig = Figure()
        self.axes = fig.add_subplot(111)

        self.ellipses = []

        super().__init__(fig)

    def update_ellipse(self, datd, dats, nodepth=False):
        """
        Update error ellipse plot.

        Parameters
        ----------
        datd : dictionary
            Dictionary containing latitudes and longitudes
        dats : list
            Data list.
        nodepth : bool, optional
            Flag to determine if there are depths. The default is False.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        if len(datd) == 0:
            self.figure.canvas.draw()
            return

        x = np.ma.masked_invalid(datd['1_longitude'])
        y = np.ma.masked_invalid(datd['1_latitude'])

        xmin = x.min()-0.5
        xmax = x.max()+0.5
        ymin = y.min()-0.5
        ymax = y.max()+0.5

        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(ymin, ymax)

        for dat in dats:
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

            self.ellipses.append(ell.get_verts())
            self.axes.add_artist(ell)


        try:
            ctx.add_basemap(self.axes, crs=pyproj.CRS.from_epsg(4326),
                            source=ctx.providers.OpenStreetMap.Mapnik)
        except:
            print('No internet')


        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_hexbin(self, data1, data2, xlbl='Time', ylbl='ML',
                      xbin=None, xrng=None):
        """
        Update the hexbin plot.

        Parameters
        ----------
        data1 : numpy array
            raster dataset to be used
        data2 : numpy array
            raster dataset to be used
        xlbl : str, optional
            X-axis label. The default is 'Time'.
        ylbl : str, optional
            Y-axis label. The default is 'ML'.
        xbin : int, optional
            Number of bins in the x direction. The default is None.
        xrng : list, optional
            X-range. The default is None.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        if len(data1) == 0 or len(data2) == 0:
            self.figure.canvas.draw()
            return

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

        hbin = self.axes.hist2d(x, y, bins=bins, cmin=1, range=rng)
        self.axes.set_xlabel(xlbl, fontsize=8)
        self.axes.set_ylabel(ylbl, fontsize=8)
        self.axes.set_xticks(np.arange(x.min(), x.max()+1))

        for tick in self.axes.get_xticklabels():
            tick.set_rotation(90)

        cbar = self.figure.colorbar(hbin[3])
        cbar.set_label('Number of Events')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_hist(self, data1, xlbl='Data Value',
                    ylbl='Number of Observations', bins='doane', rng=None):
        """
        Update the histogram plot.

        Parameters
        ----------
        data1 : numpy array.
            raster dataset to be used
        xlbl : str, optional
            X-axis label. The default is 'Data Value'.
        ylbl : str, optional
            Y-axis label. The default is 'Number of Observations'.
        bins : int or str, optional
            Number of bins or binning strategy. See matplotlib.pyplot.hist.
            The default is 'doane'.
        rng : tuple or None, optional
            Bin range. The default is None.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        if len(data1) == 0:
            self.figure.canvas.draw()
            return

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

        for tick in self.axes.get_xticklabels():
            tick.set_rotation(90)

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_bvalue(self, data1a, bins='doane'):
        """
        Update the b value plot.

        Parameters
        ----------
        data1a : numpy array
            Data array.
        bins : int or str, optional
            Number of bins or binning strategy. See matplotlib.pyplot.hist.
            The default is 'doane'.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        if len(data1a) == 0:
            self.figure.canvas.draw()
            return

        data1 = np.ma.masked_invalid(data1a)
        data1 = data1.compressed()

        # Magnitude of completeness
        dval, dcnt = np.unique(data1, return_counts=True)
        cmax = dval[dcnt == dcnt.max()][0]

        # Least squares a and b value
        bins = np.arange(data1.min()-0.05, data1.max()+.15, 0.1)

        num, bins2 = np.histogram(data1, bins)
        bins2 = bins2[:-1] + 0.05
        bins2 = np.round(bins2, 1)  # gets rid of round off error.

        num2 = np.cumsum(num[::-1])[::-1]
        num3 = np.log10(num2)

        xtmp = bins2[bins2 >= cmax]
        ytmp = num3[bins2 >= cmax]

        abvals = np.polyfit(xtmp, ytmp, 1)
        aval = np.around(abvals, 2)[1]
        bval = -np.around(abvals, 2)[0]

        # Maximum likelihood
        data2 = data1[data1 >= cmax]
        b_mle = np.log10(np.exp(1)) / (data2.mean() - data2.min())
        b_mle = np.around(b_mle, 2)

        # Plotting
        self.axes.hist(data1, bins, edgecolor='black',
                       label='Actual distribution')
        self.axes.set_yscale('log')
        self.axes.plot(bins2, num2, '.', label='Cumulative distribution')

        self.axes.plot([cmax, cmax], [0, num2.max()], 'k--',
                       label=f'Magnitude of completeness: {cmax}\n')

        txt = (f'a-value (Least Squares): {aval}\n'
               f'b-value (Least Squares): {bval}\n'
               f'b-value (Maximum Likelihood): {b_mle}')
        self.axes.plot(xtmp, 10**np.poly1d(abvals)(xtmp), 'k', label=txt)

        self.axes.set_xlabel('ML', fontsize=8)
        self.axes.set_ylabel('Number of observations', fontsize=8)

        self.axes.legend()

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def update_pres(self, data1, phase='P'):
        """
        Update the plot.

        Parameters
        ----------
        data1 : numpy array
            Data array.
        phase : str, optional
            Phase. The default is 'P'.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        if len(data1) == 0:
            self.figure.canvas.draw()
            return

        pid = np.array(data1['4_phase_id'])
        tres = np.array(data1['4_travel_time_residual'])
        weight = np.array(data1['4_weighting_indicator'])
        pid = pid[weight != 9]
        tres = tres[weight != 9]

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
        Update the residual plot.

        Parameters
        ----------
        data1 : numpy array
            Data array.
        res : str, optional
            Response type. The default is 'ML'.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        if len(dat) == 0:
            self.figure.canvas.draw()
            return

        A = {}
        T = {}
        for event in dat:
            A1 = {}
            T1 = {}
            for rec in event['4']:
                if (rec.quality+rec.phase_id).strip() in ['IAML', 'AML',
                                                          'ES', 'E']:
                    if (rec.amplitude is None or
                            rec.epicentral_distance is None):
                        continue
                    ML = (np.log10(rec.amplitude) +
                          1.149*np.log10(rec.epicentral_distance) +
                          0.00063*rec.epicentral_distance-2.04)
                    A1[rec.station_name] = ML
                if rec.travel_time_residual is not None:
                    T1[rec.station_name] = rec.travel_time_residual
            if A1:
                if np.nonzero(~np.isnan(list(A1.values())))[0].size == 0:
                    A1mean = 0
                else:
                    A1mean = np.nanmean(list(A1.values()))
            for i in A1:
                if i not in A:
                    A[i] = []
                A[i].append(A1[i]-A1mean)

            for i in T1:
                if i not in T:
                    T[i] = []
                T[i].append(T1[i])

        dmean = {}
        dstd = {}

        sname_list = list(A.keys())

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
        Update the wadati plot.

        Parameters
        ----------
        dat : list
            List of events.
        min_wad : int, optional
            Minimum data length for plot. The default is 5.
        min_vps : float, optional
            Minimum VPS. The default is 1.53.
        max_vps : float, optional
            Maximum VPS. The default is 1.93.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        if len(dat) == 0:
            self.figure.canvas.draw()
            return

        VPS = []
        for event in dat:
            P = {}
            S = {}
            for rec in event['4']:
                if rec.weighting_indicator == 9:
                    continue
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


class GraphWindow(ContextModule):
    """Graph Window - The QDialog window which will contain our image."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Graph Window')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.btn_saveshp = QtWidgets.QPushButton('Save Shapefile')

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel('Bands:')
        self.label2 = QtWidgets.QLabel('Bands:')
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
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """

    def save_shp(self):
        """
        Save shapefile.

        Returns
        -------
        None.

        """


class PlotQC(GraphWindow):
    """
    Plot Hist Class.

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.label2.hide()
        self.combobox2.hide()
        self.datd = None

    def change_band(self):
        """
        Combo box to choose band.

        Returns
        -------
        None.

        """
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
            bins = np.unique(self.datd['1_number_of_stations_used']).size
            bmin = np.nanmin(self.datd['1_number_of_stations_used'])+0.5
            bmax = np.nanmax(self.datd['1_number_of_stations_used'])+1.5
            self.mmc.update_hist(self.datd['1_number_of_stations_used'],
                                 i, bins=bins, rng=(bmin, bmax))

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
            self.mmc.update_ellipse(self.datd, self.indata['Seis'])
        elif i == 'Error Ellipse (No depth errors)':
            self.btn_saveshp.show()
            self.mmc.update_ellipse(self.datd, self.indata['Seis'], True)
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
        """
        Entry point to run routine.

        Returns
        -------
        None.

        """
        self.show()
        data = self.indata['Seis']
        self.datd = import_for_plots(data)

        if not self.datd:
            self.showlog('There is no compatible data in the file')
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
        """
        Save shapefile.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        ifile = str(filename)

        if os.path.isfile(ifile):
            tmp = ifile[:-4]
            os.remove(tmp+'.shp')
            os.remove(tmp+'.shx')
            os.remove(tmp+'.prj')
            os.remove(tmp+'.dbf')

        indata = self.mmc.ellipses
        geom = [Polygon(i) for i in indata]

        gdict = {'geometry': geom}

        gdf = gpd.GeoDataFrame(gdict)
        gdf = gdf.set_crs(4326)

        gdf.to_file(filename)

        return True


def import_for_plots(dat):
    """
    Import data to plot.

    Parameters
    ----------
    dat : list
        List of events.

    Returns
    -------
    datd : dictionary
        Dictionary of data to plot.

    """
    datd = {}

    # Next 3 lines are so that certain plots don't break
    datd['1_ML'] = []
    datd['1_ML_year'] = []
    datd['1_ML_time'] = []

    for event in dat:
        if '1' not in event:
            continue

        for rectype in event:
            if rectype in ('1', 'E'):
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
    """
    Calculate and sort eigenvalues.

    Parameters
    ----------
    cov : numpy array
        matrix to perform calculations on.

    Returns
    -------
    vals : numpy array
        Sorted eigenvalues.
    vecs : numpy array
        Sorted eigenvectors.

    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def _testfn():
    """Test routine."""
    import sys
    from pygmi.seis.iodefs import ImportSeisan

    app = QtWidgets.QApplication(sys.argv)
    tmp = ImportSeisan()
    tmp.ifile = r"D:\Workdata\PyGMI Test Data\Seismology\collect2.out"
    tmp.settings(True)

    data = tmp.outdata['Seis']

    dat = import_for_plots(data)

    tmp = PlotQC()
    tmp.indata['Seis'] = data
    tmp.run()
    tmp.exec_()


def _testfn2():
    """Test for wave files."""
    import sys
    import matplotlib.pyplot as plt
    from pygmi.seis.iodefs import ImportSeisan

    ifile = r'D:\Workdata\seismology\april2021\collect.out'

    app = QtWidgets.QApplication(sys.argv)
    tmp = ImportSeisan()
    tmp.ifile = ifile
    tmp.settings(True)

    data = tmp.outdata['Seis']

    ifile = r'D:\Workdata\seismology\april2021\mulplt.wav'

    with open(ifile, encoding='utf-8') as pntfile:
        ltmp = pntfile.read()

    ltmp = ltmp.split('\n')

    l1 = ltmp.pop(0)
    l2 = ltmp.pop(0)

    while len(ltmp) > 2:
        h1 = ltmp.pop(0)
        h2 = ltmp.pop(0).split()

        samples = int(h2[0])
        rate = float(h2[1])
        comp = h2[5]
        year = h2[6]
        month = h2[7]
        day = h2[8]

        lines = samples // 7 + 1
        y = ''.join(ltmp[:lines]).split()
        y = np.array(y, dtype=float)
        x = np.arange(0, samples*rate, rate)

        ltmp = ltmp[lines:]

        plt.title(comp)
        plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    _testfn()
