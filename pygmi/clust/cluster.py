# -----------------------------------------------------------------------------
# Name:        cluster.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2017 Council for Geoscience
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
""" This is the main Crisp Clustering set of routines """

import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np
import sklearn.cluster as skc
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from pygmi.clust.datatypes import Clust
import pygmi.menu_default as menu_default


class Cluster(QtWidgets.QDialog):
    """
    Cluster Class

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar

        self.combobox_alg = QtWidgets.QComboBox()
        self.spinbox_branchfac = QtWidgets.QSpinBox()
        self.spinbox_minsamples = QtWidgets.QSpinBox()
        self.spinbox_minclusters = QtWidgets.QSpinBox()
        self.spinbox_maxclusters = QtWidgets.QSpinBox()
        self.doublespinbox_maxerror = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_eps = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_bthres = QtWidgets.QDoubleSpinBox()
        self.spinbox_maxiterations = QtWidgets.QSpinBox()
        self.radiobutton_sscale = QtWidgets.QRadioButton()
        self.radiobutton_rscale = QtWidgets.QRadioButton()
        self.radiobutton_noscale = QtWidgets.QRadioButton()

        self.name = "Clustering"
        self.cltype = 'k-means'
        self.min_cluster = 5
        self.max_cluster = 5
        self.max_iter = 300
        self.tol = 0.0001
        self.runs = 1
        self.log = ''
        self.eps = 0.5
        self.min_samples = 5
        self.bthres = 0.5
        self.branchfac = 50

        self.setupui()

        self.combobox_alg.addItems(['k-means', 'DBSCAN', 'Birch'])
        self.combobox_alg.currentIndexChanged.connect(self.combo)
        self.combo()

        self.reportback = self.parent.showprocesslog

    def setupui(self):
        """ setup UI """
        helpdocs = menu_default.HelpButton('pygmi.clust.cluster')
        gridlayout = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        label = QtWidgets.QLabel()
        self.label_minclusters = QtWidgets.QLabel()
        self.label_maxclusters = QtWidgets.QLabel()
        self.label_maxiter = QtWidgets.QLabel()
        self.label_maxerror = QtWidgets.QLabel()
        self.label_eps = QtWidgets.QLabel()
        self.label_minsamples = QtWidgets.QLabel()
        self.label_bthres = QtWidgets.QLabel()
        self.label_branchfac = QtWidgets.QLabel()

        self.spinbox_minclusters.setMinimum(1)
        self.spinbox_minclusters.setProperty("value", self.min_cluster)
        self.spinbox_maxclusters.setMinimum(1)
        self.spinbox_maxclusters.setProperty("value", self.max_cluster)
        self.spinbox_maxiterations.setMinimum(1)
        self.spinbox_maxiterations.setMaximum(1000)
        self.spinbox_maxiterations.setProperty("value", self.max_iter)
        self.spinbox_minsamples.setMinimum(2)
        self.spinbox_minsamples.setProperty("value", self.min_samples)
        self.doublespinbox_eps.setDecimals(5)
        self.doublespinbox_eps.setProperty("value", self.eps)
        self.doublespinbox_eps.setSingleStep(0.1)
        self.doublespinbox_maxerror.setDecimals(5)
        self.doublespinbox_maxerror.setProperty("value", self.tol)
        self.radiobutton_sscale.setChecked(True)
        self.spinbox_branchfac.setMinimum(2)
        self.spinbox_branchfac.setProperty("value", self.branchfac)
        self.doublespinbox_bthres.setDecimals(5)
        self.doublespinbox_bthres.setProperty("value", self.bthres)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle("Cluster Analysis")
        label.setText("Cluster Algorithm:")
        self.label_minclusters.setText("Minimum Clusters:")
        self.label_maxclusters.setText("Maximum Clusters")
        self.label_maxiter.setText("Maximum Iterations:")
        self.label_maxerror.setText("Tolerance:")
        self.label_eps.setText("eps:")
        self.label_minsamples.setText("Minimum Samples:")
        self.radiobutton_noscale.setText("No Scaling")
        self.radiobutton_sscale.setText("Standard Scaling")
        self.radiobutton_rscale.setText("Robust Scaling")
        self.label_branchfac.setText("Branching Factor:")
        self.label_bthres.setText("Threshold:")

        gridlayout.addWidget(label, 0, 2, 1, 1)
        gridlayout.addWidget(self.combobox_alg, 0, 4, 1, 1)
        gridlayout.addWidget(self.label_minclusters, 1, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_minclusters, 1, 4, 1, 1)
        gridlayout.addWidget(self.label_maxclusters, 2, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_maxclusters, 2, 4, 1, 1)
        gridlayout.addWidget(self.label_maxiter, 3, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_maxiterations, 3, 4, 1, 1)
        gridlayout.addWidget(self.label_maxerror, 4, 2, 1, 1)
        gridlayout.addWidget(self.doublespinbox_maxerror, 4, 4, 1, 1)
        gridlayout.addWidget(self.label_bthres, 3, 2, 1, 1)
        gridlayout.addWidget(self.doublespinbox_bthres, 3, 4, 1, 1)
        gridlayout.addWidget(self.label_branchfac, 4, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_branchfac, 4, 4, 1, 1)
        gridlayout.addWidget(self.label_eps, 1, 2, 1, 1)
        gridlayout.addWidget(self.doublespinbox_eps, 1, 4, 1, 1)
        gridlayout.addWidget(self.label_minsamples, 2, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_minsamples, 2, 4, 1, 1)
        gridlayout.addWidget(self.radiobutton_noscale, 7, 2, 1, 1)
        gridlayout.addWidget(self.radiobutton_sscale, 8, 2, 1, 1)
        gridlayout.addWidget(self.radiobutton_rscale, 9, 2, 1, 1)
        gridlayout.addWidget(helpdocs, 10, 2, 1, 1)
        gridlayout.addWidget(buttonbox, 10, 4, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def combo(self):
        """ Combo box """
        i = str(self.combobox_alg.currentText())

        self.label_minclusters.hide()
        self.spinbox_minclusters.hide()
        self.label_maxclusters.hide()
        self.spinbox_maxclusters.hide()
        self.label_maxerror.hide()
        self.doublespinbox_maxerror.hide()
        self.label_maxiter.hide()
        self.spinbox_maxiterations.hide()
        self.label_minsamples.hide()
        self.spinbox_minsamples.hide()
        self.label_eps.hide()
        self.doublespinbox_eps.hide()
        self.label_branchfac.hide()
        self.spinbox_branchfac.hide()
        self.label_bthres.hide()
        self.doublespinbox_bthres.hide()

        if i == 'DBSCAN':
            self.label_eps.show()
            self.label_minsamples.show()
            self.spinbox_minsamples.show()
            self.doublespinbox_eps.show()
        elif i == 'k-means':
            self.label_minclusters.show()
            self.spinbox_minclusters.show()
            self.label_maxclusters.show()
            self.spinbox_maxclusters.show()
            self.label_maxerror.show()
            self.doublespinbox_maxerror.show()
            self.label_maxiter.show()
            self.spinbox_maxiterations.show()
        elif i == 'Birch':
            self.label_minclusters.show()
            self.spinbox_minclusters.show()
            self.label_maxclusters.show()
            self.spinbox_maxclusters.show()
            self.label_branchfac.show()
            self.spinbox_branchfac.show()
            self.label_bthres.show()
            self.doublespinbox_bthres.show()

    def settings(self):
        """ Settings """
        tst = np.unique([i.data.shape for i in self.indata['Raster']])

        if tst.size > 2:
            self.reportback('Error: Your input datasets have different ' +
                            'sizes. Merge the data first')
            return False

        self.min_samples = len(self.indata['Raster'])+1
        self.spinbox_minsamples.setProperty("value", self.min_samples)

        temp = self.exec_()
        if temp == 0:
            return False

        self.parent.process_is_active()
        self.run()
        self.parent.process_is_active(False)
        self.pbar.to_max()
        return True

    def update_vars(self):
        """ Updates the variables """
        self.cltype = str(self.combobox_alg.currentText())
        self.min_cluster = self.spinbox_minclusters.value()
        self.max_cluster = self.spinbox_maxclusters.value()
        self.max_iter = self.spinbox_maxiterations.value()
        self.tol = self.doublespinbox_maxerror.value()
        self.eps = self.doublespinbox_eps.value()
        self.min_samples = self.spinbox_minsamples.value()
        self.bthres = self.doublespinbox_bthres.value()
        self.branchfac = self.spinbox_branchfac.value()

    def run(self):
        """ Process data """
        data = copy.copy(self.indata['Raster'])
        self.update_vars()

        no_clust = range(self.min_cluster, self.max_cluster+1)

        self.reportback('Cluster analysis started')

# Section to deal with different bands having different null values.
        masktmp = data[0].data.mask
        for i in data:
            masktmp += i.data.mask
        for i, _ in enumerate(data):
            data[i].data.mask = masktmp
        X = np.array([i.data.compressed() for i in data]).T

        if self.radiobutton_sscale.isChecked():
            X = skp.StandardScaler().fit_transform(X)
        elif self.radiobutton_rscale.isChecked():
            X = skp.RobustScaler().fit_transform(X)

        dat_out = []
        for i in self.pbar.iter(no_clust):
            if self.cltype != 'DBSCAN':
                self.reportback('Number of Clusters:'+str(i))
            elif i > no_clust[0]:
                continue

            if self.cltype == 'k-means':
                cfit = skc.MiniBatchKMeans(n_clusters=i, tol=self.tol,
                                           max_iter=self.max_iter).fit(X)
            elif self.cltype == 'DBSCAN':
                cfit = skc.DBSCAN(eps=self.eps,
                                  min_samples=self.min_samples).fit(X)

            elif self.cltype == 'Birch':
                cfit = skc.Birch(n_clusters=i, threshold=self.bthres,
                                 branching_factor=self.branchfac).fit(X)

            dat_out.append(Clust())
            for k in data:
                dat_out[-1].input_type.append(k.dataid)

            zonal = np.ma.masked_all(data[0].data.shape)
            alpha = (data[0].data.mask == 0)
            zonal[alpha == 1] = cfit.labels_

            dat_out[-1].data = zonal
            dat_out[-1].nullvalue = zonal.fill_value
            dat_out[-1].no_clusters = i
            dat_out[-1].center = np.zeros([i, len(data)])
            dat_out[-1].center_std = np.zeros([i, len(data)])
            if cfit.labels_.max() > -1:
                dat_out[-1].vrc = skm.calinski_harabaz_score(X, cfit.labels_)

            if self.cltype == 'k-means':
                dat_out[-1].center = np.array(cfit.cluster_centers_)

            self.log = ("Cluster complete" + ' (' + self.cltype+')')

        for i in dat_out:
            i.xdim = data[0].xdim
            i.ydim = data[0].ydim
            i.dataid = 'Clusters: '+str(i.no_clusters)
            if self.cltype == 'DBSCAN':
                i.dataid = 'Clusters: '+str(int(i.data.max()+1))
            i.nullvalue = data[0].nullvalue
            i.extent = data[0].extent

        self.reportback("Cluster complete" + ' ('+self.cltype + ' ' + ')')

        for i in dat_out:
            i.data += 1
            i.data = i.data.astype(np.uint8)

        self.outdata['Cluster'] = dat_out
        self.outdata['Raster'] = self.indata['Raster']

        return True
