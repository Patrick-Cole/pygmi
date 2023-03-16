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
"""Main Clustering set of routines."""

import copy
import os
from PyQt5 import QtWidgets, QtCore
import numpy as np
import sklearn.cluster as skc
from sklearn.metrics import calinski_harabasz_score
import sklearn.preprocessing as skp

from pygmi.raster.datatypes import Data
from pygmi import menu_default
from pygmi.misc import BasicModule


class Cluster(BasicModule):
    """Cluster Class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combobox_alg = QtWidgets.QComboBox()
        self.spinbox_branchfac = QtWidgets.QSpinBox()
        self.spinbox_minsamples = QtWidgets.QSpinBox()
        self.spinbox_minclusters = QtWidgets.QSpinBox()
        self.spinbox_maxclusters = QtWidgets.QSpinBox()
        self.doublespinbox_maxerror = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_eps = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_bthres = QtWidgets.QDoubleSpinBox()
        self.spinbox_maxiterations = QtWidgets.QSpinBox()
        self.radiobutton_sscale = QtWidgets.QRadioButton('Standard Scaling')
        self.radiobutton_rscale = QtWidgets.QRadioButton('Robust Scaling')
        self.radiobutton_noscale = QtWidgets.QRadioButton('No Scaling')
        self.label_minclusters = QtWidgets.QLabel('Minimum Clusters:')
        self.label_maxclusters = QtWidgets.QLabel('Maximum Clusters:')
        self.label_maxiter = QtWidgets.QLabel('Maximum Iterations:')
        self.label_maxerror = QtWidgets.QLabel('Tolerance:')
        self.label_eps = QtWidgets.QLabel('eps:')
        self.label_minsamples = QtWidgets.QLabel('Minimum Samples:')
        self.label_bthres = QtWidgets.QLabel('Threshold:')
        self.label_branchfac = QtWidgets.QLabel('Branching Factor:')

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

        self.combobox_alg.addItems(['Mini Batch K-Means (fast)', 'K-Means',
                                    'DBSCAN', 'Birch'])
        self.combobox_alg.currentIndexChanged.connect(self.combo)
        self.combo()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        helpdocs = menu_default.HelpButton('pygmi.clust.cluster')
        gridlayout = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        label = QtWidgets.QLabel('Cluster Algorithm:')

        self.spinbox_minclusters.setMinimum(1)
        self.spinbox_minclusters.setProperty('value', self.min_cluster)
        self.spinbox_maxclusters.setMinimum(1)
        self.spinbox_maxclusters.setProperty('value', self.max_cluster)
        self.spinbox_maxiterations.setMinimum(1)
        self.spinbox_maxiterations.setMaximum(1000)
        self.spinbox_maxiterations.setProperty('value', self.max_iter)
        self.spinbox_minsamples.setMinimum(2)
        self.spinbox_minsamples.setProperty('value', self.min_samples)
        self.doublespinbox_eps.setDecimals(5)
        self.doublespinbox_eps.setProperty('value', self.eps)
        self.doublespinbox_eps.setSingleStep(0.1)
        self.doublespinbox_maxerror.setDecimals(5)
        self.doublespinbox_maxerror.setProperty('value', self.tol)
        self.radiobutton_sscale.setChecked(True)
        self.spinbox_branchfac.setMinimum(2)
        self.spinbox_branchfac.setProperty('value', self.branchfac)
        self.doublespinbox_bthres.setDecimals(5)
        self.doublespinbox_bthres.setProperty('value', self.bthres)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Cluster Analysis')

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
        """
        Set up combo box.

        Returns
        -------
        None.

        """
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
        elif i in ['K-Means', 'Mini Batch K-Means (fast)']:
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
            self.showprocesslog('No Raster Data.')
            return False

        tst = np.unique([i.data.shape for i in self.indata['Raster']])

        if tst.size > 2:
            self.showprocesslog('Error: Your input datasets have different '
                                'sizes. Merge the data first')
            return False

        self.min_samples = len(self.indata['Raster'])+1
        self.spinbox_minsamples.setProperty('value', self.min_samples)

        if not nodialog:
            temp = self.exec_()
            if temp == 0:
                return False
            if self.parent is not None:
                self.parent.process_is_active()

        flag = self.run()

        if not nodialog and self.parent is not None:
            self.parent.process_is_active(False)
            self.parent.pbar.to_max()
        return flag

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
        self.combobox_alg.setCurrentText(projdata['cltype'])
        self.spinbox_minclusters.setProperty('value', projdata['min_cluster'])
        self.spinbox_maxclusters.setProperty('value', projdata['max_cluster'])
        self.spinbox_maxiterations.setProperty('value', projdata['max_iter'])
        self.doublespinbox_maxerror.setProperty('value', projdata['tol'])
        self.doublespinbox_eps.setProperty('value', projdata['eps'])
        self.spinbox_minsamples.setProperty('value', projdata['min_samples'])
        self.doublespinbox_bthres.setProperty('value', projdata['bthres'])
        self.spinbox_branchfac.setProperty('value', projdata['branchfac'])

        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        self.update_vars()
        projdata = {}

        projdata['cltype'] = self.cltype
        projdata['min_cluster'] = self.min_cluster
        projdata['max_cluster'] = self.max_cluster
        projdata['max_iter'] = self.max_iter
        projdata['tol'] = self.tol
        projdata['eps'] = self.eps
        projdata['min_samples'] = self.min_samples
        projdata['bthres'] = self.bthres
        projdata['branchfac'] = self.branchfac

        return projdata

    def update_vars(self):
        """
        Update the variables.

        Returns
        -------
        None.

        """
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
        """
        Run the cluster analysis.

        Returns
        -------
        None.

        """
        data = self.indata['Raster']
        self.update_vars()

        no_clust = range(self.min_cluster, self.max_cluster+1)

        self.showprocesslog('Cluster analysis started')

        # Section to deal with different bands having different null values.
        masktmp = ~data[0].data.mask
        for i in data:
            masktmp += ~i.data.mask
        masktmp = ~masktmp

        X = []
        for band in data:
            tmp = copy.deepcopy(band)
            tmp.data.mask = masktmp
            X.append(tmp.data.compressed())
            del tmp
        X = np.transpose(X)

        if self.radiobutton_sscale.isChecked():
            self.showprocesslog('Applying standard scaling')
            X = skp.StandardScaler().fit_transform(X)
        elif self.radiobutton_rscale.isChecked():
            self.showprocesslog('Applying robust scaling')
            X = skp.RobustScaler().fit_transform(X)

        dat_out = []
        for i in self.piter(no_clust):
            if self.cltype != 'DBSCAN':
                self.showprocesslog('Number of Clusters:'+str(i))
            elif i > no_clust[0]:
                continue

            if self.cltype == 'Mini Batch K-Means (fast)':
                bsize = max(os.cpu_count()*256, 1024)
                cfit = skc.MiniBatchKMeans(n_clusters=i, tol=self.tol,
                                           max_iter=self.max_iter,
                                           # n_init='auto',
                                           batch_size=bsize).fit(X)
            elif self.cltype == 'K-Means':
                cfit = skc.KMeans(n_clusters=i, tol=self.tol, # n_init='auto',
                                  max_iter=self.max_iter).fit(X)

            elif self.cltype == 'DBSCAN':
                cfit = skc.DBSCAN(eps=self.eps,
                                  min_samples=self.min_samples).fit(X)

            elif self.cltype == 'Birch':
                X = np.ascontiguousarray(X)  # Birch gave an error without this
                cfit = skc.Birch(n_clusters=i, threshold=self.bthres,
                                 branching_factor=self.branchfac).fit(X)

            if cfit.labels_.max() < i-1 and self.cltype != 'DBSCAN':
                self.showprocesslog('Could not find '+str(i)+' clusters. '
                                    'Please change settings.')

                return False
            if cfit.labels_.max() < 0 and self.cltype == 'DBSCAN':
                self.showprocesslog('Could not find any clusters. '
                                    'Please change settings.')

                return False

            dat_out.append(Data())

            dat_out[-1].metadata['Cluster']['input_type'] = []
            for k in data:
                dat_out[-1].metadata['Cluster']['input_type'].append(k.dataid)

            zonal = np.ma.masked_all(data[0].data.shape)
            zonal[~masktmp] = cfit.labels_

            dat_out[-1].data = zonal
            dat_out[-1].nodata = zonal.fill_value
            dat_out[-1].metadata['Cluster']['no_clusters'] = i
            dat_out[-1].metadata['Cluster']['center'] = np.zeros([i, len(data)])
            dat_out[-1].metadata['Cluster']['center_std'] = np.zeros([i, len(data)])
            if cfit.labels_.max() > 0:
                dat_out[-1].metadata['Cluster']['vrc'] = calinski_harabasz_score(X, cfit.labels_)

            # Reloading this hear to save memory. Need unscaled values.
            X = []
            for band in data:
                tmp = copy.deepcopy(band)
                tmp.data.mask = masktmp
                X.append(tmp.data.compressed())
            X = np.transpose(X)

            m = []
            s = []
            for i2 in range(cfit.labels_.max()+1):
                m.append(X[cfit.labels_ == i2].mean(0))
                s.append(X[cfit.labels_ == i2].std(0))

            dat_out[-1].metadata['Cluster']['center'] = np.array(m)
            dat_out[-1].metadata['Cluster']['center_std'] = np.array(s)

            self.log = ('Cluster complete' + ' (' + self.cltype+')')

        for i in dat_out:
            i.dataid = 'Clusters: '+str(i.metadata['Cluster']['no_clusters'])
            if self.cltype == 'DBSCAN':
                i.dataid = 'Clusters: '+str(int(i.data.max()+1))
            i.nodata = data[0].nodata
            i.set_transform(transform=data[0].transform)
            i.crs = data[0].crs

        self.showprocesslog('Cluster complete' + ' ('+self.cltype + ' ' + ')')

        for i in dat_out:
            i.data += 1
            i.data = i.data.astype(np.uint8)
            i.nodata = 0

        self.outdata['Cluster'] = dat_out
        self.outdata['Raster'] = self.indata['Raster']

        return True


def _testfn():
    import sys
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster

    ifile = r"D:\Workdata\PyGMI Test Data\Classification\Cut_K_Th_U.ers"

    dat = get_raster(ifile)

    app = QtWidgets.QApplication(sys.argv)

    DM = Cluster()
    DM.indata['Raster'] = dat
    DM.settings()

    dat2 = DM.outdata['Raster']

    plt.figure(dpi=150)
    plt.imshow(dat2[0].data)
    plt.show()


def _test_marinda():
    import sys
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from pygmi.raster.iodefs import get_raster

    # Import Data
    ifile = r"D:\Workdata\testdata.hdr"

    dat = get_raster(ifile)

    dat2 = []
    for i in dat:
        if i.dataid in ['k: 1', 'th: 1', 'u: 1']:
            dat2.append(i)

    # Generate layers via cluster analysis
    app = QtWidgets.QApplication(sys.argv)

    DM = Cluster()
    DM.indata['Raster'] = dat2
    DM.spinbox_maxclusters.setProperty('value', 9)

    DM.settings()

    cdata = DM.outdata['Cluster']

    # Get cluster centers
    centers = {}
    for dat in cdata:
        num = dat.metadata['Cluster']['no_clusters']
        centers[num] = dat.metadata['Cluster']['center']

    icenter = sorted(list(centers.keys()))

    # Master center is dataset with most classes..
    master = icenter[-1]
    dist = {}
    measure = 'euclidean'

    # Relabel classes using class centers from 9 class dataset.
    for j in icenter:
        dist[j] = np.argmin(cdist(centers[master], centers[j], measure), 0)
        dist[j] = dist[j] + 1

    cdata2 = []
    for dat in cdata:
        cdata2.append(np.zeros_like(dat.data))
        cnum = dat.metadata['Cluster']['no_clusters']
        for i in range(cnum):
            cdata2[-1][dat.data == i+1] = dist[cnum][i]
        cdata2[-1] = np.ma.masked_equal(cdata2[-1], 0)

    for dat in cdata2:
        plt.figure(dpi=150)
        num = np.ma.unique(dat).compressed().size
        plt.title(f'Clusters: {num}')
        plt.imshow(dat)
        plt.show()

    plt.figure(dpi=150)
    for dat in cdata2:
        num = np.ma.unique(dat).compressed().size
        plt.title(f'Clusters: {num}')
        plt.imshow(dat, alpha=0.25)

    plt.show()

    mdata = np.ma.mean(cdata2, 0)
    plt.figure(dpi=150)
    plt.imshow(mdata)
    plt.colorbar()
    plt.show()

    breakpoint()


def _test_marinda2():
    import sys
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster
    from pygmi.raster.ginterp import norm2

    # Import Data
    ifile = r"D:\Workdata\testdata.hdr"

    dat = get_raster(ifile)

    dat2 = []
    for i in dat:
        if i.dataid in ['k: 1', 'th: 1', 'u: 1']:
            dat2.append(i)

    # Generate layers via cluster analysis
    app = QtWidgets.QApplication(sys.argv)

    DM = Cluster()
    DM.indata['Raster'] = dat2
    DM.spinbox_maxclusters.setProperty('value', 7)

    DM.settings()

    cdata = DM.outdata['Cluster']

    mask = cdata[0].data.mask
    for dat in cdata:
        mask = np.logical_or(mask, dat.data.mask)
        dat.data = dat.data.filled()

    rows, cols = cdata[0].data.shape

    # Get cluster centers
    colormap = np.ma.ones((rows, cols, 4))
    colormap[:, :, 0] = norm2(cdata[0].data)
    colormap[:, :, 1] = norm2(cdata[1].data)
    colormap[:, :, 2] = norm2(cdata[2].data)
    colormap[:, :, 3] = np.logical_not(mask)


    # colormap[:, :, 3][colormap[:, :, 0]==0] = 0

    plt.figure(dpi=150)
    plt.imshow(colormap)
    plt.colorbar()
    plt.show()

    breakpoint()


if __name__ == "__main__":
    _testfn()
