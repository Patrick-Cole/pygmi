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
        self.cmb_alg = QtWidgets.QComboBox()
        self.sb_branchfac = QtWidgets.QSpinBox()
        self.sb_minsamples = QtWidgets.QSpinBox()
        self.sb_minclusters = QtWidgets.QSpinBox()
        self.sb_maxclusters = QtWidgets.QSpinBox()
        self.dsb_maxerror = QtWidgets.QDoubleSpinBox()
        self.dsb_eps = QtWidgets.QDoubleSpinBox()
        self.dsb_bthres = QtWidgets.QDoubleSpinBox()
        self.sb_maxiterations = QtWidgets.QSpinBox()
        self.rb_sscale = QtWidgets.QRadioButton('Standard Scaling')
        self.rb_rscale = QtWidgets.QRadioButton('Robust Scaling')
        self.rb_noscale = QtWidgets.QRadioButton('No Scaling')
        self.lbl_minclusters = QtWidgets.QLabel('Minimum Clusters:')
        self.lbl_maxclusters = QtWidgets.QLabel('Maximum Clusters:')
        self.lbl_maxiter = QtWidgets.QLabel('Maximum Iterations:')
        self.lbl_maxerror = QtWidgets.QLabel('Tolerance:')
        self.lbl_eps = QtWidgets.QLabel('eps:')
        self.lbl_minsamples = QtWidgets.QLabel('Minimum Samples:')
        self.lbl_bthres = QtWidgets.QLabel('Threshold:')
        self.lbl_branchfac = QtWidgets.QLabel('Branching Factor:')

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

        self.cmb_alg.addItems(['Mini Batch K-Means (fast)', 'K-Means',
                               'Bisecting K-Means', 'DBSCAN', 'Birch'])
        self.cmb_alg.currentIndexChanged.connect(self.combo)
        self.combo()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        helpdocs = menu_default.HelpButton('pygmi.clust.cluster')
        gl_1 = QtWidgets.QGridLayout(self)

        buttonbox = QtWidgets.QDialogButtonBox()
        lbl_1 = QtWidgets.QLabel('Cluster Algorithm:')

        self.sb_minclusters.setMinimum(1)
        self.sb_minclusters.setProperty('value', self.min_cluster)
        self.sb_maxclusters.setMinimum(1)
        self.sb_maxclusters.setProperty('value', self.max_cluster)
        self.sb_maxiterations.setMinimum(1)
        self.sb_maxiterations.setMaximum(1000)
        self.sb_maxiterations.setProperty('value', self.max_iter)
        self.sb_minsamples.setMinimum(2)
        self.sb_minsamples.setProperty('value', self.min_samples)
        self.dsb_eps.setDecimals(5)
        self.dsb_eps.setProperty('value', self.eps)
        self.dsb_eps.setSingleStep(0.1)
        self.dsb_maxerror.setDecimals(5)
        self.dsb_maxerror.setProperty('value', self.tol)
        self.rb_sscale.setChecked(True)
        self.sb_branchfac.setMinimum(2)
        self.sb_branchfac.setProperty('value', self.branchfac)
        self.dsb_bthres.setDecimals(5)
        self.dsb_bthres.setProperty('value', self.bthres)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Cluster Analysis')

        gl_1.addWidget(lbl_1, 0, 2, 1, 1)
        gl_1.addWidget(self.cmb_alg, 0, 4, 1, 1)
        gl_1.addWidget(self.lbl_minclusters, 1, 2, 1, 1)
        gl_1.addWidget(self.sb_minclusters, 1, 4, 1, 1)
        gl_1.addWidget(self.lbl_maxclusters, 2, 2, 1, 1)
        gl_1.addWidget(self.sb_maxclusters, 2, 4, 1, 1)
        gl_1.addWidget(self.lbl_maxiter, 3, 2, 1, 1)
        gl_1.addWidget(self.sb_maxiterations, 3, 4, 1, 1)
        gl_1.addWidget(self.lbl_maxerror, 4, 2, 1, 1)
        gl_1.addWidget(self.dsb_maxerror, 4, 4, 1, 1)
        gl_1.addWidget(self.lbl_bthres, 3, 2, 1, 1)
        gl_1.addWidget(self.dsb_bthres, 3, 4, 1, 1)
        gl_1.addWidget(self.lbl_branchfac, 4, 2, 1, 1)
        gl_1.addWidget(self.sb_branchfac, 4, 4, 1, 1)
        gl_1.addWidget(self.lbl_eps, 1, 2, 1, 1)
        gl_1.addWidget(self.dsb_eps, 1, 4, 1, 1)
        gl_1.addWidget(self.lbl_minsamples, 2, 2, 1, 1)
        gl_1.addWidget(self.sb_minsamples, 2, 4, 1, 1)
        gl_1.addWidget(self.rb_noscale, 7, 2, 1, 1)
        gl_1.addWidget(self.rb_sscale, 8, 2, 1, 1)
        gl_1.addWidget(self.rb_rscale, 9, 2, 1, 1)
        gl_1.addWidget(helpdocs, 10, 2, 1, 1)
        gl_1.addWidget(buttonbox, 10, 4, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def combo(self):
        """
        Set up combo box.

        Returns
        -------
        None.

        """
        i = str(self.cmb_alg.currentText())

        self.lbl_minclusters.hide()
        self.sb_minclusters.hide()
        self.lbl_maxclusters.hide()
        self.sb_maxclusters.hide()
        self.lbl_maxerror.hide()
        self.dsb_maxerror.hide()
        self.lbl_maxiter.hide()
        self.sb_maxiterations.hide()
        self.lbl_minsamples.hide()
        self.sb_minsamples.hide()
        self.lbl_eps.hide()
        self.dsb_eps.hide()
        self.lbl_branchfac.hide()
        self.sb_branchfac.hide()
        self.lbl_bthres.hide()
        self.dsb_bthres.hide()

        if i == 'DBSCAN':
            self.lbl_eps.show()
            self.lbl_minsamples.show()
            self.sb_minsamples.show()
            self.dsb_eps.show()
        elif 'K-Means' in i:
            self.lbl_minclusters.show()
            self.sb_minclusters.show()
            self.lbl_maxclusters.show()
            self.sb_maxclusters.show()
            self.lbl_maxerror.show()
            self.dsb_maxerror.show()
            self.lbl_maxiter.show()
            self.sb_maxiterations.show()
        elif i == 'Birch':
            self.lbl_minclusters.show()
            self.sb_minclusters.show()
            self.lbl_maxclusters.show()
            self.sb_maxclusters.show()
            self.lbl_branchfac.show()
            self.sb_branchfac.show()
            self.lbl_bthres.show()
            self.dsb_bthres.show()

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
            self.showlog('No Raster Data.')
            return False

        tst = np.unique([i.data.shape for i in self.indata['Raster']])

        if tst.size > 2:
            self.showlog('Error: Your input datasets have different '
                         'sizes. Merge the data first')
            return False

        self.min_samples = len(self.indata['Raster'])+1
        self.sb_minsamples.setProperty('value', self.min_samples)

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

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.update_vars()

        self.saveobj(self.cmb_alg)
        self.saveobj(self.sb_branchfac)
        self.saveobj(self.sb_minsamples)
        self.saveobj(self.sb_minclusters)
        self.saveobj(self.sb_maxclusters)
        self.saveobj(self.dsb_maxerror)
        self.saveobj(self.dsb_eps)
        self.saveobj(self.dsb_bthres)
        self.saveobj(self.sb_maxiterations)

        self.saveobj(self.cltype)
        self.saveobj(self.min_cluster)
        self.saveobj(self.max_cluster)
        self.saveobj(self.max_iter)
        self.saveobj(self.tol)
        self.saveobj(self.eps)
        self.saveobj(self.min_samples)
        self.saveobj(self.bthres)
        self.saveobj(self.branchfac)

        self.saveobj(self.rb_sscale)
        self.saveobj(self.rb_rscale)
        self.saveobj(self.rb_noscale)

        self.saveobj(self.runs)
        self.saveobj(self.log)

    def update_vars(self):
        """
        Update the variables.

        Returns
        -------
        None.

        """
        self.cltype = str(self.cmb_alg.currentText())
        self.min_cluster = self.sb_minclusters.value()
        self.max_cluster = self.sb_maxclusters.value()
        self.max_iter = self.sb_maxiterations.value()
        self.tol = self.dsb_maxerror.value()
        self.eps = self.dsb_eps.value()
        self.min_samples = self.sb_minsamples.value()
        self.bthres = self.dsb_bthres.value()
        self.branchfac = self.sb_branchfac.value()

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

        self.showlog('Cluster analysis started')

        # Section to deal with different bands having different null values.
        masktmp = ~data[0].data.mask
        for i in data:
            masktmp += ~i.data.mask
        masktmp = ~masktmp

        X = []
        for band in data:
            tmp = band.copy()
            tmp.data.mask = masktmp
            X.append(tmp.data.compressed())
            del tmp
        X = np.transpose(X)

        if self.rb_sscale.isChecked():
            self.showlog('Applying standard scaling')
            X = skp.StandardScaler().fit_transform(X)
        elif self.rb_rscale.isChecked():
            self.showlog('Applying robust scaling')
            X = skp.RobustScaler().fit_transform(X)

        dat_out = []
        for i in self.piter(no_clust):
            if self.cltype != 'DBSCAN':
                self.showlog('Number of Clusters:'+str(i))
            elif i > no_clust[0]:
                continue

            cfit = None
            if self.cltype == 'Mini Batch K-Means (fast)':
                bsize = max(os.cpu_count()*256, 1024)
                cfit = skc.MiniBatchKMeans(n_clusters=i, tol=self.tol,
                                           max_iter=self.max_iter,
                                           n_init='auto',
                                           batch_size=bsize).fit(X)
            elif self.cltype == 'K-Means':
                cfit = skc.BisectingKMeans(n_clusters=i, tol=self.tol,
                                           max_iter=self.max_iter).fit(X)

            elif self.cltype == 'Bisecting K-Means':
                cfit = skc.KMeans(n_clusters=i, tol=self.tol, n_init='auto',
                                  max_iter=self.max_iter).fit(X)

            elif self.cltype == 'DBSCAN':
                cfit = skc.DBSCAN(eps=self.eps,
                                  min_samples=self.min_samples).fit(X)

            elif self.cltype == 'Birch':
                X = np.ascontiguousarray(X)  # Birch gave an error without this
                cfit = skc.Birch(n_clusters=i, threshold=self.bthres,
                                 branching_factor=self.branchfac).fit(X)

            if cfit is None:
                self.showlog('Could not find any clusters. '
                             'Please change settings.')

            if cfit.labels_.max() < i-1 and self.cltype != 'DBSCAN':
                self.showlog('Could not find '+str(i)+' clusters. '
                             'Please change settings.')

                return False
            if cfit.labels_.max() < 0 and self.cltype == 'DBSCAN':
                self.showlog('Could not find any clusters. '
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
            dat_out[-1].metadata['Cluster']['center'] = np.zeros([i,
                                                                  len(data)])
            dat_out[-1].metadata['Cluster']['center_std'] = np.zeros([i, len(data)])
            if cfit.labels_.max() > 0:
                dat_out[-1].metadata['Cluster']['vrc'] = calinski_harabasz_score(X, cfit.labels_)

            # Reloading this hear to save memory. Need unscaled values.
            X = []
            for band in data:
                tmp = band.copy()
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

            self.log = f'Cluster complete ({self.cltype})'

        for i in dat_out:
            i.dataid = 'Clusters: '+str(i.metadata['Cluster']['no_clusters'])
            if self.cltype == 'DBSCAN':
                i.dataid = 'Clusters: '+str(int(i.data.max()+1))
            i.nodata = data[0].nodata
            i.set_transform(transform=data[0].transform)
            i.crs = data[0].crs

        self.showlog('Cluster complete' + ' ('+self.cltype + ' ' + ')')

        for i in dat_out:
            i.data += 1
            i.data = np.ma.masked_equal(i.data.filled(0).astype(int), 0)
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
    DM.sb_maxclusters.setProperty('value', 9)

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
    DM.sb_maxclusters.setProperty('value', 7)

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


if __name__ == "__main__":
    _testfn()
