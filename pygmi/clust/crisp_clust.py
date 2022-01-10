# -----------------------------------------------------------------------------
# Name:        crisp_clust.py (part of PyGMI)
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
"""Main Crisp Clustering set of routines."""

import os
import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np

from pygmi.raster.datatypes import Data
import pygmi.clust.var_ratio as vr
from pygmi.misc import ProgressBarText


class CrispClust(QtWidgets.QDialog):
    """
    Crisp Cluster Class.

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
        super().__init__(parent)

        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        if parent is not None:
            self.piter = parent.pbar.iter
        else:
            self.piter = ProgressBarText().iter

        self.spinbox_maxclusters = QtWidgets.QSpinBox()
        self.combobox_alg = QtWidgets.QComboBox()
        self.doublespinbox_maxerror = QtWidgets.QDoubleSpinBox()
        self.spinbox_maxiterations = QtWidgets.QSpinBox()
        self.spinbox_repeatedruns = QtWidgets.QSpinBox()
        self.spinbox_minclusters = QtWidgets.QSpinBox()
        self.groupbox = QtWidgets.QGroupBox()
        self.label_7 = QtWidgets.QLabel()
        self.doublespinbox_constraincluster = QtWidgets.QDoubleSpinBox()
        self.radiobutton_random = QtWidgets.QRadioButton()
        self.radiobutton_manual = QtWidgets.QRadioButton()
        self.radiobutton_datadriven = QtWidgets.QRadioButton()

        self.setupui()

        self.cltype = 'k-means'
        self.min_cluster = 5
        self.max_cluster = 5
        self.max_iter = 100
        self.term_thresh = 0.00001
        self.runs = 1
        self.constrain = 0.0
        self.denorm = False
        self.init_type = 'random'
        self.type = 'crisp'
        self.log = ''

        self.combobox_alg.addItems(['k-means', 'advanced k-means', 'w-means'])
        self.combobox_alg.currentIndexChanged.connect(self.combo)
        self.combo()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        verticallayout = QtWidgets.QVBoxLayout(self.groupbox)

        buttonbox = QtWidgets.QDialogButtonBox()
        label = QtWidgets.QLabel()
        label_2 = QtWidgets.QLabel()
        label_3 = QtWidgets.QLabel()
        label_4 = QtWidgets.QLabel()
        label_5 = QtWidgets.QLabel()
        label_6 = QtWidgets.QLabel()

        self.spinbox_minclusters.setMinimum(1)
        self.spinbox_minclusters.setProperty('value', 5)
        self.spinbox_maxclusters.setMinimum(1)
        self.spinbox_maxclusters.setProperty('value', 5)
        self.spinbox_maxiterations.setMinimum(1)
        self.spinbox_maxiterations.setMaximum(1000)
        self.spinbox_maxiterations.setProperty('value', 100)
        self.doublespinbox_maxerror.setDecimals(5)
        self.doublespinbox_maxerror.setProperty('value', 1e-05)
        self.spinbox_repeatedruns.setMinimum(1)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.radiobutton_random.setChecked(True)
        self.groupbox.hide()

        self.setWindowTitle('Crisp Clustering')
        label.setText('Cluster Algorithm:')
        label_2.setText('Minimum Clusters:')
        label_3.setText('Maximum Clusters')
        label_4.setText('Maximum Iterations:')
        label_5.setText('Terminate if relative change per iteration '
                        'is less than:')
        label_6.setText('Repeated Runs:')
        self.label_7.setText('Constrain Cluster Shape '
                             '(0: unconstrained, 1: spherical)')
        self.groupbox.setTitle('Initial Guess')
        self.radiobutton_random.setText('Random')
        self.radiobutton_manual.setText('Manual')
        self.radiobutton_datadriven.setText('Data Driven')

        gridlayout.addWidget(label, 0, 2, 1, 1)
        gridlayout.addWidget(self.combobox_alg, 0, 4, 1, 1)
        gridlayout.addWidget(label_2, 1, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_minclusters, 1, 4, 1, 1)
        gridlayout.addWidget(label_3, 2, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_maxclusters, 2, 4, 1, 1)
        gridlayout.addWidget(label_4, 3, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_maxiterations, 3, 4, 1, 1)
        gridlayout.addWidget(label_5, 4, 2, 1, 1)
        gridlayout.addWidget(self.doublespinbox_maxerror, 4, 4, 1, 1)
        gridlayout.addWidget(label_6, 5, 2, 1, 1)
        gridlayout.addWidget(self.spinbox_repeatedruns, 5, 4, 1, 1)
        gridlayout.addWidget(self.label_7, 6, 2, 1, 1)
        gridlayout.addWidget(self.doublespinbox_constraincluster, 6, 4, 1, 1)
        gridlayout.addWidget(self.groupbox, 8, 2, 1, 3)
        gridlayout.addWidget(buttonbox, 9, 4, 1, 1)

        verticallayout.addWidget(self.radiobutton_random)
        verticallayout.addWidget(self.radiobutton_manual)
        verticallayout.addWidget(self.radiobutton_datadriven)

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
        if i == 'w-means':
            self.label_7.show()
            self.doublespinbox_constraincluster.show()
        else:
            self.label_7.hide()
            self.doublespinbox_constraincluster.hide()

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
        tst = np.unique([i.data.shape for i in self.indata['Raster']])
        if tst.size > 2:
            self.showprocesslog('Error: Your input datasets have different '
                                'sizes. Merge the data first')
            return False

        self.update_vars()

        if not nodialog:
            temp = self.exec_()
            if temp == 0:
                return False

            self.parent.process_is_active()
        self.run()

        if not nodialog:
            self.parent.process_is_active(False)
            self.parent.pbar.to_max()
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
        self.combobox_alg.setCurrentText(projdata['cltype'])
        self.spinbox_minclusters.setProperty('value', projdata['min_cluster'])
        self.spinbox_maxclusters.setProperty('value', projdata['max_cluster'])
        self.spinbox_maxiterations.setProperty('value', projdata['max_iter'])
        self.doublespinbox_maxerror.setProperty('value',
                                                projdata['term_thresh'])
        self.spinbox_repeatedruns.setProperty('value', projdata['runs'])
        self.doublespinbox_constraincluster.setProperty('value',
                                                        projdata['constrain'])

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
        projdata['term_thresh'] = self.term_thresh
        projdata['runs'] = self.runs
        projdata['constrain'] = self.constrain

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
        self.term_thresh = self.doublespinbox_maxerror.value()
        self.runs = self.spinbox_repeatedruns.value()
        self.constrain = self.doublespinbox_constraincluster.value()

    def run(self):
        """
        Process the data.

        Returns
        -------
        None.

        """
        data = copy.deepcopy(self.indata['Raster'])
        self.update_vars()

        cltype = self.cltype
        cov_constr = self.constrain
        no_runs = self.runs
        max_iter = self.max_iter
        term_thresh = self.term_thresh
        no_clust = np.array([self.min_cluster, self.max_cluster])
        de_norm = self.denorm

        self.showprocesslog('Crisp Clustering started')

# #############################################################################
# Section to deal with different bands having different null values.
# Start with the first entry
#         masktmp = data[0].data.mask
# # Add the all the masks to this. This promotes False values to True.
#         for i in data:
#             masktmp += i.data.mask
#         for i, _ in enumerate(data):    # Apply this to all the bands
#             data[i].data.mask = masktmp

        masktmp = ~data[0].data.mask
        for i in data:
            masktmp += ~i.data.mask
        masktmp = ~masktmp
        for i, _ in enumerate(data):
            if data[i].nodata != 0.0:
                self.showprocesslog('Setting '+data[i].dataid+' nodata to 0.')
                data[i].data = np.ma.array(data[i].data.filled(0))
            data[i].data.mask = masktmp

# #############################################################################

#        dat_in = np.array([i.data.flatten() for i in data]).T
        dat_in = np.array([i.data.compressed() for i in data]).T

#        dat_in = dat_in[dat_in != None]

#        dat_in = np.array([[x for x in data[i].data.flatten() if x != None]
#              for i in range(len(data))]).T

        if self.radiobutton_manual.isChecked() is True:
            ext = ('ASCII matrix (*.txt);;'
                   'ASCII matrix (*.asc);;'
                   'ASCII matrix (*.dat)')
            filename = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Read Cluster Centers', '.', ext)
            if filename == '':
                return False
            os.chdir(os.path.dirname(filename))

            ifile = str(filename)

            dummy_mod = np.ma.array(np.genfromtxt(ifile, unpack=True))
            [ro0, co0] = np.shape(dummy_mod)
            ro1 = np.sum(list(range(no_clust[0], no_clust[1] + 1)))
            if dat_in.shape[1] != co0 or ro0 != ro1:
                QtWidgets.QMessageBox.warning(self.parent, 'Warning',
                                              ' Incorrect matrix size!',
                                              QtWidgets.QMessageBox.Ok,
                                              QtWidgets.QMessageBox.Ok)
            cnt = -1
            for i in range(no_clust[0], no_clust[1] + 1):
                smtmp = np.zeros(i)
                for j in range(i):
                    cnt = cnt + 1
                    smtmp[j] = dummy_mod[cnt]
                startmdat = {i: smtmp}
                startmfix = {i: []}

            filename = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Read Cluster Center Constraints', '.', ext)
            if filename == '':
                QtWidgets.QMessageBox.warning(
                    self.parent, 'Warning',
                    'Running cluster analysis without constraints',
                    QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            else:
                os.chdir(os.path.dirname(filename))
                ifile = str(filename)
                dummy_mod = np.ma.array(np.genfromtxt(ifile, unpack=True))
                [ro0, co0] = np.shape(dummy_mod)
                ro1 = np.sum(list(range(no_clust[0], no_clust[1] + 1)))
                if dat_in.shape[1] != co0 or ro0 != ro1:
                    QtWidgets.QMessageBox.warning(self.parent, 'Warning',
                                                  ' Incorrect matrix size!',
                                                  QtWidgets.QMessageBox.Ok,
                                                  QtWidgets.QMessageBox.Ok)
                cnt = -1
                for i in range(no_clust[0], no_clust[1] + 1):
                    smtmp = np.zeros(i)
                    for j in range(i):
                        cnt = cnt + 1
                        smtmp = dummy_mod[cnt]
                    startmfix = {i: smtmp}

        cnt = -1
        dat_out = [Data() for i in range(no_clust[0], no_clust[1]+1)]

        for i in range(no_clust[0], no_clust[1]+1):
            self.showprocesslog('Number of Clusters:'+str(i))
            cnt = cnt + 1
            if self.radiobutton_datadriven.isChecked() is True:
                self.showprocesslog('Initial guess: data driven')
                no_samp = dat_in.shape[0]
                dno_samp = no_samp/i
#                idx=1
#                idx=[[idx,(j-1)*dno_samp] for j in range(2,i+1)]
                idx = np.arange(0, no_samp+dno_samp, dno_samp)
                idx[0] = 1
#                idx=np.array([idx,no_samp])
                startmdat = {i: np.zeros([i, dat_in.shape[1]])}
                dat_in1 = dat_in
                smtmp = np.zeros([i, dat_in.shape[1]])
                for k in range(dat_in.shape[1]):
                    for j in range(i):
                        smtmp[j, k] = np.median(dat_in1[idx[j]:idx[j+1], k])
                startmdat = {i: smtmp}
                startmfix = {i: []}
                del dat_in1

                clidx, clcent, clobj_fcn, clvrc = self.crisp_means(
                    dat_in, i, startmdat[i], startmfix[i], max_iter,
                    term_thresh, cltype, cov_constr)

            elif self.radiobutton_manual.isChecked() is True:

                self.showprocesslog('Initial guess: manual')

                clidx, clcent, clobj_fcn, clvrc = self.crisp_means(
                    dat_in, i, startmdat[i], startmfix[i], max_iter,
                    term_thresh, cltype, cov_constr)

            elif self.radiobutton_random.isChecked() is True:
                self.showprocesslog('Initial guess: random')

                clobj_fcn = np.array([np.inf])
                for j in range(no_runs):
                    self.showprocesslog('Run '+str(j+1)+' of'+str(no_runs))

                    xmins = np.minimum(dat_in, 1)
                    xmaxs = np.maximum(dat_in, 1)
                    startm1dat = {i: np.random.uniform(
                        xmins[np.zeros(i, int), :],
                        xmaxs[np.zeros(i, int), :])}
                    startmfix = {i: np.array([])}
                    clidx1, clcent1, clobj_fcn1, clvrc1 = self.crisp_means(
                        dat_in, i, startm1dat[i], startmfix[i], max_iter,
                        term_thresh, cltype, cov_constr)

                    if clobj_fcn1[-1] < clobj_fcn[-1]:
                        clidx = clidx1
                        clcent = clcent1
                        clobj_fcn = clobj_fcn1
                        clvrc = clvrc1
                        startmdat = {i: startm1dat[i]}

#            zonal = np.zeros(data[0].data.shape)-9999.0
            zonal = np.ma.masked_all(data[0].data.shape)

            alpha = (data[0].data.mask == 0)
            zonal[alpha == 1] = clidx

            cent_std = np.array([np.std(dat_in[clidx == k], 0)
                                 for k in range(i)])

#            den_cent = clcent
#            den_cent_std = np.array(cent_std, copy=True)
#            den_cent_std1 = np.array(cent_std, copy=True)
            if de_norm is True:
                pass
#                for k, _ in enumerate(data):
#                    if np.size(data[k].norm) > 0:
#                        nnorm = len(data[k].norm)
#                        for j in range(nnorm, 0, -1):
#                            if data[k].norm[j-1]['type'] == 'minmax':
#                                den_cent[:, k] = (
#                                    den_cent[:, k] *
#                                    (data[k].norm[j-1]['transform'][1, 1] -
#                                     data[k].norm[j-1]['transform'][0, 1]) +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                                den_cent_std[:, k] = (
#                                    den_cent_std[:, k] *
#                                    (data[k].norm[j-1]['transform'][1, 1] -
#                                     data[k].norm[j-1]['transform'][0, 1]) +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                                den_cent_std1[:, k] = (
#                                    den_cent_std1[:, k] *
#                                    (data[k].norm[j-1]['transform'][1, 1] -
#                                     data[k].norm[j-1]['transform'][0, 1]) +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                            elif (data[k].norm[j-1]['type'] == 'meanstd' or
#                                  data[k].norm[j-1]['type'] == 'medmad'):
#                                den_cent[:, k] = (
#                                    den_cent[:, k] *
#                                    data[k].norm[j-1]['transform'][1, 1] +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                                den_cent_std[:, k] = (
#                                    den_cent_std[:, k] *
#                                    data[k].norm[j-1]['transform'][1, 1] +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                                den_cent_std1[:, k] = (
#                                    den_cent_std1[:, k] *
#                                    data[k].norm[j-1]['transform'][1, 1] +
#                                    data[k].norm[j-1]['transform'][0, 1])
#                            elif data[k].norm[j-1]['type'] == 'histeq':
#                                den_cent[:, k] = np.interp(
#                                    den_cent[:, k],
#                                    data[k].norm[j-1]['transform'][:, 0],
#                                    data[k].norm[j-1]['transform'][:, 1])
#                                den_cent_std[:, k] = np.interp(
#                                    (den_cent[:, k] + den_cent_std[:, k]),
#                                    data[k].norm[j-1]['transform'][:, 0],
#                                    data[k].norm[j-1]['transform'][:, 1])
#                                den_cent_std1[:, k] = np.interp(
#                                    (den_cent[:, k] - den_cent_std1[:, k]),
#                                    data[k].norm[j-1]['transform'][:, 0],
#                                    data[k].norm[j-1]['transform'][:, 1])
#            else:
#                den_cent = np.array([])
#                den_cent_std = np.array([])
#                den_cent_std1 = np.array([])
            dat_out[cnt].metadata['Cluster']['input_type'] = []
            for k in data:
                dat_out[cnt].metadata['Cluster']['input_type'].append(k.dataid)
#                dat_out[cnt].proc_history.append(k.proc)

            dat_out[cnt].data = zonal
#            dat_out[cnt].data.mask = masktmp
            dat_out[cnt].nodata = zonal.fill_value
            dat_out[cnt].metadata['Cluster']['no_clusters'] = i
            dat_out[cnt].metadata['Cluster']['center'] = clcent
            dat_out[cnt].metadata['Cluster']['center_std'] = cent_std
            dat_out[cnt].metadata['Cluster']['obj_fcn'] = clobj_fcn
            dat_out[cnt].metadata['Cluster']['vrc'] = clvrc

#            dat_out[cnt].type = self.type
#            dat_out[cnt].algorithm = cltype
#            dat_out[cnt].initialization = init_type
#            dat_out[cnt].init_mod = startmdat[i]
#            dat_out[cnt].init_constrains = startmfix[i]
#            dat_out[cnt].runs = no_runs
#            dat_out[cnt].max_iterations = max_iter
#            dat_out[cnt].denormalize = de_norm
#            dat_out[cnt].term_threshold = term_thresh
#            dat_out[cnt].shape_constrain = cov_constr
#            dat_out[cnt].zonal = zonal
#            dat_out[cnt].alpha = alpha
#            dat_out[cnt].xxx = data[0].xxx
#            dat_out[cnt].yyy = data[0].yyy
#            dat_out[cnt].denorm_center = den_cent
#            dat_out[cnt].denorm_center_stdup = den_cent_std
#            dat_out[cnt].denorm_center_stdlow = den_cent_std1
#            dat_out[cnt].iterations = clobj_fcn.size

            self.log = ('Crisp Cluster complete' + ' (' + self.cltype + ' ' +
                        self.init_type+')')

        for i in dat_out:
            i.dataid = 'Crisp Cluster: '+str(i.metadata['Cluster']['no_clusters'])
            i.nodata = data[0].nodata
            i.set_transform(transform=data[0].transform)
            i.crs = data[0].crs

        self.showprocesslog('Crisp Cluster complete' + ' ('+self.cltype + ' ' +
                            self.init_type+')')

        for i in dat_out:
            i.data += 1
            i.data = i.data.astype(np.uint8)
            i.nodata = 0
        self.outdata['Cluster'] = dat_out
        self.outdata['Raster'] = self.indata['Raster']

        return True

    def crisp_means(self, data, no_clust, cent, centfix, maxit, term_thresh,
                    cltype, cov_constr):
        """
        Script enables the crisp clustering of COMPLETE multi-variate datasets.

        Parameters
        ----------
        data : numpy array
            N x P matrix containing the data to be clustered, N is number of
            samples, P is number of different attributes available for each
            sample.
        no_clust : int
            Number of clusters to be used.
        cent : numpy array
            cluster centre positions, either empty [] --> randomly guessed
            center positions will be used for initialisation or NO_CLUSTxP
            matrix
        centfix : TYPE
            Constrains the position of cluster centers, if CENTFIX is empty,
            cluster centers can freely vary during cluster analysis, otherwise
            CENTFIX is of equal size to CENT and gives an absolute deviation
            from initial center positions that should not be exceeded during
            clustering. Note, CETNFIX applies only if center values are
            provided by the user.
        maxit : int
            number of maximal allowed iterations.
        term_thresh : float
            Termination threshold, either empty [] --> go for the maximum
            number of iterations MAXIT or a scalar giving the minimum
            reduction of the size of the objective function for two consecutive
            iterations in Percent.
        cltype : str
            either 'kmeans' --> kmeans cluster analysis (spherically shaped
            cluster), 'det' --> uses the determinant criterion of Spath, H.,
            "Cluster-Formation and Analyse, chapter3" (ellipsoidal clusters,
            all cluster use the same ellipsoid), or 'vardet' --> Spath, H.,
            chapter 4 (each cluster uses its individual ellipsoid). Note: the
            latter is the crisp version of the Gustafson-Kessel algorithm
        cov_constr : float
            scalar between [0 1], values > 0 trim the covariance matrix
            to avoid needle-like ellipsoids for the clusters, applies only for
            CLTYPE='vardet', but must always be provided.

        Returns
        -------
        idx : numpy array
            cluster index number for each sample after the last iteration,
            column vector.
        cent : numpy array
            matrix with cluster centre positions after last iteration, one
            cluster centre per row
        obj_fcn : numpy array
            Vector, size of the objective function after each iteration
        vrc : numpy array
            Variance Ratio Criterion
        """
        self.showprocesslog(' ')

        no_samples = data.shape[0]
        if cent.size == 0:  # if no center values are provided
            xmins = np.minimum(data, 1)
            xmaxs = np.maximum(data, 1)
            cent = np.random.uniform(xmins[np.zeros(no_clust, int), :],
                                     xmaxs[np.zeros(no_clust, int), :])
            centfix = np.array([])

        cent_orig = cent
        centfix = np.abs(centfix)

# calculate euclidian distance for initial classification
        onetmp = np.ones([no_samples, 1], int)  # trying this for speed?
        edist = np.array([np.sqrt(np.sum(((data-onetmp*cent[j])**2), 1))
                          for j in range(no_clust)])  # initial distance -->
#                                                      Euclidian

        mindist = edist.min(0)  # 0 means row wise minimum
        idx = edist.argmin(0)

    # initial size of objective function
        obj_fcn_initial = np.sum(mindist**2)
        obj_fcn_prev = obj_fcn_initial
        obj_fcn = np.zeros(maxit)  # This is new - we must initialise this.

        for i in self.piter(range(maxit)):  # =1:maxit. loop over all iters
            cent_prev = cent  # store result of last iteration
            idx_prev = idx
            dist_prev = edist
    # calc new cluster centre positions
            cent, idx = gcentroids(data, idx, no_clust, mindist)
    # constrain the cluster center positions to keep it in  the given interval
            if centfix.size > 0:
                # constrain the center positions within the given limits
                cent_idx = cent > (cent_orig+centfix)
                cent[cent_idx == 1] = (cent_orig(cent_idx == 1) +
                                       centfix(cent_idx == 1))
                cent_idx = cent < (cent_orig-centfix)
                cent[cent_idx == 1] = (cent_orig(cent_idx == 1) -
                                       centfix(cent_idx == 1))

    # calc new cluster centre distances
            edist = gdist(data, cent, idx, no_clust, cltype, cov_constr)
    # get new index values for each data point and the distance from each
    # sample to its cluster center
            mindist = edist.min(0)
            idx = edist.argmin(0)

    # calc new objective function size
            obj_fcn[i] = np.sum(mindist**2)

            if obj_fcn[i] == 0:
                obj_fcn_dif = 0
            else:
                obj_fcn_dif = 100 * ((obj_fcn_prev-obj_fcn[i]) / obj_fcn[i])

            self.showprocesslog('Iteration: ' + str(i) + ' Threshold: ' +
                                str(term_thresh) + ' Current: ' +
                                '{:.2e}'.format(obj_fcn_dif), True)
    # if no termination threshold provided, ignore this and do all iterations
            if term_thresh > 0:
                # if the improvement between the last two iterations was less
                # than a defined threshold in percent
                if (obj_fcn_dif < term_thresh or obj_fcn[i] > obj_fcn_prev):
                    # go back to the results of the previous iteration
                    idx = idx_prev
                    cent = cent_prev
                    edist = dist_prev
                    if i == 0:
                        obj_fcn = obj_fcn_prev
                    else:
                        # changed from i-1 to i for w-means
                        obj_fcn = np.delete(obj_fcn, np.s_[i::])
                        # vrc=vr.var_ratio(data, idx, cent, edist)
                    break  # and stop the clustering right now
            obj_fcn_prev = obj_fcn[i]
        vrc = vr.var_ratio(data, idx, cent, edist)
        return idx, cent, obj_fcn, vrc


def gcentroids(data, index, no_clust, mindist):
    """
    G Centroids.

    Parameters
    ----------
    data : numpy array
        Input data.
    index : numpy array
        Cluster index number for each sample.
    no_clust : int
        Number of clusters to be used.
    mindist : numpy array
        Minimum distances.

    Returns
    -------
    centroids : numpy array
        DESC
    index : numpy array
        DESC

    """
#    no_samples=data.shape[0]
    no_datatypes = data.shape[1]
    centroids = np.tile(np.nan, (no_clust, no_datatypes))
    for j in range(no_clust):
        # find all members of the j-th cluster
        members = (index == j).nonzero()[0]
        # if j is an empty cluster, put one sample into this cluster
        if members.size == 0:
            # take the sample that has the greatest distance to its current
            # cluster and make this the center of the j-th cluster
            idx1 = mindist.argmax(0)
            centroids[j] = data[idx1]
            index[idx1] = j
            mindist[idx1] = 0
        else:
            centroids[j] = data[members].mean(0)
    return centroids, index


def gdist(data, center, index, no_clust, cltype, cov_constr):
    """
    G Dist routine.

    Parameters
    ----------
    data : numpy array
        Input data.
    center : TYPE
        DESCRIPTION.
    index : numpy array
        Cluster index number for each sample.
    no_clust : int
        Number of clusters to be used.
    cltype : str
        Clustering type.
    cov_constr : float
        scalar between [0 1].

    Returns
    -------
    bigd : numpy array
        Output data.

    """
    no_samples = data.shape[0]
    no_datasets = data.shape[1]
    bigd = np.zeros([no_clust, no_samples])
    ddd = []
    if cltype == 'k-means':
        onetmp = np.ones([no_samples, 1])  # trying this for speed?
        for j in range(no_clust):
            # Euclidian
            bigd[j] = np.sqrt(np.sum(((data-onetmp*center[j])**2), 1))
            # determinant criterion see Spath, Helmuth,
            # "Cluster-Formation and Analyse", chapter 3
    elif cltype == 'advanced k-means':
        for j in range(no_clust):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data-np.ones([no_samples, 1])*center[j]
            # grab the data belonging to cluster j
            mod_idx = (index == j)*1
            # should I use different transpose?
            # Streuungsmatrix/ covariance of the j-th cluster
            mat_a = np.dot(np.ones([no_datasets, 1])*mod_idx*dcent.T,
                           dcent/np.sum(mod_idx))
            # constrain covariance matrix if badly conditioned
            if np.linalg.cond(mat_a) > 1e10:
                # warning([' Badly conditionend covariance matrix \
                # (cond. number > 1e10) of cluster ',num2unicode(j)]);
                ed1, ev1 = np.linalg.eig(mat_a)
                edmax = np.max(ed1)
                ed1[1e10*ed1 < edmax] = edmax/1e10
                mat_a = np.dot(np.dot(ev1, (ed1*np.eye(no_datasets))),
                               np.linalg.inv(ev1))
            if j == 0:  # sum all covariance matrices for all clusters
                mat_a0 = mat_a
            else:
                mat_a0 = mat_a0 + mat_a
# calc new distances using the same covariance matrix for all clusters -->
# ellisoidal clusters, all clusters use equal ellipsoids
        for j in range(no_clust):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data-np.ones([no_samples, 1])*center[j]
            # does this need to be in this loop?
            mbig = (np.linalg.det(mat_a0)**(1.0/no_datasets) *
                    np.linalg.pinv(mat_a0))
            ddd.append(np.sum((np.dot(dcent, mbig)*dcent), 1).T)
        bigd = np.sqrt(ddd)
# cluster adapted determinant criterion see Spath, Helmuth,
# "Cluster-Formation and Analyse", chapter 4 --> equivalent to crisp GK
# algorithm
    elif cltype == 'w-means':
        for j in range(no_clust):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data-np.ones([no_samples, 1])*center[j]
            mod_idx = (index == j)*1  # grab data belonging to cluster j
#    '*dcent/sum(mod_idx); % Streuungsmatrix/ covariance of the j-th cluster
            mat_a = np.dot(np.ones([no_datasets, 1])*mod_idx*dcent.T,
                           dcent/np.sum(mod_idx))
            mat_a0 = np.eye(mat_a.shape[0])
# if cov_constr>0, this enforces not to elongated ellipsoids -->
# avoid the needle-like cluster
            mat_a = (1-cov_constr)*mat_a+cov_constr*(mat_a0/no_samples)
# constrain covariance matrix if badly conditioned and cluster contains
# more than 1 sample
            if np.linalg.cond(mat_a) > 1e10 and np.sum(mod_idx) > 1:
                # warning([' Badly conditionend covariance matrix '+
                #   '(cond. number > 1e10) of cluster ',num2unicode(j)])
                ed1, ev1 = np.linalg.eig(mat_a)
                edmax = np.max(ed1)
                ed1[1e10*ed1 < edmax] = edmax/1e10
                mat_a = np.dot(np.dot(ev1, (ed1*np.eye(no_datasets))),
                               np.linalg.inv(ev1))
# assume spherical shape of clusters with only one sample
            elif np.sum(mod_idx) == 1:
                mat_a = mat_a0
            mbig = (np.linalg.det(mat_a)**(1.0/no_datasets) *
                    np.linalg.pinv(mat_a))
# calc cluster to sample distances using mahalanobis distance for each
# cluster, ellipsoidal clusters, each cluster has its own individually
# oriented and shaped ellipsoid
            ddd.append(np.sum((np.dot(dcent, mbig)*dcent), 1).T)
        bigd = np.sqrt(ddd)
    return bigd.real
