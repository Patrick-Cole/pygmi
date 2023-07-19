# -----------------------------------------------------------------------------
# Name:        fuzzy_clust.py (part of PyGMI)
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
"""Fuzzy clustering."""

import os
import copy
from PyQt5 import QtWidgets, QtCore
import numpy as np

from pygmi.raster.datatypes import Data
from pygmi.clust import var_ratio as vr
from pygmi.misc import BasicModule


class FuzzyClust(BasicModule):
    """Fuzzy Clustering class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combobox_alg = QtWidgets.QComboBox()
        self.doublespinbox_maxerror = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_fuzzynessexp = QtWidgets.QDoubleSpinBox()
        self.doublespinbox_constraincluster = QtWidgets.QDoubleSpinBox()
        self.spinbox_maxclusters = QtWidgets.QSpinBox()
        self.spinbox_maxiterations = QtWidgets.QSpinBox()
        self.spinbox_repeatedruns = QtWidgets.QSpinBox()
        self.spinbox_minclusters = QtWidgets.QSpinBox()
        self.label_7 = QtWidgets.QLabel()
        self.radiobutton_random = QtWidgets.QRadioButton()
        self.radiobutton_manual = QtWidgets.QRadioButton()
        self.radiobutton_datadriven = QtWidgets.QRadioButton()

        self.setupui()

        self.cltype = 'fuzzy c-means'
        self.min_cluster = 5
        self.max_cluster = 5
        self.max_iter = 100
        self.term_thresh = 0.00001
        self.runs = 1
        self.constrain = 0.0
        self.denorm = False
        self.init_type = 'random'
        self.type = 'fuzzy'
        self.fexp = 1.5

        self.combobox_alg.addItems(['fuzzy c-means',
                                    'advanced fuzzy c-means',
                                    'Gustafson-Kessel'])
#                                   , 'Gath-Geva'])
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
        groupbox = QtWidgets.QGroupBox(self)
        verticallayout = QtWidgets.QVBoxLayout(groupbox)

        buttonbox = QtWidgets.QDialogButtonBox(self)
        label = QtWidgets.QLabel()
        label_2 = QtWidgets.QLabel()
        label_3 = QtWidgets.QLabel()
        label_4 = QtWidgets.QLabel()
        label_5 = QtWidgets.QLabel()
        label_6 = QtWidgets.QLabel()
        label_8 = QtWidgets.QLabel()

        self.spinbox_maxclusters.setMinimum(1)
        self.spinbox_maxclusters.setProperty('value', 5)
        self.spinbox_maxiterations.setMinimum(1)
        self.spinbox_maxiterations.setMaximum(1000)
        self.spinbox_maxiterations.setProperty('value', 100)
        self.spinbox_repeatedruns.setMinimum(1)
        self.spinbox_minclusters.setMinimum(1)
        self.spinbox_minclusters.setProperty('value', 5)
        self.doublespinbox_maxerror.setDecimals(5)
        self.doublespinbox_maxerror.setProperty('value', 1e-05)
        self.doublespinbox_fuzzynessexp.setProperty('value', 1.5)
        self.radiobutton_random.setChecked(True)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Fuzzy Clustering')
        groupbox.setTitle('Initial Guess')
        groupbox.hide()
        label.setText('Cluster Algorithm:')
        label_2.setText('Minimum Clusters:')
        label_3.setText('Maximum Clusters')
        label_4.setText('Maximum Iterations:')
        label_5.setText('Terminate if relative change per iteration is less '
                        'than:')
        label_6.setText('Repeated Runs:')
        self.label_7.setText('Constrain Cluster Shape (0: unconstrained, '
                             '1: spherical)')
        label_8.setText('Fuzzyness Exponent')
        self.radiobutton_random.setText('Random')
        self.radiobutton_manual.setText('Manual')
        self.radiobutton_datadriven.setText('Data Driven')

        gridlayout.addWidget(label, 0, 2, 1, 1)
        gridlayout.addWidget(label_2, 1, 2, 1, 1)
        gridlayout.addWidget(label_3, 2, 2, 1, 1)
        gridlayout.addWidget(label_4, 3, 2, 1, 1)
        gridlayout.addWidget(label_5, 4, 2, 1, 1)
        gridlayout.addWidget(label_6, 5, 2, 1, 1)
        gridlayout.addWidget(self.label_7, 6, 2, 1, 1)
        gridlayout.addWidget(label_8, 7, 2, 1, 1)
        gridlayout.addWidget(groupbox, 9, 2, 1, 3)

        gridlayout.addWidget(self.combobox_alg, 0, 4, 1, 1)
        gridlayout.addWidget(self.spinbox_minclusters, 1, 4, 1, 1)
        gridlayout.addWidget(self.spinbox_maxclusters, 2, 4, 1, 1)
        gridlayout.addWidget(self.spinbox_maxiterations, 3, 4, 1, 1)
        gridlayout.addWidget(self.doublespinbox_maxerror, 4, 4, 1, 1)
        gridlayout.addWidget(self.spinbox_repeatedruns, 5, 4, 1, 1)
        gridlayout.addWidget(self.doublespinbox_constraincluster, 6, 4, 1, 1)
        gridlayout.addWidget(self.doublespinbox_fuzzynessexp, 7, 4, 1, 1)
        gridlayout.addWidget(buttonbox, 10, 4, 1, 1)

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
        if i in ('Gath-Geva', 'Gustafson-Kessel'):
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
        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
            return False

        tst = np.unique([i.data.shape for i in self.indata['Raster']])
        if tst.size > 2:
            self.showlog('Error: Your input datasets have different '
                         'sizes. Merge the data first')
            return False

        if not nodialog:
            self.update_vars()
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
        self.doublespinbox_fuzzynessexp.setProperty('value',
                                                    projdata['fexp'])
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
        projdata['fexp'] = self.fexp

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
        self.fexp = self.doublespinbox_fuzzynessexp.value()

    def run(self):
        """
        Run.

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
        expo = self.fexp

        self.showlog('Fuzzy Clustering started')

# #############################################################################
# Section to deal with different bands having different null values.
#         masktmp = data[0].data.mask   # Start with the first entry
# # Add the masks to this.This promotes False values to True if necessary
#         for i in data:
#             masktmp += i.data.mask
#         for i, _ in enumerate(data):    # Apply this to all the bands
#             data[i].data.mask = masktmp

        masktmp = ~data[0].data.mask
        for i in data:
            masktmp += ~i.data.mask
        masktmp = ~masktmp
        for datai in data:
            if datai.nodata != 0.0:
                self.showlog('Setting '+datai.dataid+' nodata to 0.')
                datai.data = np.ma.array(datai.data.filled(0))
            datai.data.mask = masktmp

# #############################################################################

        dat_in = np.array([i.data.compressed() for i in data]).T

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
            [row, col] = np.shape(dummy_mod)
            ro1 = np.sum(list(range(no_clust[0], no_clust[1] + 1)))

            if dat_in.shape[1] != col or row != ro1:
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
                ifile = str(filename)
                dummy_mod = np.ma.array(np.genfromtxt(ifile, unpack=True))
                [row, col] = np.shape(dummy_mod)
                ro1 = np.sum(list(range(no_clust[0], no_clust[1] + 1)))
                if dat_in.shape[1] != col or row != ro1:
                    QtWidgets.QMessageBox.warning(
                        self.parent, 'Warning', ' Incorrect matrix size!',
                        QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                cnt = -1
                for i in range(no_clust[0], no_clust[1] + 1):
                    smtmp = np.zeros(i)
                    for j in range(i):
                        cnt = cnt + 1
                        smtmp = dummy_mod[cnt]
                    startmfix = {i: smtmp}

        cnt = -1
        dat_out = [Data() for i in range(no_clust[0], no_clust[1] + 1)]

        for i in range(no_clust[0], no_clust[1] + 1):
            self.showlog(f'Number of Clusters: {i}')
            cnt = cnt + 1
            if self.radiobutton_datadriven.isChecked() is True:
                self.showlog('Initial guess: data driven')

                no_samp = dat_in.shape[0]
                dno_samp = no_samp / i
                idx = np.arange(0, no_samp + dno_samp, dno_samp)
                idx[0] = 1
                startmdat = {i: np.zeros([i, dat_in.shape[1]])}
                dat_in1 = dat_in
                smtmp = np.zeros([i, dat_in.shape[1]])
                for k in range(dat_in.shape[1]):
                    for j in range(i):
                        smtmp[j, k] = np.median(dat_in1[idx[j]:idx[j + 1], k])
                startmdat = {i: smtmp}
                startmfix = {i: np.array([])}
                del dat_in1

                clu, clcent, clobj_fcn, clvrc, clnce, clxbi = self.fuzzy_means(
                    dat_in, i, startmdat[i], startmfix[i], max_iter,
                    term_thresh, expo, cltype, cov_constr)

            elif self.radiobutton_manual.isChecked() is True:
                self.showlog('Initial guess: manual')

                clu, clcent, clobj_fcn, clvrc, clnce, clxbi = self.fuzzy_means(
                    dat_in, i, startmdat[i], startmfix[i],
                    max_iter, term_thresh, expo, cltype, cov_constr)

            elif self.radiobutton_random.isChecked() is True:
                self.showlog('Initial guess: random')

                clobj_fcn = np.array([np.Inf])
                for j in range(no_runs):
                    self.showlog(f'Run {j+1} of {no_runs}')

                    xmins = np.minimum(dat_in, 1)
                    xmaxs = np.maximum(dat_in, 1)
                    startm1dat = {i: np.random.uniform(
                        xmins[np.zeros(i, int), :],
                        xmaxs[np.zeros(i, int), :])}
                    startmfix = {i: np.array([])}
                    clu1, clcent1, clobj_fcn1, clvrc1, clnce1, clxbi1 = \
                        self.fuzzy_means(dat_in, i, startm1dat[i],
                                         startmfix[i], max_iter, term_thresh,
                                         expo, cltype, cov_constr)

                    if clobj_fcn1[-1] < clobj_fcn[-1]:
                        clu = clu1
                        clcent = clcent1
                        clobj_fcn = clobj_fcn1
                        clnce = clnce1
                        clxbi = clxbi1
                        clvrc = clvrc1
                        startmdat = {i: startm1dat[i]}

            clalp = np.array(clu).max(0)
            clidx = np.array(clu).argmax(0)
            clalp = clalp - (1.0 / clcent.shape[0])
            clalp = clalp / clalp.max()
            clalp[clalp > 1] = 1
            clalp[clalp < 0] = 0
            zonal = np.ma.zeros(data[0].data.shape)-9999.0
            alpha = np.ma.zeros(data[0].data.shape)-9999.0
            alpha1 = (data[0].data.mask == 0)
            zonal[alpha1 == 1] = clidx
            alpha[alpha1 == 1] = clalp
            zonal.mask = data[0].data.mask
            alpha.mask = data[0].data.mask

            cent_std = np.array([np.std(dat_in[clidx == k], 0)
                                 for k in range(i)])

            if de_norm == 1:
                pass

            dat_out[cnt].metadata['Cluster']['input_type'] = []
            for k in data:
                dat_out[cnt].metadata['Cluster']['input_type'].append(k.dataid)

            dat_out[cnt].data = np.ma.array(zonal)
            dat_out[cnt].metadata['Cluster']['no_clusters'] = i
            dat_out[cnt].metadata['Cluster']['center'] = clcent
            dat_out[cnt].metadata['Cluster']['center_std'] = cent_std

            dat_out[cnt].metadata['Cluster']['memdat'] = []
            for k in range(clcent.shape[0]):
                dummy = np.ones(data[0].data.shape) * np.nan
                alpha1 = (data[0].data.mask == 0)
                dummy[alpha1 == 1] = clu[k, :]
                dummy = np.ma.masked_invalid(dummy)
                dat_out[cnt].metadata['Cluster']['memdat'].append(dummy)
            dat_out[cnt].metadata['Cluster']['vrc'] = clvrc
            dat_out[cnt].metadata['Cluster']['nce'] = clnce
            dat_out[cnt].metadata['Cluster']['xbi'] = clxbi
            dat_out[cnt].metadata['Cluster']['obj_fcn'] = clobj_fcn

        for i in dat_out:
            i.dataid = ('Fuzzy Cluster: ' +
                        str(i.metadata['Cluster']['no_clusters']))
            i.nodata = data[0].nodata
            i.set_transform(transform=data[0].transform)
            i.data += 1
            i.crs = data[0].crs

        self.showlog(f'Fuzzy Cluster complete ({self.cltype} {self.init_type})')

        self.outdata['Cluster'] = dat_out
        self.outdata['Raster'] = self.indata['Raster']

        return True

    def fuzzy_means(self, data, no_clust, init, centfix, maxit, term_thresh,
                    expo, cltype, cov_constr):
        """
        Fuzzy clustering.

        Finds NO_CLUST clusters in the data set DATA.. Supported algorithms are
        fuzzy c-means, Gustafson-Kessel, advanced fuzzy c-means.


        Parameters
        ----------
        data : numpy array
            DATA is size M-by-N, where M is the number of samples
            and N is the number of coordinates (attributes) for each sample.
        no_clust : int
            Number of clusters.
        init : TYPE
            INIT may be set to [], in this case the FCM generates random
            initial center locations to start the algorithm. Alternatively,
            INIT can be of matrix type, either containing a user-given
            membership matrix [NO_CLUST M] or a cluster center matrix
            [NO_CLUST, N].
        centfix : TYPE
            DESCRIPTION.
        maxit : int
            MAXIT give the maximum number of iterations..
        term_thresh : float
            Gives the required minimum improvement in per cent per
            iteration. (termination threshold)
        expo : float
            Fuzzification exponent.
        cltype : str
            either 'FCM' for fuzzy c-means (spherically shaped clusters),
            'DET' for advanced fuzzy c-means (ellipsoidal clusters, all
            clusters use the same ellipsoid), or 'GK' for Gustafson-Kessel
            clustering (ellipsoidal clusters, each cluster uses its own
            ellipsoid).
        cov_constr : float
            COV_CONSTR applies only to the GK algorithm. constrains the cluster
            shape towards spherical clusters to avoid needle-like clusters.
            COV_CONSTR = 1 make the GK algorithm equal to the FCM algorithm,
            COV_CONSTR = 0 results in no constraining of the covariance
            matrices of the clusters.

        Returns
        -------
        uuu : numpy array
            This membership function matrix contains the grade of
            membership of each data sample to each cluster.
        cent : numpy array
            The coordinates for each cluster center are returned in the rows
            of the matrix CENT.
        obj_fcn : numpy array
            At each iteration, an objective function is minimized to find the
            best location for the clusters and its values are returned in
            OBJ_FCN.
        vrc : numpy array
            Variance ration criterion.
        nce :
            Normalised class entropy.
        xbi : numpy array
            Xie beni index.
        """
        self.showlog(' ')

        if cltype == 'fuzzy c-means':
            cltype = 'fcm'
        if cltype == 'advanced fuzzy c-means':
            cltype = 'det'
        if cltype == 'Gustafson-Kessel':
            cltype = 'gk'
        if cltype == 'Gath-Geva':
            cltype = 'gg'

        no_samples = data.shape[0]
        data_types = data.shape[1]

        uuu = []  # dummy definition of membership matrix

        # if neither initial centers nor initial meberships are provided ->
        # random guess
        if init.size == 0:
            xmins = np.minimum(data, 1)
            xmaxs = np.maximum(data, 1)
            # initial guess of centroids
            cent = np.random.uniform(xmins[np.zeros(no_clust, int), :],
                                     xmaxs[np.zeros(no_clust, int), :])
            # GK and det clustering require center and memberships for distance
            # calculation: here initial guess for uuu assuming spherically
            # shaped clusters
            # calc distances of each data point to each cluster centre assuming
            # spherical clusters
            edist = self.fuzzy_dist(cent, data, [], [], 'fcm', cov_constr)
            tmp = edist ** (-2 / (expo - 1))  # calc new U, suppose expo != 1
            uuu = tmp / (np.ones([no_clust, 1]) * np.sum(tmp, 0))
            m_f = uuu ** expo   # m_f matrix after exponential modification
        # if center matrix is provided
        elif init.shape[0] == no_clust and init.shape[1] == data_types:
            cent = init
            # calc distances of each data point to each cluster centre assuming
            # spherical clusters
            edist = fuzzy_dist(cent, data, [], [], 'fcm', cov_constr)
            tmp = edist ** (-2.0 / (expo - 1))  # calc new U, suppose expo != 1
            uuu = tmp / (np.ones([no_clust, 1]) * np.sum(tmp, 0))
            m_f = uuu ** expo  # MF matrix after exponential modification

        # if membership matrix is provided
        elif init.shape[0] == no_clust and init.shape[1] == no_samples:
            if init[init < 0].size > 0:  # check for negative memberships
                self.showlog('No negative memberships allowed!')
                # scale given memberships to a column sum of unity
            uuu = init / (np.ones([no_clust, 1]) * init.sum())
            # MF matrix after exponential modification
            m_f = uuu ** expo
            # new inital center matrix based on the given membership
            cent = m_f * data / ((np.ones([np.size(data, 2), 1]) *
                                  (m_f.T).sum()).T)
            # calc distances of each data point to each cluster centre assuming
            # spherical clusters
            edist = fuzzy_dist(cent, data, [], [], 'fcm', cov_constr)

        centfix = abs(centfix)

        obj_fcn = np.zeros(maxit)

        for i in self.piter(range(maxit)):  # loop over all iterations
            cent_prev = cent  # store result of last iteration
            uprev = uuu
            dist_prev = edist
            if i > 0:
                # calc new centers
                cent = np.dot(m_f, data) / ((np.ones([data.shape[1], 1]) *
                                             np.sum(m_f, 1)).T)
            # calc distances of each data point to each cluster centre
            edist = fuzzy_dist(cent, data, uuu, expo, cltype, cov_constr)
            tmp = edist ** (-2 / (expo - 1))  # calc new uuu, suppose expo != 1
            uuu = tmp / np.sum(tmp, 0)
            m_f = uuu ** expo
            obj_fcn[i] = np.sum((edist ** 2) * m_f)  # objective function
            if i > 0:
                self.showlog('Iteration: ' + str(i) + ' Threshold: ' +
                             str(term_thresh) + ' Current: ' +
                             '{:.2e}'.format(100*((obj_fcn[i - 1] -
                                                   obj_fcn[i]) /
                                                  obj_fcn[i - 1])), True)

                # if objective function has increased
                if obj_fcn[i] > obj_fcn[i - 1]:
                    uuu = uprev  # use memberships and
                    cent = cent_prev  # centers od the previous iteration
                    #  eliminate last value for objective function and
                    obj_fcn = np.delete(obj_fcn, np.s_[i::])
                    edist = dist_prev
                    break  # terminate
                # if improvement less than given termination threshold
                if (obj_fcn[i-1]-obj_fcn[i])/obj_fcn[i-1] < term_thresh/100:
                    break  # terminate

        idx = np.argmax(uuu, 0)
        vrc = vr.var_ratio(data, idx, cent, edist)
        nce = (-1.0 * (np.sum(uuu * np.log10(uuu)) / np.shape(uuu)[1]) /
               np.log10(np.shape(uuu)[0]))
        xbi = xie_beni(data, expo, uuu, cent, edist)

        return uuu, cent, obj_fcn, vrc, nce, xbi


def fuzzy_dist(cent, data, uuu, expo, cltype, cov_constr):
    """
    Fuzzy distance.

    Parameters
    ----------
    cent : numpy array
        DESCRIPTION.
    data : numpy array
        Input data.
    uuu : numpy array
        Membership function matrix.
    expo : float
        Fuzzification exponent.
    cltype : str
        Clustering type.
    cov_constr : float
        Applies only to the GK algorithm. constrains the cluster shape towards
        spherical clusters.

    Returns
    -------
    ddd : numpy array
        Output data.

    """
    no_samples = data.shape[0]
    no_datasets = data.shape[1]
    no_cent = cent.shape[0]
    ddd = np.zeros([cent.shape[0], no_samples])

    # FCM
    if cltype in ('FCM', 'fcm'):
        for j in range(no_cent):
            ddd[j, :] = np.sqrt(np.sum(((data - np.ones([no_samples, 1]) *
                                         cent[j])**2), 1))
        # determinant criterion see Spath, Helmuth,
        # "Cluster-Formation and Analyse", chapter 3
    elif cltype in ('DET', 'det'):
        m_f = uuu ** expo
        for j in range(no_cent):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data - np.ones([no_samples, 1]) * cent[j]
            # Covar of the j-th cluster
            aaa = np.dot(np.ones([no_datasets, 1]) * m_f[j] * dcent.T,
                         dcent / np.sum(m_f[j], 0))

            # constrain covariance matrix if badly conditioned
            if np.linalg.cond(aaa) > 1e10:
                e_d, e_v = np.linalg.eig(aaa)
                edmax = np.max(e_d)
                e_d[1e10 * e_d < edmax] = edmax / 1e10
                aaa = np.dot(np.dot(e_v, (e_d * np.eye(no_datasets))),
                             np.linalg.inv(e_v))
            if j == 0:  # sum all covariance matrices for all clusters
                aaa0 = aaa
            else:
                aaa0 = aaa0 + aaa
        mmm = np.linalg.det(aaa0)**(1.0/no_datasets)*np.linalg.pinv(aaa0)
        dtmp = []
        # calc new distances using the same covariance matrix for all
        # clusters --> ellisoidal clusters, all clusters use equal
        # ellipsoids
        for j in range(no_cent):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data - np.ones([no_samples, 1]) * cent[j]
            dtmp.append(np.sum(np.dot(dcent, mmm) * dcent, 1).T)
        ddd = np.sqrt(np.array(dtmp))
    # GK
    elif cltype in ['GK', 'gk']:
        m_f = uuu ** expo
        dtmp = []
        for j in range(no_cent):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data - np.ones([no_samples, 1]) * cent[j]
            # Covariance of the j-th cluster
            aaa = np.dot(np.ones([no_datasets, 1]) * m_f[j] * dcent.T,
                         dcent / np.sum(m_f[j], 0))
            aaa0 = np.eye(no_datasets)
            # if cov_constr>0, this enforces not to elongated ellipsoids -->
            # avoid the needle-like cluster
            aaa = (1.0-cov_constr)*aaa + cov_constr*(aaa0/no_samples)
            # constrain covariance matrix if badly conditioned
            if np.linalg.cond(aaa) > 1e10:
                e_d, e_v = np.linalg.eig(aaa)
                edmax = np.max(e_d)
                e_d[1e10 * e_d < edmax] = edmax / 1e10
                aaa = np.dot(np.dot(e_v, (e_d * np.eye(no_datasets))),
                             np.linalg.inv(e_v))
            # GK Code
            mmm = np.linalg.det(aaa)**(1.0/no_datasets)*np.linalg.pinv(aaa)
            dtmp.append(np.sum(np.dot(dcent, mmm) * dcent, 1).T)
        ddd = np.sqrt(np.array(dtmp))
    # GG
    elif cltype in ['GG', 'gg']:
        m_f = uuu ** expo
        dtmp = []
        for j in range(no_cent):
            # difference between each sample attribute to the corresponding
            # attribute of the j-th cluster
            dcent = data - cent[j]
            # Covariance of the j-th cluster
            aaa = np.dot(m_f[j] * dcent.T, dcent / np.sum(m_f[j], 0))
            aaa0 = np.eye(no_datasets)
            # if cov_constr>0, this enforces not to elongated ellipsoids -->
            # avoid the needle-like cluster
            aaa = (1.0-cov_constr)*aaa + cov_constr*(aaa0/no_samples)

            # constrain covariance matrix if badly conditioned
            if np.linalg.cond(aaa) > 1e10:
                e_d, e_v = np.linalg.eig(aaa)
                edmax = np.max(e_d)
                e_d[1e10 * e_d < edmax] = edmax / 1e10
                aaa = np.dot(np.dot(e_v, (e_d * np.eye(no_datasets))),
                             np.linalg.inv(e_v))
            # GG code
            ppp = 1.0 / no_samples * np.sum(m_f[j])

            t_1 = np.linalg.det(aaa)**0.5/ppp
            t_4 = np.linalg.pinv(aaa)
            t_5 = np.dot(dcent, t_4) * dcent * 0.5
            t_7 = np.exp(t_5)
            t_9 = t_1 * t_7
            t_10 = np.sum(t_9, 1).T
            dtmp.append(t_10)

        ddd = np.sqrt(np.array(dtmp))
    ddd[ddd == 0] = 1e-10  # avoid, that a data point equals a cluster center
    if (ddd == np.inf).max() == True:
        ddd[ddd == np.inf] = np.random.normal() * 1e-10  # solve break

    return ddd


def xie_beni(data, expo, uuu, center, edist):
    """
    Xie Beni.

    Calculates the Xie-Beni index
    accepts missing values when given as nan elements in the data base)
    min xbi is optimal

    Parameters
    ----------
    data : numpy array
        input dataset
    expo : float
    uuu : numpy array
        membership matrix (FCM) or cluster index values (k-means)
    center : numpy array
        cluster centers
    edist : numpy array

    Returns
    -------
    xbi : numpy array
        xie beni index

    """
    if edist.size == 0:  # calc euclidian distances if no distances are
        #                  provided
        for k in range(center.shape[0]):  # no of clusters
            # squared distance of all data values to the k-th cluster,
            # contains nan for missing values
            dummy = np.dot(data - np.ones(np.size(data, 1), 1),
                           center[k] ** 2).T
            # put in zero distances for all missing values, now all nans are
            # replaced by zeros.
            dummy[np.isnan(dummy) == 1] = 0
            # calc distance matrix from dat points to centres
            # (equals distfcm_mv.m)
            edist[k] = np.sqrt(np.sum(dummy))

    m_f = uuu ** expo
    # equal to objective function without spatial constraints
    numerator = np.sum((edist ** 2) * m_f)

    min_cdist = np.inf  # set minimal centre distance to infinity
    cdist = []
    for i in range(center.shape[0]):  # no of clusters
        dummy_cent = center
        # eliminate the i th row from center
        dummy_cent = np.delete(dummy_cent, i, 0)
        # no of cluster minus one row
        for j in range(dummy_cent.shape[0]):
            # calc squared distance between the selected two clustercentrs,
            # incl. nan if center values are nan
            cdist.append((center[i] - dummy_cent[j]) ** 2)
    cdist = np.array(cdist)
    cdist1 = np.sum(cdist, 1)
    min_cdist = cdist1.min()
    xbi = numerator / (data.shape[0] * min_cdist)
    return xbi
