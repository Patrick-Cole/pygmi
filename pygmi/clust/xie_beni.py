# -----------------------------------------------------------------------------
# Name:        xie_beni.py (part of PyGMI)
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
"""Xie Beni."""

import numpy as np


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
            dummy = np.dot(((data - np.ones(np.size(data, 1), 1),
                             center[k]) ** 2).T)
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
#    cnt = 0
    cdist = []
    for i in range(center.shape[0]):  # no of clusters
        dummy_cent = center
        # eliminate the i th row from center
        dummy_cent = np.delete(dummy_cent, i, 0)
        # no of cluster minus one row
        for j in range(dummy_cent.shape[0]):
            #            cnt += 1
            # calc squared distance between the selected two clustercentrs,
            # incl. nan if center values are nan
            cdist.append((center[i] - dummy_cent[j]) ** 2)
    cdist = np.array(cdist)
# mv=nanmin(cdist)
# dummy_mv=ones(cnt,1)*mv
# cdist(isnan(cdist(:))==1)=dummy_mv(isnan(cdist(:))==1)
    cdist1 = np.sum(cdist, 1)
    min_cdist = cdist1.min()
    xbi = numerator / (data.shape[0] * min_cdist)
    return xbi
