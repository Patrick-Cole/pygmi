# -----------------------------------------------------------------------------
# Name:        var_ratio.py (part of PyGMI)
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
""" Variance Ratio """

import numpy as np


def var_ratio(data, uuu, center, dist_orig):
    """
    Variance Ratio

    Calculates the Variance ratio criterion after Calinski and Harabasz,
    1974. Does not accept missing data.
    Max VRC is optimal.
    U can either be membership matrix (FCM) or cluster index values (k-means)

    Parameters
    ----------
    data : numpy array
        input dataset
    uuu : numpy array
        membership matrix (FCM) or cluster index values (k-means)
    center : numpy array
        cluster centers
    dist_orig : numpy array

    Returns
    -------
    vrc : numpy array
        variance ration criterion
    """

    if uuu.ndim == 1:   # check whether fuzzy or crisp info is given
        crisp = uuu
    else:
        # alp = uuu.max(0)
        crisp = uuu.argmin(0)

# sum of squared dist between cluster cent
# calc the overall mean for each data type considering the entire data set
    m_all = np.nanmean(data)
# squared dist of cluster centres from overall mean
    dis_cent = (center - (np.ones([center.shape[0], 1]) * m_all)) ** 2
    dis_cent = np.sum(dis_cent, 1)

    icd = 0
    for i in range(center.shape[0]):
        # use [0] to get rid of tuple
        # grab indices of data values grouped into cluster ii
        index = np.nonzero(i == crisp)[0]
        # no of data points falling into the i th cluster
        no_clmem = len(index)
        dis_cent[i] = dis_cent[i] * no_clmem
        if dist_orig.size > 0:
            edist = dist_orig[i, index] ** 2
        else:
            # calc squared distances of all values from the centre belonging
            # to cluster i
            edist = (data[index] - (np.ones([no_clmem, 1]) * center[i])) ** 2

        icd += np.sum(edist)

# normalize squared sum by no of clusters minus one
    numerator = np.sum(dis_cent) / (center.shape[0] - 1)
# normalization on number of data and number of clusters
    denom = icd / (data.shape[0] - center.shape[0])
    vrc = numerator / denom

    return vrc
