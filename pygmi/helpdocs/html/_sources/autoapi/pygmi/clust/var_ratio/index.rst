pygmi.clust.var_ratio
=====================

.. py:module:: pygmi.clust.var_ratio

.. autoapi-nested-parse::

   Variance Ratio.



Functions
---------

.. autoapisummary::

   pygmi.clust.var_ratio.var_ratio


Module Contents
---------------

.. py:function:: var_ratio(data, uuu, center, dist_orig)

   Variance Ratio.

   Calculates the Variance ratio criterion after Calinski and Harabasz,
   1974. Does not accept missing data. Max VRC is optimal.

   :param data: input dataset
   :type data: numpy array
   :param uuu: membership matrix (FCM) or cluster index values (k-means)
   :type uuu: numpy array
   :param center: cluster centers
   :type center: numpy array
   :param dist_orig:
   :type dist_orig: numpy array

   :returns: **vrc** -- variance ration criterion
   :rtype: numpy array


