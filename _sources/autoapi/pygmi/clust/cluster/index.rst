pygmi.clust.cluster
===================

.. py:module:: pygmi.clust.cluster

.. autoapi-nested-parse::

   Cluster Analysis.

   The cluster module performs unsupervised classification using the
   scikit-learn library.



Classes
-------

.. autoapisummary::

   pygmi.clust.cluster.Cluster


Functions
---------

.. autoapisummary::

   pygmi.clust.cluster.cluster


Module Contents
---------------

.. py:class:: Cluster(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Cluster analysis GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: combo()

      Set up combo box, used to choose clustering algorithm.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: update_vars()

      Update the variables.

      :rtype: None.



.. py:function:: cluster(data, cltype='K-Means', sscale=True, rscale=False, min_cluster=5, max_cluster=5, tol=0.0001, max_iter=300, eps=0.5, bthres=0.5, branchfac=50, xi=0.05, min_samples=5, showlog=print, piter=iter)

   Run the cluster analysis.

   This function uses the scikit learn library.

   :param data: List of PyGMI data (pygmi.raster.datatypes.Data).
   :type data: list
   :param cltype: Cluster analysis type. Can be one of 'K-Means',
                  'Mini Batch K-Means (fast)', 'Bisecting K-Means', 'DBSCAN', 'OPTICS',
                  'Birch'. The default is 'K-Means'.
   :type cltype: str, optional
   :param sscale: Use standard scaling. The default is True.
   :type sscale: bool, optional
   :param rscale: Use robust scaling. The default is False.
   :type rscale: bool, optional
   :param min_cluster: Minimum number of clusters to find. The default is 5.
   :type min_cluster: int, optional
   :param max_cluster: Maximum number of clusters to find. The default is 5.
   :type max_cluster: int, optional
   :param tol: Tolerance (K-Means only). The default is 0.0001.
   :type tol: float, optional
   :param max_iter: Maximum number of iterations (K-Means only). The default is 300.
   :type max_iter: int, optional
   :param eps: Epsilon factor (DBSCAN only). The default is 0.5.
   :type eps: float, optional
   :param bthres: Threshold for Birch. The default is 0.5.
   :type bthres: float, optional
   :param branchfac: Branching factor for Birch. The default is 50.
   :type branchfac: float, optional
   :param xi: Minimum steepness on the reachability plot for OPTICS.
              The default is 0.05.
   :type xi: float, optional
   :param min_samples: Minimum samples for DBSCAN. The default is 5.
   :type min_samples: int, optional
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: List of raster datasets of classes.
   :rtype: list


