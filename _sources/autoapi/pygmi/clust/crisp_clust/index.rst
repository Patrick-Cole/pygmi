pygmi.clust.crisp_clust
=======================

.. py:module:: pygmi.clust.crisp_clust

.. autoapi-nested-parse::

   Crisp clustering is a set of clustering routines.

   This uses standard statistical methods, as opposed to fuzzy methods.



Classes
-------

.. autoapisummary::

   pygmi.clust.crisp_clust.CrispClust


Functions
---------

.. autoapisummary::

   pygmi.clust.crisp_clust.crispclust
   pygmi.clust.crisp_clust.crisp_means
   pygmi.clust.crisp_clust.gcentroids
   pygmi.clust.crisp_clust.gdist


Module Contents
---------------

.. py:class:: CrispClust(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Crisp cluster GUI class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: combo()

      Set up combo box to choose algorithm.

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



.. py:function:: crispclust(data, cltype='k-means', min_cluster=5, max_cluster=5, cov_constr=0.0, no_runs=1, max_iter=100, term_thresh=1e-05, init_type='random', ifiles=None, showlog=print, piter=iter)

   Crisp Clustering.

   :param data: List of PyGMI data (pygmi.raster.datatypes.Data).
   :type data: list
   :param cltype: Clustering method, by default 'k-means'
   :type cltype: str, optional
   :param min_cluster: minimum number of clusters, by default 5
   :type min_cluster: int, optional
   :param max_cluster: maximum number of clusters, by default 5
   :type max_cluster: int, optional
   :param cov_constr: scalar between [0 1], by default 0.
   :type cov_constr: _type_, optional
   :param no_runs: number of runs, by default 1
   :type no_runs: int, optional
   :param max_iter: maximum iterations, by default 100
   :type max_iter: int, optional
   :param term_thresh: terminating threshold, by default 0.00001
   :type term_thresh: float, optional
   :param init_type: initial guess, by default 'random'
   :type init_type: str, optional
   :param ifiles: Files used in 'manual' initial guess, by default None
   :type ifiles: list, optional
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **dat_out** -- List of raster datasets of classes.
   :rtype: list


.. py:function:: crisp_means(data, no_clust, cent, centfix, maxit, term_thresh, cltype, cov_constr, showlog=print, piter=iter)

   Script enables the crisp clustering of COMPLETE multi-variate datasets.

   :param data: N x P matrix containing the data to be clustered, N is number of
                samples, P is number of different attributes available for each
                sample.
   :type data: numpy array
   :param no_clust: Number of clusters to be used.
   :type no_clust: int
   :param cent: cluster centre positions, either empty [] --> randomly guessed
                center positions will be used for initialisation or NO_CLUSTxP
                matrix
   :type cent: numpy array
   :param centfix: Constrains the position of cluster centers, if CENTFIX is empty,
                   cluster centers can freely vary during cluster analysis, otherwise
                   CENTFIX is of equal size to CENT and gives an absolute deviation
                   from initial center positions that should not be exceeded during
                   clustering. Note, CETNFIX applies only if center values are
                   provided by the user.
   :type centfix: numpy array
   :param maxit: number of maximal allowed iterations.
   :type maxit: int
   :param term_thresh: Termination threshold, either empty [] --> go for the maximum
                       number of iterations MAXIT or a scalar giving the minimum
                       reduction of the size of the objective function for two consecutive
                       iterations in Percent.
   :type term_thresh: float
   :param cltype: either 'kmeans' --> kmeans cluster analysis (spherically shaped
                  cluster), 'det' --> uses the determinant criterion of Spath, H.,
                  "Cluster-Formation and Analyse, chapter3" (ellipsoidal clusters,
                  all cluster use the same ellipsoid), or 'vardet' --> Spath, H.,
                  chapter 4 (each cluster uses its individual ellipsoid). Note: the
                  latter is the crisp version of the Gustafson-Kessel algorithm
   :type cltype: str
   :param cov_constr: scalar between [0 1], values > 0 trim the covariance matrix
                      to avoid needle-like ellipsoids for the clusters, applies only for
                      cltype='vardet', but must always be provided.
   :type cov_constr: float
   :param showlog: Show information using a function. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: * **idx** (*numpy array*) -- cluster index number for each sample after the last iteration,
               column vector.
             * **cent** (*numpy array*) -- matrix with cluster centre positions after last iteration, one
               cluster centre per row
             * **obj_fcn** (*numpy array*) -- Vector, size of the objective function after each iteration
             * **vrc** (*numpy array*) -- Variance Ratio Criterion


.. py:function:: gcentroids(data, index, no_clust, mindist)

   G Centroids.

   :param data: Input data.
   :type data: numpy array
   :param index: Cluster index number for each sample.
   :type index: numpy array
   :param no_clust: Number of clusters to be used.
   :type no_clust: int
   :param mindist: Minimum distances.
   :type mindist: numpy array

   :returns: * **centroids** (*numpy array*) -- Centroids
             * **index** (*numpy array*) -- Index


.. py:function:: gdist(data, center, index, no_clust, cltype, cov_constr)

   G Dist routine.

   :param data: Input data.
   :type data: numpy array
   :param center: center of each class.
   :type center: numpy array
   :param index: Cluster index number for each sample.
   :type index: numpy array
   :param no_clust: Number of clusters to be used.
   :type no_clust: int
   :param cltype: Clustering type.
   :type cltype: str
   :param cov_constr: scalar between [0 1].
   :type cov_constr: float

   :returns: **bigd** -- Output data.
   :rtype: numpy array


