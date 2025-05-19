pygmi.clust.fuzzy_clust
=======================

.. py:module:: pygmi.clust.fuzzy_clust

.. autoapi-nested-parse::

   Fuzzy clustering is a set of clustering routines.

   This makes use of fuzzy logic.



Classes
-------

.. autoapisummary::

   pygmi.clust.fuzzy_clust.FuzzyClust


Functions
---------

.. autoapisummary::

   pygmi.clust.fuzzy_clust.fuzzyclust
   pygmi.clust.fuzzy_clust.fuzzy_means
   pygmi.clust.fuzzy_clust.fuzzy_dist
   pygmi.clust.fuzzy_clust.xie_beni


Module Contents
---------------

.. py:class:: FuzzyClust(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Fuzzy clustering GUI class.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


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



.. py:function:: fuzzyclust(data, cltype='fuzzy c-means', min_cluster=5, max_cluster=5, cov_constr=0.0, no_runs=1, max_iter=100, expo=1.5, term_thresh=1e-05, init_type='random', ifiles=None, showlog=print, piter=iter)

   Run.

   :rtype: None.


.. py:function:: fuzzy_means(data, no_clust, init, centfix, maxit, term_thresh, expo, cltype, cov_constr, showlog=print, piter=iter)

   Fuzzy clustering.

   Finds NO_CLUST clusters in the data set DATA.. Supported algorithms are
   fuzzy c-means, Gustafson-Kessel, advanced fuzzy c-means.


   :param data: DATA is size M-by-N, where M is the number of samples
                and N is the number of coordinates (attributes) for each sample.
   :type data: numpy array
   :param no_clust: Number of clusters.
   :type no_clust: int
   :param init: INIT may be set to [], in this case the FCM generates random
                initial center locations to start the algorithm. Alternatively,
                INIT can be of matrix type, either containing a user-given
                membership matrix [NO_CLUST M] or a cluster center matrix
                [NO_CLUST, N].
   :type init: numpy array
   :param centfix: Constrains the position of cluster centers.
   :type centfix: numpy array
   :param maxit: MAXIT give the maximum number of iterations..
   :type maxit: int
   :param term_thresh: Gives the required minimum improvement in per cent per
                       iteration. (termination threshold)
   :type term_thresh: float
   :param expo: Fuzzification exponent.
   :type expo: float
   :param cltype: either 'FCM' for fuzzy c-means (spherically shaped clusters),
                  'DET' for advanced fuzzy c-means (ellipsoidal clusters, all
                  clusters use the same ellipsoid), or 'GK' for Gustafson-Kessel
                  clustering (ellipsoidal clusters, each cluster uses its own
                  ellipsoid).
   :type cltype: str
   :param cov_constr: COV_CONSTR applies only to the GK algorithm. constrains the cluster
                      shape towards spherical clusters to avoid needle-like clusters.
                      COV_CONSTR = 1 make the GK algorithm equal to the FCM algorithm,
                      COV_CONSTR = 0 results in no constraining of the covariance
                      matrices of the clusters.
   :type cov_constr: float

   :returns: * **uuu** (*numpy array*) -- This membership function matrix contains the grade of
               membership of each data sample to each cluster.
             * **cent** (*numpy array*) -- The coordinates for each cluster center are returned in the rows
               of the matrix CENT.
             * **obj_fcn** (*numpy array*) -- At each iteration, an objective function is minimized to find the
               best location for the clusters and its values are returned in
               OBJ_FCN.
             * **vrc** (*numpy array*) -- Variance ration criterion.
             * *nce* -- Normalised class entropy.
             * **xbi** (*numpy array*) -- Xie beni index.


.. py:function:: fuzzy_dist(cent, data, uuu, expo, cltype, cov_constr)

   Fuzzy distance calculation.

   :param cent: Class centers.
   :type cent: numpy array
   :param data: Input data.
   :type data: numpy array
   :param uuu: Membership function matrix.
   :type uuu: numpy array
   :param expo: Fuzzification exponent.
   :type expo: float
   :param cltype: Clustering type.
   :type cltype: str
   :param cov_constr: Applies only to the GK algorithm. constrains the cluster shape towards
                      spherical clusters.
   :type cov_constr: float

   :returns: **ddd** -- Output data.
   :rtype: numpy array


.. py:function:: xie_beni(data, expo, uuu, center, edist)

   Calculate the Xie-Beni index.

   Accepts missing values when given as nan elements in the data base). A
   small Xie-Beni index is optimal.

   :param data: input dataset
   :type data: numpy array
   :param expo:
   :type expo: float
   :param uuu: membership matrix (FCM) or cluster index values (k-means)
   :type uuu: numpy array
   :param center: cluster centers
   :type center: numpy array
   :param edist:
   :type edist: numpy array

   :returns: **xbi** -- xie beni index
   :rtype: numpy array


