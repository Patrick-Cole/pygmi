pygmi.clust.cluster
===================

.. py:module:: pygmi.clust.cluster

.. autoapi-nested-parse::

   The cluster module performs unsupervised classification using the
   scikit-learn library.



Classes
-------

.. autoapisummary::

   pygmi.clust.cluster.Cluster


Module Contents
---------------

.. py:class:: Cluster(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Cluster analysis GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


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



   .. py:method:: acceptall()

      Run the cluster analysis.

      :rtype: None.



