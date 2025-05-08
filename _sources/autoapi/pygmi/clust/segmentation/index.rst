pygmi.clust.segmentation
========================

.. py:module:: pygmi.clust.segmentation

.. autoapi-nested-parse::

   Image segmentation routines, following Baatz and Schäpe (2000).



Classes
-------

.. autoapisummary::

   pygmi.clust.segmentation.ImageSeg


Functions
---------

.. autoapisummary::

   pygmi.clust.segmentation.segment1
   pygmi.clust.segmentation.get_l


Module Contents
---------------

.. py:class:: ImageSeg(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Image Segmentation GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

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



.. py:function:: segment1(data, *, scale=500, wcolor=0.5, wcompact=0.5, doshape=True, showlog=print, piter=iter)

   Perform image segmentation.

   :param data: Input data.
   :type data: numpy array
   :param scale: Scale. The default is 500.
   :type scale: int, optional
   :param wcolor: Colour weight. The default is 0.5.
   :type wcolor: float, optional
   :param wcompact: Compactness weight. The default is 0.5.
   :type wcompact: float, optional
   :param doshape: Perform shape segmentation. The default is True.
   :type doshape: bool, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **omap** -- Output data.
   :rtype: numpy array


.. py:function:: get_l(data)

   Get bounding box length.

   :param data: Input data.
   :type data: numpy array

   :returns: **ltmp** -- Bounding box length.
   :rtype: int


