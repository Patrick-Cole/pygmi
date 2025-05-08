pygmi.pfmod.misc
================

.. py:module:: pygmi.pfmod.misc

.. autoapi-nested-parse::

   These are miscellaneous functions for the program.



Classes
-------

.. autoapisummary::

   pygmi.pfmod.misc.ProgressBar
   pygmi.pfmod.misc.MergeMod3D


Functions
---------

.. autoapisummary::

   pygmi.pfmod.misc.update_lith_lw
   pygmi.pfmod.misc.gmerge


Module Contents
---------------

.. py:function:: update_lith_lw(lmod, lwidget)

   Update the lithology list widget.

   :param lmod: 3D model.
   :type lmod: LithModel
   :param lwidget: List widget.
   :type lwidget: QListWidget

   :rtype: None.


.. py:class:: ProgressBar(pbar, pbarmain)

   Wrapper for a progress bar. It consists of two progress bars.

   :param par: Progress bar.
   :type par: pygmi.misc.ProgressBar
   :param pbarmain: Main progress bar.
   :type pbarmain: pygmi.misc.ProgressBar


   .. py:method:: incr()

      Increase value by one.

      :rtype: None.



   .. py:method:: iter(iterable)

      Iterate routine.

      :param iterable: Iterable.
      :type iterable: iterable

      :Yields: **obj** (*object*) -- Object in iterable.



   .. py:method:: incrmain(i=1)

      Increase value by i.

      :param i: Iteration step. The default is 1.
      :type i: int, optional

      :rtype: None.



   .. py:method:: maxall()

      Set all progress bars to maximum value.

      :rtype: None.



   .. py:method:: resetall(maximum=1, mmax=1)

      Set min and max and resets all bars to 0.

      :param maximum: Maximum value. The default is 1.
      :type maximum: int, optional
      :param mmax: Maximum value. The default is 1.
      :type mmax: int, optional

      :rtype: None.



   .. py:method:: resetsub(maximum=1)

      Set min and max and resets sub bar to 0.

      :param maximum: Maximum value. The default is 1.
      :type maximum: int, optional

      :rtype: None.



   .. py:method:: busysub()

      Busy.

      :rtype: None.



.. py:class:: MergeMod3D(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Perform Merge of two models.

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



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :returns: True if successful, False otherwise
      :rtype: bool



.. py:function:: gmerge(master, slave, xrange=None, yrange=None)

   Routine used to merge two grids.

   :param master: PyGMI raster dataset.
   :type master: pygmi.raster.datatypes.Data
   :param slave: PyGMI raster dataset.
   :type slave: pygmi.raster.datatypes.Data
   :param xrange: List containing range of minimum and maximum X. The default is None.
   :type xrange: list, optional
   :param yrange: List containing range of minimum and maximum Y. The default is None.
   :type yrange: list, optional

   :returns: PyGMI raster dataset.
   :rtype: pygmi.raster.datatypes.Data


