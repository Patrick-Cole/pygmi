pygmi.rsense.change
===================

.. py:module:: pygmi.rsense.change

.. autoapi-nested-parse::

   Calculate change detection indices.



Classes
-------

.. autoapisummary::

   pygmi.rsense.change.CalculateChange


Functions
---------

.. autoapisummary::

   pygmi.rsense.change.calc_change
   pygmi.rsense.change.calc_mean
   pygmi.rsense.change.calc_sam
   pygmi.rsense.change.coefv
   pygmi.rsense.change.imean
   pygmi.rsense.change.match_data
   pygmi.rsense.change.sam
   pygmi.rsense.change.scm
   pygmi.rsense.change.stddev


Module Contents
---------------

.. py:class:: CalculateChange(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to calculate change indices.

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

      :rtype: None.



   .. py:method:: setindices()

      Set the available indices.

      :rtype: None.



   .. py:method:: invert_selection()

      Invert the selected indices.

      :rtype: None.



   .. py:method:: set_selected_indices()

      Set the selected indices.

      :rtype: None.



.. py:function:: calc_change(flist, ilist=None, showlog=print, piter=iter)

   Calculate Change Indices.

   :param flist: List of batch file list data.
   :type flist: list of RasterMeta.
   :param ilist: List of strings describing index to calculate.
   :type ilist: list, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **datfin** -- List of PyGMI Data.
   :rtype: list of pygmi.raster.datatypes.Data


.. py:function:: calc_mean(flist, showlog=print, piter=iter)

   Load data and calculate iterative Mean.

   :param flist: List of batch file list data.
   :type flist: list of RasterMeta
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: * **meandat** (*dictionary of pygmi.raster.datatypes.Data.*) -- PyGMI Data representing means.
             * **cnt** (*dictionary of numpy arrays*) -- Count of values which made up mean.
             * **M** (*dictionary of numpy arrays*) -- Variance parameter, where Variance = M/cnt.


.. py:function:: calc_sam(flist, showlog=print, piter=iter)

   Load data and calculate spectral angle between two times.

   :param flist: List of batch file list data.
   :type flist: list of RasterMeta.
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: **angle** -- PyGMI Data of SAM angles.
   :rtype: pygmi.raster.datatypes.Data


.. py:function:: coefv(mean, std)

   Calculate coefficient of variation.

   :param mean: numpy array of mean values.
   :type mean: numpy array
   :param std: numpy array of standard deviation values.
   :type std: numpy array

   :returns: **cv** -- Array of coefficient of variation values.
   :rtype: numpy array


.. py:function:: imean(mean, newdat, cnt=None, M=None)

   Calculate mean and variance parameters.

   :param mean: existing mean values.
   :type mean: numpy array
   :param newdat: new data to be added to mean..
   :type newdat: numpy array
   :param cnt: cnt of values which made up mean. The default is None.
   :type cnt: numpy array, optional
   :param M: Variance parameter, where Variance = M/cnt. The default is None.
   :type M: numpy array, optional

   :returns: * **mean** (*numpy array*) -- Updated mean of data.
             * **cnt** (*numpy array*) -- Updated cnt of values which made up mean.
             * **M** (*numpy array*) -- Updated variance parameter, where Variance = M/cnt.


.. py:function:: match_data(flist, showlog=print, piter=iter)

   Match two datasets.

   This routine also puts the datasets in order of date.

   :param flist: List of batch file list data.
   :type flist: list of RasterMeta or Data lists
   :param showlog: Display information. The default is print.
   :type showlog: function, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional

   :returns: * **dat1** (*list of pygmi.raster.datatypes.Data*) -- First dataset with matched bands only.
             * **dat2** (*list of pygmi.raster.datatypes.Data*) -- Second dataset with matched bands only.


.. py:function:: sam(s1, s2)

   Calculate Spectral Angle Mapper (SAM).

   :param s1: Spectrum 1.
   :type s1: numpy array
   :param s2: Spectrum 2.
   :type s2: numpy array

   :returns: **result** -- Output angles.
   :rtype: numpy array


.. py:function:: scm(s1, s2)

   SCM or MSAM.

   :param s1: Spectrum 1.
   :type s1: numpy array
   :param s2: Spectrum 2.
   :type s2: numpy array

   :returns: **result** -- Output angles.
   :rtype: numpy array


.. py:function:: stddev(M, cnt)

   Calculate std deviation.

   :param M: Variance parameter, where Variance = M/cnt.
   :type M: numpy array
   :param cnt: cnt of values which made up mean.
   :type cnt: numpy array

   :returns: **std** -- Calculated standard deviation.
   :rtype: numpy array


