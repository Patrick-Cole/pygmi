pygmi.rsense.transforms
=======================

.. py:module:: pygmi.rsense.transforms

.. autoapi-nested-parse::

   Transforms such as PCA and MNF.



Classes
-------

.. autoapisummary::

   pygmi.rsense.transforms.MNF
   pygmi.rsense.transforms.PCA


Functions
---------

.. autoapisummary::

   pygmi.rsense.transforms.get_noise
   pygmi.rsense.transforms.mnf_calc
   pygmi.rsense.transforms.pca_calc
   pygmi.rsense.transforms.pca_calc_fitlist
   pygmi.rsense.transforms.blockwise_cov
   pygmi.rsense.transforms.blockwise_dot


Module Contents
---------------

.. py:class:: MNF(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to perform MNF transform.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: changeoutput()

      Change the interface to reflect whether full calculation is needed.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:class:: PCA(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to perform PCA transform.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: pygmi.main.MainWidget, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: settings(nodialog=False)

      Entry point into item.

      :param nodialog: Run settings without a dialog. The default is False.
      :type nodialog: bool, optional

      :returns: True if successful, False otherwise.
      :rtype: bool



   .. py:method:: changeoutput()

      Change the interface to reflect whether full calculation is needed.

      :rtype: None.



   .. py:method:: saveproj()

      Save project data from class.

      :rtype: None.



   .. py:method:: acceptall()

      Accept option.

      Updates self.outdata, which is used as input to other modules.

      :rtype: None.



.. py:function:: get_noise(x2d, mask, noisetype='', piter=iter)

   Calculate noise dataset from original data.

   :param x2d: Input array, of dimension (MxNxChannels).
   :type x2d: numpy array
   :param mask: mask of dimension (MxN).
   :type mask: numpy array
   :param noisetype: Noise type to calculate. Can be 'diagonal', 'hv average' or ''.
                     The default is ''.
   :type noisetype: str, optional

   :returns: * **nevals** (*numpy array*) -- Noise eigenvalues.
             * **nevecs** (*numpy array*) -- Noise eigenvectors.


.. py:function:: mnf_calc(dat, *, ncmps=None, noisetxt='hv average', showlog=print, piter=iter, fwdonly=True)

   MNF Calculation.

   :param dat: List of PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param ncmps: Number of components to use for filtering. The default is None
                 (meaning all).
   :type ncmps: int or None, optional
   :param noisetxt: Noise type. Can be 'diagonal', 'hv average' or 'quad'. The default is
                    'hv average'.
   :type noisetxt: txt, optional
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param piter: Iteration function, used for progress bars. The default is iter.
   :type piter: function, optional
   :param fwdonly: Option to perform forward calculation only. The default is True.
   :type fwdonly: bool, optional

   :returns: * **odata** (*list of pygmi.raster.datatypes.Data.*) -- Output list of PyGMI Data. Can be forward or inverse transformed data.
             * **ev** (*numpy array*) -- Explained variance, from PCA.


.. py:function:: pca_calc(dat, ncmps=None, showlog=print, piter=iter, fwdonly=True)

   PCA Calculation.

   :param dat: List of PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param ncmps: Number of components to use for filtering. The default is None
                 (meaning all).
   :type ncmps: int or None, optional
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param piter: Iteration function, used for progress bars. The default is iter.
   :type piter: function, optional
   :param fwdonly: Option to perform forward calculation only. The default is True.
   :type fwdonly: bool, optional

   :returns: * **odata** (*list of pygmi.raster.datatypes.Data.*) -- Output list of PyGMI Data. Can be forward or inverse transformed data.
             * **ev** (*numpy array*) -- Explained variance, from PCA.


.. py:function:: pca_calc_fitlist(flist, ncmps=None, showlog=print, piter=iter, fwdonly=True)

   PCA Calculation with using list of files in common fit.

   :param dat: List of PyGMI Data.
   :type dat: list of pygmi.raster.datatypes.Data.
   :param ncmps: Number of components to use for filtering. The default is None
                 (meaning all).
   :type ncmps: int or None, optional
   :param showlog: Function for printing text. The default is print.
   :type showlog: function, optional
   :param piter: Iteration function, used for progress bars. The default is iter.
   :type piter: function, optional
   :param fwdonly: Option to perform forward calculation only. The default is True.
   :type fwdonly: bool, optional

   :returns: * **odata** (*list of pygmi.raster.datatypes.Data.*) -- Output list of PyGMI Data.Can be forward or inverse transformed data.
             * **ev** (*numpy array*) -- Explained variance, from PCA.


.. py:function:: blockwise_cov(A)

   Blockwise covariance.

   :param A: Matrix.
   :type A: numpy array

   :returns: **ncov** -- Covariance matrix.
   :rtype: numpy array


.. py:function:: blockwise_dot(A, B, max_elements=int(2**27))

   Compute the dot product of two matrices in a block-wise fashion.

   Only blocks of `A` with a maximum size of `max_elements` will be
   processed simultaneously.

   from : https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays

   :param A: MxN matrix.
   :type A: numpy array
   :param B: NxO matrix.
   :type B: Numpy array
   :param max_elements: Maximum number of elements in a block. The default is int(2**27).
   :type max_elements: int, optional

   :returns: **out** -- Output dot product.
   :rtype: numpy array


