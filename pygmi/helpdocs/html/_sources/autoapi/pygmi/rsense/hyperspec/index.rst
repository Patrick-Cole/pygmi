pygmi.rsense.hyperspec
======================

.. py:module:: pygmi.rsense.hyperspec

.. autoapi-nested-parse::

   Hyperspectral Interpretation Routines.



Classes
-------

.. autoapisummary::

   pygmi.rsense.hyperspec.GraphMap
   pygmi.rsense.hyperspec.AnalSpec
   pygmi.rsense.hyperspec.ProcFeatures


Functions
---------

.. autoapisummary::

   pygmi.rsense.hyperspec.calcfeatures
   pygmi.rsense.hyperspec.indexcalc
   pygmi.rsense.hyperspec.fproc
   pygmi.rsense.hyperspec.cubic_calc
   pygmi.rsense.hyperspec.phulljit
   pygmi.rsense.hyperspec.phull
   pygmi.rsense.hyperspec.readsli


Module Contents
---------------

.. py:class:: GraphMap(parent=None)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   Graph Map widget.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: init_graph()

      Initialise the graph.

      :rtype: None.



   .. py:method:: compute_initial_figure()

      Compute initial figure.



.. py:class:: AnalSpec(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   Analyse spectra GUI.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: button_press_callback(event)

      Button press callback.

      :param event: Mouse Event.
      :type event: matplotlib.backend_bases.MouseEvent

      :rtype: None.



   .. py:method:: disp_splib(row)

      Change library spectra for display.

      :param row: row of table, unused.
      :type row: int

      :rtype: None.



   .. py:method:: feature_change()

      Change depth marker combo.

      :rtype: None.



   .. py:method:: hull()

      Change whether hull is removed or not.

      :rtype: None.



   .. py:method:: load_splib()

      Load ENVI spectral library data.

      :rtype: None.



   .. py:method:: on_combo()

      On combo.

      :rtype: None.



   .. py:method:: rotate_view()

      Rotates view.

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



.. py:class:: ProcFeatures(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   GUI to process hyperspectral features.

   :param parent: Reference to the parent routine. The default is None.
   :type parent: parent, optional


   .. py:method:: setupui()

      Set up UI.

      :rtype: None.



   .. py:method:: product_change()

      Change product combo.

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



.. py:function:: calcfeatures(dat, mineral, feature, ratio, product, *, cryst=None, rfilt=True, piter=iter)

   Calculate feature dataset.

   :param dat: Input PyGMI data.
   :type dat: list of pygmi.raster.datatypes.Data
   :param mineral: Mineral description.
   :type mineral: str
   :param feature: Dictionary containing the hyperspectral features.
   :type feature: dictionary
   :param ratio: Dictionary containing string definitions of ratios.
   :type ratio: dictionary
   :param product: Final hyperspectral products. Each dictionary value, is a list of
                   features or ratios with thresholds to be combined.
   :type product: dictionary
   :param cryst: Crystallinity of the product, if available
   :type cryst: dictionary, optional
   :param rfilt: Flag to decide whether to filter final ratio products less than 1.0
   :type rfilt: bool
   :param piter: Progress bar iterable. The default is iter.
   :type piter: function, optional

   :returns: **datfin** -- Output datasets.
   :rtype: list of pygmi.raster.datatypes.Data.


.. py:function:: indexcalc(formula, dat)

   Calculate an index using numexpr.

   :param formula: string expression containing index formula.
   :type formula: str
   :param dat: Dictionary of variables to be used in calculation.
   :type dat: dict

   :returns: **out** -- This can be a masked array.
   :rtype: numpy array


.. py:function:: fproc(fdat, ptmp, dtmp, i1a, i2a, xdat, mtmp)

   Feature process.

   This function finds the minimum value of a feature.

   :param fdat: Feature data
   :type fdat: numpy array
   :param ptmp: Feature wavelengths.
   :type ptmp: numpy array
   :param dtmp: Feature depths.
   :type dtmp: numpy array
   :param i1a: Start index of feature definition.
   :type i1a: int
   :param i2a: End Index of feature definition.
   :type i2a: int
   :param xdat: Wavelengths of feature definition.
   :type xdat: numpy array

   :returns: * **ptmp** (*numpy array*) -- Feature wavelengths.
             * **dtmp** (*numpy array*) -- Feature depths.


.. py:function:: cubic_calc(xdat, crem, imin)

   Find minimum of function using an analytic cubic calculation for speed.

   :param xdat: wavelengths - x data.
   :type xdat: numpy array
   :param crem: continuum removed data - y data.
   :type crem: numpy array
   :param imin: Index for estimated minimum.
   :type imin: int

   :returns: * **x** (*float*) -- wavelength at minimum.
             * **y** (*float*) -- y value at minimum.


.. py:function:: phulljit(sample1)

   Hull Calculation.

   This is only here to be called from the jit routines

   :param sample1: Sample to create a hull for.
   :type sample1: numpy array

   :returns: **out** -- Output hull.
   :rtype: numpy array


.. py:function:: phull(y)

   Calculate Continuum/hull.

   Based on: https://stackoverflow.com/questions/73382974/how-to-apply-continuum-removal-in-spectral-graph

   :param y: Sample to create a hull for.
   :type y: numpy array

   :returns: **out** -- Output hull.
   :rtype: numpy array


.. py:function:: readsli(ifile)

   Read an ENVI sli file.

   :param ifile: Input sli spectra file.
   :type ifile: str

   :returns: **spectra** -- Dictionary of spectra with wavelengths and reflectances.
   :rtype: dictionary


