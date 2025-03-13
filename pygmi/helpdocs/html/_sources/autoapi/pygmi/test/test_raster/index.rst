pygmi.test.test_raster
======================

.. py:module:: pygmi.test.test_raster

.. autoapi-nested-parse::

   These are tests. Run pytest on this file from within this directory to do
   the tests.



Functions
---------

.. autoapisummary::

   pygmi.test.test_raster.test_gradients
   pygmi.test.test_raster.test_dratio
   pygmi.test.test_raster.test_thgrad
   pygmi.test.test_raster.test_vertical
   pygmi.test.test_raster.test_viz
   pygmi.test.test_raster.test_check_dataid
   pygmi.test.test_raster.test_trimraster
   pygmi.test.test_raster.test_equation
   pygmi.test.test_raster.test_hmode
   pygmi.test.test_raster.test_aspect
   pygmi.test.test_raster.test_shader
   pygmi.test.test_raster.test_histcomp
   pygmi.test.test_raster.test_histeq
   pygmi.test.test_raster.test_img2rgb
   pygmi.test.test_raster.test_norm
   pygmi.test.test_raster.test_norm255
   pygmi.test.test_raster.test_corr2d
   pygmi.test.test_raster.smalldata
   pygmi.test.test_raster.test_io_rasterio
   pygmi.test.test_raster.test_io_ascii
   pygmi.test.test_raster.test_io_xyz
   pygmi.test.test_raster.test_normalisation
   pygmi.test.test_raster.test_smooth
   pygmi.test.test_raster.test_agc


Module Contents
---------------

.. py:function:: test_gradients()

   test directional derivative.


.. py:function:: test_dratio()

   test derivative ratio.


.. py:function:: test_thgrad()

   test total horizontal gradient.


.. py:function:: test_vertical()

   test vertical derivative.


.. py:function:: test_viz()

   test visibility.


.. py:function:: test_check_dataid()

   test check dataid.


.. py:function:: test_trimraster()

   test trim raster.


.. py:function:: test_equation()

   tests equation editor.


.. py:function:: test_hmode()

   tests hmode.


.. py:function:: test_aspect()

   tests aspect.


.. py:function:: test_shader()

   tests shader.


.. py:function:: test_histcomp()

   tests histogram compaction.


.. py:function:: test_histeq()

   tests histogram equalisation.


.. py:function:: test_img2rgb()

   tests img to RGB.


.. py:function:: test_norm()

   tests norm2.


.. py:function:: test_norm255()

   tests norm255.


.. py:function:: test_corr2d()

   tests corr2d.


.. py:function:: smalldata()

   Small test dataset.


.. py:function:: test_io_rasterio(smalldata, ext, drv)

   Tests IO for rasterio files.


.. py:function:: test_io_ascii(smalldata)

   Tests IO for ascii files.


.. py:function:: test_io_xyz(smalldata)

   Tests IO for xyz files.


.. py:function:: test_normalisation()

   Tests for normalisation.


.. py:function:: test_smooth()

   Tests for smoothing.


.. py:function:: test_agc()

   Tests for AGC.


