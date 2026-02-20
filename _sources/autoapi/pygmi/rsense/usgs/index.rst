pygmi.rsense.usgs
=================

.. py:module:: pygmi.rsense.usgs

.. autoapi-nested-parse::

   USGS SPECPR import.



Functions
---------

.. autoapisummary::

   pygmi.rsense.usgs.SPECPR
   pygmi.rsense.usgs.case1
   pygmi.rsense.usgs.unpack_icflag


Module Contents
---------------

.. py:function:: SPECPR(ifile)

   SPECPR import function.

   :param ifile: Input file.
   :type ifile: str

   :returns: **spec** -- Output spectra.
   :rtype: dict


.. py:function:: case1(dat)

   Case 1.

   :param dat: Binary record.
   :type dat: bytes

   :returns: **rec** -- Output record.
   :rtype: dict


.. py:function:: unpack_icflag(icflag)

   Unpack the bits from icflag.

   :param icflag: Binary icflag.
   :type icflag: bytes

   :returns: **b** -- Unpacked bits.
   :rtype: list


