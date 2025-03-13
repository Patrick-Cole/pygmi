pygmi.mag.igrf
==============

.. py:module:: pygmi.mag.igrf

.. autoapi-nested-parse::

   IGRF calculations.

   This code is based on the GEOMAG software, with information given below. It was
   translated into Python from the GEOMAG code.

   | This program, originally written in FORTRAN, was developed using subroutines
   | written by   : A. Zunde
   |                USGS, MS 964, Box 25046 Federal Center, Denver, Co.  80225
   |                and
   |                S.R.C. Malin & D.R. Barraclough
   |                Institute of Geological Sciences, United Kingdom.

   | Translated
   | into C by    : Craig H. Shaffer
   |                29 July, 1988

   | Rewritten by : David Owens
   |                For Susan McLean

   | Maintained by: Stefan Maus
   | Contact      : stefan.maus@noaa.gov
   |                National Geophysical Data Center
   |                World Data Center-A for Solid Earth Geophysics
   |                NOAA, E/GC1, 325 Broadway,
   |                Boulder, CO  80303



Classes
-------

.. autoapisummary::

   pygmi.mag.igrf.IGRF


Functions
---------

.. autoapisummary::

   pygmi.mag.igrf.calc_igrf
   pygmi.mag.igrf.getshc
   pygmi.mag.igrf.extrapsh
   pygmi.mag.igrf.interpsh
   pygmi.mag.igrf.shval3
   pygmi.mag.igrf.dihf


Module Contents
---------------

.. py:class:: IGRF(parent=None)

   Bases: :py:obj:`pygmi.misc.BasicModule`


   IGRF field calculation.

   This produces two datasets. The first is an IGRF dataset for the area of
   interest, defined by some input magnetic dataset. The second is the IGRF
   corrected form of that input magnetic dataset.

   To do this, the input dataset must be reprojected from its local projection
   to degrees, where the IGRF correction will take place. This is done within
   this class.

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



.. py:function:: calc_igrf(data, sdate, *, alt=100, wkt=None, igrfonly=True, piter=iter, showlog=print)

   Calculate IGRF.

   :param data: Input magnetic data.
   :type data: pygmi.raster.datatypes.Data
   :param sdate: Survey date.
   :type sdate: Date
   :param alt: Sensor clearance. The default is 100.
   :type alt: float, optional
   :param wkt: WKT projection. The default is None.
   :type wkt: str, optional
   :param igrfonly: Output IGRF only. The default is True.
   :type igrfonly: bool, optional
   :param piter: Progress bar iterator. The default is iter.
   :type piter: function, optional
   :param showlog: Display information. The default is print.
   :type showlog: function, optional

   :returns: * **outdata** (*list*) -- List of output PyGMI Data (pygmi.raster.datatypes.Data).
             * **fmean** (*float*) -- Total intensity mean.
             * **imean** (*float*) -- Inclination mean.
             * **dmean** (*float*) -- Declination mean.


.. py:function:: getshc(file, iflag, strec, nmax_of_gh, igh, gh)

   Read spherical harmonic coefficients from the specified model.

   Reads spherical harmonic coefficients from the specified model into an
   array (Schmidt quasi-normal internal spherical harmonic coefficients).

   | FORTRAN: Bill Flanagan, NOAA CORPS, DESDIS, NGDC, 325 Broadway,
   | Boulder CO.  80301
   | C: C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

   :param file: reference to a file object
   :type file: file
   :param iflag: Flag for SV equal to 1 or not equal to 1 for designated read
                 statements
   :param strec: Starting record number to read from model
   :type strec: int
   :param nmax_of_gh: Maximum degree and order of model
   :type nmax_of_gh: int
   :param igh: Index for Schmidt quasi-normal internal spherical harmonic
               coefficients.
   :type igh: int
   :param gh: Schmidt quasi-normal internal spherical harmonic coefficients.
   :type gh: numpy array

   :returns: **gh** -- Schmidt quasi-normal internal spherical harmonic coefficients.
   :rtype: numpy array


.. py:function:: extrapsh(date, dte1, nmax1, nmax2, igh, gh)

   Extrapolate a spherical harmonic model.

   Extrapolates linearly a spherical harmonic model with a rate-of-change
   model. Updates Schmidt quasi-normal internal spherical
   harmonic coefficients.

   | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
   | CO. 80225
   | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

   :param date: date of resulting model (in decimal year)
   :type date: float
   :param dte1: date of base model
   :type dte1: float
   :param nmax1: maximum degree and order of base model
   :type nmax1: int
   :param nmax2: maximum degree and order of rate-of-change model
   :type nmax2: int
   :param igh: Index of gh.
   :type igh: int
   :param gh: Schmidt quasi-normal internal spherical harmonic coefficients of
              base model and rate-of-change model
   :type gh: numpy array

   :returns: * **nmax** (*int*) -- maximum degree and order of resulting model
             * **gh** (*numpy array*) -- Schmidt quasi-normal internal spherical harmonic coefficients of
               base model and rate-of-change model


.. py:function:: interpsh(date, dte1, nmax1, dte2, nmax2, igh, gh)

   Temporal Interpolation between two spherical harmonic models.

   Interpolates linearly, in time, between two spherical harmonic
   models.

   Updates Schmidt quasi-normal internal spherical harmonic
   coefficients.

   | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
   | CO. 80225
   | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

   :param date: date of resulting model (in decimal year)
   :type date: float
   :param dte1: date of earlier model
   :type dte1: float
   :param nmax1: maximum degree and order of earlier model
   :type nmax1: int
   :param dte2: date of later model
   :type dte2: float
   :param nmax2: maximum degree and order of later model
   :type nmax2: int
   :param gh: Schmidt quasi-normal internal spherical harmonic coefficients of
              earlier model and internal model
   :type gh: numpy array

   :returns: * **nmax** (*int*) -- maximum degree and order of resulting model
             * **gh** (*numpy array*) -- Schmidt quasi-normal internal spherical harmonic coefficients of
               earlier model and internal model


.. py:function:: shval3(igdgc, flat, flon, elev, nmax, igh, gh)

   Calculate field components from spherical harmonic (sh) models.

   This routine updates self.x, self.y, self.z (Northward, Eastward and
   vertically downward components respectively NED)

   Based on subroutine 'igrf' by D. R. Barraclough and S. R. C. Malin,
   report no. 71/1, institute of geological sciences, U.K.

   | FORTRAN : Norman W. Peddie, USGS, MS 964, box 25046 Federal Center,
   | Denver, CO. 80225
   | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

   :param igdgc: indicates coordinate system used set equal to 1 if geodetic, 2 if
                 geocentric
   :type igdgc: int
   :param flat: north latitude, in degrees
   :type flat: float
   :param flon: east longitude, in degrees
   :type flon: float
   :param elev: WGS84 altitude above ellipsoid (igdgc=1), or radial distance from
                earth's center (igdgc=2)
   :type elev: float
   :param nmax: maximum degree and order of coefficients
   :type nmax: int
   :param gh: Schmidt quasi-normal internal spherical harmonic coefficients of
              earlier model and internal model
   :type gh: numpy array

   :returns: * **x** (*float*) -- Northward component (NED)
             * **y** (*float*) -- Eastward component (NED)
             * **z** (*float*) -- Vertically downward component (NED)


.. py:function:: dihf(x, y, z)

   Compute the geomagnetic d, i, h, and f from x, y, and z.

   This updates self.d, self.i, self.h and self.f (declination,
   inclination, horizontal intensity and total intensity).

   | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
   | CO. 80225
   | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

   :param x: northward component
   :type x: float
   :param y: eastward component
   :type y: float
   :param z: vertically-downward component
   :type z: float

   :returns: * **h** (*float*) -- Horizontal Intensity
             * **f** (*float*) -- Total Intensity
             * **i** (*float*) -- Inclination
             * **d** (*float*) -- Declination


