=========
Changelog
=========

v2.2.3, 10 November 2014
------------------------
* Fix for bad values in reprojections.
* New display of point data.
* Equation editor improved to use numexpr.
* Fixed a bug regarding duplicate data names in interpretation module.
* Added a few reports in 3D modelling module.
* Improved the multi-band select by making it a context menu.
* Update help reference.
* Update to python 3.4.2 - includes a dependency on numba. No longer use cython
* Added some seismology routines.
* Fixed writing of null value to file when exporting ENVI format.
* Query for which datasets to connect added.
* Added new gridding technique. and fixed bugs related to vector imports.
* Add a custom data range to the profile view on the modelling module.

v2.2.2, 22 September 2014
-------------------------
* Fixed problems with the potential field calculations
* Fixed bugs with the equation editor
* Fixed a bug with basic statistics and masked values
* Fixed a bug fix in the summing of calculations for modelling
* Fixed a problem when exporting color bars
* Fixed sunshade bug
* IGRF bug fixes
* Fixed problem with high colors in geotiff export
* Fixed a bug saving geotiffs
* Fixed bug on apply regional in modelling
* ASCII Import fixed
* Minor bug fixes and formatting
* Fixed imports into modules to allow for relative imports
* Fixed a bug in setup.py
* Fixed a bad reference to pygmi.point in setup.py. It should now be pygmi.vector
* Improvements to calculation speed
* Regional model merge
* Allows merging of a regional model with primary model

v2.2.1, 22 August 2014
----------------------
* Multiprocessing support added to potential field calculation.
* Fixed bug with ascii model export
* ASCII model export bug fixed
* Export is renamed from xyz to csv
* Fixed IGRF bugs
* Organisation of graph routines
* Rose Diagrams and shape files added
* Fixed progress bar on forward modelling

v2.2, 12 August 2014
--------------------
* Implemented multiprocessing on forward modelling
* Added custom profile display
* Testing routine
* Added a testing routine for forward modelling.
* I/O bug fixes
* Import and export bug fixes, especially with null values
* Fixes to name mangling
* Fixes to Smoothing and data cutting
* Converted code to functions for easier library access.
* Modified smoothing algorithm and added better comments
* Python 2.7 Compatibility changes
* Fixed import problem with pickle
* A module was moved and this prevented some data being loaded. This was fixed
* Fixed a bug which caused figures to pop up independent of the GUI
* Increased the decimal precision of the density input in the modelling module
* Changes to make PyGMI functions accessible
* Exposed some raster functions

v2.1, 17 July 2014
------------------
* Initial Release
