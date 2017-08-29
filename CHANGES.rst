=========
Changelog
=========

v2.4.1, 29 August 2017
----------------------
* Added axis and orthographic projection option to 3D display view.
* Added perspective change to beachball plots
* Corrected clustering label. 
* Bugfix on lithmodel. 
* Correction to profile coordinates to place profile in centre of cell, as opposed to beginning of it. 
* Added IGRF report backs.
* Fixed bugs with tensor calculations
* fix for error exporting text columns
* fixed a bug with calculating changes only on model
* made changed to the way matplotlib calls are made.
* speed improvements to gravity and magnetic calcs
* Fuzzy and Crisp clustering replaced by scikit_learn cluster analysis.
* Import of csv point data enhanced and new cut tool added for point data.
* Dependancies updated. Minor bugs fixed

v2.3.0, 11 May 2017
-------------------
* Removed the auto update check due to problems it was giving on many pc's
* Fixed bugs with smoothing data and merging data
* Got rid of excessive old code.
* Fixed a bug with null values from equation editor.
* Fixed null value bug exporting rgb tiffs. (8-bit)
* Changed the profile views so that calculated data is drawn over observed data.
* Fixed bugs relating to selection of raster bands going to modelling, and saving of those raster bands.
* Fixed bug on data import for Qt5
* Added Lith Merge
* Migrated to Qt5

v2.2.15, 6 March 2017
---------------------
* Fixed incorrect calculation of remanence.
* 3d import fix.
* Fix for 3d import from text files.
* Anaglyph tests.
* Minor maintenence.
* Seismology Focmec format update.
* Added feature to calculate only changes to model.
* Fixed leapfrog import bug when header is in csv file.
* Readme update.

v2.2.14, 15 November 2016
-------------------------
* Added import of Leapfrog csv blockfiles
* Fixed bug exporting ER Mapper files using SA custion projection
* Fixed a bug in 3D model software
* Bug fix for merge module
* Added a tool to merge two models
* reactivated a progress bar display
* alpha speed update
* update modelling calculation using multi processing.
* numerous bug fixes

v2.2.13, 11 October 2016
------------------------
* Fixed some setup bugs

v2.2.12, 10 October 2016
------------------------
* Fixed an bug saving and opening files, introduced in previous commit.
* Bug fixes and prep for PyQt5
* Fixed a bug exporting 3D image.
* New version also checks for an update on pypi
* Fixed a bug with no mask exported from modeller.

v2.2.11, 12 July 2016
---------------------
* Added aster GED (binary) and fixed a bug on hdr aster GED import.
* Update to misc function
* Fixed a bug when resizing a model
* Bugfix in kmz export and in quarry event removal algorithm
* Fixes to shapefile 3D export
* Update to beachball, vertical gradient and export 3d model to shapefile
* Update to picture overlay on 3D modelling
* Fault plane solutions
* Update readme taking into account anaconda bug

v2.2.10, 10 March 2016
----------------------
* Added some Raster imports
* Fixed a bug preventing the saving of an image in the 3D viewer. It was caused by a changing library API.
* Fixed the reduction to the pole module.
* Removed pdb in crisp clust
* Fixed bug affecting export of integer datasets
* Arcinfo grid
* Fixed a new bug with equation editor
* Added save message for 3D model save.
* Equation editor fix: Added null values, Fixed masking of null values
* Added alpha version Vertical Gradients - but there is still lots of work to be done. It does not play well with null values.
* Bugfix with export csv
* New exports all profiles from a 3-d model

v2.2.9, 2 October 2015
----------------------
* Fixed a bug crashing regional test
* Fixed a bug where null values were not set correctly in the normalisation routine.
* Fixed a problem with an offset on calculated magnetic data, introduced in v2.2.8
* Fixed a bug when using the Seismology Delete Records option.

v2.2.8, 1 October 2015
----------------------
* Removed libraries not needed etc
* Fixed a problem with adding a gravity regional dataset to calculated gravity.
* Updates to the speed of the calculation for magnetic data.

v2.2.7, 18 June 2015
--------------------
* Update to setup for hosting on pypi
* Added the possibility for ENVI files to have .dat extension
* Allowed uint files to have a no data value of 0 where none is defined
* Fixed bug with surfer export
* Fixed bug with regional test
* Equation editor bug fix
* Fixed a bug where profiles were not saving to images correctly

v2.2.6, 10 April 2015
---------------------
* Progress Bar on Main Interface. New progress bars include time left.
* Reprojecting bug fix for datasets with negative values.
* Fixes to tilt depth and new progress bars
* Added Column to tilt depth to specify contour id. Also removed redundant
  progress bars. Sped up smoothing with median.
* Added tilt depth algorithm.
* Bug Fix with tilt angle.
* Added RTP.
* Cluster and Fuzzy analysis had a bug when connecting external data
* Changed where rows and cols displays on modelling software, for people
  with lower resolution screens. Made small improvement to drawing speed on
  profile view.
* Added references to the help.
* Modelling now has variable size cursor.
* Change to modelling cursor.
* Updates the behaviour of the slider on the profile view of the 3D
  modelling module.
* Grids on kmz export were upside down
* Mag and Grav calculation buttons simplified.
* Gravity regional addition (scalar add) in modelling program now modify
  calculated data, instead of observed data - so that original data is
  honoured.
* Update to kmz export. The export now can allow smooth models. Update
  also allows new projection format for igrf, data reprojection and kmz
  files.
* Projections improved and expanded.
* Geotiff now save tfw world file. Contour Geotiffs are now 3 times
  bigger, to improve resolution. Contour lines now have double thickness.
* Sunshaded Geotiff is now the same as the on screen version.
* Add save model to 3D modelling module.
* Primary Help completed.
* First version with a helpdoc button on main interface.
* Fixed a bug on the Geosoft import.

v2.2.5, 12 February 2015
------------------------
* Fixed a display bug in modeller where data was not visible.
* Added Geosoft grid import
* Added Geopak grid import
* Fixed a python 2.7 print function bug

v2.2.4, 12 December 2014
------------------------
* Increased size of font for ternary colorbar.
* Corrected issues with modelling information display, especially w.r.t. remanence.

v2.2.3, 10 December 2014
------------------------
* Added ternary colorbar
* Fixed ability to save 3D images on new smoothing
* Bug fix - masking problem with ER Mapper import
* Added extra 3D display functionality
* Added smooth model
* Added marching cubes
* Forced full field recalc to avoid bug
* Fixed layer import bug
* Bug Fix in model import
* Fixed bug when resizing some models
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
