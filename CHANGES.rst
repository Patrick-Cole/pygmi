=========
Changelog
=========

v3.2.5.12, 24 August 2022
-------------------------
*Fixed a bug where gravity profiles exported from the 3D model had incorrect values.
*Raster import can now import multiple files at once.
*Fixed a bug where a crash occurred when sun shading was deselected.

v3.2.5.9, 21 July 2022
----------------------
*Removed GeoTIFF ZSTD export due to library issues.
*Updates to help files.
*Fixed spelling mistakes.
*Removed redundant code.
*Bugfix, scrollbars now match on main interface.
*Bugfix, MNF forward transform bands now labelled correctly.
*MNF and PCA will now output correct number of bands on inverse transform.
*Band ratios and condition indices now includes Landsat 9 data.
*Worldview Pan tile import bug fixed.
*Worldview data import sped up.
*Bugfix for crash when no land surface temperature data in condition indices.
*Satellite import now sorts bands
*Sentinel-2 import states band resolution to avoid duplicate band names.
*Added a button to reset the light, so light direction is reset to new rotation.
*MT edit EDI no longer crashes due to an error in the resize event.
*Fixed a bug where Birch cluster analysis needed c-contiguous arrays.
*Fixed a bug where some data entry points disappeared.
*Crash in gradients fixed.
*Ternary images can now display full histograms.
*AGC test added
*Thgrad test added
*Vertical test added.
*Landsat composite now allows for the target date to be manually set.
*Small updates to code and comments.
*Inversion tests
*Bugfix for cursor width and height not being integers
*Observed data minimum is now correct in pfmod, if null values are present.
*In lstack, if masterid is True, and dxy is not null, dxy value will now be used.
*Magnetic inversion now allows for custom numbers of classes.
*Bugfix to mean mosaic
*Fixed bug mosaicing with different nodata and dtype values.
*Added Landsat composite
*Layer stacking now checks extents in addition to rows and columns
*Bugfix - spinbox setvalue now an integer.
*MNF forward transform now allows custom number of components
*Band sorting can be disabled for RGB images
*Explained variance ratio added to PCA band names.
*PCA added
*Fixed bug where ternary images were showing strange colors when data was clipped
*Raster Export will now sort the output bands, so that satellite data is in order.
*Bugfix, RTP now puts projection into output.
*Clustering is more memory efficient.
*Mini batch k-means has been added.
*New aster ratio added.
*Sentinel-2 zip files are now accepted in batch processing
*Cut raster will use first overlapping polygon in a multipolygon
*Some MultiPolygon support has been added.
*Fixed a bug which caused a crash if the text progressbar activates in a windows console.
*Added 7/5, 6/2 and 7/3 Landsat 8 ratios to band ratio tool
*Sentinel data is now recognized in the batch import by S2A and S2B prefixes.
*Merge data will now merge based on shifting the last data to the mean overlap value.

v3.2.5.6, 10 June 2022
----------------------
*Added landsat composite
*Layerstacking now checks extents in addition to rows and columns
*Bugfix - spinbox setValue now an integer.
*MNF forward transform now allows custom number of components
*Band sorting can be disabled for RGB images
*Explained variance ratio added to PCA band names.
*PCA added.
*Fixed bug where ternary images were showing strange colors when data was clipped
*Raster Export will now sort the output bands, so that satellite data is in order.
*Bugfix, RTP now puts projection into output.
*Clustering is more memory efficient.
*Mini batch k-means has been added.
*New aster ratio added.
*Sentinel-2 zip files are now accepted in batch processing
*Cut raster will use first overlapping polygon in a multipolygon
*Some MultiPolygon support has been added.
*Fixed a bug which caused a crash if the text progressbar activates in a windows console.
*Added 7/5, 6/2 and 7/3 Landsat 8 ratios to band ratio tool
*Sentinel data is now recognized in the batch import by S2A and S2B prefixes.
*Merge data will now merge based on shifting the last data to the mean overlap value.

v3.2.5.2, 22 April 2022
-----------------------
*Fixed a nodata value problem with magnetic inversion.
*Fixed the import of some MODIS data types
*Fixed a crash which occurs in seismology QA, when no data is found.
*Changed the x labels to vertical orientation, to fit more in.
*Bugfix to remove nodata from inversion.
*Fixed a bug where 'Other' datasets were not included in a model merge, causing a crash
*Fixed a bug where static shift was incorrectly applied when applying to all stations
*Fixed bad reference to wkt in pfmod
*Fixed a bug where null values could cause artifacts for second order tilt angle
*Exploration seismics viewer removed.
*All classes are now shown after loading a shapefile for supervised classification.
*Gridding will now apply a blanking distance for all methods.
*Fixed a bug causing a crash when using cut vector
*Fixed bug where rows and columns displayed were zero on startup, with some data.
*Hexbin plot will add data units, if defined.
*Visibility now accepts windows from 5 an up only.
*AGC has some null value issues fixed.
*Batch file import and related condition index and ratio calculations now support  normal raster imports
*Landsat import bugfix.
*Ratio bugfix
*Masks will be taken only from bands used in ratio
*Added support for WV-3 and WV-2 tile import, as well as in ratios.
*Updated Magnetic inversion.
*Added magnetic inversion via the SimPEG library

*Added support for Landsat 9.
*Fixed a bug causing supervised classification to crash
*Gravity processing will now calculate drift based on datetime, and not on the order found in CG-5 file.
*GMT import now added to raster file imports
*Reproject will use specified input projection, as opposed to what was defined in the input data.
*Give more information for gravity drift
*A local projection is now assigned to datasets without a projection, to avoid errors later.
*Fixed a bug displaying too much information on Line Map.
*Data export now adds statistics for ease of use in ArcGIS
*Fixed misinterpretation of unicode strings in line data.
*Added encoding to open statements
*Layer stacking (API) now allows a master dataset to which all other layers are clipped.
*Modelling will not correctly use the dem
*Added space delimited text files
*Fixed a bug causing vector reprojection to crash.
*Long projection information will now wrap correctly
*ENVI data import now correctly stores fwhm information
*Correlation graph now has a color bar and improved numbering
*2d correlation coefficient will now corrected take into account layers with differing mask.
*Text will use complementary colors.
*Layer stacking will us a common data type when data types of input bands are mixed.
*Raster image display defaults to no interpolation to avoid incorrect interpolation of null values.
*Modis import now import LULC layer.
*Modis import update
*Condition index now used a common mask between datasets, to avoid edge effects.
*A null value of 1e=20 is also enforced.
*Landsat level 2 science product images now convert DN to reflectance.
*Condition indices have been moved to a new module, and the calculation has been corrected.
*Equation editor now ensures that the output data type is the same as the input data type.
*Bugfix - fixed a bug where the ratio list was not displaying on startup.
*Updated error messages where no raster datasets are connected to a module or where there is no projection.
*Bug fixed where analyze spectra could tried to create spectra outside of the image.
*ASTER, Landsat and Sentinel-2 now store wavelength information properly
*Analyze spectra will sort spectra beforehand, and advise stacking when necessary
*Corrected wavelengths on hyperion import
*Added support for Hyperion L1T data

v3.2.4.5, 14 January 2022
-------------------------
*Added a warning if data has no projection
*Fixed a bug where the hyperbolic tilt angle had an invalid mask
*Fixed a bug where pressing 'OK' in display metadata caused a crash.

v3.2.4.4, 13 January 2022
-------------------------
*Minor bugfix for surfer 7 export
*Layer stacking will now give an error if input data has no projection.
*Fixed a bug in RTP where nan were generated in FFT preparation, resulting in no output.
*ZSTD compression option added for exporting GeoTiffs
*Surfer 7 export now replaces surfer 6 export.
*Fixed a bug where ASTER hdf was not correctly retrieving coordinates.
*Added longitude and latitude labels to plots, where necessary
*Added code to convert PolygonZ to polygon type when cutting out a raster
*Get raster now allows the nodata value to be specified
*Amended f2160 feature
*Fixed a bug where layer stacking with common mask changed the nodata value in a dataset to 1e+20
*Added VCU calculation for Sentinel-2
*Added invert selection to ratio dialog.

v3.2.4.2, 03 November 2021
--------------------------
*Bugfix to model saving for changes
*PyGMI now allows a user to continue from where they left off in 3D calculations
*Fixed some bugs with new ratio calculations
*Added VCI, EVI, TCI, VHI to ratios
*Added EVI calculation for ratios
*Fixed a bug with 3D model profile resizing.
*Added code for nodata being wrong type
*Added code when importing old models, to make grids more compatible
*Bugfix for RGB images
*Added option to filter out values less than 1 if final product is a ratio.
*Fixed bugs caused by rasterio to 3D modelling
*Fixed a bug with AGC grid boundary.
*Fixed a bug importing 3D models
*Changed clip percentage labels
*Changed raster data interpretation to raster data display.
*Updated sentinel 5p to rasterio
*Update to raster cutting using polygons
*Multiple profiles can be extracted from raster.
*Sentinel-2 bugfix
*Fixed some bugs with testing routines
*Added import of ASTER GED data
*Update ASTER import
*Fixed a bug with reprojection when the input data has no projection
*Updated modis, landsat and sentinel2 to rasterio
*Changed  Data.nullvalue to Data.nodata
*Fixed a bug with nodata values in new reprojection tool
*Updated cut_raster to rasterio
*Rasterio updates
*Fixed masking on some ratios
*Fixed new export to raster projection issue

v3.2.4.1, 20 September 2021
---------------------------
*Fixed a bug where layerstack was not loading properly.

v3.2.4.0, 17 September 2021
---------------------------
*Fixed bug with new IGRF data correction.
*Minor bugs
*Fixed a bug in merge tool when * is in band name
*Scatterplot tool displays classes using discrete colorbar.
*Membership maps now display between 0 and 1 only.
*Ginterp now includes membership data.
*Copy.copy has been converted to copy.deepcopy in some instances.
*Merge/mosaic now writes to disk to avoid memory slow downs.
*Fixed bug where masks could eliminate data for classification, if no data at that point in another layer.
*Fixed a bug on exiting mosaic
*Added a warning if no feature found in process fetures.
*Fixed a bug with importing xyz models, not having background.
*Comment corrections for headings
*Faster calculation of std dev in standard raster display.
*Large update to speed for ginterp with big images.
*Added projection information to Sentinel  5P import.
*Z value on  graphs fixed
*Fixed some problems with modis import
*Fixed geosoft xyz import
*Fixed a bug in minc
*Bugfix for display of z values when zoomed into raster modest image
*Memory for MNF reduced on 1 step
*Fixed bug with min and max merging
*Merge tool has new options
*Improved detection of minimum feature value.
*IGRF code modified to allow for API calls and calculate faster
*Batch export from process features now describes the mineral in the file name.
*Updated help and option description.
*Bugfix merge tool
*Export raster will now export wavelengths and fwhm to ENVI, if present
*Update to MNF help
*MNF memory problem bugfix
*Added MNF
*Minimum curvature bugfix
*Process features bug
*Fixed a bug when importing bil files with nan values
*Update processing features to allow for feature depth thresholds
*Process features can now have new features added in features.py
*Raster export will export wavelengths and reflectance scale factors, if available.
*Any routine using hull removal is significantly faster, (e.g. feature processing)
*Merge has been renames layer stack.
*New merge module added to merge adjacent datasets (from rasterio)
*Added text progressbars to all classes where necessary (for testing)
*Merging can now have a common mask for all bands.
*Scipy nearest neighbour replaces quickgrid, linear and cubic options added to gridding as well as minimum curvature
*Fixed issue with merge assigning wrong nullvalue
*Change to output IGRF bands only.
*Added resampling of dtm to match magnetic data.
*Fixed a bug when exporting geotiffs from ginterp.py
*Allows for datatype to be set, eg to uint8.
*Add sentinel 2 zip file functionality
*Added an import for Sentinel 2 data, with bands only
*Added AGC
*Fixed a labelling bug after merge
*Adds lower and upper clip functionality as well as display of clip values to histogram.
*Added total horizontal gradient

v3.2.3.0, 01 June 2021
----------------------
*Cluster - reports when no classes are found.
*Segmentation - now has DBSCAN for grouping segments
*Raster cut - will give a better error when the polygon is not overlapping the data
*Added *.tiff as opposed to *.tif to imports
*Fixed some display bugs with ginterp.
*Fixed a big with integer datasets when smoothing
*Fixed an issue where projection information was not save in supervised classification results
*Started to use modest_image for some of the raster image displays
*Fixed a bug on analyze spectra causing scale problems
*Calculate metrics now allows for saving of metrics to excel
*Fixed bug importing some class shapefiles
*Fixed a bug causing ratios to break when using sentinel 2 data.
*Fixed an issue causing some dataset units to be imported incorrectly
*Import wavelengths for ASTER and Landsat data
*Merge tool will now have progress bar when use in export.
*S2 data will store wavelengths for use in ArcGIS
*Remove commas from remote sensing band names to ensure ESRI compatibility
*Vector reprojection now warns if inf values are output.
*Change hexbin colormap to inferno
*Added colormaps to show raster and show surface
*Set histogram number of bins to 'sqrt' as opposed to a fixed 50 bins.
*Simplified ENVI import
*Added log option for y-axis of histogram.
*Added export for SEISAN to xlsx and shp
*Bugfix - fixed a bug displaying some point vector data.
*Seismology - to beachball plot import and b value calculation
*Updates to spelling and help files
*Updates QC graphs to take into account weight 9 for record type 4
*Fixed bugs importing some thermal aster data and L1T data
*Hyperspectral analysis tool.
*Project load bugfix
*Added spectral interpretation tool.
*Changed mosaic in equation editor to overlay grids.
*Progress bar bugfix for RTP
*Added 2.5 stddev to quick raster display.
*Outlines of all polygons now remain on the map.
*Confusion matrix display now has appropriate labels.
*Shapefiles save and load class names
*Fixed a bug which caused supervised classification to crash if a class was empty
*Added equal area rose diagrams
*Code to make sure only ENVI, ERS and EHdr  can use BIL routine
*Added faster import for BIL binary files. (i.e. ENVI and ER Mapper)
*Added drift rate curve to gravity QC
*MT tipper graphs added

v3.2.2.4, 20 November 2020
--------------------------
*Fixed library installer problem in windows installer.
*Added features to numpy_to_pygmi for convenience.
*Added shapefile functionality to vector reprojection.
*Update get raster to read sections of files.

v3.2.2.3, 10 November 2020
--------------------------
*Geophysical interpretation resizes window smoother.
*Merge/resize tool will fill null values after resize.
*Improved geophysics interpretation tool.
*Fixed a bug where in some cases residual plots did not display data
*Fixed a bug preventing PyGMI from running in Linux.
*Updated install instructions for Anaconda
*Export gdal routines now uses the progress bar
*Fixed a bug preventing sentinel-2 data imported from ENVI files from having ratios calculated.
*Geotiff output now has correct band names, especially for ternary images.
*Fixed some bugs with showprocesslog calls
*Updated ratios to accept data labelled Band 1, Band 2 etc using raster import.
*For supervised classification, zoom tool and panning will no longer create polygon points
*Fixed a bug in supervised classification where first point of new poly was on top left corner of grid.
*Fixed a bug where only the edge of a polygon was used in class definitions
*Reverted graph tool to plotting maps with matplotlib library instead of modestimage, because of bugs in zooming.
*Custom profile will now show beginning and end of user coords as a +
*Bugfix causing profiles with directions greater than 90 degrees to not work.
*Added automatic detection of some x and y columns.
*Made a change to gravity import allowing for e,w,s,n, in gps coords
*Fixed some matplotlib issues due to API changes.
*Changed method to call cm in Matplotlib
*Changed library calls for matplotlib to be more compatible with pylint
*Stopped using picker due to future matplotlib depreciation.
*Fixed resize for picked features on line profile
*Stopped redirecting stdout globally in favor of a more elegant approach
*Fixed a scaling bug when viewing SEG-Y data.
*Sentinel-2 import now divides DN by 10000
*Comment update
*Update to modis v6 import
*TDEM additions
*MODIS16 import
*Change detection viewer now saves gif animations.
*Changed FFT preparation padding to use a mach faster routine taking into account null values.
*Added more bins for linear stretch in interpretation module.
*Changed band labels for sentinel 2 import.
*Added text toolbar class.
*Added alpha channel support to RGB import.

v3.2.1.1, 05 August 2020
----------------------
*Added 99% linear stretch to geophysical interp.
*Created a magnetic menu for modules which are magnetic only.
*Updated more graphs to have thousands separator.
*Updated modelling to allow for data grids with only one column.
*Fixed extents issue with gridding data.
*Fixes an issue if there is missing geometry in a shapefile.
*Fixed some issues with axis labels on graphs
*Fixed a bug causing incorrect stats for supervised classification if null values were in the dataset.
*Added comma as thousands separator for raster and vector graphs
*Added upward and downward continuation.
*Added general orders to vertical derivative functions
*Fixed a bug exporting null values for 32 bit float datasets.
*Fixed a recent bug preventing saving of data from geophysical interpretation tool
*Added units for some remote sensing imports (sentinel-2 and aster)
*Added modest_image support for display raster option
*Fixed a bug causing a crash in interpretation tool when receiving results from cluster analysis
*Added crisp and fuzzy cluster analysis settings
*Added image segmentation settings
*Added export for shapefiles
*Added saved proj settings for cluster analysis
*Added color to point shapefile display
*Fixed a bug displaying incorrect utm values in EDI metadata
*Fixed the message displayed from DBSCAN cluster analysis
*Fixed a bug causing cut vector files to not be plotted.
*Fixed bug in band select
*Fixed a bug exporting saga data, when dataset had multiple bands
*Reorganised code.
*Updates to project save.
*Added project save and load.
*Will save workflow but only certain modules have settings saved at this stage.
*Delete key now deletes arrows or items
*Tests updated to reflect recent fixes.
*File imports will display filename in information
*Band ratio labels replace divide sign with div, for ESRI compatibility
*Bugfixes in ratio import with a single file.
*Data class will store the filename of the dataset imported.
*Changed description on surfer grids.
*Fixed a bug which occurs for some padding of RTP datasets
*Fixed a bug in RTP calculation
*Alpha version of ratios
*Fixed a bug where PyGMI would crash when double clicking on an arrow.
*Added a mosaic function to the equation editor, for a simple mosaic of two datasets.
*Moved importing of remote sensing data to remote sensing menu.
*Started work on a ratio function (remote sensing), with batch capabilities
*Undo custom window size
*Added import for sentinel 5P data
*Fixed bug which reset last lithology whenever background layer has changes applied.
*Changes will no longer be applied automatically
*Bugfix, profile add
*Custom profile now correctly deletes, and reports if it is outside the model area
*Fixed a bug with drawing lines.
*Added save complete when saving model in modelling interface.
*Fixed the odd sizing of the cursor, and related drawing of lithologies.
*Improved listboxes for modelling and 3D display
*Fixed an issue where a custom profile image was not being saved with a 3D model
*Fixed a bug when reimporting a model with rgb image inside it.
*Updated readme files

v3.1.0, 24 March 2020
---------------------
*Updates to gravity routines to report duplicated stations.
*Fix bug where reprojected coordinates were not properly written to exported file for vector datasets.
*Fixed incorrect label on RTP.
*Fixed entry into scatter plot tool, and tool now correctly only displays selected parts of the histogram.
*Disabled windows context help.
*Maps will now have plain coordinates rather than scientific notation.
*Fixed a bug with equation editor causing iall variable to not work properly.
*Added occam1d warning for no executable.
*Fixed a bug causing clipping in saved sunshaded images.
*Fixed interpolation on model (caused by API change) Fixed a bug in drift correction for gravity.
*Added reprojection of line data.
*Unified Line and point data - they are now the same thing.
*Adopted pandas and geopandas as point, line and shapefile format
*Misc updates

v3.0.2, 5 March 2020
--------------------
* SimPEG 1D TDEM inversion (pre release alpha)
* Improved line map scaling.
* Separated MT and EM routines.
* Made exit returns from routines more consistant.
* Added ability to tie in local gravity base station to a known base station.
* Fixed column labelling of gps data in gravity module.
* Fixed output of ternary colorbar.
* fixed bug when using 2% clip on sunshading
* Fixed bug due to gdal axis api change in 3.0
* Added 2% data clip to interpretation.
* Changed way PyGMI uses processlog on the main window. It now redirected from stdout.
* Fixed a scaling bug with derivative calculations. The calculations now correctly take into account cell spacing.
* Update pygmi.grav.iodefs.importpointdata.html
* Now able to grid line data.
* Corrected some errors in gravity processing.
* Fixed some problems with gravity processing and visualisation.
* Added seismology description corrections.
* Added new seismology tools.
* Fix colorbar export for ginterp.
* Added docstrings to many routines.
* Worked on model merge bug.

v3.0.1, 6 December 2019
-----------------------
* Added custom profiles to the modelling interface
* Added test routines for PyGMI modules
* Added change detection viewer
* Added BIRRP interface
* Added supervised classification
* Added segmentation
* Fixed a bug causing the measured data in the 3D modeller to shift in the wrong place
* Added MT processing and inversion
* Added import and display of SEG-Y data
* Added basic gravity processing
* Change line direction to be 0 degrees in N direction. Added parallel processing to magnetic calculations
* Added parallel processing to core calculations for forward modelling

v3.0.0, 22 August 2019
----------------------
* New 3D modelling interface
* QC for seismology events
* Added tilt depth to 3D model functionality
* Gridding now has an option for a null value.
* Added geosoft line data import and display.
* Added older crisp and fuzzy cluster routines
* Numerous bug fixes and improvements.

v2.4.3, 7 March 2019
----------------------
* Fixed bug in IGRF for linux systems
* Fixed dependency on winsound for linux systems
* Fixed bug on metadata for linux systems

v2.4.1.2, 1 March 2018
----------------------
* Added updated IGRF coefficients
* Bug fixes in saving of 3d model, when it is used by another process, and in reading csv vector data.
* IGRF bugfix: fixed a bug relating to newer numpy
* Maintenance: Cleaned code in the equation editor.
* Add more control to Anaglyphs
* Fixed the orientation of anaglyph contours
* Fixed a bug in beachball code.
* Introduced anaglyphs for raster data.
* Updated color bar list to new standards
* Minor changes and a bugfix between ginterp.py and the latest matplotlib.
* Added directional lighting to 3D display

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
