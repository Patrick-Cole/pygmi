3D Magnetic Inversion
---------------------
This dialog sets up the parameters for 3D magnetic inversion which uses the SimPEG library (Cockett et al., 2015) (https://simpeg.xyz/). The routine uses magnetic and DTM data as input, and these datasets should be imported using :doc:`Import Raster Data<raster.dm.importrasterdata>`. Data are output as a number of classes, in line with the lithology principle which PyGMI uses. The output from inversion can therefore be directly input into the modelling module.

It is important to ensure the following:

* All datasets must be projected (not in geographic coordinates) and they must be in the same projection. To check the projection of each dataset, look at the :doc:`raster’s metadata<raster.cm.meta>`.
* Ensure that the band names are distinct and that they clearly identify the type of data. This can be changed by accessing the :doc:`raster’s metadata<raster.cm.meta>`.
* If you use magnetic data, make sure that it is not total magnetic intensity data. The :doc:`IGRF must be removed<mag.dm.igrf>`.

Select the **3D Magnetic Inversion** function from the :doc:`Potential Field Modelling<pfmod>` menu and connect the input rasters to this module. Double-click on the **3D Magnetic Inversion** module to bring up the **Inverse Modelling Parameters** interface:

1. **Dataset Information** – The imported rasters are connected to the correct datasets. 

  * **DTM Dataset** – This is a Digital Terrain Model or Digital Elevation Model (DEM) of your area of study.
  * **Magnetic Dataset** – This is the magnetic data to be modelled.

2. **General Properties** – These general properties are important for the inversion. If they are incorrect, or if there is extensive remanence in the area, the inversion results may be poor.

  * **Height of observation - Magnetic** – The magnetic sensor height in metres.
  * **Magnetic Field Intensity (nT)** – The intensity of the magnetic field at the time of the survey. This can be calculated using magnetic calculators provided by National Oceanic and Atmospheric Administration (NOAA) at https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm 
  * **Magnetic Inclination** – The inclination of the magnetic field at the time of the survey (in degrees). This can be gotten from NOAA.
  * **Magnetic Declination** – This is the declination of the magnetic field at the time of the survey (in degrees). This can be gotten from NOAA.

3. **Model Extent Properties** – The model extents control the area size, depth and resolution of the model. Typically, one of your datasets are used to fill in most of the fields automatically.

  * **Get Study Area from following Dataset** - Usually the DTM dataset is chosen, but it can be any of the datasets.
  * **Upper Top Left X Coordinate** – The north-west corner’s X coordinate of the model in metres.
  * **Upper Top Left Y Coordinate** – The north-west corner’s Y coordinate of your model in meters.
  * **Upper Top Left Z Coordinate** – The north-west corner Z coordinate of the model in metres above sea level.
  * **Total X Extent** – The distance from east to west of the model in metres.
  * **Total Y Extent** – The distance from north to south of the model in metres.
  * **Total Z Extent (Depth)** – The depth range of the model in metres.
  * **X and Y Cell Size** – The width of each model cube in the X and Y direction (in metres).
  * **Z Cell Size** – The height in metres of each model cube in Z direction.

4. The number of cells/voxels in the model with the specified extent parameters:

  * **Columns (X)** – This is calculated automatically from extents and cell sizes.
  * **Rows (Y)** – This is calculated automatically from extents and cell sizes.
  * **Layers (Z)** – This is calculated automatically from extents and cell sizes.

5. **Number of output classes** – The number of classes (lithologies) generated during the inversion process.

.. figure:: _images/pfmodinvmag.png

   Inverse Modelling Parameters interface for 3D inversion of magnetic data.

.. tip::

   * Start with a low resolution model to get a general model in place first. 20x20x20 (cells x rows x layers) should be fine,  since any bigger will get slow to model.
   * Once your low resolution model is complete, you can increase the model’s resolution to a better resolution. As a rule of thumb, models bigger than 100x100x100 may be slow, although this will depend on the hardware available


Once your low resolution model is complete, you can increase the model's resolution to a better resolution. Please note that bigger than 100x100x100 will cause calculations to become very slow (unless you have lots of memory).

References
^^^^^^^^^^

 Cockett, Rowan, Seogi Kang, Lindsey J. Heagy, Adam Pidlisecky, and Douglas W. Oldenburg. "SimPEG: An Open Source Framework for Simulation and Gradient Based Parameter Estimation in Geophysical Applications" Computers & Geosciences, September 2015. doi:10.1016/j.cageo.2015.09.015.

 Lindsey J. Heagy, Rowan Cockett, Seogi Kang, Gudni K. Rosenkjaer, Douglas W. Oldenburg. "A framework for simulation and inversion in electromagnetics" Computers & Geosciences, September 2017. doi:10.1016/j.cageo.2017.06.018
 
