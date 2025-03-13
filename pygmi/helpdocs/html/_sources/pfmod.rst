Potential Field Modelling
=========================

Potential field modelling within PyGMI is used to model magnetic and gravity data in three dimensions. The model consists of voxels and the response of each voxel is calculated using the code developed by Blakely (1996). The full 3D model is displayed using a context menu for the corresponding module.

Description of Modules
----------------------
The **Potential Field Modelling** module can be used to create synthetic models without observed data and to create geological models based on observed gravity and/or magnetic data. Furthermore, a DEM can be used to define the surface of the model. The specific options on the **Potential Field Modelling** menu are:

* :doc:`Import 3D Model<pfmod.dm.import3dmodel>` – Imports a model created in PyGMI. The models use the NumPy **NPZ** format. Each **NPZ** file contains the voxel model and the raster datasets associated with that model. These rasters are the DEM and/or magnetic and/or gravity and/or gravity regional data, and/or RGB images (e.g. a geological map) that have been loaded into the model previously.
* :doc:`Model Creation and Editing<pfmod.dm.modelcreate>` – This module is the potential field modelling section of PyGMI. It is used to create a model or edit an existing model imported using the **Import 3D Model** option. The output from here is a voxel model which can be displayed in 3D using the right-click context menu on a green module. The module comprises a series of Tabs, which are described below. The actual calculation of the magnetic and gravity field data is based on the work by Blakely (1996).
* :doc:`Merge two 3D Models<pfmod.dm.merge3dmodels>` – This allows for two 3D models to be merged into one single model.
* :doc:`3D Magnetic Inversion<pfmod.dm.inv3d>` – A user interface that set up the parameters for the inversion routine of the SimPEG library (Cockett et al., 2015).

.. toctree::
    :titlesonly:

    pfmod.dm.modelcreate
    pfmod.dm.import3dmodel
    pfmod.dm.merge3dmodels
    pfmod.dm.inv3d

Context Menu
------------
The context menu for **Potential Field Modelling** can be accessed by right-clicking on the **Import 3d Model** and **Potential Field Modelling modules.** Since the imported and processed data are raster datasets, many functions on the context menu are the same as the context menu for raster data.


.. toctree::
    :titlesonly:

    pfmod.cm.show3dmodel
    pfmod.cm.stat
    pfmod.cm.export3dmodel

.. figure:: _images/pfmodcm.png

   Context menu for the potential field modelling modules.
   
References
^^^^^^^^^^
 Blakely, R.J. 1996. Potential Theory in Gravity and Magnetic Applications. Cambridge University Press, Cambridge, UK, 441 pp., 441 pp.
 
 Cockett, R., Kang, S., Heagy, L.J., Pidlisecky, A. and Oldenburg, D.W. 2015. SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers and Geosciences, 85, 142–154, https://doi.org/10.1016/j.cageo.2015.09.015.