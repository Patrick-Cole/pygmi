Sen2Cor: Sentinel-2 Atmospheric Correction
------------------------------------------
Level 2A atmospherically corrected data can be downloaded from https://dataspace.copernicus.eu/. This dataset had a topographic correction applied, generally using the 30 m SRTM DEM (this can be checked by searching for DEM in the MTD_MSIL2A.xml file in the L2A SAFE folder). However, in areas where shadows are problematic the user may have access to a higher resolution DEM that will yield better results. The **Sen2Cor: Sentinel-2 Atmospheric Correction** and :doc:`Topographic Correction<rsense.dm.topo>` functions gives the user the ability to do that. **Sen2Cor: Sentinel-2 Atmospheric Correction** is run first on Level 1C data without applying a :doc:`Topographic Correction<rsense.dm.topo>`, and the resulting Level 2A file is then used in conjunction with a DEM file to produce a topographically corrected L2A file.

PyGMI uses the Sen2Cor software package (Pignatale, 2024) provided by the European Space Agency (ESA) to do the atmospheric correction. Therefore, this software must first be downloaded and installed. It is available at the `Sen2Cor website <https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-12/>`_. A document describing the installation steps can also be downloaded from this website (Sen2cor_Dev_Team and OMPC-Team, 2025). Extract the downloaded ZIP file in the desired location (e.g. the root of the C-drive). Go into the folder and run L2A_Process.bat once to set up a folder with a configuration file. This only has to be done once, when you use Sen2Cor for the first time. PyGMI has a built-in version of the configuration file that excludes the topographic correction.

When selecting **Sen2Cor: Sentinel-2 Atmospheric Correction** the **Sen2Cor** module appears in the main PyGMI interface. Double-click on the module to bring up the **Sen2Cor: Sentinel-2 Atmospheric Correction** dialog box. The user needs to provide links to two folders:

1. **Sen2Cor Directory** – The folder where the Sen2Cor software was extracted to (within this folder you should see some subfolders and a **BAT** and **TXT** file). 
2. **Sentinel-2 L1C.SAFE** Directory – The L1C Sentinel data which is downloaded in ZIP format must be extracted so that a **SAFE** folder is visible. Select this folder. 

.. figure:: _images/rsenses2c1.png

   Sen2Cor: Sentinel-2 Atmospheric Correction dialog box.

The atmospherically corrected data are stored in a SAFE folder which contains MSIL2A in the name.

.. figure:: _images/rsenses2c2.png

   Sentinel-2 data before and after atmospheric correction.

References
^^^^^^^^^^
 Pignatale, F.C. 2024. Sen2Cor 2.12.03 Configuration and User Manual. Copernicus Space Component Sentinel Optical Imaging Mission Performance Cluster Service, OMPC.TPZ.SUM.002, 52 pp.
 
 Sen2cor_Dev_Team and OMPC-Team 2025. Sen2Cor v 2.12.03: Installation, Configuration and Processing V. Copernicus Space Component Sentinel Optical Imaging Mission Performance Cluster Service, v. 2.1.0, 23 pp.