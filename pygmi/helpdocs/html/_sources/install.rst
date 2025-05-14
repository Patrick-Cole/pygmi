Installation Instructions
=========================
The PyGMI software is available for download from https://github.com/Patrick-Cole/pygmi/releases. A pre-built Windows installer or source code can be downloaded. Linux users will need to download the source code from https://github.com/Patrick-Cole/pygmi or install via https://pypi.org/project/pygmi/

Hardware requirements
---------------------
The hardware required for running PyGMI depends on the application. 32 GB is adequate for most applications but more memory (at least 64 GB) is recommended when processing large data sets such as satellite imagery.

Software requirements
---------------------
PyGMI will run on both Windows and Linux. The software has been tested on Windows 10 and 11 and Ubuntu 24.10. It should be noted that the main development is done in Python 3 on Windows. For the latest version information regarding PyGMI, Python and the libraries used in PyGMI, please visit the the GitHub repository (https://github.com/Patrick-Cole/pygmi).

Installation
------------
The simplest installation of PyGMI is on Windows, using a pre-built 64-bit installer (`PyGMIx64_<version number>.exe <https://github.com/Patrick-Cole/pygmi/releases>`_).

If you prefer building from source, you can use PyPi or Conda.

Once built using PyPi, running pygmi can be done at the command prompt as follows:

   pygmi

If you are in python, you can run PyGMI by using the following commands:

   from pygmi.main import main

   main()

If you prefer not to install pygmi as a library, download the source code and execute the following command to run it manually:

   python quickstart.py

PyPi - Windows
^^^^^^^^^^^^^^
Windows users can use the `WinPython <https://winpython.github.io/>`_ distribution as an alternative to Anaconda. It comes with most libraries preinstalled, so using pip should be sufficient.

Install with the following command.

   pip install pygmi

Should you wish to manually install binaries, related binaries can be obtained at the `website <https://github.com/cgohlke/geospatial-wheels/>`_ by Christoph Gohlke.

If you wish to update GDAL, you will need to download and install:

* fiona
* GDAL
* pyproj
* rasterio
* Rtree
* shapely

All these binaries should be downloaded since they have internal co-dependencies.


PyPi - Linux
^^^^^^^^^^^^
Linux normally comes with python installed, but the additional libraries will still need to be installed.

The process is as follows:

   sudo apt-get install pipx
   
   pipx ensurepath

   pipx install pygmi

Once installed, running pygmi can be done at the command prompt as follows:

   pygmi

If you get the following error: *qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.*, then you can try the following command, since this is Linux issue:

   sudo apt-get install libxcb-xinerama0

Anaconda
^^^^^^^^
Anaconda users are advised not to use pip since it can break PyQt5. However, one package is installed only by pip, so a Conda environment should be created.

The process to install is as follows:

   conda create -n pygmi python=3.12

   conda activate pygmi

   conda config --env --add channels conda-forge

   conda install pyqt

   conda install fiona

   conda install matplotlib

   conda install psutil

   conda install numexpr

   conda install rasterio

   conda install geopandas

   conda install natsort

   conda install scikit-image

   conda install pyopengl

   conda install simpeg

   conda install shapelysmooth

   conda install openpyxl

   conda install h5netcdf

   conda install rioxarray

   conda install pytest

   conda install beautifulsoup4

   pip install mtpy

   conda update --all

Once this is done, download pygmi, extract (unzip) it to a directory, and run it from its root directory with the following command:

   python quickstart.py
