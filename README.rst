Overview
========

PyGMI stands for Python Geoscience Modelling and Interpretation. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets.

PyGMI is developed at the `Council for Geoscience <http://www.geoscience.org.za>`_ (Geological Survey of South Africa).

It includes:

* Magnetic and Gravity 3D forward modelling.
* Cluster Analysis, including use of scikit-learn libraries.
* Routines for cutting, reprojecting and doing simple modifications to data.
* Convenient display of data using pseudo-color, ternary and sunshaded representation.
* MT processing and 1D inversion using MTpy.
* Gravity processing.
* Seismological functions for SEISAN data.
* Remote sensing ratios and improved imports.

It is released under the `Gnu General Public License version 3.0 <http://www.gnu.org/copyleft/gpl.html>`_

The PyGMI `Wiki <http://patrick-cole.github.io/pygmi/index.html>`_ pages, include installation and full usage!

The latest release version can be found `here <https://github.com/Patrick-Cole/pygmi/releases>`_.

You may need to install the `Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_.

If you have any comments or queries, you can contact the author either through `GitHub <https://github.com/Patrick-Cole/pygmi>`_ or via email at pcole@geoscience.org.za

Requirements
------------
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.8 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* python 3.8.5
* discretize 0.5.1
* fiona 1.8.17
* gdal 3.1.4
* geopandas 0.8.1
* llvmlite 0.34.0
* matplotlib 3.3.2
* mtpy 1.1.3
* numba 0.51.2
* numexpr 2.7.2
* numpy 1.19.2+mkl
* pandas 1.1.3
* pillow 8.0.1
* pymatsolver 0.1.2
* pyopengl 3.1.5
* PyQt5 5.12.3
* pytest 6.0.1
* scikit-image 0.17.2
* scikit-learn 0.23.2
* scipy 1.5.3
* segyio 1.9.3
* shapely 1.7.1
* SimPEG 0.14.2
* openpyxl 3.0.6

Installation
------------
General (Not Anaconda)
----------------------
The easiest way to install pygmi if you are working in a python environment is to use the pip command as follows:

   pip install pygmi

This will download pygmi from PyPI and install it within your python repository. Please note the use of pip when installing PyGMI may cause Anaconda installations to break. Anaconda users should follow the instructions below.

Alternatively, if you satisfy the requirements, you can download pygmi either from Github or PyPI, extract it and run the following command from within the extracted directory:

   python setup.py install

In either case, running pygmi can be now done at the command prompt as follows:

   pygmi

If you are in python, you can run PyGMI by using the following commands:

   import pygmi

   pygmi.main()

If you prefer not to install pygmi as a library, or if there is a problem with running it in that matter, you can simply execute the following command to run it manually:

   python quickstart.py

Windows Users
-------------
Installers are available in `64-bit <https://github.com/Patrick-Cole/pygmi/releases>`_

Alternatively, you can use the instructions above to run PyGMI with your local python installation. You may need to install some dependencies using downloaded binaries, because of compilation requirements. Therefore, if you do get an error, you can try installing precompiled binaries before installing PyGMI.

Examples of binaries you may need to get are:

* numexpr
* numba
* llvmlite
* GDAL

They can be obtained from the `website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ by Christoph Gohlke.

Linux
-----
Linux normally comes with python installed, but the additional libraries will still need to be installed. One convenient option is to install the above libraries through `Anaconda Python <https://www.anaconda.com/>`_.

Anaconda
--------
Anaconda users are advised not to use pip since it can break PyQt5. However, two packages are installed only by pip, so a Conda environment should be created. Note that I installed all packages from the 'defaults' conda channel, except where the command specifies otherwise.

The process to install is as follows:

   conda create -n pygmi python=3.8

   conda activate pygmi

   conda install pyqt

   conda install numpy

   conda install scipy

   conda install numexpr

   conda install gdal

   conda install pillow

   conda install matplotlib

   conda install numba

   conda install pandas

   conda install scikit-learn

   conda install scikit-image

   conda install geopandas

   conda install pyopengl

   conda install -c conda-forge segyio

   conda install -c conda-forge simpeg

   pip install mtpy


Once this is done, download pygmi, extract it to a directory, and run it from its root directory with the following command:

   python quickstart.py

Alternatively, if you satisfy the requirements, you can run the following command from within the extracted directory:

   python setup_anaconda.py install

Running pygmi can be now done at the command prompt as follows:

   pygmi
