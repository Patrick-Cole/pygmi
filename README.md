#PyGMI Readme

PyGMI stands for *Python Geophysical Modelling and Interpretation*. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets. It includes:
* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation

It is released under the Gnu General Public License version 3.0

For license information see the file LICENSE.txt

##Requirements
PyGMI will run on both Windows and Linux, as well as Python 2.7 and Python 3.3. 
It should be noted that the main development is done in Python 3.3 on Windows.

PyGMI requires the following libraries in order to function:

Python 3.3.5 or 2.7.6
NumPy 1.8.1
SciPy 0.13.3
Matplotlib 1.3.1
PySide 1.2.1
QtOpenGL for PySide (1.1.0 or greater)
GDAL 1.10.1
numexpr 2.3.1

It is possible that it migt work on earlier versions, especially on non-windows operating systems. Under windows, there are compiled .pyd files (python version of a dll) which require the NumPy version to match.

##Installation

In general, if you satisfy the requirements, and have installed PyGMI as a library, you can run PyGMI by using the following commands:

	import pygmi
	pygmi.main()

###Installation - Windows (Fast)
On Windows, the simplest is to either use WinPython or download an installer which includes WinPython.

The installer can be found on [Google Drive](https://209f493642c7e79b2a878320662bfff73a2946cf.googledrive.com/host/0B6BP_95afhWzN01tZzh5VG1aNk0/)

WinPython does not by default install itself into your registry, so it will not conflict with other versions of python.
