#PyGMI Readme

PyGMI stands for *Python Geophysical Modelling and Interpretation*. It is a modelling and interpretation suite aimed at magnetic, gravity and other datasets. It includes:
* Magnetic and Gravity 3D forward modelling
* Cluster Analysis
* Routines for cutting, reprojecting and doing simple modifications to data
* Convenient display of data using pseudo-color, ternary and sunshaded representation

It is released under the Gnu General Public License version 3.0

For license information see the file LICENSE.txt

##Requirements
PyGMI will run on both Windows and Linux. It should be noted that the main development is done in Python 3.4 on Windows.

PyGMI is developed and has been tested with the following libraries in order to function:

* Python 3.4.2
* NumPy 1.8.2
* SciPy 0.14.0
* Matplotlib 1.4.1
* PyQt 4.10.4
* GDAL 1.11.1
* numexpr 2.4
* numba 0.15.1

It is possible that it might work on earlier versions, especially on non-windows operating systems. Under windows, there are compiled .pyd files (python version of a dll) which require the NumPy and Python version to match. However, for actual releases, I do post windows binaries which include a standalone python distribution (that will not interfere with existing installations), so this should not be a problem.

##Installation
###General
If you satisfy the requirements, and wish to run PyGMI from within your python environment, you can install PyGMI as a library using the following command from within the root of the PyGMI directory:

python setup.py install
Once you are in python, you can run PyGMI by using the following commands:

	import pygmi
	pygmi.main()


###Windows
I have now made available convenient installers for PyGMI, thanks to Cyrille Rossant. If you have already installed PyGMI using this installer, you can simply update it using the smaller pygmi_update.exe file. This smaller update file contains only the python code, if you are only interested in that.

Running the software can be acheived through the shortcut on your desktop.

###Linux
Linux normally comes with python installed, but the additional libraries will still need to be installed. The software has been tested on Ubuntu 13.10 linux, using Python 2.7

Install libraries (shown below for Ubuntu 13.10):

	sudo apt-get upgrade
	sudo apt-get install python-numpy
	sudo apt-get install python-scipy
	sudo apt-get install python-qt4
	sudo apt-get install python-matplotlib
	sudo apt-get install python-gdal
	sudo apt-get install python-numexpr
	sudo apt-get install cython
	sudo apt-get install libqt4-opengl
	sudo apt-get install python-qt4-gl

Extract the zip file.

Execute the following command from within the root of the PyGMI directory to run the program:

	python quickstart.py

Alternatively you can install it as a library as shown above.


