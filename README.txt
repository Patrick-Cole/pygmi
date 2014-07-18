=============
PyGMI Readme
=============

For licence information see the file LICENSE.txt

Requirements
============
PyGMI will run on both Windows and Linux, as well as Python 2.7 and Python 3.3. 
It should be noted that the main development is done in Python 3.3 on Windows.

PyGMI requires the following libraries in order to function:

Python 3.3.5 or 2.7.6
Numpy 1.8.1
Scipy 0.13.3
Matplotlib 1.3.1
PySide 1.2.1
QtOpenGL for PySide (1.1.0 or greater)
GDAL 1.10.1
numexpr 2.3.1

On Windows, the simplest is to use WinPython. This software is developed using WinPython 3.3

This can be found at:
http://winpython.sourceforge.net/

WinPython does not by default install itself into your registry,
so it will not conflict with other versions of python.

On Linux, Python 2.7 is normally installed, but the libraries must be installed seperately

Installation
============
Please check the internet for the most detailed installation instructions

Installation - Windows and WinPython
------------------------------------
1) download WinPython
2) download GDAL
3) install WinPython ONLY
4) Open WinPython Control Panel (WinPython Control Panel.exe, found in install directory)
5) Drag GDAL Binary onto the control panel and click 'Install Package'
6) unzip pygmi files into a directory of your choice
7) Edit PyGMI2.cmd to point at your python install. An example of the relevant line for 
	winpython version 3.3.2.3 is shown below:

	call C:\WinPython-64bit-3.3.2.3\scripts\env.bat

The PyGMI.cmd file allows PyGMI to run without needing WinPython
installed into the registry.

Execute the cmd file to run the program.

Installation - Linux
------------------------------------
Linux normally comes with python installed, but the additional 
libraries will still need to be installed. The software has 
been tested on Ubuntu 13.10 linux, using Python 2.7

1) Install libraries (shown below for Ubuntu 13.10)
sudo apt-get upgrade
sudo apt-get install python-numpy
sudo apt-get install python-scipy
sudo apt-get install python-pyside
sudo apt-get install python-matplotlib
sudo apt-get install python-gdal
sudo apt-get install python-numexpr
sudo apt-get install cython
sudo apt-get install python-opengl

2) Extract the zip file.
3) Execute the following command from within the root of the PyGMI directory to run the program:
		python main.py
