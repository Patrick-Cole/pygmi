#PyGMI Readme

For licence information see the file LICENSE.txt

##Requirements
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

##Installation
Please check the internet for the most detailed installation instructions

###Installation - Windows
On Windows, the simplest is to either use WinPython or download an installer which includes WinPython.

The installer can be found on [Google Drive](https://209f493642c7e79b2a878320662bfff73a2946cf.googledrive.com/host/0B6BP_95afhWzN01tZzh5VG1aNk0/)

WinPython does not by default install itself into your registry, so it will not conflict with other versions of python.

##Running as a library
PyGMI can be imported simply by using the following commands:
	
	import pygmi
	pygmi.main.main()
