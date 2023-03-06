echo off
echo.
echo PyGMI
echo =====
echo.
echo This console window is used to display error messages  
echo Errors are also stored in %cd%\err.log
echo If you do have any errors, you can send the message to:
echo pcole@geoscience.org.za
echo.
echo Loading PyGMI...
set GDAL_DATA=%cd%\python\Lib\site-packages\osgeo\data\gdal
set PROJ_DATA=%cd%\python\Lib\site-packages\pyproj\proj_dir\share\proj
.\python\python.exe quickstart.py > err.log 2>&1
echo.
echo Latest Errors and Messages:
type err.log
pause