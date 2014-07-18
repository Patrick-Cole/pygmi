echo off
echo.
echo PyGMI
echo =====
echo.
echo This console window is used to display error messages. 
echo If you do have any errors, you can send the message to:
echo pcole@geoscience.org.za
echo.
echo Loading python environment...
call .\WinPython-64bit-3.3.5.0\scripts\env.bat
echo Loading PyGMI...
.\WinPython-64bit-3.3.5.0\python-3.3.5.amd64\python.exe main.py > err.log 2>&1
echo.
echo Latest Errors and Messages (also stored in err.log):
type err.log
pause