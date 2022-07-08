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
call ".\python\scripts\env.bat"
python.exe quickstart.py > err.log 2>&1
echo.
echo Latest Errors and Messages:
type err.log
pause