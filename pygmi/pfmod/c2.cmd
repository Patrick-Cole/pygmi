@echo off
cl  grvmagc_%PVER%_%BITS%.c /I"%WINPYDIR%\include" /I"%WINPYDIR%\lib\site-packages\numpy\core\include" /link /dll /libpath:"%WINPYDIR%\libs" /out:grvmagc_%PVER%_%BITS%.pyd
del *.lib, *.obj, *.exp, *.c, grvmagc_%PVER%_%BITS%.pyx
