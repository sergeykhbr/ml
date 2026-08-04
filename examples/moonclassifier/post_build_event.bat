set COPY_TO=%1

xcopy %QT_PATH%\plugins\platforms\qwindows.dll %COPY_TO%\platforms\ /F /R /Y /D
xcopy %QT_PATH%\plugins\platforms\qwindowsd.dll %COPY_TO%\platforms\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Cored.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Cored.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Guid.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Guid.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Widgetsd.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Widgetsd.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Core.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Core.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Gui.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Gui.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\Qt6Widgets.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\Qt6Widgets.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\lib\libEGL.lib %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\libEGL.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\icudt*.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\icuin*.dll %COPY_TO%\ /F /R /Y /D
xcopy %QT_PATH%\bin\icuuc*.dll %COPY_TO%\ /F /R /Y /D

@echo off
rem @echo mftprosim.exe -c ../../targets/mftpro.json > %COPY_TO%\_run_mftpro_sim.bat


