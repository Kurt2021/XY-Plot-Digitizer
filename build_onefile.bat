@echo off
rem Builds app_onefile\XY_Digitizer.exe: ONE file that holds everything (Python, libraries, OCR models,
rem source code and license texts). Passing on this exe alone is enough to run the program.
rem Uses the build environment made by build_exe.bat (run build_exe.bat once first).
rem Details of every step: build_onefile_log.txt in this folder.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0build_onefile_log.txt"
echo XY Plot Digitizer: building the single-file exe. Details: build_onefile_log.txt
echo [%date% %time%] one-file build started > "%LOG%"
if not exist build_env\Scripts\pyinstaller.exe goto noenv

echo 1/3 source code and licenses to pack into the exe ...
set "B=%~dp0build_tmp\onefile_bundled"
if exist "%B%" rmdir /s /q "%B%"
mkdir "%B%\source\src"
build_env\Scripts\python.exe src\collect_licenses.py "%B%\licenses" --onefile >> "%LOG%" 2>&1
if errorlevel 1 goto fail
copy /y LICENSE.txt "%B%\" >> "%LOG%" 2>&1
copy /y LICENSE-AGPL-3.0.txt "%B%\" >> "%LOG%" 2>&1
copy /y src\*.py "%B%\source\src\" >> "%LOG%" 2>&1
xcopy /e /i /y src\icon "%B%\source\src\icon" >> "%LOG%" 2>&1
xcopy /e /i /y src\gnu_licenses "%B%\source\src\gnu_licenses" >> "%LOG%" 2>&1
for %%f in (build_exe.bat build_onefile.bat XY_Digitizer_onefile.spec run_from_source.bat README.txt LICENSE.txt LICENSE-AGPL-3.0.txt) do copy /y %%f "%B%\source\" >> "%LOG%" 2>&1

echo 2/3 PyInstaller, one file (a few minutes) ...
echo [%date% %time%] pyinstaller >> "%LOG%"
build_env\Scripts\pyinstaller.exe --noconfirm --clean --distpath "%~dp0app_onefile" --workpath "%~dp0build_tmp\onefile" "%~dp0XY_Digitizer_onefile.spec" >> "%LOG%" 2>&1
if errorlevel 1 goto fail
if not exist "app_onefile\XY_Digitizer.exe" goto fail

echo 3/3 test: the same test as build_exe.bat, on the single-file exe, and the packed source and licenses ...
echo [%date% %time%] test >> "%LOG%"
set "XYD_EXE=%~dp0app_onefile\XY_Digitizer.exe"
call "%~dp0test\run_test.bat" nopause >> "%LOG%" 2>&1
set "CHK=%~dp0build_tmp\onefile_check"
if exist "%CHK%" rmdir /s /q "%CHK%"
start "" /wait "%XYD_EXE%" --save-bundled "%CHK%"
set "BOK=FAIL"
if exist "%CHK%\source\src\xydapp.py" if exist "%CHK%\LICENSE-AGPL-3.0.txt" if exist "%CHK%\licenses\Qt_PySide6__NOTICE.txt" set "BOK=PASS"
echo SOURCE AND LICENSES INSIDE THE EXE: %BOK%>> "%~dp0test\compare.txt"
echo.
if exist "%~dp0test\compare.txt" type "%~dp0test\compare.txt"
echo.
echo [%date% %time%] BUILD FINISHED >> "%LOG%"
echo Finished: app_onefile\XY_Digitizer.exe
echo This window closes in 60 s (or press a key).
timeout /t 60 > nul
exit /b 0

:noenv
echo The build environment is missing: run build_exe.bat once first.
echo [%date% %time%] no build_env >> "%LOG%"
:fail
echo [%date% %time%] BUILD FAILED >> "%LOG%"
echo.
echo BUILD FAILED. See build_onefile_log.txt in this folder.
echo This window closes in 60 s (or press a key).
timeout /t 60 > nul
exit /b 1
