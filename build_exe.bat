@echo off
rem Builds app\XY_Digitizer\XY_Digitizer.exe, then runs the test in the "test" folder.
rem Every step writes its details to build_log.txt in this folder.
setlocal
cd /d "%~dp0"
set "LOG=%~dp0build_log.txt"
set "PY=C:\Users\Kurt\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PY%" set "PY=python"

echo XY Plot Digitizer: building the exe. Details: build_log.txt
echo [%date% %time%] build started > "%LOG%"
"%PY%" --version >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo 1/5 build environment ...
if exist build_env\Scripts\python.exe goto have_env
"%PY%" -m venv build_env >> "%LOG%" 2>&1
if errorlevel 1 goto fail
:have_env

echo 2/5 packages (the first build downloads about 300 MB) ...
build_env\Scripts\python.exe -m pip install --upgrade pip >> "%LOG%" 2>&1
build_env\Scripts\python.exe -m pip install --upgrade numpy scipy matplotlib PySide6-Essentials opencv-python PyMuPDF openpyxl rapidocr onnxruntime pyinstaller >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo 3/5 PyInstaller (a few minutes) ...
echo [%date% %time%] pyinstaller >> "%LOG%"
build_env\Scripts\pyinstaller.exe --noconfirm --clean --windowed --name XY_Digitizer --paths "%~dp0src" ^
  --hidden-import matplotlib.backends.backend_qtagg ^
  --collect-all rapidocr --collect-data omegaconf ^
  --icon "%~dp0src\icon\xyd.ico" --add-data "%~dp0src\icon\xyd.ico;icon" --add-data "%~dp0src\icon\xyd.png;icon" ^
  --exclude-module pymupdf.layout --exclude-module pymupdf4llm ^
  --exclude-module tkinter --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module torch --exclude-module IPython ^
  --distpath "%~dp0app" --workpath "%~dp0build_tmp" --specpath "%~dp0build_tmp" "%~dp0src\xydapp.py" >> "%LOG%" 2>&1
if errorlevel 1 goto fail
if not exist "app\XY_Digitizer\XY_Digitizer.exe" goto fail

echo 4/5 licenses and source code ...
build_env\Scripts\python.exe src\collect_licenses.py "%~dp0app\XY_Digitizer\licenses" >> "%LOG%" 2>&1
copy /y LICENSE.txt "%~dp0app\XY_Digitizer\LICENSE.txt" >> "%LOG%" 2>&1
copy /y LICENSE-AGPL-3.0.txt "%~dp0app\XY_Digitizer\LICENSE-AGPL-3.0.txt" >> "%LOG%" 2>&1
rem PyMuPDF 2.0 imports its layout package (ONNX models, 60 MB, not used here) at start-up if present;
rem it is left out above, so that "import pymupdf" does not look for the model files
rem the program reads no video: leave out the FFmpeg plugin of OpenCV (30 MB, LGPL)
del /q "%~dp0app\XY_Digitizer\_internal\cv2\opencv_videoio_ffmpeg*.dll" >> "%LOG%" 2>&1
rem complete source code next to the exe (AGPL-3.0)
set "SRC=%~dp0app\XY_Digitizer\source"
if exist "%SRC%" rmdir /s /q "%SRC%"
mkdir "%SRC%\src"
copy /y src\*.py "%SRC%\src\" >> "%LOG%" 2>&1
xcopy /e /i /y src\icon "%SRC%\src\icon" >> "%LOG%" 2>&1
xcopy /e /i /y src\gnu_licenses "%SRC%\src\gnu_licenses" >> "%LOG%" 2>&1
copy /y build_exe.bat "%SRC%\" >> "%LOG%" 2>&1
copy /y run_from_source.bat "%SRC%\" >> "%LOG%" 2>&1
copy /y build_onefile.bat "%SRC%\" >> "%LOG%" 2>&1
copy /y XY_Digitizer_onefile.spec "%SRC%\" >> "%LOG%" 2>&1
copy /y README.txt "%SRC%\" >> "%LOG%" 2>&1
copy /y LICENSE.txt "%SRC%\" >> "%LOG%" 2>&1
copy /y LICENSE-AGPL-3.0.txt "%SRC%\" >> "%LOG%" 2>&1

echo 5/5 test: test\iqe.png against test\Sheet1.dat, and a PDF (test\pdf_check.pdf) ...
echo [%date% %time%] test >> "%LOG%"
call "%~dp0test\run_test.bat" nopause >> "%LOG%" 2>&1
echo.
if exist "%~dp0test\compare.txt" type "%~dp0test\compare.txt"
echo.
echo [%date% %time%] BUILD FINISHED >> "%LOG%"
echo Finished: app\XY_Digitizer\XY_Digitizer.exe
echo This window closes in 60 s (or press a key).
timeout /t 60 > nul
exit /b 0

:fail
echo [%date% %time%] BUILD FAILED >> "%LOG%"
echo.
echo BUILD FAILED. See build_log.txt in this folder.
echo This window closes in 60 s (or press a key).
timeout /t 60 > nul
exit /b 1
