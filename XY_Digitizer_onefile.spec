# -*- mode: python ; coding: utf-8 -*-
# Single-file build of XY Plot Digitizer; run by build_onefile.bat (pyinstaller XY_Digitizer_onefile.spec).
# Same content as the folder build of build_exe.bat, plus the source code and license texts packed inside
# the exe (folder "bundled", filled by build_onefile.bat before this runs).
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = SPECPATH
src = os.path.join(root, "src")
bundled = os.path.join(root, "build_tmp", "onefile_bundled")
ico = os.path.join(src, "icon", "xyd.ico")

datas = [(ico, "icon"), (os.path.join(src, "icon", "xyd.png"), "icon"), (bundled, "bundled")]
binaries = []
hiddenimports = ["matplotlib.backends.backend_qtagg"]
datas += collect_data_files("omegaconf")
d, b, h = collect_all("rapidocr")
datas += d; binaries += b; hiddenimports += h

a = Analysis(
    [os.path.join(src, "xydapp.py")],
    pathex=[src],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pymupdf.layout", "pymupdf4llm", "tkinter", "PyQt5", "PyQt6", "torch", "IPython"],
    noarchive=False,
    optimize=0,
)
# the program reads no video: leave out the FFmpeg plugin of OpenCV (30 MB, LGPL)
a.binaries = [x for x in a.binaries if "opencv_videoio_ffmpeg" not in os.path.basename(x[0]).lower()]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="XY_Digitizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon=[ico],
)
