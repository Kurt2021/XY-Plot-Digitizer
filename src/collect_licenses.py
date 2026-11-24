"""Copies the license files of all installed packages (build environment) next to the built exe,
and adds the notice and license texts for Qt / PySide6 (LGPL-3.0) and PyMuPDF (AGPL-3.0)."""
import os, shutil, sys, sysconfig, glob
dest = sys.argv[1]
onefile = "--onefile" in sys.argv[2:]     # single-file exe: Qt is packed inside the exe
os.makedirs(dest, exist_ok=True)
site = sysconfig.get_paths()["purelib"]
n = 0
for dist in glob.glob(os.path.join(site, "*.dist-info")):
    name = os.path.basename(dist).split("-")[0]
    files = [p for p in glob.glob(os.path.join(dist, "**", "*"), recursive=True)
             if os.path.isfile(p) and any(k in os.path.basename(p).upper() for k in ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS"))]
    for p in files:
        rel = os.path.relpath(p, dist).replace(os.sep, "_")
        shutil.copy(p, os.path.join(dest, name + "__" + rel + ("" if rel.lower().endswith(".txt") else ".txt")))
        n += 1
# packages that keep their license files inside the package folder, not in the .dist-info folder
for pkg, names in (("onnxruntime", ("LICENSE", "ThirdPartyNotices.txt")),):
    for nm in names:
        p = os.path.join(site, pkg, nm)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(dest, pkg + "__" + nm + ("" if nm.endswith(".txt") else ".txt")))
            n += 1
# Python itself
base = sys.base_prefix
for cand in ("LICENSE.txt", "LICENSE"):
    p = os.path.join(base, cand)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(dest, "Python__LICENSE.txt")); n += 1
        break

# Qt and PySide6
QT_FOLDER = """The libraries are shipped unmodified, as separate files in the program folder
(_internal\\PySide6 and _internal\\shiboken6). You may replace them with other builds of the same
libraries, and you may modify XY Plot Digitizer for your own use and reverse engineer it to debug such
modifications.
"""
QT_ONEFILE = """The libraries are shipped unmodified. In this single-file executable they are packed inside
XY_Digitizer.exe and unpacked to a temporary folder at each start. To use the program with other builds
of the same libraries, rebuild it from its complete source code (saved with the button "Save source code
and licenses" in the tab About; build_onefile.bat) against those libraries, or use the folder build
(build_exe.bat), where they are separate files. You may modify XY Plot Digitizer for your own use and
reverse engineer it to debug such modifications.
"""
try:
    import PySide6
    from PySide6 import QtCore
    versions = "PySide6 %s, Qt %s" % (PySide6.__version__, QtCore.qVersion())
except Exception:
    versions = "PySide6 / Qt"
with open(os.path.join(dest, "Qt_PySide6__NOTICE.txt"), "w", encoding="utf-8") as fh:
    fh.write(("""This program uses Qt and Qt for Python (%s).
Copyright (C) The Qt Company Ltd. and other contributors.

Qt and PySide6 are used here under the GNU Lesser General Public License version 3 (LGPL-3.0).

%s
License texts: Qt_PySide6__LGPL-3.0.txt and Qt_PySide6__GPL-3.0.txt in this folder
               (https://www.gnu.org/licenses/lgpl-3.0.txt, https://www.gnu.org/licenses/gpl-3.0.txt)
Source code:   https://download.qt.io/official_releases/qt/
               https://download.qt.io/official_releases/QtForPython/pyside6/
""") % (versions, QT_ONEFILE if onefile else QT_FOLDER))
n += 1

# PyMuPDF / MuPDF
try:
    import pymupdf
    pv, mv = pymupdf.__version__, pymupdf.mupdf_version
except Exception:
    pv, mv = "?", "?"
with open(os.path.join(dest, "PyMuPDF__NOTICE.txt"), "w", encoding="utf-8") as fh:
    fh.write("""This program uses PyMuPDF %s and MuPDF %s to read PDF pages.
Copyright (C) Artifex Software, Inc.

PyMuPDF and MuPDF are licensed under the GNU Affero General Public License version 3 (AGPL-3.0),
or under a commercial license from Artifex. They are used here under the AGPL-3.0, which is also the
license of XY Plot Digitizer (LICENSE.txt). They are shipped unmodified.

License text: PyMuPDF__AGPL-3.0.txt in this folder (https://www.gnu.org/licenses/agpl-3.0.txt)
Source code of the versions shipped here:
              PyMuPDF  https://pypi.org/project/PyMuPDF/%s/#files (source archive .tar.gz),
                       https://github.com/pymupdf/PyMuPDF/releases
              MuPDF    https://mupdf.com/releases (mupdf-%s-source), https://github.com/ArtifexSoftware/mupdf
""" % (pv, mv, pv, mv))
n += 1

# GNU license texts, kept with the sources (src/gnu_licenses), so the build needs no network for them
here = os.path.dirname(os.path.abspath(__file__))
for prefix, name in (("Qt_PySide6", "LGPL-3.0"), ("Qt_PySide6", "GPL-3.0"), ("PyMuPDF", "AGPL-3.0")):
    shutil.copy(os.path.join(here, "gnu_licenses", name + ".txt"), os.path.join(dest, "%s__%s.txt" % (prefix, name)))
    n += 1
print("license files copied:", n)
