# XY Plot Digitizer

XY Plot Digitizer reads the data of an XY line or scatter plot and exports it as CSV or XLSX. The plot can come from a picture (clipboard, PNG, JPEG, TIFF and other image files) or from a figure in a PDF, SVG, EPS or PS file. It is a Windows program with a graphical interface, written in Python (PySide6, NumPy, SciPy, OpenCV, PyMuPDF, RapidOCR) and packaged with PyInstaller.

Version: 1.1.34
Author: Kurt Taretto
License: GNU AGPL-3.0

## What it can do

### Input
- Import a plot from the clipboard (Ctrl+V) or from an image file.
- Import a figure from a PDF or SVG page: choose the page, then click an outlined figure or drag a rectangle around the plot.
- Read vector figures in a PDF directly from the drawing paths, which gives exact values. The import dialog says whether the chosen figure is vector graphics or an embedded picture.
- Find and outline the figures on scanned papers (one picture per page, also with a tilted scan line or an OCR text layer).
- Open EPS and PS files (needs Ghostscript).

### Axes
- Detect the plot frame and read the tick labels by OCR.
- Handle linear, log, ln and reciprocal (1/x) axis scales; the exported values are the printed ones.
- Handle a second Y axis on the right with its own scale; a dialog asks which axis each curve belongs to.
- Let the user correct any axis by typing two values or by picking two positions with a cross-hair.

### Curves and symbols
- Separate curves by colour and by line style (solid, dashed, dash-dot), also where curves cross.
- Take the centroid of each scatter symbol as the data point.
- Find open symbols (circles, squares, triangles) that touch a curve or each other. If the curve passes through the symbol centres, the plot is taken as one line with symbols; if not, the symbols become a separate data set and the curve is traced past them.
- Ignore grid lines, shaded bands behind the curves, insets and most text inside the plot.
- Handle semi-transparent curves and stacked panels.
- Optional guided trace ("Trace a curve"): click points along one curve to extract a curve the automatic tracing mixed up, for example two curves of the same colour that cross.
- Optional analysis of blurred pictures at full size.

### Checking the result
- Show the extracted points over the imported picture, next to a redrawn plot in the original curve colours.
- List the series found; untick what is not data, and rename a series (the name heads its columns in the export).
- Report what the analysis found in a "Messages" window.

### Output
- Export to CSV or XLSX.
- With "interpolate x-values" (on by default), all series share one x column; without it, each series gets its own x and y columns.
- Command-line mode without interface:
  `XY_Digitizer.exe --digitize PLOT --out BASE`
  writes BASE.csv, BASE.xlsx, BASE_common.csv, BASE_overlay.png and BASE_log.txt (last line: RESULT: OK, INCOMPLETE or FAILED).

### Interface
- Dark or light theme and adjustable text size.
- Colour-blind safe palette (IBM) for the interface.
- Settings are kept between sessions.

## Known limits
- One plot per picture. No top X axis, no broken axes, no rotated scans, no offset texts such as "x10^3".
- Curves lying on top of each other cannot be told apart in a picture; use the PDF if the figure is vector graphics.
- Pictures that are compressed (JPEG), enlarged or blurred give curves in pieces; check the result.
- Text inside the plot and legend samples are not always removed; untick those series before export.
- A filled area that is itself the data (area plot) is ignored, like a shaded band.
- Series names and guided traces are not kept when the plot is processed again.

## Download and install
1. Download `XY_Digitizer.zip` from the Releases page.
2. Unzip it and keep the `XY_Digitizer` folder together.
3. Run `XY_Digitizer.exe`. No installation is needed.

## Contents of the zip
| Item | What it is |
|---|---|
| `XY_Digitizer.exe` | the program |
| `_internal\` | Python runtime and libraries used by the program |
| `source\` | complete source code of this version and the build scripts |
| `licenses\` | licenses of the third-party components |
| `LICENSE.txt`, `LICENSE-AGPL-3.0.txt` | license of the program |

## Build from source
In `source\`: run `build_exe.bat` to build the program folder, or `build_onefile.bat` (after `build_exe.bat`) for a single-file exe. `run_from_source.bat` runs the program from the Python files. The first build downloads about 300 MB of packages.

## License
XY Plot Digitizer is free software under the GNU Affero General Public License, version 3 (AGPL-3.0). Whoever passes the program on must pass on, or offer, the source code under the same license. Third-party components keep their own licenses (see the `licenses` folder). Qt and PySide6 are used under the GNU LGPL version 3; PyMuPDF and MuPDF under the GNU AGPL version 3.
