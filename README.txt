XY Plot Digitizer 1.1.33: data from pictures, PDF, SVG and EPS figures of XY line and scatter plots

Build the exe:   double-click build_exe.bat
                 -> app\XY_Digitizer\XY_Digitizer.exe   (keep the XY_Digitizer folder together; make a desktop shortcut to the exe)
                 The first build downloads about 300 MB of packages. Details of every step: build_log.txt.
                 The build ends with the test below.
Single file:     double-click build_onefile.bat (after build_exe.bat has run once)
                 -> app_onefile\XY_Digitizer.exe   one file with everything inside, also the source code and
                 the license texts (tab About, button "Save source code and licenses"). Passing on this exe alone
                 is enough. It unpacks itself to a temporary folder at each start, so it starts slower than the
                 folder build. Details: build_onefile_log.txt; ends with the same test, plus a check of the
                 packed source code and licenses (last line of test\compare.txt).
Run without exe: run_from_source.bat (after build_exe.bat has run once)
Test:            test\run_test.bat runs the exe on test\iqe.png and compares the extracted curves with the
                 measured data in test\Sheet1.dat (1/alpha in cm, IQE). Result: test\compare.txt, last line
                 TEST: PASS or TEST: FAIL; test\result_overlay.png shows the extracted points over the picture;
                 test\window.png is a picture of the program window after processing the same plot.

Sources:         src\xydapp.py (interface, Qt/PySide6), src\xydtheme.py (colours and style sheet, taken from IV Analysis),
                 src\xydcore.py (image analysis, PDF reading, export), src\collect_licenses.py (build helper),
                 src\gnu_licenses (LGPL-3.0, GPL-3.0 and AGPL-3.0 texts, copied into the licenses folder),
                 src\make_icon.py and src\icon (program icon: xyd.ico, xyd.png)
License:         GNU AGPL-3.0 (LICENSE.txt, full text LICENSE-AGPL-3.0.txt). The build puts the complete source
                 code into app\XY_Digitizer\source and the third-party licenses into app\XY_Digitizer\licenses.
                 To pass the program on, give the whole app\XY_Digitizer folder (with "source" and "licenses"),
                 or only app_onefile\XY_Digitizer.exe (source and licenses are inside).
Exports go by default to D:\Resources\Software\My Digitizer (change in tab "1. Settings").

Use:             tab "2. Digitize": Import from clipboard (or Ctrl+V) / Import from file, Process, check, Export.
                 For a PDF the dialog says whether the chosen figure is vector graphics (data read from the drawing
                 paths, exact) or an embedded picture (image analysis).
                 After Process: imported plot on the left with the extracted points drawn over it, extracted plot on
                 the right, list of series (untick what is not data), detected axes (edit, or Pick two positions, then
                 Apply axes). Pick turns the pointer into a cross-hair: click the position (click again to move
                 it), double click to finish.
                 Two Y axes: when the right-hand axis carries another scale than the left one, a dialog asks which
                 axis the curves belong to; single series are switched by a click in the column "y axis".
                 Row "Y right" of box 4 defines or corrects that axis by hand; empty the row to drop it.
                 Scale (tick spacing) of each axis: linear, log, ln (same spacing as log), 1/x (reciprocal).
                 The exported values are the printed ones.
                 Export: CSV or XLSX. With "interpolate x-values" (tab "1. Settings", on by default) all lines
                 share one x column (x, S1_y, S2_y, ...); without it, one x and one y column per ticked series.
                 Without interface:  XY_Digitizer.exe --digitize PLOT --out BASE
                 writes BASE.csv, BASE.xlsx, BASE_common.csv (interpolated x), BASE_overlay.png, BASE_log.txt (last line RESULT: OK, INCOMPLETE or FAILED).

Look:            as IV Analysis: dark or light theme ("Theme" and "Text" at the right end of the tab row), IBM colour-blind
                 safe palette in the interface (blue #648FFF, purple #785EF0, magenta #DC267F, orange #FE6100,
                 yellow #FFB000). Messages are blue (done), yellow (attention) or magenta (problem).
                 The extracted plot keeps the colours of the original curves, so that the two plots can be compared;
                 a curve too close to the background colour is drawn in the text colour.
                 Theme, text size, table font, symbol position and output folder are remembered
                 (%APPDATA%\XY_Digitizer\config.json).
Left out on purpose (present in IV Analysis): languages, sessions, undo/redo, PDF report.

Limits:          one plot per picture; no top X axis, no broken axes, no rotated scans.
                 EPS / PS files need Ghostscript installed (gswin64c.exe); PDF and pictures do not.
                 Curves lying on top of each other over most of their length cannot be told apart in a picture.
                 Results of the test battery (synthetic plots with known data, run on Linux from the sources):
                 see dev\HANDOFF.md for the numbers of this version.
                 Axes are read reliably; series from compressed, enlarged or blurred pictures need checking.
                 Short straight lines next to longer series (tangents, reference marks) are listed but not ticked.

Changes in 1.1.33
- Scanned papers (Daub and Wuerfel, PRL 74, 1021 (1995): each page one 1-bit scan with an OCR text layer).
  1. The X axis was read from about its middle on. A scanned axis line is tilted by a pixel or so, and the frame
     was taken from the lowest pixel row of the bottom line, which only holds the part past the last 1-px step.
     The left axis was then not found next to that end, the right Y axis was taken as the Y axis, only 2 or 3 X
     labels were left inside, and X was read as log. Now the line is followed from row to row to its real ends (a
     row change counts only where the line goes on for at least 3 % of the width, so frames of pictures that are
     not tilted stay exactly as before). Fig. 3: frame from the left axis, X linear 0.9 to 1.35 (5/5 labels), both
     log Y axes, three curves; Fig. 4: 2 series, none ticked -> the curve and 16 of the 22 open circles.
  2. Tick marks left inside the frame edge formed "rows of dashes", so the curve of Fig. 4 was taken for a solid and
     a dashed curve of one colour and its symbols were dropped. Pieces at the frame edge no longer count as dashes.
  3. No figure was outlined in the PDF dialog: the axes frames were smaller than 15 % of the page. When one picture
     covers the page, frames down to 15 % of the page width are found, each grown over its labels and titles and
     cut at captions and paragraphs of the OCR text layer. Pages 2 and 3: Fig. 1 (chosen at once), Fig. 2, 3, 4.
  Battery, n,k, crossings, markers, panels, regions, SI regions, SI curves, twins, guided trace, superscripts,
  iqe.png / nk_two_axes.png / twins_abs.png frames and the GUI tests give the same output as 1.1.32, except
  twins_abs.png enlarged 2 times: mean 0.064 -> 0.822 (its frame is now found whole). Test: dev\tests\test_daub.py
  PATH_TO_PDF (1.1.32: FAIL, 1.1.33: PASS). Not changed: in Fig. 4 the line above 1.07 eV, where the circles touch
  each other, is lost with the circles (use "Trace a curve"); in Fig. 2 the circles are traced as part of a line
  that runs down the right frame edge (as in 1.1.32).

Changes in 1.1.32
- Tick labels read by OCR in a light font that spaces the digits wide: the '1' and '0' of 10^-8 (Aalbers et al.,
  Nat Commun 2024, Fig. 4c, page 6) were 0.41 x the character height apart, above the 0.4 x limit for joining
  characters into one label, so each X label was read as '1' and '0^-8' and no X axis was found. A gap of up to
  0.6 x now joins when every such gap is less than a quarter of the smallest gap between labels.
  Fig. 4c: X log 1e-8 to 1e-6 (3/3 labels), Y linear 150 to 350. Battery, n,k, crossings, markers, panels, twins,
  guided trace, SI curves, superscripts and the iqe.png compare test give the same output as 1.1.31 (d5 and
  test_regions.py not run: they need Ghostscript). Test: dev\tests\test_aalbers.py (Fig. 4c added).

Changes in 1.1.31
- An axis read on its own is used at once: when Process reads one axis and not the other (Aalbers et al., Nat Commun
  2024, Fig. 4c: Y linear 150 to 350 read, X log not read), the extracted plot shows the read axis in data units and
  only the missing one in pixels; 1.1.30 drew both in pixels until the missing axis was typed in. The status line
  and the plot title name the missing axis. Export still needs both axes. Test: dev\gui_partial_axes_test.py.

Changes in 1.1.30
- PDF figures drawn as vector paths over a picture layer (Aalbers et al., Nat Commun 2024, page 3: shaded bands of
  Fig. 1c, symbols of Fig. 2a/b and the device schematic of Fig. 1a are JPEG layers under the vector drawing):
  * a picture that spreads over two or more vector plots is no longer offered as a figure of its own (the box over
    Fig. 2a and 2b, the box over all of Fig. 1);
  * a region counts as vector graphics also when a picture covers more than half of it, if at least 60 stroked
    path pieces are drawn there: Fig. 1b and 1c are read from the paths, so the shaded bands no longer matter
    (Fig. 1c: 13 series from image analysis -> EQE, blackbody line and one arrow);
  * text inside the plot drawn as outlines (no text layer) is found by OCR, its glyphs are not taken as symbols,
    and stacked labels mark a legend as with real text; a curve drawn as short dashes that passes through such a
    legend zone keeps its dashes there;
  * one dash drawn as an open polyline of several points is a dash, not a symbol (the blackbody line was two
    scatter series);
  * tick labels drawn as outlines: '1.2' in a light font was read as '1' and '.2' (the x axis came out log); the
    right-hand Y labels are now read by OCR too (Fig. 1c: log 1e-10 to 1e10, Fig. 2b: linear 0.8 to 2.4); a
    narrow '1' of '10' next to the axis no longer cuts the label off.
  Test: dev\tests\test_aalbers.py PATH_TO_PDF. Battery (run_all, nk, cross, twins, markers, panels, SI regions
  and curves, iqe.png, nk_two_axes.png, twins_abs.png, pdf_check.pdf) unchanged.

Changes in 1.1.29
- "No data series found" on plots with shaded bands, from a PDF or the clipboard (Jiang et al., Nat Commun 2025,
  Fig. 3e). The bands, the curves over them and the text formed one piece whose outline filled its box, so the whole
  piece was erased as a "filled box" (label background), curves included. Now, when the fill colour of such a piece
  forms wide areas and ink of other colours inside runs across much of it (curves, not letters), only the fill
  colour is erased. Label backgrounds with text and inset plots (a thin frame with a plot inside) are still erased
  whole. The Fig. 3e panel from the PDF: 0 -> 12 series; rendered as a picture at 96 / 120 / 150 / 200 dpi (as a
  clipboard copy): 0 -> 12 to 16 series, 9 to 13 ticked, the nine curves among them. jiang.jpg unchanged (20).
  All tests unchanged.

Changes in 1.1.28
- Shaded areas behind the curves (bands marking a range, filled regions) are ignored: ink of one flat colour much
  wider than any line, covering at least 1 % of the plot; only pixels of that fill colour go, so curves drawn over
  it stay. Before, a band touching the curves was traced as dozens of pieces. jiang.jpg (EQE / IQE / R, two light
  green bands): 1.1.27 gave 56 series (36 ticked), 1.1.28 gives 20 (13 ticked): the nine curves (EQE, IQE and R of
  the three blends) each as one ticked series, plus 4 ticked pieces to untick (band edges, legend, the blue
  "87.7 %" text). All test results unchanged, except two si_example.pdf regions that are not plots (a journal logo,
  a page without axes).

Changes in 1.1.27
- Guided trace (button "Trace a curve" next to Messages, after Process; the automatic tracing stays as it is):
  for a curve the automatic tracing mixes up, such as a solid and a dash-dot curve of the same colour crossing
  each other. Click points along ONE curve in the imported picture: both ends, every peak and dip, and on both
  sides of each place where it crosses or runs into a curve of the same colour; double click to finish, Esc
  cancels. The colour is the one most clicks sit on; whether the curve is solid or dashed is also read from the
  clicks (on long strokes or on short pieces of that colour). Between two clicks the trace is the cheapest path
  through the picture: cheap on the curve's own kind of ink (and, for a dashed curve, in the gaps between its
  dashes), dearer on the other kind of ink of that colour and on ink of other colours (where it runs under them),
  dearest on the background; it never turns back in x. Clicks are moved onto the nearest ink of the colour within
  6 px. The trace is added as a new series (note "guided trace ..."); untick the automatic series it replaces.
- Results (dev\tests\test_guided.py: the six curves of test\twins_abs.png at 0.8, 1, 1.25 and 1.4 times, 6 to 11
  clicks per curve, each click up to 4 px off): every curve within 0.03 of the reference over 0.97 to 1.00 of
  415-985 nm (the automatic tracing, 1.1.26: 0.54 to 1.00 per curve). Too few clicks (no click at a peak or a
  crossing with the same-colour partner) can still put a stretch on the wrong curve: click more there.
- New tests: dev\tests\test_guided.py, dev\gui_trace_test.py (mouse clicks and double click in the picture, Esc,
  Process while tracing). The automatic results are byte for byte those of 1.1.26.

Changes in 1.1.26
- Extracted plot after Process again: it was not redrawn (it kept the curves of the run before), and from then on
  ticking or unticking a series did not change it either. Cause: the picture is reloaded at each Process, which
  deletes the drawn traces, and the program then failed removing them a second time. Fixed; new test
  dev\gui_reprocess_test.py.
- Solid and dash-dot curves of the same colour, at other picture sizes: 1.1.25 separated them on test\twins_abs.png
  as captured (932 x 671 px) but not on the same plot at another size. At 1.25 times (a screenshot with Windows
  display scaling at 125 %) it gave 12 series with the pairs mixed again. Changes:
  the solid curve of such a pair may cross a gap only where ink covers it (it has no gaps of its own; before it
  jumped onto dashes across the white background); runs of dashes merged into the solid curve and branching off it
  are dropped from the solid curve and given back to the dashes; two pieces of one curve that end in the same point
  (also at a sharp peak) are joined; two dashed pieces are joined along their solid twin where the dashes ran into
  it, also when the dashed end is still up to 6 % of the plot diagonal off the solid curve but closing in on it; two
  pieces of one colour whose ends face each other a few pixels apart in x are joined when the way between lies on
  ink. The U-turn rule of 1.1.25 now also holds when one of the two ends is set back (both pieces on the same side).
- Results (dev\tests\run_twins_sizes.py, share of each true curve traced by one series; 1.1.25 -> 1.1.26):
  12 synthetic pairs plots 0.868 -> 0.921; test\twins_abs.png at 0.8 / 1 / 1.25 / 1.5 / 2 times
  0.848 / 1.000 / 0.843 / 0.734 / 0.064 -> 0.844 / 0.998 / 0.998 / 0.795 / 0.064. test_twins.py at 1.25 times:
  PASS (1.1.25: FAIL).
- Still open: at 0.8 times the black and blue pairs are cut where the dash-dot crosses the solid curve (a run of
  merged dashes is taken as a branch of the solid curve); at 1.5 and 2 times (enlarged, so blurred, pictures
  that are analysed reduced) the black dash-dot curve is lost, and at 2 times the Y axis is read as logarithmic.
- The single-file exe (app_onefile) is only as new as its last build_onefile.bat run; check the version in the
  About tab.

Changes in 1.1.25
- A solid and a dash-dot (or dashed) curve of the same colour that run close together are no longer traced into
  each other. Absorption plot of P3HT, PM6 and D18, neat (dash-dot) and blended with Y6 (solid), test\twins_abs.png:
  1.1.24 gave 8 series, among them both black curves as one loop, a U-turn from the red blend at the right frame
  into the PM6 tail, and the D18 blend running back along the D18 curve; 1.1.25 gives 7 series, one per curve.
  How: in such a colour the long strokes and the short dashes are traced in two separate passes, so a dash is never
  linked into the solid curve; thick spots where a dash merges into the solid curve are no longer cut out as
  symbols; where the dashes run inside the solid curve (PM6 at 624-653 nm), the dash-dot curve is joined along
  the solid one.
- Two curve ends that leave the frame side by side (both going out to the right) are no longer joined into a U-turn.
- Left: the blue D18 curve starts at 350 nm on the D18:Y6 curve (0.37 instead of 0.25) up to about 380 nm, where
  all seven curves meet; the D18:Y6 curve is traced from 417 to 938 nm and the P3HT curve from 407 nm.
- Tab Settings: the text of "analyse blurred pictures at full size" says that a picture which was not reduced
  gives the same result either way (this picture was not reduced: the setting changes nothing for it).
- New test: dev\tests\test_twins.py -> TWINS: PASS (1.1.24: FAIL).

Changes in 1.1.24
- SVG files are listed in the import dialog. They were already read as vector figures (through PyMuPDF), but the
  dialog did not show them. MuPDF drops the dash pattern of SVG strokes, so dashed curves are named "solid";
  the data are not affected.

Changes in 1.1.23
- A dashed curve that crosses a solid curve of the same colour is no longer cut there. Chen et al., Appl. Phys.
  Lett. 2015, Fig. 2a: the black dashed k curve came out in two pieces split at 419 nm, and the first one
  (200-419 nm, with the peak) was not ticked; now it is one series from 200 to 1661 nm. Cause: where a dash
  merges into the crossing line, the traced centre line had a small closed loop around a hole in the ink, and the
  dashed curve was linked to that loop; such loops are now dropped.
- Same cause, Rodkey 2024 SI Fig. S11: the forward (dash-dot) trace ends at 0.873 V instead of running on along
  the reverse curve to 0.90 V.

Changes in 1.1.22
- A curve whose turn at a dip or peak is hidden under another curve is joined again: the piece that goes on
  points straight back at the end of the first one. Rodkey 2024 SI Fig. S4b at full size: the black curve
  goes on after its minimum (to 836 nm, before it stopped at 825 nm) and the blue curve is one piece.
- Fig. S12a: the red LED curve is followed through its 392 nm peak and 405 nm dip.

Changes in 1.1.21
- Setting "analyse blurred pictures at full size" (tab Settings, off by default). Blurred or enlarged
  pictures are analysed at reduced size, which keeps lines whole in enlarged screenshots but can cut narrow
  peaks and mix close curves. Rodkey 2024 SI Fig. S4b at full size: both grey peaks are traced and the light
  grey runs to about 810 nm (reduced: it stops at 775 nm and the mid grey peak is notched). No measure found
  tells such a picture from an enlarged one, so the choice is left to the user; Messages says when a picture
  was reduced.

Changes in 1.1.20
- Two curves of similar colour running a few pixels apart are no longer mixed: the colour of a dark curve
  does not spread into a lighter curve next to it (Rodkey 2024 SI Fig. S4b: between 375 and 395 nm the
  black trace ran on the dark grey curve, up to 1.2 off; now within 0.3).
- S4b: the blue curve is followed from 363 nm, with its peak at 375 nm (before: from 395 nm, where the
  black curve crosses it).
- A light pixel inside the first symbol of a curve no longer cuts off the start of the trace.
- test\nk_two_axes.png: the black n curve is ticked over its whole length, in two pieces (1.1.19 stopped
  at 855 nm and added 4 false triangle markers). The arrow pointing at the curves is still traced into it.

Changes in 1.1.19
- Specks left by the tick marks are removed. A row of them along the frame was traced as a dotted line and
  joined to curves ending at the frame (Rodkey 2024 SI Fig. S4b: a trace around the frame and down to the
  axis; Fig. S12b: the black curve pulled down the left frame from 65 to 60 %).
- A light grey and a mid grey curve in one plot are told apart (S4b: two LiF curves were one colour).
- Narrow peaks and dips: the trace goes through the tip instead of being cut flat or broken there (S4b blue
  peak 23.1 instead of 22.3, black curve now whole to 827 nm).
- Thick spots on a line are reported as symbols only if there are at least 5 of them, about evenly spaced:
  no more false symbols where curves cross or at peaks (S4b: 9, S12a: 44, S9b: 3).

Changes in 1.1.18
- Where two curves cross, a trace no longer makes a small S or loop (it doubled back in x for a few pixels);
  the doubled stretch is bridged by the curve through its neighbours. Kerremans 2020 SI Fig. S9: the light
  grey n curve at 605 nm and the dark grey one at 750 nm.

Changes in 1.1.17
- Lines with small symbols packed closely along them (e.g. n,k plots with a symbol every few nm, light
  symbols on a pale line): traced as one line through the symbols. Before, the symbols on peaks and steep
  parts were cut out as separate markers, and the line came out in pieces, some with a jump at the end.

Changes in 1.1.16
- Single-file exe: build_onefile.bat builds app_onefile\XY_Digitizer.exe, which holds the program, all
  libraries, the OCR models, the source code and the license texts. Tab About, button "Save source code
  and licenses" copies the packed source and licenses to a folder (without interface:
  XY_Digitizer.exe --save-bundled FOLDER). The folder build (build_exe.bat) is unchanged.

Changes in 1.1.15
- Axis titles next to the tick labels (e.g. "Reflectance (%)") are no longer read as part of a label; in
  1.1.14 a title letter next to "50" gave a y axis up to 1e15.
- A small plot inside the plot (inset with its own frame, labels and title) is ignored. To digitize the
  inset, select it alone.
- Traces that run along the frame (thick or blurred frame lines) are dropped.

Changes in 1.1.14
- Renaming a series by double click (or F2) works; in 1.1.13 the table did not open the editor. Only the
  name can be edited.

Changes in 1.1.13
- Series names can be edited in the series table (double click): the name heads the series' columns in
  the CSV / XLSX export (NAME_x, NAME_y) and appears in the plot legend. Commas, semicolons, quotes, tabs
  and line breaks become spaces; an empty name goes back to S<n>; a name used twice gets " (2)".
- The Messages box is gone; its text opens in a window with the button "Messages" below the series table.
  The axes and series boxes share the width.

Changes in 1.1.12
- Traced lines in pictures no longer wiggle. Solid lines: light smoothing along the curve (Gaussian, 0.6 x
  the stroke width), since the centre of a stroke scatters by a fraction of a pixel from point to point.
  Dotted curves: one point per dot (its centre) and a smooth curve through the centres. Rodkey 2024
  Fig. 3a, steep part: wiggle 0.66 -> 0.30 px rms (black), 0.61 -> 0.23 px (blue dots); every dot of the
  blue curve within 2 px of the trace (except the two touching the frame). IQE test: rms 0.15 -> 0.10 %.
- Lines carry about one point per pixel of their length (more points in the export).

Changes in 1.1.11
- Freeze when zooming the imported plot: the wheel zoom had no limit (scale x1.25 per step without end,
  until Qt's scroll range overflowed). Now between half the fitted size and 32 screen pixels per picture
  pixel; above 3x the pixels are drawn plain (faster, sharper).
- Dotted curves: dots at a sharp turn (top of a peak) were left out and the trace passed under the peak
  (Rodkey 2024 Fig. 3a, 600 nm: 10 px below the top dot). Such dots are now put into the gap.
- A line covered at its edge by symbols of another colour (blue dots over the black EQE curve) is filled
  where they cover it, so its trace stays on the line (1.1.10: 79 of 120 trace points in that stretch lay
  beside the black line, now 20 of 121).
- Legend labels with Greek letters ("1 um" written with a micro sign) are recognised: their line samples
  are removed (the "1 um" sample was a ticked series).
- Filled boxes (the yellow background of the "Monofacial" label) and patches are ignored (3 junk series).

Changes in 1.1.10
- Tab "2. Digitize": the axes box (4) is now under the imported plot and the series box (5) under the
  extracted plot.
- A curve running just under (or over) a text label is no longer erased with the label: flat pieces in the
  top or bottom margin of the text box stay (Fig. 1a of Rodkey et al., Adv. Energy Mater. 2024: the
  dash-dot-dot "Optimal Index" curve was missing; now 371 to 811 nm).
- Inward tick marks as thick as the frame line are removed (Fig. 1b: the ticks along the top and bottom
  were traced as dotted lines and the curves ran up the frame).
- Dashes at the start or end of a broken curve that did not link are appended when they lie straight ahead.
- Not changed: the short leader line from a label that touches a curve still gives a small spike on it
  (removing narrow spikes would also remove real narrow peaks).

Changes in 1.1.9
- Straight lines that run from one edge of the frame to another are data, not guides (power laws f^n cut
  off by the frame: Figs. 12 and 13 of the Zurich Instruments PLL white paper; 3 of 5 lines were unticked).
  Vertical lines across the frame (reference marks) stay unticked.
- Button "Recent folder" after "Import from file": opens the folder of the last picture or PDF loaded
  (remembered between sessions). "Import from file" starts in the output folder.
- About, AI assistance: the main recognition methods in five lines.

Changes in 1.1.8 (pictures: crossing solid and dashed curves, Fig. 5 of Tabernig et al., ACS Nano 2022)
- Labels written along a curve ("planar", "patterned"), in the colour of the curve: only the letters are
  erased now, inside the turned text box. Before, everything of that colour inside an upright box around
  the word was erased, which cut the curves into pieces (the EQE curves were lost from 950 to 1100 nm).
- Short dashes (as long as about twice their thickness) were read as scatter symbols, or as symbols sitting
  on a line ("+" marks in the overlay). Dashes stretched along the row they lie in are now line pieces;
  their direction is taken from their shape, so the gaps are bridged along the curve.
- Darker centres of dashes have a colour between orange and purple and were dropped as "mixed colour";
  a shade that touches only one curve now belongs to that curve.
- A line is "solid" when ink covers at least 80 % of its length.
- Markers of a line that is not ticked (dashes of an arrow) are not ticked either.
- Result on that figure: 4 ticked series for the 4 curves, each from about 310 to 1195 nm.

Changes in 1.1.7
- Every PDF failed to open in the exe built after 2026-10-18 ("No such file or directory: ...pymupdf\layout\
  resources\onnx\layout_rf2.4.1+imf1.yaml"). The build updates PyMuPDF; version 2.0 loads its layout package
  (ONNX models) whenever PyMuPDF is imported, and PyInstaller had not copied the model files. The build now
  leaves the layout package out (not used here; the program folder is about 75 MB smaller).
- The build test also opens a PDF (test\pdf_check.pdf): line "PDF TEST: PASS" in test\compare.txt.
- PDF regions: in 1.1.6 a figure with stacked panels and an inset could lose a panel (with the DejaVu Sans
  font of matplotlib: 3 regions for 4 plots). All plots of the page now compete for the labels, and a label
  is given to one plot only.

Changes in 1.1.6
- PDF figures: panels in rows AND columns are now offered separately (a 2 x 2 figure gave only the right
  column, e.g. Fig. 3 of Ndione et al., J. Mater. Chem. C, page 11 of the user's PDF: now 4 regions, each
  with its tick labels and axis title). Glyph outlines inside a figure count as its labels.
- Log axes: an exponent printed nearly as large as the 10 (10^5) is read as the exponent. Before, the 5 was
  taken as another tick label or as a third digit (105). The exponent is told by its raised baseline.
- Pick (manual axes): cross-hair pointer until a double click; clicking again moves the position.
- Window and taskbar icon: loaded from xyd.png (no Qt image plugin needed) and set through Windows as well.
- The buttons of row 1 stay packed to the left before and after an import.
- The title row is gone (the name is in the window title); Theme and Text are at the right of the tab row.

Changes in 1.1.5
- License: GNU AGPL-3.0 (was non-commercial), so the program may be passed on freely together with PyMuPDF
  (AGPL-3.0). The build ships the source code next to the exe and leaves out OpenCV's video plugin
  (FFmpeg), which the program does not use. ONNX Runtime's license files are now collected too.
- Program icon (exe, window and taskbar): three neon curves on a dark red tile with white axes.
- Curves hidden under another curve: the hidden stretch may now follow that curve (also when it bends),
  and pieces that touch at a bend, or that overlap at their ends, are joined. Pieces of a similar colour
  that run side by side over the same x range are still never joined, and a join may not turn back.

Changes in 1.1.4
- A semi-transparent curve that comes out darker where it lies over other curves (e.g. a light orange
  curve over grey and blue ones) was split into several series, and a bend hidden under another curve
  broke it. Pieces of one hue at a similar strength are now joined when they continue each other, when
  one piece runs on from the end of another, or when both ends point at one corner that lies under a
  curve of another colour. Pieces of a similar colour that run side by side over the same x range are
  never joined (two shades of blue stay two curves).

Changes in 1.1.3
- Tab "1. Settings": checkbox "interpolate x-values" (on by default). Export then writes one x column shared by
  all ticked lines, and each line is interpolated onto it (linear, in plot coordinates, so log axes are
  interpolated in decades). Step: about one value per pixel of the plot, rounded down to 1, 2, 2.5 or 5 x 10^n
  (in decades for a log x axis). A line is blank outside its own x range (no extrapolation). Symbol series and
  lines that pass the same x twice at different y (loops, a trace that ran into an annotation) keep their own
  x and y columns. The XLSX sheet "info" says how the x values were made.
- The build test checks both export layouts (test\result.csv and test\result_common.csv).

Changes in 1.1.2
- PDF pictures stored as several strips are joined into one figure (a single plot was offered in pieces).
- Several plots in one picture (side by side or stacked) are offered as separate regions.
- Flat data lines at tick values are no longer removed as grid lines or as legend samples.
- A few thick spots where curves cross are no longer taken as symbols of a line.

Changes in 1.1.1
- Stacked panels and insets in PDFs as separate regions; shared x labels; inset curves left out of the outer plot.
- Extracted plot: wheel zoom and scroll bars instead of the matplotlib toolbar.

Changes in 1.1.0
- Right-hand Y axis is read; dialog and column "y axis" to assign series to the left or right axis; manual row "Y right".
- Axis scale types linear, log, ln and 1/x (tick spacing).
- PDF import: proposed figure regions hold the plot with its tick labels and axis titles, without paragraph text or caption.
- PDF import: stacked panels and insets are offered as separate regions; a panel without x labels takes those of the
  panel below with the same x axis; the curves of an inset are not read as data of the plot around it.
- The extracted plot zooms with the mouse wheel and scrolls like the imported picture (the matplotlib toolbar is gone).
- Vector figures: tick labels written as 3 x 10^-3 are read as one number; a label at a frame corner is placed on the frame.
- Pictures: dashes are no longer taken as symbols (a symbol must be thicker than the lines of its colour); steep
  anti-aliased dashes are traced (they were lost in thinning); only the legend sample is erased left of a legend
  label, not everything in that zone; legend numbers and single letters are removed as text; mixed colours where
  two curves touch are ignored; pieces of a curve are joined across parts hidden by another curve; a solid line
  interrupted by other curves is named solid; small pieces (ellipses, arrows) are listed but not ticked.
Development files: dev\HANDOFF.md, dev\tests (regression battery, two-axis test plot, PDF region test).
