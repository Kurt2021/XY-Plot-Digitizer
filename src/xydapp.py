"""
XY Plot Digitizer: desktop app (Qt / PySide6 + matplotlib).
Same look and options as IV Analysis: dark and light theme with the IBM colour-blind safe palette
(xydtheme.py), text size, table font size and output folder, remembered between sessions.
Extraction is in xydcore.py.

Without the interface:   XY_Digitizer.exe --digitize PICTURE_OR_PDF --out BASENAME
writes BASENAME.csv, BASENAME.xlsx, BASENAME_overlay.png and BASENAME_log.txt.
Single-file exe only:    XY_Digitizer.exe --save-bundled FOLDER   copies the source code and licenses packed in it.
"""
import json
import os
import re
import shutil
import sys
import traceback

if sys.stdout is None:                     # windowed exe: no console; libraries that print must not fail
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QPushButton, QLineEdit, QComboBox, QSpinBox, QCheckBox, QGroupBox, QTabWidget,
                               QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog,
                               QMessageBox, QTextBrowser, QPlainTextEdit, QSizePolicy, QSplitter, QGraphicsView,
                               QGraphicsScene, QDialog, QDialogButtonBox, QProgressDialog)
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import xydcore as C
import xydtheme as TH

APP_NAME = "XY Plot Digitizer"
APP_VERSION = "1.1.34"
APP_YEAR = "2026"
APP_AUTHOR = "Kurt Taretto"
DEFAULT_OUT = r"D:\Resources\Software\My Digitizer"
FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
TABS = [("howto", "How to use"), ("settings", "1. Settings"), ("digitize", "2. Digitize"), ("about", "About")]
AXIS_NAME = {"x": "X", "y": "Y", "y2": "Y right"}
SCALES = ["linear", "log", "ln", "1/x"]                    # tick spacing; ln is the same spacing as log
MPL_MARKER = {"circle": "o", "square": "s", "triangle up": "^", "triangle down": "v", "triangle left": "<",
              "triangle right": ">", "diamond": "D", "plus": "+", "cross": "x", "star": "*", "dot": ".",
              "pentagon": "p", "hexagon": "h", "polygon": "o"}
MPL_LINE = {"solid": "-", "dashed": "--", "dotted": ":", "dash-dot": "-."}

HOWTO = """WORKFLOW
1. Settings: symbol position, font size of the tables, interpolate x-values, and the output folder.
2. Digitize:
   - Import from clipboard (or Ctrl+V): a copied picture or a screenshot of the plot.
   - Import from file: png, jpg, bmp, tif, pdf, svg, eps, ps. For a PDF choose the page, then click an outlined
     figure or drag a rectangle around the plot. The dialog says whether the chosen region is vector graphics
     (the data are then read from the drawing paths, exact) or an embedded picture (image analysis).
     A figure made of stacked panels, or with an inset, shows one outline per panel and per inset; panels
     without x labels of their own take them from the panel below when both share the x axis.
   - Recent folder: the same, starting in the folder of the last picture or PDF loaded (remembered).
   - Process: a progress bar shows the steps. The axes type (linear or log) and range are read from the
     tick labels. If the plot has a second Y axis on the right with another scale, a dialog asks which
     axis the curves belong to; single series are switched by clicking their cell in the column 'y axis'.
   - Compare the imported plot (left, extracted points drawn over it) with the extracted plot (right).
   - Data series: untick what is not data. Double click a name to rename it: the name heads the series'
     columns in the export (commas and quotes become spaces). 'Messages' opens what the analysis found.
   - Trace a curve (guided trace, for a curve the automatic tracing mixed up, e.g. a solid and a dashed
     curve of the same colour crossing each other): click points along ONE curve in the imported picture,
     from one end to the other: both ends, every peak and dip, and on both sides of each place where it
     crosses or runs into a curve of the same colour; double click to finish, Esc cancels. The trace follows
     the ink of the clicked colour between the points and never turns back in x (for a curve that does, click
     at each turning point). It is added as a new series; untick the automatic series it replaces.
     Short straight lines beside longer series (tangents, reference marks) are listed but not ticked.
   - Axes: if a value is wrong, type the right one and press Apply axes. 'Pick' turns the pointer into a
     cross-hair: click a position in the picture (click again to move it), double click to finish
     (pixel 1 / pixel 2 are the two reference positions of that axis). Scale is the spacing
     of the ticks: linear, log, ln (the same spacing as log) or 1/x (reciprocal). The exported values are
     the printed ones. Fill in the row 'Y right' to define a right-hand axis by hand; empty it to drop it.
   - Export: CSV or XLSX. With 'interpolate x-values' (Settings, on by default) the lines share one x column
     (x, S1_y, S2_y, ...), each interpolated onto it; symbols keep their own S_x, S_y columns. Without it,
     one x and one y column per ticked series (S1_x, S1_y, S2_x, ...). The XLSX has a second sheet with the
     series list, the axes and how the x values were made.

WHAT IS RECOGNISED
Lines: solid, dashed, dotted, dash-dot; the centre of the line is traced, gaps of broken lines are filled by
interpolation. Curves are separated by colour and by line type; crossings are followed straight through.
Symbols: circle, square, triangle, diamond, plus, cross, star, filled or open; symbols that touch each other
or a line are separated. Symbols on a line are listed as their own series, next to the line.
A symbol must be clearly thicker than the lines of its colour; a piece as thick as the line is a dash.
Removed before tracing: grid lines, tick marks, legend boxes, text, legend samples.
A curve that disappears under another curve is joined across the hidden part when both ends point at
each other; the note of the series says so. Small pieces (ellipses, arrows) are listed but not ticked.

LIMITS
- One plot per picture. A top X axis, broken axes and rotated scans are not handled.
- Best input: the PDF of the paper (vector figure), or a PNG screenshot at 100 % zoom or more.
  JPEG, enlarged or blurred pictures: the axes are still read, but lines can come out in pieces.
  Blurred pictures are analysed at reduced size; if narrow peaks are cut or close curves mixed, tick
  'analyse blurred pictures at full size' (tab Settings) and press Process again.
- Symbols smaller than about 8 pixels cannot be named reliably (circle, square or triangle).
- Symbol position 'centroid' or 'bounding-box centre' differ only for triangles.
- EPS / PS files need Ghostscript installed.
- Curves that lie on top of each other over most of their length cannot be told apart in a picture:
  only the one drawn last is found. Use the PDF of the figure when it is vector graphics.
- Always check the overlay on the imported plot before using the numbers.

THE PICTURE AND THE EXTRACTED PLOT
Mouse wheel: zoom in or out around the pointer; scroll bars move the view. Double click: fit to the box.
"""


def to_float(s, default=None):
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return default


def config_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "XY_Digitizer")


def bundled_dir():
    """Single-file exe: the folder (unpacked at start) with the source code and the license texts; else None."""
    d = os.path.join(getattr(sys, "_MEIPASS", ""), "bundled")
    return d if getattr(sys, "frozen", False) and os.path.isdir(d) else None


def save_bundled(dest):
    """Copy the source code and the license texts packed in the single-file exe to the folder dest."""
    shutil.copytree(bundled_dir(), dest, dirs_exist_ok=True)
    return dest


def load_config():
    try:
        with open(os.path.join(config_dir(), "config.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_config(cfg):
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(os.path.join(config_dir(), "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
    except Exception:
        pass


def setup_matplotlib_fonts():
    have = set(font_manager.get_font_names())
    matplotlib.rcParams["font.family"] = [f for f in ("Segoe UI", "DejaVu Sans") if f in have] or ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = True


# ------------------------------------------------------------------ small widgets
class _NoWheel:
    """The mouse wheel does not change a value by accident."""

    def wheelEvent(self, e):
        e.ignore()


class Combo(_NoWheel, QComboBox):
    pass


class Spin(_NoWheel, QSpinBox):
    pass


class Canvas(FigureCanvasQTAgg):
    """Matplotlib canvas whose mouse wheel and double click go to the PlotView around it."""
    view = None

    def wheelEvent(self, e):
        if self.view is not None:
            self.view.zoom(e)

    def mouseDoubleClickEvent(self, e):
        if self.view is not None:
            self.view.fit()


class PlotView(QScrollArea):
    """The extracted plot, handled like the imported picture: the mouse wheel zooms in and out around
    the pointer (scroll bars appear), a double click fits the plot to the box again."""

    def __init__(self, canvas):
        super().__init__()
        self.canvas, self.f = canvas, 1.0
        canvas.view = self
        self.setWidget(canvas)
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.NoFrame)

    def _size(self):
        v = self.viewport().size()
        return QtCore.QSize(int(v.width() * self.f), int(v.height() * self.f))

    def fit(self):
        self.f = 1.0
        self.canvas.resize(self._size())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.canvas.resize(self._size())

    def zoom(self, e):
        f = min(6.0, max(1.0, self.f * (1.25 if e.angleDelta().y() > 0 else 0.8)))
        if f == self.f:
            return
        pos = e.position()                                 # keep the point under the pointer in place
        fx, fy = pos.x() / max(1, self.canvas.width()), pos.y() / max(1, self.canvas.height())
        at = self.canvas.mapTo(self.viewport(), pos.toPoint())
        self.f = f
        self.canvas.resize(self._size())
        self.horizontalScrollBar().setValue(int(fx * self.canvas.width() - at.x()))
        self.verticalScrollBar().setValue(int(fy * self.canvas.height() - at.y()))


class Page(QScrollArea):
    """Vertically scrolling tab page; content goes into .lay. With fit=True the content is shrunk to the
    window down to its minimum size before a scroll bar appears (pages built around plots)."""

    def __init__(self, fit=False):
        super().__init__()
        self.fit = fit
        self.setWidgetResizable(not fit)
        self.setFrameShape(QFrame.NoFrame)
        self.inner = QWidget()
        self.inner.setObjectName("page")
        self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(16, 12, 16, 14)
        self.lay.setSpacing(10)
        self.setWidget(self.inner)
        self.verticalScrollBar().setSingleStep(30)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit:
            m = self.inner.minimumSizeHint()
            self.inner.resize(max(self.viewport().width(), m.width()), max(self.viewport().height(), m.height()))


def hrow(*widgets, spacing=10, stretch=True):
    """Widgets side by side, packed to the left."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if isinstance(w, int):
            lay.addSpacing(w)
        elif w is not None:
            lay.addWidget(w, 0, Qt.AlignVCenter)
    if stretch:
        lay.addStretch(1)
    return lay


class ImageView(QGraphicsView):
    """Zoomable picture. Emits clicked(x, y) in picture pixels and dragged(rect). In cross mode the pointer
    is a cross-hair and a double click emits finished() instead of fitting the picture."""
    clicked = QtCore.Signal(float, float)
    dragged = QtCore.Signal(QtCore.QRectF)
    finished = QtCore.Signal()

    def __init__(self, rubber=False):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.has_image, self._press, self._band, self._zoomed = False, None, None, False
        self.cross = False
        if rubber:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.rubberBandChanged.connect(self._on_band)

    def set_cross(self, on):
        self.cross = bool(on)
        if self.cross:
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.cross and self.viewport().cursor().shape() != Qt.CrossCursor:   # the view may reset it
            self.viewport().setCursor(Qt.CrossCursor)

    def set_image(self, arr):
        self.scene().clear()
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(img))
        self.scene().setSceneRect(0, 0, w, h)
        self.has_image, self._zoomed = True, False
        self.fit()

    def fit(self):
        if self.has_image and not self._zoomed:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self.setRenderHint(QPainter.SmoothPixmapTransform, self.transform().m11() < 3.0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()

    def showEvent(self, e):
        super().showEvent(e)
        self.fit()

    def wheelEvent(self, e):
        """Zoom between half the fitted size and 32 screen pixels per picture pixel. Without a limit the
        scale grows without end (1.25 per step) until Qt's scroll range overflows and painting stalls."""
        if not self.has_image or e.angleDelta().y() == 0:
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        r = self.scene().sceneRect()
        fit = min(self.viewport().width() / max(r.width(), 1.0), self.viewport().height() / max(r.height(), 1.0))
        cur = self.transform().m11()
        new = min(32.0, max(0.5 * fit, cur * f))
        if abs(new - cur) < 1e-9:
            return
        self._zoomed = True
        self.scale(new / cur, new / cur)
        self.setRenderHint(QPainter.SmoothPixmapTransform, new < 3.0)   # large: plain pixels (faster, sharper)

    def mouseDoubleClickEvent(self, e):
        if self.cross:
            self.set_cross(False)
            self.finished.emit()
            return
        self._zoomed = False
        self.fit()

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        p = e.position().toPoint()
        if self._press is not None and (p - self._press).manhattanLength() < 4:
            q = self.mapToScene(p)
            self.clicked.emit(q.x(), q.y())
        elif self._band is not None:
            self.dragged.emit(self._band)
        self._press = self._band = None

    def _on_band(self, rect, a, b):
        if not rect.isNull():
            self._band = QtCore.QRectF(a, b).normalized()


class FigurePicker(QDialog):
    """Choose the figure on a PDF page; reports whether the chosen region is vector graphics."""

    def __init__(self, app, doc, title):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Choose the plot: %s" % title)
        self.resize(940, 800)
        self.doc, self.zoom, self.clip, self.result, self.cache = doc, 2.0, None, None, {}
        lay = QVBoxLayout(self)
        self.spin = Spin()
        self.spin.setRange(1, len(doc))
        self.spin.setFixedWidth(80)
        whole = app.button("Whole page", lambda: self.choose(*self.page.rect))
        top = hrow(QLabel("Page"), self.spin,
                   QLabel("of %d.   Click an outlined figure, or drag a rectangle around the plot." % len(doc)), stretch=False)
        top.addStretch(1)
        top.addWidget(whole)
        lay.addLayout(top)
        self.view = ImageView(rubber=True)
        lay.addWidget(self.view, 1)
        self.status = QLabel("No region chosen.")
        self.status.setWordWrap(True)
        self.status.setFont(app.f_bold)
        lay.addWidget(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setProperty("kind", "accent")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.spin.valueChanged.connect(self.show_page)
        self.view.clicked.connect(self.on_click)
        self.view.dragged.connect(lambda r: self.choose(r.x() / self.zoom, r.y() / self.zoom, r.right() / self.zoom, r.bottom() / self.zoom))
        self.show_page(1)

    def say(self, text, role):
        self.status.setText(text)
        self.status.setStyleSheet("color: %s;" % self.app.tok(role))

    def show_page(self, number):
        self.page = self.doc[number - 1]
        if number not in self.cache:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.page.remove_rotation()
                dr = self.page.get_drawings()
                self.cache[number] = (dr, C.figure_candidates(self.page, dr))
            finally:
                QApplication.restoreOverrideCursor()
        self.drawings, self.cands = self.cache[number]
        img, _ = C.render_region(self.page, self.page.rect, self.zoom)
        self.view.set_image(img)
        for r, kind in self.cands:
            pen = QPen(QColor(TH.IBM["blue"] if kind == "vector" else TH.IBM["orange"]), 2)
            pen.setCosmetic(True)
            self.view.scene().addRect(r.x0 * self.zoom, r.y0 * self.zoom, r.width * self.zoom, r.height * self.zoom, pen)
        self.sel_item, self.clip = None, None
        if len(self.cands) == 1:
            self.choose(*self.cands[0][0])
        elif not self.cands:
            self.choose(*self.page.rect)
        else:
            self.say("%d figures found on this page (blue: vector drawing, orange: embedded picture). Choose one." % len(self.cands), "text")
            self.ok.setEnabled(False)

    def on_click(self, x, y):
        import pymupdf
        p = pymupdf.Point(x / self.zoom, y / self.zoom)
        hit = sorted((r for r, _ in self.cands if p in r), key=lambda r: r.get_area())
        if hit:
            self.choose(*hit[0])

    def choose(self, x0, y0, x1, y1):
        import pymupdf
        clip = pymupdf.Rect(x0, y0, x1, y1) & self.page.rect
        if clip.width < 20 or clip.height < 20:
            return
        self.clip = clip
        if self.sel_item is not None:
            self.view.scene().removeItem(self.sel_item)
        pen = QPen(QColor(TH.IBM["magenta"]), 2, Qt.DashLine)
        pen.setCosmetic(True)
        self.sel_item = self.view.scene().addRect(clip.x0 * self.zoom, clip.y0 * self.zoom, clip.width * self.zoom, clip.height * self.zoom, pen)
        info = C.region_info(self.page, clip, self.drawings)
        if info["vector"]:
            self.say("Chosen region is VECTOR graphics (%d paths, %d segments): the data will be read from the paths "
                     "themselves." % (info["paths"], info["items"]), "ok")
        elif info["image_fraction"] >= 0.5:
            self.say("Chosen region is an embedded PICTURE (about %.0f px/inch), not vector graphics: it will be analysed "
                     "as an image." % info["image_ppi"], "warn")
        else:
            self.say("Chosen region has almost no vector paths (%d segments): it will be analysed as an image." % info["items"], "warn")
        self.ok.setEnabled(True)

    def accept(self):
        if self.clip is not None:
            self.result = C.pdf_source(self.page, self.clip, self.drawings)
            self.result["label"] = self.status.text()
        super().accept()


class Worker(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, src, full=False):
        super().__init__()
        self.src, self.full = src, full

    def run(self):
        try:
            self.done.emit(C.run_source(self.src, lambda p, m="": self.progress.emit(int(p), m), self.full))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ------------------------------------------------------------------ the app
class App(QMainWindow):
    def __init__(self, qapp):
        super().__init__()
        self.qapp = qapp
        cfg = load_config()
        self.theme = cfg.get("theme", "dark") if cfg.get("theme", "dark") in TH.THEMES else "dark"
        self.textpt = int(min(16, max(8, to_float(cfg.get("textpt"), 10) or 10)))
        self.tablefont = int(min(40, max(8, to_float(cfg.get("tablefont"), 11) or 11)))
        self.marker_ref = cfg.get("marker_ref", "centroid") if cfg.get("marker_ref") in ("centroid", "bbox") else "centroid"
        self.show_overlay = bool(cfg.get("overlay", True))
        self.common_x = bool(cfg.get("common_x", True))
        self.full_res = bool(cfg.get("full_res", False))
        out = cfg.get("outdir") or DEFAULT_OUT
        self.outdir = out if os.path.isdir(out) else os.path.expanduser("~")
        last = cfg.get("lastdir") or ""
        self.lastdir = last if os.path.isdir(last) else ""     # folder of the last picture or PDF loaded
        self.setWindowTitle("%s %s" % (APP_NAME, APP_VERSION))
        self.setMinimumSize(980, 600)
        setup_matplotlib_fonts()

        # ---- state (survives a rebuild of the interface)
        self.src, self.res, self.src_label = None, None, ""
        self.log_text, self.status_state = "", None
        self.pick, self.docs, self.guides, self.overlay_items = None, [], {}, []
        self.trace, self.trace_items = None, []             # guided trace: clicked points while it is running
        self.auto = False                                  # True: no dialogs (build check)
        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.timeout.connect(lambda: self.build_ui(self.current_tab()))

        self.apply_theme()
        self.build_ui("digitize")
        QShortcut(QKeySequence.Paste, self, activated=self.paste_key)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.cancel_trace)
        avail = self.screen().availableGeometry()
        if avail.width() < 1460 or avail.height() < 960:
            self.resize(min(1420, avail.width() - 40), min(920, avail.height() - 60))
            self._maximize = True
        else:
            self.resize(1420, 920)
            self._maximize = False

    # ================================================================ generic helpers (as in IV Analysis)
    def tok(self, key):
        return TH.tokens(self.theme)[key]

    def apply_theme(self):
        f = QFont(FONT, self.textpt)
        self.qapp.setFont(f)
        self.f_bold = QFont(f)
        self.f_bold.setBold(True)
        self.f_small = QFont(FONT, max(7, self.textpt - 1))
        self.f_table = QFont(FONT, self.tablefont)
        self.f_table_b = QFont(self.f_table)
        self.f_table_b.setBold(True)
        icons = TH.make_icons(os.path.join(config_dir(), "theme"), self.theme)
        self.qapp.setPalette(TH.palette(self.theme))
        self.qapp.setStyleSheet(TH.stylesheet(self.theme, icons))

    def save_cfg(self):
        save_config(dict(theme=self.theme, textpt=self.textpt, tablefont=self.tablefont, marker_ref=self.marker_ref,
                         overlay=self.show_overlay, outdir=self.outdir, common_x=self.common_x,
                         full_res=self.full_res, lastdir=self.lastdir))

    def rebuild_later(self):
        """Rebuild the interface once the widget that asked for it has finished handling its event."""
        self.rebuild_timer.start(0)

    def set_theme(self, name):
        if name != self.theme:
            self.theme = name
            self.save_cfg()
            self.apply_theme()
            self.rebuild_later()

    def set_textpt(self, pt):
        if pt != self.textpt:
            self.textpt = int(pt)
            self.save_cfg()
            self.apply_theme()
            self.rebuild_later()

    def set_tablefont(self, pt):
        if pt != self.tablefont:
            self.tablefont = int(pt)
            self.save_cfg()
            self.apply_theme()
            self.rebuild_later()

    def lab(self, text, role=None, wrap=False, bold=False, small=False):
        w = QLabel(text)
        if wrap:
            w.setWordWrap(True)
        if bold:
            w.setFont(self.f_bold)
        elif small:
            w.setFont(self.f_small)
        if role:
            w.setStyleSheet("color: %s;" % self.tok(role))
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return w

    def msg(self, label, text, role="text", status=False):
        label.setText(text)
        label.setStyleSheet("color: %s;" % self.tok(role))
        label.setVisible(bool(text))
        if status and text:
            self.say(text, role)

    def say(self, text, role="text"):
        """Status line at the bottom of the window."""
        self.statusline.setText(text.replace("\n", "   "))
        self.statusline.setStyleSheet("color: %s;" % self.tok(role))

    def set_status(self, text, role):
        self.status_state = (text, role)
        self.msg(self.status, text, role, status=True)

    def button(self, text, cmd, kind=None, tip=None, minw=0):
        b = QPushButton(text)
        if kind:
            b.setProperty("kind", kind)
        if kind == "primary":
            b.setFont(self.f_bold)
            minw = max(minw, QFontMetrics(self.f_bold).horizontalAdvance(text) + 48)
        if minw:
            b.setMinimumWidth(minw)
        b.setCursor(Qt.PointingHandCursor)
        b.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        b.clicked.connect(lambda checked=False: cmd())
        if tip:
            b.setToolTip(tip)
        return b

    def box(self, title):
        """Titled box; returns (box, vertical layout of its content)."""
        g = QGroupBox(title)
        lay = QVBoxLayout(g)
        lay.setSpacing(8)
        return g, lay

    def group(self, layout, title, note=""):
        g, lay = self.box(title)
        if note:
            lay.addWidget(self.lab(note, "muted", wrap=True))
        layout.addWidget(g)
        return lay

    def make_table(self, fonts=None):
        tv = QTableWidget()
        tv._fonts = fonts or (self.f_table, self.f_table_b)
        tv.setFont(tv._fonts[0])
        tv.horizontalHeader().setFont(tv._fonts[1])
        tv.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tv.horizontalHeader().setHighlightSections(False)
        tv.horizontalHeader().setSectionsClickable(False)
        tv.verticalHeader().hide()
        tv.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tv.setSelectionBehavior(QAbstractItemView.SelectRows)
        tv.setWordWrap(False)
        tv.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        tv.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        tv.setFocusPolicy(Qt.ClickFocus)
        return tv

    def warn_box(self, text):
        QMessageBox.warning(self, APP_NAME, text)

    def current_tab(self):
        return TABS[self.tabs.currentIndex()][0] if hasattr(self, "tabs") else "digitize"

    # ================================================================ the window
    def build_ui(self, select="digitize"):
        self.guides, self.overlay_items, self.pick = {}, [], None
        self.trace, self.trace_items = None, []
        self.root = QWidget()
        self.root.setObjectName("root")
        outer = QVBoxLayout(self.root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- theme and text size, at the right end of the tab row (the program name is in the window title)
        top = QWidget()
        top.setObjectName("tabcorner")
        top.setAttribute(Qt.WA_StyledBackground, True)
        bar = QHBoxLayout(top)
        bar.setContentsMargins(0, 3, 10, 3)
        bar.setSpacing(2)
        bar.addWidget(QLabel("Theme"))
        bar.addSpacing(6)
        tm = Combo()
        tm.addItems(["\u263e  Dark", "\u2600  Light"])
        tm.setCurrentIndex(0 if self.theme == "dark" else 1)
        tm.setToolTip("Dark or light appearance")
        tm.activated.connect(lambda i: self.set_theme("dark" if i == 0 else "light"))
        bar.addWidget(tm)
        bar.addSpacing(14)
        bar.addWidget(QLabel("Text"))
        bar.addSpacing(6)
        ts = Spin()
        ts.setRange(8, 16)
        ts.setValue(self.textpt)
        ts.setFixedWidth(74)
        ts.setKeyboardTracking(False)
        ts.setToolTip("Text size of the interface (pt)")
        ts.valueChanged.connect(self.set_textpt)
        bar.addWidget(ts)

        # ---- tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("main")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setUsesScrollButtons(True)
        self.page = {}
        for key, title in TABS:
            pg = Page(fit=(key == "digitize"))
            self.page[key] = pg
            self.tabs.addTab(pg, title.replace("&", "&&"))
        self.tabs.setCornerWidget(top, Qt.TopRightCorner)
        outer.addWidget(self.tabs, 1)
        self.statusline = QLabel("Ready.")
        self.statusline.setObjectName("status")
        outer.addWidget(self.statusline)

        self.tab_howto()
        self.tab_settings()
        self.tab_digitize()
        self.tab_about()
        self.setCentralWidget(self.root)          # the previous interface is deleted here
        self.tabs.setCurrentIndex([k for k, _ in TABS].index(select if select in dict(TABS) else "digitize"))
        self.restore_view()

    # ---------------------------------------------------------------- How to use
    def tab_howto(self):
        tb = QTextBrowser()
        tb.setPlainText("%s %s, \u00a9 %s %s\n\n" % (APP_NAME, APP_VERSION, APP_YEAR, APP_AUTHOR) + HOWTO)
        tb.setMinimumHeight(400)
        self.page["howto"].lay.addWidget(tb, 1)

    # ---------------------------------------------------------------- 1. Settings
    def tab_settings(self):
        L = self.page["settings"].lay
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(8)
        self.group(L, "Digitizing and display").addLayout(g)
        g.addWidget(self.lab("position of a symbol"), 0, 0)
        mr = Combo()
        mr.addItems(["centroid of the symbol", "centre of its bounding box"])
        mr.setCurrentIndex(1 if self.marker_ref == "bbox" else 0)
        mr.setMinimumWidth(260)

        def set_ref(i):
            self.marker_ref = "bbox" if i == 1 else "centroid"
            self.save_cfg()
            self.refresh()
        mr.activated.connect(set_ref)
        g.addWidget(mr, 0, 1, Qt.AlignLeft)
        g.addWidget(self.lab("The two differ only for triangles. Plots made with matplotlib put the data point at the centre of "
                             "the bounding box.", "muted", wrap=True, small=True), 0, 2)
        g.addWidget(self.lab("font size of the tables (pt)"), 1, 0)
        tf = Spin()
        tf.setRange(8, 40)
        tf.setValue(self.tablefont)
        tf.setFixedWidth(76)
        tf.setKeyboardTracking(False)
        tf.valueChanged.connect(self.set_tablefont)
        g.addWidget(tf, 1, 1, Qt.AlignLeft)
        cx = QCheckBox("interpolate x-values")
        cx.setChecked(self.common_x)

        def set_cx(on):
            self.common_x = bool(on)
            self.save_cfg()
        cx.toggled.connect(set_cx)
        g.addWidget(cx, 2, 1, Qt.AlignLeft)
        g.addWidget(self.lab("Export: all lines share one x column, each line interpolated onto it (one value per pixel "
                             "of the plot, rounded to a step of 1, 2, 2.5 or 5 x 10^n; blank outside a line's own range). "
                             "Symbols and lines that turn back in x keep their own x column. Off: every series has its "
                             "own x column with the traced points.", "muted", wrap=True, small=True), 2, 2)
        fr = QCheckBox("analyse blurred pictures at full size")
        fr.setChecked(self.full_res)

        def set_fr(on):
            self.full_res = bool(on)
            self.save_cfg()
        fr.toggled.connect(set_fr)
        g.addWidget(fr, 3, 1, Qt.AlignLeft)
        g.addWidget(self.lab("Off (default): a blurred or enlarged picture is reduced to the resolution it really has, "
                             "which keeps lines whole in enlarged screenshots. On: no reduction; try it when narrow peaks "
                             "are cut or close curves are mixed (Messages says when a picture was reduced; a picture "
                             "that was not reduced gives the same result either way). Press "
                             "Process again after changing it.", "muted", wrap=True, small=True), 3, 2)
        g.setColumnStretch(2, 1)
        b2 = self.group(L, "Output folder (default for exported data)")
        self.e_out = QLineEdit(self.outdir)
        self.e_out.setFixedWidth(640)

        def out_done():
            d = self.e_out.text().strip()
            if os.path.isdir(d):
                self.outdir = d
                self.save_cfg()
            else:
                self.e_out.setText(self.outdir)

        def browse():
            d = QFileDialog.getExistingDirectory(self, APP_NAME, self.outdir)
            if d:
                self.e_out.setText(os.path.normpath(d))
                out_done()
        self.e_out.editingFinished.connect(out_done)
        b2.addLayout(hrow(self.e_out, self.button("Browse\u2026", browse, tip="Choose the folder for exported files")))
        L.addStretch(1)

    # ---------------------------------------------------------------- 2. Digitize
    def tab_digitize(self):
        L = self.page["digitize"].lay
        b1 = self.group(L, "1. Import and process")
        self.b_clip = self.button("Import from clipboard", self.import_clipboard, "accent",
                                  "Take the picture that was copied (Ctrl+V)", 170)
        self.b_file = self.button("Import from file", self.import_file, "accent", "Picture, PDF, SVG, EPS or PS file", 150)
        self.b_recent = self.button("Recent folder", lambda: self.import_file(self.lastdir), "accent", "", 130)
        self.update_recent()
        self.b_proc = self.button("Process", self.process, "primary", "Find the axes and extract the data series", 170)
        self.b_exp = self.button("Export", self.export, None, "Write the ticked series to a CSV or XLSX file", 110)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setFont(self.f_bold)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row = hrow(self.b_clip, self.b_file, self.b_recent, 8, self.b_proc, 8, self.b_exp, stretch=False)
        row.addSpacing(10)
        row.addWidget(self.status, 20)
        row.addStretch(1)                                  # takes the free width while the status is hidden
        b1.addLayout(row)
        self.source = self.lab("No plot imported yet.", "muted", wrap=True, small=True)
        b1.addWidget(self.source)

        # ---- imported and extracted, side by side
        gl, ll = self.box("2. Imported plot")
        self.view = ImageView()
        self.view.setMinimumSize(320, 220)
        self.view.clicked.connect(self.on_image_click)
        self.view.finished.connect(self.end_pick)
        ll.addWidget(self.view, 1)
        self.c_overlay = QCheckBox("draw the extracted points over the picture")
        self.c_overlay.setChecked(self.show_overlay)

        def set_overlay(on):
            self.show_overlay = bool(on)
            self.save_cfg()
            self.refresh()
        self.c_overlay.toggled.connect(set_overlay)
        ll.addLayout(hrow(self.c_overlay, 12, self.lab("wheel: zoom, double click: fit", "muted", small=True)))
        gr, lr = self.box("3. Extracted data")
        self.fig = Figure(dpi=100)
        self.canvas = Canvas(self.fig)
        self.plotview = PlotView(self.canvas)
        self.plotview.setMinimumSize(320, 220)
        lr.addWidget(self.plotview, 1)
        lr.addLayout(hrow(self.lab("wheel: zoom, double click: fit", "muted", small=True)))
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(gl)
        mid.addWidget(gr)
        mid.setSizes([700, 700])
        mid.setChildrenCollapsible(False)

        # ---- series, axes, messages
        gs, ls = self.box("5. Data series of the extracted plot (untick what is not data)")
        self.table = self.make_table()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["series", "colour", "type", "style / symbol", "points", "y axis", "note"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnHidden(5, True)                # shown when the plot has a right-hand Y axis
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)   # names only
        self.table.itemChanged.connect(self.on_table)
        self.table.cellClicked.connect(self.on_cell)
        ls.addWidget(self.table, 1)
        self.b_msg = self.button("Messages", self.show_messages, None, "What the analysis found and did (opens a window)", 110)
        self.b_trace = self.button("Trace a curve", self.start_trace, None,
                                   "Guided trace for a curve the automatic tracing mixed up: click points along it in the "
                                   "imported picture, double click to finish (Esc cancels). Added as a new series.", 130)
        self.b_trace.setEnabled(False)
        ls.addLayout(hrow(self.b_msg, 8, self.b_trace, 10, self.lab("double click a name to rename it (used as column name in the export)",
                                                   "muted", small=True)))
        ga, la = self.box("4. Axes of the imported plot (detected; edit and press Apply to correct)")
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)
        la.addLayout(g)
        for c, t in enumerate(("", "scale", "pixel 1", "value 1", "", "pixel 2", "value 2", "")):
            g.addWidget(self.lab(t, "muted", small=True), 0, c)
        self.ax_w = {}
        for r, name in ((1, "x"), (2, "y"), (3, "y2")):
            g.addWidget(self.lab(AXIS_NAME[name], bold=True), r, 0)
            combo = Combo()
            combo.addItems(SCALES)
            combo.setToolTip("Spacing of the ticks. ln is the same spacing as log; 1/x is reciprocal spacing.")
            combo.setMinimumWidth(110)
            g.addWidget(combo, r, 1)
            w = [combo]
            for k in (0, 1):
                px, val = QLineEdit(), QLineEdit()
                px.setFixedWidth(74)
                val.setMinimumWidth(96)
                pick = self.button("Pick", lambda n=name, kk=k: self.start_pick(n, kk), tip="Then click that position in the imported picture")
                g.addWidget(px, r, 2 + 3 * k)
                g.addWidget(val, r, 3 + 3 * k)
                g.addWidget(pick, r, 4 + 3 * k)
                w += [px, val]
            self.ax_w[name] = w
        g.setColumnStretch(3, 1)
        g.setColumnStretch(6, 1)
        self.ax_info = self.lab("", "muted", wrap=True, small=True)
        row = hrow(self.button("Apply axes", self.apply_axes, "accent", "Use the values typed above", 130), stretch=False)
        row.addSpacing(8)
        row.addWidget(self.ax_info, 1)
        la.addLayout(row)
        la.addStretch(1)
        self.msg_dlg = QDialog(self)               # messages: in their own window, opened by the button
        self.msg_dlg.setWindowTitle("%s: messages" % APP_NAME)
        self.msg_dlg.resize(620, 420)
        dl = QVBoxLayout(self.msg_dlg)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        dl.addWidget(self.logbox, 1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.msg_dlg.hide)
        dl.addWidget(close)
        bottom = QSplitter(Qt.Horizontal)
        for wdg in (ga, gs):                       # axes under the imported plot, series under the extracted plot
            bottom.addWidget(wdg)
        bottom.setSizes([700, 700])
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        bottom.setChildrenCollapsible(False)
        split = QSplitter(Qt.Vertical)
        split.addWidget(mid)
        split.addWidget(bottom)
        split.setSizes([540, 250])
        split.setStretchFactor(0, 1)               # extra height goes to the two plots
        split.setStretchFactor(1, 0)
        split.setChildrenCollapsible(False)
        L.addWidget(split, 1)

    def show_messages(self):
        self.logbox.setPlainText(self.log_text)
        self.msg_dlg.show()
        self.msg_dlg.raise_()
        self.msg_dlg.activateWindow()

    # ---------------------------------------------------------------- About
    def tab_about(self):
        L = self.page["about"].lay
        title = QLabel(APP_NAME)
        title.setFont(QFont(FONT, self.textpt + 9, QFont.Bold))
        L.addWidget(title)
        L.addWidget(self.lab("Data extraction from XY line and scatter plots: pictures, PDF and EPS figures", "muted"))
        g = QGridLayout()
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(7)
        self.group(L, "Version and author").addLayout(g)
        info = [("Version", APP_VERSION), ("Author", APP_AUTHOR),
                ("Copyright", "\u00a9 %s %s" % (APP_YEAR, APP_AUTHOR)),
                ("License", "Free software under the GNU Affero General Public License version 3 (AGPL-3.0): you may use, "
                            "copy, modify and pass on the program, also commercially, provided that what you pass on stays "
                            "under the AGPL-3.0 and comes with its source code. " +
                            ("The source code and the license texts are packed inside this single-file executable: "
                             "button 'Save source code and licenses' below." if bundled_dir() else
                             "The source code is in the folder 'source' next to the executable; the license text is "
                             "LICENSE-AGPL-3.0.txt.")),
                ("Warranty", "Provided \"as is\", without warranty of any kind. The author is not liable for any damage or for "
                             "results derived from its use; check the extracted data against the original plot before "
                             "publishing them."),
                ("AI assistance", "Requirements, test data and acceptance: %s. The Python code of this program was written with "
                                  "the assistance of an AI model (Claude, Anthropic). Results were checked against synthetic "
                                  "plots with known data, one measured data set and figures from published papers. "
                                  "Methods used:" % APP_AUTHOR)]
        for i, (k, v) in enumerate(info):
            g.addWidget(self.lab(k, "muted"), i, 0, Qt.AlignTop)
            g.addWidget(self.lab(v, wrap=True), i, 1)
        g.setColumnStretch(1, 1)
        methods = (
            "1. Vector figures (PDF, SVG, EPS, PS): curve coordinates come from the drawing paths, tick labels from the text layer "
            "(character recognition when the text is drawn as outlines); each axes frame on the page is offered as one plot.",
            "2. Axes in pictures: frame and tick marks from long dark lines and short stubs; tick labels by character "
            "recognition (RapidOCR), exponents by their raised baseline; linear or log is the scale that explains most labels.",
            "3. Clean-up and colours: grid, legend boxes and text are removed (rotated labels only letter by letter); the ink "
            "is split into colours tolerant to anti-aliasing, and a shade that touches only one curve stays with that curve.",
            "4. Lines: each colour is thinned to its centre line; short dashes are replaced by segments along their axis; "
            "pieces are linked across gaps by distance and direction, also where another curve covers them; then sub-pixel centring.",
            "5. Symbols and guides: symbols are named by shape and separated when they touch; pieces stretched along their row "
            "are dashes, not symbols; straight lines that are short and do not run from edge to edge are listed unticked.")
        row0 = len(info)
        for j, line in enumerate(methods):
            g.addWidget(self.lab(line, wrap=True), row0 + j, 1)
        if bundled_dir():
            g.addWidget(self.button("Save source code and licenses", self.save_sources,
                                    tip="Copies the source code and all license texts packed in this exe to a folder"),
                        row0 + len(methods), 1, Qt.AlignLeft)
        b3 = self.group(L, "Python code and third-party components (versions of this build)")
        rows = [["component", "version", "license", "use in this program"]] + self.components()
        tv = self.make_table(fonts=(QFont(FONT, self.textpt), self.f_bold))
        tv.setColumnCount(4)
        tv.setRowCount(len(rows) - 1)
        tv.setHorizontalHeaderLabels(rows[0])
        for r, row in enumerate(rows[1:]):
            for c, v in enumerate(row):
                tv.setItem(r, c, QTableWidgetItem(str(v)))
        tv.resizeColumnsToContents()
        tv.resizeRowsToContents()
        tv.horizontalHeader().setStretchLastSection(True)
        tv.setFixedHeight(tv.horizontalHeader().sizeHint().height() + sum(tv.rowHeight(r) for r in range(len(rows) - 1)) + 22)
        tv.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        b3.addWidget(tv)
        b3.addWidget(self.lab("PyMuPDF and MuPDF are used under the GNU AGPL v3, the license of this program; where to get their "
                              "source code is in licenses\\PyMuPDF__NOTICE.txt.", "muted", wrap=True, small=True))
        b3.addWidget(self.lab("Qt and PySide6 are used under the GNU LGPL v3. " +
                              ("In this single-file executable they are packed inside the exe; to use other builds of them, "
                               "rebuild the program from its source code (build_onefile.bat) or use the folder build "
                               "(build_exe.bat), where they are separate files." if bundled_dir() else
                               "They are shipped as separate library files in the program folder and can be replaced by other "
                               "builds of the same libraries."), "muted", wrap=True, small=True))
        b3.addWidget(self.lab("Program code: xydapp.py (interface), xydtheme.py (look) and xydcore.py (extraction), in Python. The "
                              "bundled components keep their own licenses; their full texts are in the 'licenses' folder " +
                              ("saved with the button above." if bundled_dir() else "next to the executable."),
                              "muted", wrap=True, small=True))
        L.addStretch(1)

    def save_sources(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for the source code and licenses", self.outdir)
        if not d:
            return
        try:
            out = save_bundled(os.path.join(d, "XY_Digitizer %s source and licenses" % APP_VERSION))
            self.set_status("Source code and licenses saved to %s" % os.path.normpath(out), "ok")
        except Exception as e:
            self.set_status("Saving the source code failed: %s" % e, "err")

    def components(self):
        import platform
        import PySide6

        def ver(mod, attr="__version__"):
            try:
                return str(getattr(__import__(mod), attr))
            except Exception:
                return "?"
        try:
            from importlib.metadata import version
            rocr = version("rapidocr")
        except Exception:
            rocr = "?"
        try:
            import PyInstaller  # present only in the build environment
            piv = PyInstaller.__version__
        except Exception:
            piv = "build tool"
        return [["Python", platform.python_version(), "PSF License Agreement", "language and standard library"],
                ["NumPy", ver("numpy"), "BSD-3-Clause", "arrays, numerical routines"],
                ["SciPy", ver("scipy"), "BSD-3-Clause", "nearest-neighbour search, interpolation"],
                ["OpenCV", ver("cv2"), "Apache-2.0", "image analysis"],
                ["PyMuPDF", ver("pymupdf"), "AGPL-3.0", "reading PDF pages: paths, text, rendering"],
                ["RapidOCR", rocr, "Apache-2.0", "reading tick labels and text in pictures"],
                ["ONNX Runtime", ver("onnxruntime"), "MIT", "runs the character recognition models"],
                ["openpyxl", ver("openpyxl"), "MIT", "XLSX export"],
                ["Matplotlib", ver("matplotlib"), "Matplotlib License (PSF-based, BSD-compatible)", "plot of the extracted data"],
                ["Qt", QtCore.qVersion(), "LGPL-3.0", "windows, widgets and theme"],
                ["PySide6 (Qt for Python)", PySide6.__version__, "LGPL-3.0", "windows, widgets and theme"],
                ["PyInstaller", piv, "GPL-2.0 with bootloader exception", "packaging into the .exe"],
                ["OCR models (PaddleOCR PP-OCRv6, via RapidOCR)", "bundled", "Apache-2.0", "text detection and recognition"],
                ["other bundled libraries", "bundled", "BSD / MIT / Apache-2.0 / HPND / MPL-2.0 / LGPL-2.1 / LGPL-3.0",
                 "Pillow, Shapely (GEOS: LGPL-2.1), pyclipper, PyYAML, OmegaConf, protobuf, kiwisolver, fonttools, "
                 "contourpy, cycler, pyparsing, packaging, certifi (MPL-2.0), tqdm (MPL-2.0 / MIT), shiboken6"]]

    # ================================================================ import
    def restore_view(self):
        """Show the current picture and result again (after start-up or a rebuild of the interface)."""
        self.b_proc.setEnabled(self.src is not None)
        self.b_exp.setEnabled(False)
        self.source.setText(self.src_label or "No plot imported yet.")
        self.logbox.setPlainText(self.log_text)
        if self.src is not None:
            self.view.set_image(self.res.image if self.res is not None else self.src["image"])
        if self.status_state:
            self.msg(self.status, *self.status_state, status=True)
        else:
            self.msg(self.status, "")
            self.say("Ready. Import a plot from the clipboard (Ctrl+V) or from a file.", "muted")
        if self.res is not None:
            self.show_result()
        else:
            self.draw_plot()

    def set_source(self, src, label):
        self.src, self.res, self.src_label, self.log_text = src, None, label, ""
        self.guides, self.overlay_items = {}, []
        self.trace, self.trace_items = None, []
        self.view.set_cross(False)
        self.b_trace.setEnabled(False)
        self.tabs.setCurrentIndex([k for k, _ in TABS].index("digitize"))
        self.view.set_image(src["image"])
        self.table.setRowCount(0)
        self.logbox.setPlainText("")
        self.ax_info.setText("")
        for name in AXIS_NAME:
            for w in self.ax_w[name][1:]:
                w.clear()
        self.table.setColumnHidden(5, True)
        self.source.setText(label)
        self.b_proc.setEnabled(True)
        self.b_exp.setEnabled(False)
        self.draw_plot()
        self.set_status("Plot imported. Press Process.", "text")

    def paste_key(self):
        if not isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit)):
            self.import_clipboard()

    def import_clipboard(self):
        cb = QApplication.clipboard()
        img = cb.image()
        if not img.isNull():
            img = img.convertToFormat(QImage.Format_RGBA8888)
            a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]
            a = a.reshape(img.height(), img.width(), 4).astype(np.float32)
            alpha = a[:, :, 3:4] / 255.0
            rgb = np.ascontiguousarray((a[:, :, :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8))
            self.set_source(dict(kind="image", image=rgb), "Picture from the clipboard, %d x %d px." % (rgb.shape[1], rgb.shape[0]))
            return
        urls = [u.toLocalFile() for u in cb.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(urls[0])
        else:
            self.set_status("The clipboard holds no picture. Copy the plot (or a screenshot of it) first.", "warn")

    def import_file(self, folder=None):
        flt = ("Plots (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp *.pdf *.svg *.eps *.epsf *.epsi *.ps);;"
               "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp);;PDF (*.pdf);;SVG (*.svg);;PostScript (*.eps *.epsf *.epsi *.ps);;All files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "Import a plot", folder or self.outdir, flt)
        if path:
            self.open_path(path)

    def update_recent(self):
        ok = bool(self.lastdir) and os.path.isdir(self.lastdir)
        self.b_recent.setEnabled(ok)
        self.b_recent.setToolTip("Import from the folder of the last picture or PDF: %s" % self.lastdir if ok
                                 else "Available after a picture or PDF has been loaded from a file")

    def open_path(self, path):
        name, low = os.path.basename(path), path.lower()
        if not self.auto and os.path.isdir(os.path.dirname(os.path.abspath(path))):   # not the build test
            self.lastdir = os.path.dirname(os.path.abspath(path))
            self.save_cfg()
            if hasattr(self, "b_recent"):
                self.update_recent()
        try:
            if low.endswith(C.IMAGE_EXT):
                img = C.load_image(path)
                self.set_source(dict(kind="image", image=img), "%s: picture, %d x %d px." % (name, img.shape[1], img.shape[0]))
                return
            import pymupdf
            if low.endswith(C.PS_EXT):
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    pdf = C.ps_to_pdf(path)
                finally:
                    QApplication.restoreOverrideCursor()
                doc = pymupdf.open(pdf)
            else:
                doc = pymupdf.open(path)
            if doc.needs_pass:
                raise ValueError("The PDF is password protected.")
            self.docs.append(doc)                      # keep the document open while its page is in use
            dlg = FigurePicker(self, doc, name)
            if dlg.exec() == QDialog.Accepted and dlg.result is not None:
                self.set_source(dlg.result, "%s, page %d. %s" % (name, dlg.spin.value(), dlg.result["label"]))
        except Exception as e:
            self.warn_box("Could not open %s:\n%s" % (name, e))

    # ================================================================ processing
    def process(self):
        if self.src is None:
            return
        self.dlg = QProgressDialog("Processing ...", None, 0, 100, self)
        self.dlg.setWindowTitle(APP_NAME)
        self.dlg.setWindowModality(Qt.WindowModal)
        self.dlg.setMinimumDuration(0)
        self.dlg.setAutoClose(False)
        self.dlg.setAutoReset(False)
        self.dlg.setMinimumWidth(420)
        self.dlg.setValue(1)
        self.worker = Worker(self.src, self.full_res)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, pct, msg):
        self.dlg.setValue(pct)
        if msg:
            self.dlg.setLabelText(msg + " ...")

    def on_failed(self, tb):
        self.dlg.close()
        self.log_text = tb
        self.logbox.setPlainText(tb)
        self.set_status("Processing failed. Details: press Messages (below the series table).", "err")

    def on_done(self, res):
        self.dlg.setValue(100)
        self.dlg.close()
        self.res = res
        self.guides, self.overlay_items = {}, []           # set_image below clears the scene and deletes them
        self.trace, self.trace_items = None, []
        self.view.set_cross(False)
        lines = ["Read from: %s." % ("vector paths (exact coordinates)" if res.source == "vector" else "image analysis"),
                 "%d series found, %d ticked." % (len(res.series), sum(s.use for s in res.series))]
        lines += ["%s: %s" % (s.name, s.note) for s in res.series if s.note] + res.log
        self.log_text = "\n".join(lines)
        self.logbox.setPlainText(self.log_text)
        self.view.set_image(res.image)
        if res.yaxis2 is not None and res.series and not self.auto:
            self.ask_y_axis()
        self.show_result()
        if res.xaxis is None and res.yaxis is None:
            self.set_status("Axes not read: enter two reference values per axis (box 4) and press Apply axes.", "warn")
        elif res.xaxis is None or res.yaxis is None:
            miss, got = ("X", "Y") if res.xaxis is None else ("Y", "X")
            self.set_status("%s axis not read (%s axis applied): enter its scale and two reference values in box 4, "
                            "then press Apply axes." % (miss, got), "warn")
        elif not res.series:
            self.set_status("No data series found in this plot.", "warn")
        else:
            self.set_status("Done: %d series. Check the two plots, untick what is not data, then Export." % len(res.series), "ok")

    def show_result(self):
        res = self.res
        x0, y0, x1, y1 = res.box
        for name, ax, p1, p2 in (("x", res.xaxis, x0, x1), ("y", res.yaxis, y1, y0), ("y2", res.yaxis2, y1, y0)):
            w = self.ax_w[name]
            w[0].setCurrentIndex(0 if ax is None else SCALES.index(ax.kind))
            if name == "y2" and ax is None:                # no right-hand axis: the row stays empty
                for q in w[1:]:
                    q.clear()
                continue
            w[1].setText("%.1f" % p1)
            w[3].setText("%.1f" % p2)
            w[2].setText("" if ax is None else "%.5g" % ax.to_data(p1))
            w[4].setText("" if ax is None else "%.5g" % ax.to_data(p2))
        self.table.blockSignals(True)
        self.table.setRowCount(len(res.series))
        for r, s in enumerate(res.series):
            it = QTableWidgetItem(s.name)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if s.use else Qt.Unchecked)
            self.table.setItem(r, 0, it)
            sw = QTableWidgetItem("")
            sw.setFlags(Qt.ItemIsEnabled)
            sw.setBackground(QColor(*s.color))
            self.table.setItem(r, 1, sw)
            for c, t in enumerate((s.kind, s.style, str(len(s.px)), s.yref, s.note), 2):
                cell = QTableWidgetItem(t)
                cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)     # only the name can be edited
                self.table.setItem(r, c, cell)
            for c in range(7):
                self.table.item(r, c).setToolTip("Click to switch this series between the left and the right Y axis." if c == 5
                                                 else "Double click to rename; the name heads its columns in the export." if c == 0
                                                 else s.note)
        self.table.setColumnHidden(5, res.yaxis2 is None)
        self.b_trace.setEnabled(res.image is not None)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        for c in range(6):
            self.table.setColumnWidth(c, self.table.columnWidth(c) + 6)
        self.table.blockSignals(False)
        self.refresh()

    def on_table(self, item):
        if self.res is None or item.column() != 0:
            return
        r = item.row()
        s = self.res.series[r]
        name = self.clean_name(item.text(), r)
        if name != item.text():                            # show the name as it will be exported
            self.table.blockSignals(True)
            item.setText(name)
            self.table.blockSignals(False)
        s.name = name
        s.use = item.checkState() == Qt.Checked
        self.table.resizeColumnToContents(0)
        self.table.setColumnWidth(0, self.table.columnWidth(0) + 6)
        self.refresh()

    def clean_name(self, text, row):
        """Series name as typed, made safe as a column name: no commas, semicolons, quotes, tabs or line
        breaks (the CSV is written without quoting); empty -> the default S<n>; a name used by another
        series gets ' (2)', ' (3)' ..."""
        t = " ".join(re.sub(r'[,;"\t\r\n]', " ", text).split())
        if not t:
            t = "S%d" % (row + 1)
        others = {q.name for i, q in enumerate(self.res.series) if i != row}
        base, k = t, 2
        while t in others:
            t = "%s (%d)" % (base, k)
            k += 1
        return t

    def on_cell(self, row, col):
        """A click in the column 'y axis' moves the series to the other Y axis."""
        if self.res is None or col != 5 or self.res.yaxis2 is None:
            return
        s = self.res.series[row]
        s.yref = "left" if s.yref == "right" else "right"
        self.table.blockSignals(True)
        self.table.item(row, 5).setText(s.yref)
        self.table.blockSignals(False)
        self.refresh()

    def ask_y_axis(self):
        """The plot has two different Y scales: ask which one the series are read on."""
        res = self.res
        x0, y0, x1, y1 = res.box
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Question)
        box.setText("This plot has two Y axes with different scales.\n\n"
                    "Left:   %s\nRight:  %s\n\nWhich axis do the curves belong to?\n"
                    "Single series can be switched afterwards: click their cell in the column 'y axis'."
                    % (C.describe_axis(res.yaxis, y1, y0).split(" (")[0], C.describe_axis(res.yaxis2, y1, y0).split(" (")[0]))
        b_left = box.addButton("Left axis", QMessageBox.AcceptRole)
        b_right = box.addButton("Right axis", QMessageBox.AcceptRole)
        box.setDefaultButton(b_left)
        box.exec()
        ref = "right" if box.clickedButton() is b_right else "left"
        for s in res.series:
            s.yref = ref

    # ================================================================ axes by hand
    def start_pick(self, name, k):
        if self.src is None:
            return
        self.cancel_trace(quiet=True)
        self.pick = (name, k)
        self.say("Click the position of %s reference %d in the imported picture; double click when it is right."
                 % (AXIS_NAME[name], k + 1), "text")
        self.view.set_cross(True)

    def end_pick(self):
        """Double click in the picture: the position is kept, the pointer returns to normal."""
        if self.trace is not None:
            self.finish_trace()
            return
        if self.pick is None:
            return
        name, k = self.pick
        self.pick = None
        self.view.set_cross(False)
        self.ax_w[name][2 + 2 * k].setFocus()
        self.ax_w[name][2 + 2 * k].selectAll()
        self.say("Now type the value of %s reference %d, then Apply axes." % (AXIS_NAME[name], k + 1), "text")

    def on_image_click(self, x, y):
        if self.trace is not None:
            self.add_trace_point(x, y)
            return
        if self.pick is None:
            return
        name, k = self.pick
        v = (x if name == "x" else y) - 0.5
        self.ax_w[name][1 + 2 * k].setText("%.1f" % v)
        old = self.guides.pop((name, k), None)
        if old is not None:
            self.view.scene().removeItem(old)
        pen = QPen(QColor(TH.IBM["magenta"]), 1, Qt.DashLine)
        pen.setCosmetic(True)
        r = self.view.scene().sceneRect()
        self.guides[(name, k)] = (self.view.scene().addLine(x, 0, x, r.height(), pen) if name == "x"
                                  else self.view.scene().addLine(0, y, r.width(), y, pen))
        self.say("%s reference %d at %.1f px. Click again to move it, double click to finish." % (AXIS_NAME[name], k + 1, v), "text")

    # ================================================================ guided trace
    def start_trace(self):
        if self.res is None or self.res.image is None:
            return
        self.pick = None
        self.cancel_trace(quiet=True)
        self.trace = []
        self.view.set_cross(True)
        self.say("Guided trace: click points along ONE curve in the imported picture, from one end to the other: both "
                 "ends, every peak and dip, and on both sides of places where it crosses or runs into a curve of the "
                 "same colour. Double click to finish, Esc to cancel.", "text")

    def add_trace_point(self, x, y):
        if self.trace and abs(x - self.trace[-1][0]) + abs(y - self.trace[-1][1]) < 1.5:
            return                                         # the second half of a double click
        self.trace.append((x - 0.5, y - 0.5))              # scene: pixel centres at +0.5
        d = max(2.5, 0.005 * max(self.res.image.shape[:2]))
        pen = QPen(QColor(TH.IBM["magenta"]), 2)
        pen.setCosmetic(True)
        self.trace_items.append(self.view.scene().addEllipse(x - d, y - d, 2 * d, 2 * d, pen))
        self.say("Guided trace: %d point%s. Go on clicking along the curve; double click to finish, Esc to cancel."
                 % (len(self.trace), "" if len(self.trace) == 1 else "s"), "text")

    def clear_trace_marks(self):
        for it in self.trace_items:
            try:
                self.view.scene().removeItem(it)
            except RuntimeError:                           # already deleted with the scene
                pass
        self.trace_items = []

    def cancel_trace(self, quiet=False):
        if self.trace is None:
            return
        self.trace = None
        self.clear_trace_marks()
        self.view.set_cross(False)
        if not quiet:
            self.say("Guided trace cancelled.", "text")

    def finish_trace(self):
        pts, self.trace = self.trace, None
        self.clear_trace_marks()
        self.view.set_cross(False)
        res = self.res
        if res is None or len(pts) < 2:
            self.say("Guided trace: click at least two points along the curve.", "warn")
            return
        try:
            s = C.guided_trace(res.image, pts, res.box)
        except Exception as e:
            self.set_status("Guided trace failed: %s" % e, "err")
            return
        s.name = self.clean_name("S%d" % (len(res.series) + 1), len(res.series))
        res.series.append(s)
        self.log_text += "\n%s: %s, colour rgb%s, %s" % (s.name, s.note, s.color, s.style)
        self.show_result()
        self.set_status("Guided trace added as %s (%s, %d points clicked). Untick the automatic series it replaces, "
                        "then Export." % (s.name, s.style, len(pts)), "ok")

    def apply_axes(self):
        if self.res is None:
            return
        try:
            axes = {}
            for name, label in AXIS_NAME.items():
                w = self.ax_w[name]
                kind = {"ln": "log"}.get(w[0].currentText(), w[0].currentText())
                if name == "y2" and not any(w[i].text().strip() for i in (1, 2, 3, 4)):
                    axes[name] = None                      # empty row: no right-hand axis
                    continue
                vals = [to_float(w[i].text()) for i in (1, 2, 3, 4)]
                if any(v is None for v in vals):
                    raise ValueError("%s: fill in the two pixel positions and the two values." % label)
                p1, v1, p2, v2 = vals
                if abs(p1 - p2) < 1 or v1 == v2:
                    raise ValueError("%s: the two references must differ." % label)
                if kind == "log" and (v1 <= 0 or v2 <= 0):
                    raise ValueError("%s: a log axis needs positive values." % label)
                if kind == "1/x" and (v1 == 0 or v2 == 0):
                    raise ValueError("%s: a 1/x axis cannot have a reference at zero." % label)
                axes[name] = C.Axis.from_two_points(p1, v1, p2, v2, kind)
        except ValueError as e:
            self.set_status("Axes not applied. %s" % e, "err")
            return
        self.res.xaxis, self.res.yaxis, self.res.yaxis2 = axes["x"], axes["y"], axes["y2"]
        if axes["y2"] is None:
            for s in self.res.series:
                s.yref = "left"
        self.show_result()
        self.set_status("Axes applied.", "ok")

    # ================================================================ display
    def refresh(self, *_):
        if self.res is None:
            return
        res = self.res
        x0, y0, x1, y1 = res.box
        self.ax_info.setText("X: %s\nY: %s" % (C.describe_axis(res.xaxis, x0, x1), C.describe_axis(res.yaxis, y1, y0))
                             + ("" if res.yaxis2 is None else "\nY right: %s" % C.describe_axis(res.yaxis2, y1, y0)))
        self.b_exp.setEnabled(res.xaxis is not None and res.yaxis is not None and any(s.use for s in res.series))
        for it in self.overlay_items:
            self.view.scene().removeItem(it)
        self.overlay_items = []
        if self.show_overlay:
            for s in res.series:
                if not s.use:
                    continue
                pen = QPen(QColor(TH.IBM["magenta"]) if sum(s.color) < 250 else QColor(0, 0, 0), 1.3)
                pen.setCosmetic(True)
                px = (s.px_alt if (s.kind == "scatter" and self.marker_ref == "bbox" and s.px_alt is not None) else s.px) + 0.5
                path = QPainterPath()
                if s.kind == "line":
                    path.moveTo(*px[0])
                    for q in px[1:]:
                        path.lineTo(*q)
                else:
                    d = max(3.0, 0.006 * max(res.image.shape[:2]))
                    for x, y in px:
                        path.moveTo(x - d, y)
                        path.lineTo(x + d, y)
                        path.moveTo(x, y - d)
                        path.lineTo(x, y + d)
                self.overlay_items.append(self.view.scene().addPath(path, pen))
        self.draw_plot()

    def plot_color(self, rgb):
        """Colour of a series in the extracted plot: the original one, unless it would not show on the background."""
        bgc = QColor(self.tok("ax"))
        if max(abs(rgb[0] - bgc.red()), abs(rgb[1] - bgc.green()), abs(rgb[2] - bgc.blue())) < 90:
            return self.tok("text")
        return tuple(c / 255.0 for c in rgb)

    def style_fig(self, fig):
        """Theme colours for a figure: page-coloured background, faint grid, left and bottom axes only."""
        p = TH.tokens(self.theme)
        fig.set_facecolor(p["fig"])
        for ax in fig.axes:
            ax.set_facecolor(p["ax"])
            ax.tick_params(colors=p["text"], which="both")
            for side, s in ax.spines.items():
                s.set_color(p["edge"])
                s.set_visible(side in ("left", "bottom") or (side == "right" and len(fig.axes) > 1))
            if ax is fig.axes[0] and ax.get_xticks().size and ax.get_yticks().size:
                ax.grid(True, which="major", color=p["grid"], lw=0.7)
                ax.set_axisbelow(True)
            ax.title.set_color(p["muted"])
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(p["ax"])
                leg.get_frame().set_edgecolor(p["grid"])
                for tx in leg.get_texts():
                    tx.set_color(p["text"])
            for tx in ax.texts:
                tx.set_color(p["muted"])

    def draw_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        res = self.res
        if res is None:
            ax.text(0.5, 0.5, "The extracted plot appears here after Process", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.style_fig(self.fig)
            self.canvas.draw_idle()
            return
        x0, y0, x1, y1 = res.box
        calx, caly = res.xaxis is not None, res.yaxis is not None   # each axis drawn calibrated on its own
        two = caly and res.yaxis2 is not None and any(s.use and s.yref == "right" for s in res.series)
        ax2 = ax.twinx() if two else None                  # series read on the right-hand axis
        handles = []
        for s in res.series:
            if not s.use:
                continue
            px = s.px_alt if (s.kind == "scatter" and self.marker_ref == "bbox" and s.px_alt is not None) else s.px
            yax = res.yaxis2 if (s.yref == "right" and res.yaxis2 is not None) else res.yaxis
            d = np.c_[res.xaxis.to_data(px[:, 0]) if calx else px[:, 0],
                      yax.to_data(px[:, 1]) if caly else -px[:, 1]]
            col = self.plot_color(s.color)
            right = two and s.yref == "right"
            name = s.name + (" (right)" if right else "")
            if s.kind == "line":
                handles += (ax2 if right else ax).plot(d[:, 0], d[:, 1], MPL_LINE.get(s.style, "-"), color=col, lw=1.5, label=name)
            else:
                mk = MPL_MARKER.get(s.style.replace("open ", ""), "o")
                handles += (ax2 if right else ax).plot(d[:, 0], d[:, 1], mk, color=col, mfc="none" if s.style.startswith("open ") else col, ms=6, label=name)
        if caly:
            for a, yx in ((ax, res.yaxis), (ax2, res.yaxis2)):
                if a is not None:
                    if yx.log:
                        a.set_yscale("log")
                    lim = sorted(float(v) for v in yx.to_data([y1, y0]))
                    if np.all(np.isfinite(lim)):
                        a.set_ylim(*(lim if yx.to_data(y1) <= yx.to_data(y0) else lim[::-1]))
        else:
            ax.set_ylim(-y1, -y0)
        if calx:
            if res.xaxis.log:
                ax.set_xscale("log")
            lim = [float(v) for v in res.xaxis.to_data([x0, x1])]
            if np.all(np.isfinite(lim)):
                ax.set_xlim(*lim)
        else:
            ax.set_xlim(x0, x1)
        if not (calx and caly):
            miss = "axes" if not (calx or caly) else "X axis" if not calx else "Y axis"
            ax.set_title("pixel coordinates: %s not read" % miss, fontsize=9)
        ax.set_box_aspect((y1 - y0) / max(x1 - x0, 1))
        if ax2 is not None:
            ax2.set_box_aspect((y1 - y0) / max(x1 - x0, 1))
        if 0 < len(handles) <= 12:
            (ax2 or ax).legend(handles=handles, fontsize=8, framealpha=0.7)
        self.fig.subplots_adjust(left=0.12, right=0.88 if ax2 is not None else 0.97, top=0.95, bottom=0.1)
        self.style_fig(self.fig)
        self.canvas.draw_idle()

    # ================================================================ export
    def export(self):
        if self.res is None:
            return
        start = os.path.join(self.outdir, "extracted.csv")
        path, flt = QFileDialog.getSaveFileName(self, "Export data", start, "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith((".csv", ".xlsx")):
            path += ".xlsx" if "xlsx" in flt else ".csv"
        try:
            k, n = C.export_result(self.res, path, self.marker_ref, self.common_x)
            self.set_status("%d series exported (up to %d rows) to %s" % (k, n, os.path.normpath(path)), "ok")
        except Exception as e:
            self.set_status("Export failed: %s" % e, "err")


def cli_digitize(source, base):
    """Digitize without the interface (used by the test script). Returns 0 if the axes and some data were found."""
    log = []
    try:
        if source.lower().endswith(C.IMAGE_EXT):
            src = dict(kind="image", image=C.load_image(source))
        else:
            import pymupdf
            doc = pymupdf.open(C.ps_to_pdf(source) if source.lower().endswith(C.PS_EXT) else source)
            page = doc[0]
            src = C.pdf_source(page, page.rect, page.get_drawings())
        res = C.run_source(src)
        x0, y0, x1, y1 = res.box
        log += ["source: %s" % source, "read from: %s" % res.source,
                "x axis: %s" % C.describe_axis(res.xaxis, x0, x1), "y axis: %s" % C.describe_axis(res.yaxis, y1, y0)]
        if res.yaxis2 is not None:
            log.append("y axis (right): %s" % C.describe_axis(res.yaxis2, y1, y0))
        log += ["%s %s %-7s %-10s rgb%s n=%d %s" % (s.name, "[x]" if s.use else "[ ]", s.kind, s.style, s.color, len(s.px), s.note)
                for s in res.series] + res.log
        import cv2
        cv2.imwrite(base + "_overlay.png", C.overlay_image(res)[:, :, ::-1])
        ok = res.xaxis is not None and res.yaxis is not None and any(s.use for s in res.series)
        if ok:
            C.export_result(res, base + ".csv")
            C.export_result(res, base + ".xlsx")
            C.export_result(res, base + "_common.csv", common_x=True)
        log.append("RESULT: %s" % ("OK" if ok else "INCOMPLETE"))
    except Exception:
        ok = False
        log += [traceback.format_exc(), "RESULT: FAILED"]
    with open(base + "_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    return 0 if ok else 1


def set_taskbar_icon(win, ico):
    """Give the window its icon through Windows itself (WM_SETICON), from the .ico file or, in the exe, from
    the icon resource of the exe. The taskbar then shows it even if Qt could not set it."""
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        u32.LoadImageW.restype = wintypes.HANDLE
        u32.LoadImageW.argtypes = (wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT)
        u32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        u32.GetSystemMetrics.argtypes = (ctypes.c_int,)
        hwnd = int(win.winId())
        for which, cx, cy in ((1, 11, 12), (0, 49, 50)):          # ICON_BIG: SM_CXICON, ICON_SMALL: SM_CXSMICON
            w, h = u32.GetSystemMetrics(cx), u32.GetSystemMetrics(cy)
            hicon = u32.LoadImageW(None, ico, 1, w, h, 0x10) if os.path.isfile(ico) else None   # IMAGE_ICON, LR_LOADFROMFILE
            if not hicon and getattr(sys, "frozen", False):
                ctypes.windll.shell32.ExtractIconW.restype = wintypes.HANDLE
                ctypes.windll.shell32.ExtractIconW.argtypes = (wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT)
                hicon = ctypes.windll.shell32.ExtractIconW(None, sys.executable, 0)
            if hicon:
                u32.SendMessageW(hwnd, 0x0080, which, hicon)         # WM_SETICON
    except Exception:
        pass


def main():
    if "--save-bundled" in sys.argv:            # single-file exe: copy the packed source code and licenses (build check)
        if not bundled_dir():
            sys.exit(1)
        save_bundled(sys.argv[sys.argv.index("--save-bundled") + 1])
        sys.exit(0)
    if "--digitize" in sys.argv:
        i = sys.argv.index("--digitize")
        base = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.splitext(sys.argv[i + 1])[0] + "_extracted"
        sys.exit(cli_digitize(sys.argv[i + 1], base))
    if sys.platform == "win32":                  # own taskbar button and icon, also when run from the sources
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("KurtTaretto.XYPlotDigitizer")
        except Exception:
            pass
    qapp = QApplication(sys.argv)
    qapp.setStyle("Fusion")
    # window and taskbar icon: the PNG first, Qt reads it without an image plugin (the .ico needs qico)
    folder = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "icon")
    icon = QIcon()
    for f in ("xyd.png", "xyd.ico"):
        pm = QPixmap(os.path.join(folder, f))
        if icon.isNull() and not pm.isNull():
            icon = QIcon(pm)
    if not icon.isNull():
        qapp.setWindowIcon(icon)
    win = App(qapp)
    if not icon.isNull():
        win.setWindowIcon(icon)
    if win._maximize:
        win.showMaximized()
    else:
        win.show()
    if sys.platform == "win32":
        set_taskbar_icon(win, os.path.join(folder, "xyd.ico"))
    if "--screenshot" in sys.argv:              # build check: start, optionally process a picture, save a
        target = sys.argv[sys.argv.index("--screenshot") + 1]          # picture of the window, quit
        snap = lambda *_: QTimer.singleShot(1500, lambda: (win.grab().save(target), qapp.quit()))
        win.auto = True
        if "--load" in sys.argv:
            win.open_path(sys.argv[sys.argv.index("--load") + 1])
            win.process()
            win.worker.done.connect(snap)
            win.worker.failed.connect(snap)
        else:
            snap()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
