"""Look of the interface (same as IV Analysis): colour tokens, Qt palette and style sheet, small icons drawn at start-up.
Accent and signal colours come from the IBM colour-blind safe palette (IBM Design Library):
blue #648FFF, purple #785EF0, magenta #DC267F, orange #FE6100, yellow #FFB000."""
import os

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen

IBM = dict(blue="#648FFF", purple="#785EF0", magenta="#DC267F", orange="#FE6100", yellow="#FFB000")

THEMES = {
    "dark": dict(
        bg="#24282E", base="#16191D", border="#383D44", ctl="#464C54", header="#383D44", text="#E4E7EB",
        muted="#9AA1AA", disabled="#6B727B", hover="#2D3239", press="#383D44", field="#24282E",
        accent=IBM["blue"], accent_text=IBM["blue"], accent_bg="#202E52", sel="#2C3F73", indicator="#B4B9C0",
        primary=IBM["orange"], primary_hover="#FF772B", primary_press="#E25600", on_primary="#FFFFFF",
        ok=IBM["blue"], warn=IBM["yellow"], err="#F2569D",
        fig="#24282E", ax="#24282E", grid="#383D44", edge="#7A818B"),
    "light": dict(
        bg="#F5F6F8", base="#E3E6EB", border="#CDD2D9", ctl="#B4BAC3", header="#E1E5EA", text="#1F2328",
        muted="#5F6670", disabled="#A3A9B2", hover="#EAEDF1", press="#DDE1E7", field="#FFFFFF",
        accent=IBM["blue"], accent_text="#3A62D8", accent_bg="#E0E8FF", sel="#CFDCFF", indicator="#5F6670",
        primary=IBM["orange"], primary_hover="#FF772B", primary_press="#E25600", on_primary="#FFFFFF",
        ok="#3A62D8", warn="#9A6700", err="#C21E6F",
        fig="#F5F6F8", ax="#FFFFFF", grid="#DDE1E7", edge="#6B727B"),
}


def tokens(name):
    return THEMES.get(name, THEMES["dark"])


# ---------------------------------------------------------------- icons (check mark, plus, minus, combo bar)
def _image(w, h, scale):
    img = QImage(int(w * scale), int(h * scale), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(scale, scale)
    return img, p


def make_icons(folder, name):
    """Writes the few bitmaps the style sheet needs (normal and @2x for high-DPI screens); returns their paths."""
    t = tokens(name)
    os.makedirs(folder, exist_ok=True)
    out = {}
    for key in ("check", "plus", "minus", "bar"):
        for scale, suffix in ((1, ""), (2, "@2x")):
            if key == "check":
                img, p = _image(16, 16, scale)
                pen = QPen(QColor("#FFFFFF"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                p.setPen(pen)
                p.drawPolyline([QPointF(4.0, 8.4), QPointF(7.0, 11.2), QPointF(12.2, 5.2)])
            elif key == "bar":
                img, p = _image(10, 4, scale)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t["indicator"]))
                p.drawRoundedRect(QRectF(0, 0, 10, 4), 1, 1)
            else:
                img, p = _image(9, 9, scale)
                p.setRenderHint(QPainter.Antialiasing, False)
                p.setPen(QPen(QColor(t["indicator"]), 1))
                p.drawLine(1, 4, 7, 4)
                if key == "plus":
                    p.drawLine(4, 1, 4, 7)
            p.end()
            path = os.path.join(folder, "%s_%s%s.png" % (key, name, suffix))
            img.save(path, "PNG")
            if not suffix:
                out[key] = path.replace("\\", "/")
    return out


# ---------------------------------------------------------------- palette (used by widgets the style sheet leaves alone)
def palette(name):
    t = tokens(name)
    c = QColor
    pal = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "field"),
                      (QPalette.AlternateBase, "hover"), (QPalette.Text, "text"), (QPalette.Button, "bg"),
                      (QPalette.ButtonText, "text"), (QPalette.ToolTipBase, "header"), (QPalette.ToolTipText, "text"),
                      (QPalette.Highlight, "sel"), (QPalette.HighlightedText, "text"), (QPalette.PlaceholderText, "muted"),
                      (QPalette.Link, "accent_text"), (QPalette.Light, "ctl"), (QPalette.Midlight, "border"),
                      (QPalette.Mid, "border"), (QPalette.Dark, "base"), (QPalette.Shadow, "base")):
        pal.setColor(role, c(t[key]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c(t["disabled"]))
    return pal


# ---------------------------------------------------------------- style sheet
QSS = """
QWidget { color: %(text)s; }
QMainWindow, QWidget#root, QWidget#page, QWidget#plain { background: %(bg)s; }
QScrollArea { background: %(bg)s; border: none; }
QLabel { background: transparent; }
QToolTip { background: %(header)s; color: %(text)s; border: 1px solid %(ctl)s; padding: 5px 8px; }

/* ---- top bar: flat text buttons */
QWidget#tabcorner { background: %(base)s; }
QToolButton#menu { background: transparent; border: none; border-radius: 4px; padding: 6px 14px; }
QToolButton#menu:hover { background: %(hover)s; }
QToolButton#menu:pressed { background: %(press)s; }
QToolButton#menu:disabled { color: %(disabled)s; }
QFrame#vsep { background: %(border)s; max-width: 1px; min-width: 1px; }
QLabel#status { background: %(bg)s; padding: 5px 10px; border-top: 1px solid %(border)s; }

/* ---- tabs: dark strip, selected tab raised in the page colour with accent text */
QTabWidget#main { background: %(base)s; }
QTabWidget#main::pane { border: none; border-top: 1px solid %(border)s; background: %(bg)s; top: -1px; }
QTabWidget#main > QTabBar { background: %(base)s; }
QTabBar::tab { background: %(base)s; color: %(text)s; padding: 7px 17px; margin-top: 3px;
               border: 1px solid transparent; border-bottom: none;
               border-top-left-radius: 9px; border-top-right-radius: 9px; }
QTabBar::tab:hover:!selected { background: %(hover)s; }
QTabBar::tab:selected { background: %(bg)s; color: %(accent_text)s; border-color: %(border)s; }
QTabBar QToolButton { background: %(base)s; border: none; }

/* ---- group boxes: thin rounded frame, title on the frame */
QGroupBox { border: 1px solid %(border)s; border-radius: 9px; margin-top: 11px; padding: 14px 10px 8px 10px;
            background: %(bg)s; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0px 6px;
                   background: %(bg)s; color: %(text)s; }

/* ---- buttons: outlined; accent = suggested step; primary = the action that computes */
QPushButton { background: %(bg)s; border: 1px solid %(ctl)s; border-radius: 5px; padding: 5px 14px; min-height: 18px; }
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed { background: %(press)s; }
QPushButton:disabled { background: %(base)s; color: %(disabled)s; border-color: %(border)s; }
QPushButton[kind="accent"] { background: %(accent_bg)s; border: 1px solid %(accent)s; }
QPushButton[kind="accent"]:hover { border: 2px solid %(accent)s; padding: 4px 13px; }
QPushButton[kind="primary"] { background: %(primary)s; color: %(on_primary)s; border: none; border-radius: 6px;
                              padding: 8px 20px; }
QPushButton[kind="primary"]:hover { background: %(primary_hover)s; }
QPushButton[kind="primary"]:pressed { background: %(primary_press)s; }
QPushButton[kind="primary"]:disabled { background: %(border)s; color: %(disabled)s; }
QToolButton#info { border: 1px solid %(ctl)s; border-radius: 9px; min-width: 16px; max-width: 16px;
                   min-height: 16px; max-height: 16px; color: %(muted)s; font-style: italic; padding: 0px; }
QToolButton#info:hover { border-color: %(accent)s; color: %(accent_text)s; }

/* ---- input fields */
QLineEdit, QComboBox, QAbstractSpinBox { background: %(field)s; border: 1px solid %(ctl)s; border-radius: 5px;
        padding: 3px 8px; min-height: 20px; selection-background-color: %(accent)s; selection-color: #FFFFFF; }
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border-color: %(accent)s; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { background: %(base)s; color: %(disabled)s;
        border-color: %(border)s; }
QComboBox { padding-right: 26px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }
QComboBox::down-arrow { image: url("%(i_bar)s"); width: 10px; height: 4px; }
QComboBox QAbstractItemView { background: %(bg)s; color: %(text)s; border: 1px solid %(ctl)s; outline: 0;
        selection-background-color: %(sel)s; selection-color: %(text)s; padding: 2px; }
QAbstractSpinBox { padding-right: 22px; }
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 19px;
        border-left: 1px solid %(ctl)s; border-top-right-radius: 5px; background: transparent; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 19px;
        border-left: 1px solid %(ctl)s; border-bottom-right-radius: 5px; background: transparent; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: %(hover)s; }
QAbstractSpinBox::up-arrow { image: url("%(i_plus)s"); width: 9px; height: 9px; }
QAbstractSpinBox::down-arrow { image: url("%(i_minus)s"); width: 9px; height: 9px; }

/* ---- check boxes and radio buttons: accent fill */
QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(ctl)s; border-radius: 4px; background: %(bg)s; }
QCheckBox::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url("%(i_check)s"); }
QCheckBox::indicator:disabled { border-color: %(border)s; background: %(base)s; }
QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid %(ctl)s; border-radius: 9px; background: %(bg)s; }
QRadioButton::indicator:hover { border-color: %(accent)s; }
QRadioButton::indicator:checked { width: 6px; height: 6px; border: 6px solid %(accent)s; border-radius: 9px;
                                  background: %(bg)s; }

/* ---- sliders */
QSlider::groove:horizontal { height: 4px; background: %(ctl)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0px; border-radius: 8px; background: %(accent)s;
                             border: 1px solid %(bg)s; }

/* ---- tables */
QTableWidget { background: %(bg)s; alternate-background-color: %(bg)s; border: 1px solid %(border)s;
               border-radius: 6px; gridline-color: %(border)s; selection-background-color: %(sel)s;
               selection-color: %(text)s; outline: 0; }
QTableWidget::item { padding: 3px 10px; }
QHeaderView { background: %(header)s; border: none; }
QHeaderView::section { background: %(header)s; color: %(text)s; border: none; padding: 6px 10px; }
QTableCornerButton::section { background: %(header)s; border: none; }
QWidget#filetable { background: %(bg)s; border: 1px solid %(border)s; border-radius: 6px; }
QWidget#th { background: %(header)s; }
QLabel#thl { background: %(header)s; padding: 6px 10px; }
QLabel#td { padding: 2px 10px; }

/* ---- text pages */
QTextBrowser, QPlainTextEdit { background: %(bg)s; border: 1px solid %(border)s; border-radius: 9px; padding: 10px;
                               selection-background-color: %(accent)s; selection-color: #FFFFFF; }

/* ---- scroll bars */
QScrollBar:vertical { background: %(bg)s; width: 13px; margin: 0px; border: none; }
QScrollBar::handle:vertical { background: %(ctl)s; border-radius: 4px; min-height: 36px; margin: 2px 3px; }
QScrollBar:horizontal { background: %(bg)s; height: 13px; margin: 0px; border: none; }
QScrollBar::handle:horizontal { background: %(ctl)s; border-radius: 4px; min-width: 36px; margin: 3px 2px; }
QScrollBar::handle:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- plot toolbar, dialogs, menus */
QToolBar { background: %(bg)s; border: none; spacing: 6px; padding: 2px; }
QToolBar QToolButton { background: transparent; border: none; border-radius: 4px; padding: 4px; }
QToolBar QToolButton:hover { background: %(hover)s; }
QToolBar QToolButton:checked { background: %(accent_bg)s; border: 1px solid %(accent)s; }
QToolBar QLabel { color: %(muted)s; }
QMessageBox, QDialog { background: %(bg)s; }
QMenu { background: %(bg)s; border: 1px solid %(ctl)s; }
QMenu::item:selected { background: %(sel)s; }

/* ---- additions for the digitizer: tick boxes inside tables, progress bar, splitters, picture view */
QTableWidget::indicator { width: 16px; height: 16px; border: 1px solid %(ctl)s; border-radius: 4px; background: %(bg)s; }
QTableWidget::indicator:hover { border-color: %(accent)s; }
QTableWidget::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url("%(i_check)s"); }
QProgressBar { background: %(field)s; border: 1px solid %(ctl)s; border-radius: 5px; text-align: center; min-height: 16px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 4px; }
QSplitter::handle { background: %(bg)s; }
QSplitter::handle:horizontal { width: 8px; }
QSplitter::handle:vertical { height: 8px; }
QGraphicsView { background: %(base)s; border: 1px solid %(border)s; border-radius: 6px; }
"""


def stylesheet(name, icons):
    d = dict(tokens(name))
    d.update(i_check=icons["check"], i_plus=icons["plus"], i_minus=icons["minus"], i_bar=icons["bar"])
    return QSS % d
