"""
XY Plot Digitizer - extraction core (no interface code).
Reads the data of one XY line / scatter plot from a picture (process_raster) or from the vector paths
of a PDF page (process_vector; EPS / PS are converted with Ghostscript first).
The interface is in xydapp.py, the look in xydtheme.py.
"""
import colorsys
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

APP_NAME = "XY Plot Digitizer"

# ----------------------------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------------------------


@dataclass
class Axis:
    """Tick spacing: value = a*pixel + b (linear), 10**(a*pixel + b) (log; ln spacing is the same),
    or 1 / (a*pixel + b) (reciprocal, '1/x')."""
    log: bool = False
    a: float = 1.0
    b: float = 0.0
    n: int = 0          # tick labels consistent with the fit
    ntot: int = 0       # tick labels read
    inv: bool = False

    @property
    def kind(self):
        return "log" if self.log else "1/x" if self.inv else "linear"

    def to_data(self, p):
        v = self.a * np.asarray(p, float) + self.b
        if self.log:
            return 10.0 ** v
        if self.inv:
            with np.errstate(divide="ignore"):
                return 1.0 / v
        return v

    @staticmethod
    def from_two_points(p1, v1, p2, v2, kind):
        """kind: 'linear', 'log' or '1/x' (True / False are taken as log / linear)."""
        kind = {True: "log", False: "linear"}.get(kind, kind)
        f1, f2 = (math.log10(v1), math.log10(v2)) if kind == "log" else (1.0 / v1, 1.0 / v2) if kind == "1/x" else (v1, v2)
        a = (f2 - f1) / (p2 - p1)
        return Axis(kind == "log", a, f1 - a * p1, 2, 2, kind == "1/x")


@dataclass
class Series:
    kind: str                    # 'line' or 'scatter'
    px: np.ndarray               # (N, 2) image pixel coordinates (x, y)
    color: tuple                 # RGB 0-255
    style: str                   # solid / dashed / dotted / dash-dot, or the marker name
    note: str = ""
    name: str = ""
    px_alt: np.ndarray = None    # scatter only: bounding-box centres (px holds centroids)
    use: bool = True
    yref: str = "left"           # 'left' or 'right': the Y axis this series is read on


@dataclass
class Result:
    image: np.ndarray            # RGB image shown to the user
    box: tuple                   # plot frame (x0, y0, x1, y1) in pixels
    xaxis: Axis
    yaxis: Axis
    series: list
    source: str                  # 'vector' or 'raster'
    log: list = field(default_factory=list)
    yaxis2: Axis = None          # right-hand Y axis, when the plot has one that differs from the left


def _noprog(pct, msg=""):
    pass


# ----------------------------------------------------------------------------------------------
# Numbers and axis calibration
# ----------------------------------------------------------------------------------------------

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-", "‐": "-", "﹣": "-", "－": "-", "⁻": "-", "_": "-"})
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺", "0123456789+")
_LOOKALIKE = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "°": "0", "B": "8"})


def parse_number(s):
    s = s.strip().translate(_MINUS).translate(_SUP).replace(" ", "").rstrip("%")
    if re.fullmatch(r"-?[1-9]\d{0,2}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", s):
        s = s.replace(",", ".")
    m = re.fullmatch(r"(-?\d*\.?\d+)[x×*·]10\^?(-?\d+)", s)
    if m:
        return float(m[1]) * 10.0 ** int(m[2])
    m = re.fullmatch(r"10\^(-?\d+)", s)
    if m:
        return 10.0 ** int(m[1])
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def power_value(base, exp):
    """Value of a tick label written as base^exp (e.g. 10 and -3)."""
    exp = re.sub(r"[^0-9+\-.]", "", exp.translate(_MINUS).translate(_SUP))
    base = base.strip().translate(_MINUS).replace(" ", "")
    try:
        e = float(exp)
        m = re.fullmatch(r"(-?\d*\.?\d+)[x×*·]10", base)
        if m:
            return float(m[1]) * 10.0 ** e
        return float(base) ** e
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def fit_axis(pos, val, length, increasing):
    """Fit pixel -> value from tick labels. Linear or log is chosen by the number of labels each
    model explains (misread labels are left out as outliers)."""
    pos, val = np.asarray(pos, float), np.asarray(val, float)
    tol = max(1.5, 0.006 * length)
    best, best_key = None, None
    for log in (False, True):
        ok = val > 0 if log else np.ones(len(val), bool)
        p, f = pos[ok], (np.log10(val[ok]) if log else val[ok])
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if f[i] == f[j] or abs(p[i] - p[j]) < 3 * tol:
                    continue
                c = (p[j] - p[i]) / (f[j] - f[i])
                inl = np.abs(c * f + (p[i] - c * f[i]) - p) < tol
                if len(np.unique(f[inl])) < 2:
                    continue
                a, b = np.polyfit(p[inl], f[inl], 1)
                rms = float(np.sqrt(np.mean((a * p[inl] + b - f[inl]) ** 2)) / abs(a))
                n = int(inl.sum())
                pow10 = bool(np.all(np.abs(np.log10(val[ok][inl]) % 1) < 1e-9)) if np.all(val[ok][inl] > 0) else False
                prefer = (log and pow10) if n == 2 else (not log)
                key = (n, (a > 0) == increasing, prefer, -rms)
                if best_key is None or key > best_key:
                    best_key, best = key, Axis(log, float(a), float(b), n, len(pos))
    return best


def snap_to_ticks(labels, ticks, tol):
    """labels [(pos, value)], ticks [(pos, length)] -> labels moved onto the tick mark they belong to:
    the nearest of the long ticks within tol. "Long" is judged among the ticks near that label, so one
    over-long stub elsewhere (a curve that touches the axis) does not switch the snapping off."""
    if len(ticks) < 2:
        return labels
    tp = np.array([t[0] for t in ticks], float)
    tl = np.array([t[1] for t in ticks], float)
    out = []
    for p, v in labels:
        near = np.abs(tp - p) <= tol
        if not near.any():
            out.append((p, v))
            continue
        long_ = near & (tl >= 0.75 * tl[near].max())
        k = np.nonzero(long_)[0][np.abs(tp[long_] - p).argmin()]
        out.append((float(tp[k]), v))
    return out


def describe_axis(ax, p_lo, p_hi):
    if ax is None:
        return "not detected"
    lo, hi = ax.to_data(p_lo), ax.to_data(p_hi)
    return f"{ax.kind}, {lo:.4g} to {hi:.4g} ({ax.n}/{ax.ntot} tick labels used)"


def second_axis(yax, yax2, p_top, p_bot, log):
    """Decide what to do with the Y axis read on the right side of the frame: None when it repeats the
    left axis, the only axis when the left one has no labels, else a second axis. Returns (yax, yax2)."""
    if yax2 is None or yax2.n < 3:
        return yax, None
    if yax is None:
        log.append("Y axis read from the right side of the frame.")
        return yax2, None
    if yax.log == yax2.log:
        f = (lambda v: np.log10(v)) if yax.log else (lambda v: v)
        a, b = f(yax.to_data([p_top, p_bot])), f(yax2.to_data([p_top, p_bot]))
        if np.all(np.abs(a - b) < 0.02 * (abs(a[0] - a[1]) or 1.0)):
            return yax, None                               # the same scale on both sides
    log.append(f"second Y axis on the right: {describe_axis(yax2, p_bot, p_top)}")
    return yax, yax2


# ----------------------------------------------------------------------------------------------
# OCR (RapidOCR: pure pip install, models are inside the wheel)
# ----------------------------------------------------------------------------------------------


class OCR:
    _eng = None
    error = None

    @classmethod
    def engine(cls):
        if cls._eng is None and cls.error is None:
            try:
                from rapidocr import RapidOCR
                cls._eng = RapidOCR(params={"Global.log_level": "error"})
            except Exception as e:  # not installed, or model failed to load
                cls.error = f"{type(e).__name__}: {e}"
        return cls._eng

    @classmethod
    def read(cls, crop, bg):
        """Recognise one text snippet (RGB crop). Returns (text, score)."""
        eng = cls.engine()
        if eng is None or crop.size == 0:
            return "", 0.0
        col = [int(c) for c in bg]
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=col)
        s = float(np.clip(48.0 / crop.shape[0], 1.0, 6.0))
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        try:
            r = eng(crop[:, :, ::-1].copy(), use_det=False, use_cls=False, use_rec=True)
            return (r.txts[0], float(r.scores[0])) if r.txts else ("", 0.0)
        except Exception:
            return "", 0.0

    @classmethod
    def detect(cls, img):
        """Find text in a whole RGB image. Returns [(text, score, x0, y0, x1, y1, quad)]: the upright box and
        the four corners of the (possibly rotated) text box."""
        eng = cls.engine()
        if eng is None:
            return []
        try:
            r = eng(img[:, :, ::-1].copy(), use_det=True, use_cls=False, use_rec=True)
            if r.boxes is None:
                return []
            return [(t, float(s), float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max()),
                     np.asarray(b, float).reshape(4, 2))
                    for b, t, s in zip(r.boxes, r.txts, r.scores)]
        except Exception:
            return []


def is_real_text(text, score):
    """Text detectors also 'read' markers (o, x, squares ...). Returns 2 for a confident word or
    formula, 1 for a confident single character that does not look like a marker, 0 otherwise."""
    if any(ord(c) >= 0x2E80 for c in text):
        return 0
    letters = set(c for c in text if c.isalpha())                 # also Greek (\u00b5m, \u03bbs)
    if score >= 0.8 and len(letters) >= 2:
        return 2
    t = text.strip()
    if score >= 0.9 and len(t) >= 5 and len(re.findall(r"\d", t)) >= 3 and re.search(r"[=/()+.,]", t):
        return 2                                           # numbers written into the plot, e.g. (1+0.85)/2=0.925
    if score >= 0.75 and len(t) >= 5 and len(re.findall(r"\d", t)) >= 3 and re.search(r"[.,x\u00d7]", t):
        return 1                                           # numbers of a legend (1.7 x 10^19): taken when stacked
    return 1 if (score >= 0.9 and len(t) == 1 and t.isalnum() and t not in "oO0QDxXvVA8lI1T") else 0


def legend_like(boxes):
    """Indices of text boxes stacked and left-aligned with another one (legend rows)."""
    idx = set()
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            h = 0.5 * ((a[3] - a[1]) + (b[3] - b[1]))
            dy = abs(0.5 * (a[1] + a[3]) - 0.5 * (b[1] + b[3]))
            if abs(a[0] - b[0]) < 0.7 * h and 0.7 * h < dy < 2.8 * h:
                idx.add(i)
    return idx


# ----------------------------------------------------------------------------------------------
# Raster: frame, tick labels
# ----------------------------------------------------------------------------------------------


def dominant_color(pix):
    pix = pix.reshape(-1, 3)
    if len(pix) > 200000:
        pix = pix[:: len(pix) // 200000 + 1]
    q = pix.astype(np.int32) >> 3
    key = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    return np.median(pix[key == vals[cnt.argmax()]], axis=0)


def _cc(mask, conn=8):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=conn)


def find_frame(img):
    """Locate the axes: the lowest long horizontal line and the vertical line that meets it."""
    h, w = img.shape[:2]
    bg = dominant_color(img)
    diff = np.abs(img.astype(np.int16) - bg).max(axis=2)
    panel = diff <= 12                                     # grey-panel styles: the most common colour is the plot area
    rows, cols = np.nonzero(panel.mean(axis=1) > 0.3)[0], np.nonzero(panel.mean(axis=0) > 0.3)[0]
    if len(rows) and len(cols) and rows[0] > 3 and rows[-1] < h - 4 and cols[0] > 3 and cols[-1] < w - 4 \
            and (rows[-1] - rows[0]) * (cols[-1] - cols[0]) > 0.2 * h * w:
        return dict(box=(int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])), t=(0, 0, 0, 0), panel=True)
    fg = (diff > 30).astype(np.uint8)
    fg[:2] = 0; fg[-2:] = 0; fg[:, :2] = 0; fg[:, -2:] = 0
    oh = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((1, max(8, int(0.35 * w))), np.uint8))
    ov = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((max(8, int(0.35 * h)), 1), np.uint8))
    rows = np.nonzero(oh.any(axis=1))[0]
    if len(rows) == 0:
        return None
    yb = int(rows.max())
    xs = np.nonzero(oh[yb])[0]
    xl, xr = int(xs.min()), int(xs.max())
    # a scanned line is tilted: its lowest pixel row holds only the part past the last 1-px step. The line is
    # followed from row to row (up to 6 rows above) where it goes on there for a long stretch; a line cap or a
    # tick going on for a pixel or two does not move the end
    def walk(x0, step):
        x, r, need = x0, yb, max(8, int(0.03 * w))
        while 0 <= x + step < w:
            if fg[r, x + step]:
                x += step
                continue
            for rr in (r - 1, r + 1):
                if yb - 6 <= rr <= yb and fg[rr, x + step]:
                    seg = fg[rr, x + step::step] if step > 0 else fg[rr, x + step::-1]
                    run = int(np.argmin(seg)) if not seg.all() else len(seg)
                    if run >= need:
                        r = rr
                        break
            else:
                break
        return x if r != yb else x0                        # not tilted: the end as found before
    xl, xr = walk(xl, -1), walk(xr, +1)
    cols = np.nonzero(ov.any(axis=0))[0]
    near_l = cols[(cols >= xl - 3) & (cols <= xl + 0.02 * w)]
    near_r = cols[(cols <= xr + 3) & (cols >= xr - 0.02 * w)]
    if len(near_l):
        xl = int(min(xl, near_l.min()))
        yt = int(np.nonzero(ov[:, near_l].any(axis=1))[0].min())
    elif len(near_r):
        yt = int(np.nonzero(ov[:, near_r].any(axis=1))[0].min())
    else:
        yt = int(rows.min()) if rows.min() < yb - 0.2 * h else 2
    if xr - xl < 20 or yb - yt < 20:
        return None

    def thick(get, start, step):
        k = 0
        while k < 3 and get(start + step * k) <= 0.7:      # line caps can overshoot the crossing line
            k += 1
        if k == 3:
            return 0
        t = k
        while t < k + 12 and get(start + step * t) > 0.7:
            t += 1
        return t
    tb = thick(lambda r: oh[r, xl:xr + 1].mean(), yb, -1)
    tt = thick(lambda r: oh[r, xl:xr + 1].mean(), yt, +1)
    tl = thick(lambda c: ov[yt:yb + 1, c].mean(), xl, +1)
    tr = thick(lambda c: ov[yt:yb + 1, c].mean(), xr, -1)
    if max(tb, tt, tl, tr) >= 12:          # a filled axes patch, not a line frame
        tb = tt = tl = tr = 0
    return dict(box=(xl, yt, xr, yb), t=(tl, tt, tr, tb))


def _read_label(img, comps, bg):
    """comps: character boxes (x0, y0, x1, y1) of one tick label. Returns (value, text, box, y of the base)."""
    comps = sorted(comps)
    x0 = min(c[0] for c in comps); y0 = min(c[1] for c in comps)
    x1 = max(c[2] for c in comps); y1 = max(c[3] for c in comps)
    hs = [c[3] - c[1] + 1 for c in comps]
    hb = max(hs)
    # an exponent is told by its raised baseline, not by its size: some fonts print it nearly as large as the 10
    tall = [c for c, hh in zip(comps, hs) if hh >= 0.6 * hb]
    bottom = float(max(c[3] for c in tall))
    base = [c for c in tall if c[3] >= bottom - 0.25 * hb]
    bx1 = max(c[2] for c in base)
    sup = [c for c, hh in zip(comps, hs) if c[3] < bottom - 0.3 * hb and hh <= 1.05 * hb and c[0] > bx1 - 1]

    def dash(c):
        return (c[2] - c[0] + 1) >= 1.6 * (c[3] - c[1] + 1) and (c[3] - c[1] + 1) <= 0.35 * hb
    by0, by1 = min(c[1] for c in base), max(c[3] for c in base)
    if sup:
        xs = min(c[0] for c in sup)
        btxt, _ = OCR.read(img[y0:y1 + 1, x0:xs], bg)
        neg = any(dash(c) for c in sup)
        dig = [c for c in sup if not dash(c)]
        if not dig:
            return None, btxt, (x0, y0, x1, y1), 0.5 * (by0 + by1)
        etxt, _ = OCR.read(img[min(c[1] for c in dig):max(c[3] for c in dig) + 1,
                               min(c[0] for c in dig):max(c[2] for c in dig) + 1], bg)
        etxt = re.sub(r"[^0-9]", "", etxt.translate(_SUP).translate(_LOOKALIKE))
        btxt = re.sub(r"[^0-9.x×*·\-]", "", btxt.translate(_MINUS).translate(_LOOKALIKE))
        val = power_value(btxt, ("-" if neg else "") + etxt) if etxt else None
        return val, f"{btxt}^{'-' if neg else ''}{etxt}", (x0, y0, x1, y1), 0.5 * (by0 + by1)
    txt, _ = OCR.read(img[y0:y1 + 1, x0:x1 + 1], bg)
    t = re.sub(r"[^0-9eE.,+\-x×*·^]", "", txt.translate(_MINUS).translate(_SUP).translate(_LOOKALIKE)).lstrip("-+")
    neg = dash(comps[0]) and comps[0][3] < bottom - 0.2 * hb
    val = parse_number(t)
    if val is not None and neg:
        val = -val
    return val, ("-" if neg else "") + t, (x0, y0, x1, y1), 0.5 * (by0 + by1)


def margin_labels(img, frame, side, read=True, words=None):
    """Tick labels and tick marks outside the frame. side 'x': below the frame, 'y': left of it,
    'y2': right of it. words: quads [4 x 2] of words found by the text detector (axis titles); their ink
    is left out, so that a title written close to the labels is not read as part of them.
    Returns (labels [(pos, value)], ticks [(pos, length)], char height)."""
    h, w = img.shape[:2]
    xl, yt, xr, yb = frame["box"]
    excl = np.zeros((h, w), np.uint8)
    for q in words or []:
        cv2.fillPoly(excl, [np.round(np.asarray(q)).astype(np.int32)], 1)
    if side == "x":
        pad = int(0.08 * w)
        ox, oy = max(0, xl - pad), yb + 1
        band = img[oy:h, ox:min(w, xr + pad + 1)]
        bex = excl[oy:h, ox:min(w, xr + pad + 1)]
    else:                                   # rotate so that the axis is the top edge of the band
        pad = int(0.08 * h)
        ox, oy = 0, max(0, yt - pad)
        if side == "y2":
            band = np.ascontiguousarray(img[oy:min(h, yb + pad + 1), xr + 1:w].transpose(1, 0, 2))
            bex = np.ascontiguousarray(excl[oy:min(h, yb + pad + 1), xr + 1:w].T)
        else:
            band = img[oy:min(h, yb + pad + 1), 0:xl]
            band = np.ascontiguousarray(band[:, ::-1].transpose(1, 0, 2))
            bex = np.ascontiguousarray(excl[oy:min(h, yb + pad + 1), 0:xl][:, ::-1].T)
    if band.shape[0] < 6 or band.shape[1] < 6:
        return [], [], 0
    bg = dominant_color(band)
    fg = np.abs(band.astype(np.int16) - bg).max(axis=2) > 70
    fg &= bex == 0
    k = 0
    while k < 3 and fg[k].mean() > 0.4:     # anti-aliased edge of the axis line
        fg[k] = False
        k += 1
    n, lab, st, _ = _cc(fg)
    ticks, chars = [], []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if y <= k + 1 and cw <= max(6, 0.012 * min(h, w)) and ch <= max(8, 0.05 * min(h, w)):
            ticks.append((x + (cw - 1) / 2.0, int(ch)))
        elif y > k + 1 and area >= 2:
            chars.append((int(x), int(y), int(x + cw - 1), int(y + ch - 1)))
    if not chars:
        return [], ticks_out(ticks, side, ox, oy), 0
    # first block of text next to the axis (the axis title is the next block)
    occ = np.zeros(band.shape[0], bool)
    for c in chars:
        occ[c[1]:c[3] + 1] = True
    hchar = float(np.median([c[3] - c[1] + 1 for c in chars] if side == "x" else [c[2] - c[0] + 1 for c in chars]))
    gap_ok = 1 if side == "x" else max(2, int(0.45 * hchar))
    if side != "x":                                        # many small tick pieces outside the frame pull the median
        hlen = float(np.percentile([c[2] - c[0] + 1 for c in chars], 80))   # down; then the gap between '1' and '0'
        gap_ok = max(gap_ok, int(0.45 * hlen))                              # of '10' split a label
    runs, r = [], 0                                        # blocks of text rows, going away from the axis
    while r < len(occ):
        if occ[r]:
            e, miss = r, 0
            for q in range(r, len(occ)):
                if occ[q]:
                    e, miss = q, 0
                else:
                    miss += 1
                    if miss > gap_ok:
                        break
            runs.append((r, e))
            r = e + 1
        else:
            r += 1
    r0 = r1 = None
    for a, b in runs:                                      # the first block that looks like a row of labels
        inside = [c for c in chars if a <= 0.5 * (c[1] + c[3]) <= b]
        if b - a + 1 >= max(5, 0.012 * min(h, w)) and len(inside) >= 2:
            r0, r1 = a, b
            break
    if r0 is None:
        return [], ticks_out(ticks, side, ox, oy), 0
    chars = [c for c in chars if r0 <= 0.5 * (c[1] + c[3]) <= r1]
    # back to image coordinates
    if side == "x":
        chars = [(c[0] + ox, c[1] + oy, c[2] + ox, c[3] + oy) for c in chars]
    elif side == "y2":                      # band (row, col) = image (xr+1+row, col+oy)
        chars = [(xr + 1 + c[1], c[0] + oy, xr + 1 + c[3], c[2] + oy) for c in chars]
    else:                                   # band (row, col) = image (xl-1-row, col+oy)
        chars = [(xl - 1 - c[3], c[0] + oy, xl - 1 - c[1], c[2] + oy) for c in chars]
    hs = sorted(c[3] - c[1] + 1 for c in chars)
    hchar = float(hs[int(0.8 * (len(hs) - 1))])
    if not read:                                           # only the character height is wanted
        return [], ticks_out(ticks, side, ox, oy), hchar
    # group characters into labels
    groups = []
    if side == "x":
        dot = lambda c: c[3] - c[1] + 1 <= 0.3 * hchar and c[2] - c[0] + 1 <= 0.3 * hchar   # a decimal point
        # light fonts space the digits of one label wide ('1' and '0' of 10^-8 in Aalbers Fig. 4c: 0.41 x the
        # character height): allow up to 0.6 x when every such gap is far below the gaps between labels
        cs, gaps, right = sorted(chars), [], None
        for c in cs:
            if right is not None:
                gaps.append(c[0] - right)
            right = c[2] if right is None else max(right, c[2])
        wide = [g for g in gaps if g > 0.6 * hchar]
        mid = [g for g in gaps if 0.4 * hchar < g <= 0.6 * hchar]
        join = 0.6 * hchar if wide and mid and max(mid) < 0.25 * min(wide) else 0.4 * hchar
        for c in cs:
            gap = c[0] - max(g[2] for g in groups[-1]) if groups else None
            raised = groups and c[3] < max(g[3] for g in groups[-1]) - 0.25 * hchar     # an exponent
            point = groups and (dot(c) or dot(max(groups[-1], key=lambda g: g[2])))  # light fonts space '1.2' wide
            if groups and (gap <= join or (raised and gap <= 0.9 * hchar) or (point and gap <= 0.7 * hchar)):
                groups[-1].append(c)
            else:
                groups.append([c])
    else:
        for c in sorted(chars, key=lambda c: c[1]):
            if groups and c[1] - max(g[3] for g in groups[-1]) <= 0.15 * hchar:
                groups[-1].append(c)
            else:
                groups.append([c])
    labels = []
    for g in groups:
        val, txt, box, ybase = _read_label(img, g, bg)
        if val is not None:
            labels.append((0.5 * (box[0] + box[2]) if side == "x" else ybase, val))
    return labels, ticks_out(ticks, side, ox, oy), hchar


def ticks_out(ticks, side, ox, oy):
    return [(p + ox, ln) for p, ln in ticks] if side == "x" else [(p + oy, ln) for p, ln in ticks]


# ----------------------------------------------------------------------------------------------
# Raster: clean-up of the plot area
# ----------------------------------------------------------------------------------------------


def remove_stubs(fg, lmax, frame_t=0):
    """Erase short stubs attached to the border of the plot area (inward tick marks).
    Returns their positions along the bottom, left and right borders as [(pos, length)]."""
    found = {}
    wmax = max(5, int(0.012 * min(fg.shape)), 2 * int(frame_t) + 2)   # stub width: thicker in a high-resolution or
                                                                     # enlarged picture (as thick as the frame line)
    views = dict(bottom=fg[::-1, :], top=fg, left=fg.T, right=fg[:, ::-1].T)
    for name, a in views.items():
        depth = min(lmax + 2, a.shape[0])
        strip = a[:depth]
        run = np.where(strip.all(axis=0), depth, np.argmin(strip, axis=0))
        cand = (run >= 2) & (run <= lmax)
        out, c = [], 0
        while c < len(cand):
            if cand[c]:
                e = c
                while e + 1 < len(cand) and cand[e + 1]:
                    e += 1
                if e - c + 1 <= wmax:
                    for q in range(c, e + 1):
                        strip[:run[q], q] = False
                    out.append((0.5 * (c + e), int(run[c:e + 1].max())))
                c = e + 1
            else:
                c += 1
        found[name] = out
    return found["bottom"], found["left"], found["right"]



def erase_crumbs(fg, stubs):
    """Specks left by the tick marks (their anti-aliased tips, 1 px deep remnants at the border): pieces of up
    to 6 px that touch an erased tick or lie within 2 px of the border. A row of them along the frame was traced
    as a dotted line and joined to the curves ending there; two of them pulled a curve end down the frame.
    Anti-aliasing leaves specks of 1 to 3 px at any picture size; larger pieces (a symbol cut by a tick) stay."""
    amax = 6
    zone = cv2.dilate(stubs.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    zone[:2] = zone[-2:] = True
    zone[:, :2] = zone[:, -2:] = True
    n, lab, st, _ = _cc(fg)
    hit = np.unique(lab[zone & fg])
    small = [i for i in hit if i > 0 and st[i][4] <= amax]
    if small:
        fg[np.isin(lab, small)] = False
    return len(small)


def _blend_dist(pix, bg, col):
    """Distance of pixels from the segment bg -> col (anti-aliased pixels of a 'col' line lie on it)."""
    d = col - bg
    t = np.clip(((pix - bg) @ d) / max(float(d @ d), 1e-6), 0, 1.2)
    return np.linalg.norm(pix - bg - t[..., None] * d, axis=-1)


def remove_grid(work, fg, bg, xticks, yticks, log):
    """Erase grid lines: thin lines of one colour crossing the plot area at tick positions.
    A grid has a line at most of the major ticks: at least two lines of one colour, and at least half
    as many as there are major ticks inside the plot, so flat data lines that happen to lie at a tick
    (constant values such as 1.1 and 1.2) are kept."""
    cands, nmajor = [], {}
    for axis, ticks in ((0, yticks), (1, xticks)):
        a = fg if axis == 0 else fg.T
        c = work if axis == 0 else work.transpose(1, 0, 2)
        n, m = a.shape
        nb = 8
        bs = m // nb
        if bs < 4:
            continue
        cov = np.sort(a[:, :bs * nb].reshape(n, nb, bs).mean(axis=2), axis=1)
        cand = (a.mean(axis=1) >= 0.25) & (cov[:, 2] >= 0.15)       # may be hidden under a legend in places
        tk = np.array([t[0] for t in ticks]) if len(ticks) >= 2 else None
        if tk is not None:
            lmax = max(t[1] for t in ticks)
            nmajor[axis] = sum(1 for t in ticks if t[1] >= 0.75 * lmax and 3 < t[0] < n - 4)
        r = 0
        while r < n:
            if not cand[r]:
                r += 1
                continue
            e = r
            while e + 1 < n and cand[e + 1]:
                e += 1
            if e - r + 1 <= 4 and (tk is None or np.abs(tk - 0.5 * (r + e)).min() <= 3):
                px = c[r:e + 1][a[r:e + 1]].astype(np.float32)
                col = np.median(px, axis=0)
                if (np.abs(px - col).max(axis=1) < 45).mean() >= 0.75:
                    cands.append((axis, r, e, col))
            r = e + 1
    count = 0
    for axis, r, e, col in cands:
        same = sum(o[0] == axis and np.abs(col - o[3]).max() < 40 for o in cands)
        if same < max(2, 0.5 * nmajor.get(axis, 0)):
            continue
        a = fg if axis == 0 else fg.T
        c = work if axis == 0 else work.transpose(1, 0, 2)
        lo, hi = max(0, r - 1), min(a.shape[0], e + 2)
        kill = (_blend_dist(c[lo:hi].astype(np.float32), bg.astype(np.float32), col) < 30) & a[lo:hi]
        c[lo:hi][kill] = bg
        a[lo:hi][kill] = False
        count += 1
    if count:
        log.append(f"removed {count} grid lines")


def find_legend_boxes(fg, hchar=10.0):
    """Rectangular frames inside the plot area (legend or text boxes)."""
    hh, ww = fg.shape
    u = fg.astype(np.uint8)
    oh = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((1, max(12, int(0.05 * ww))), np.uint8))
    ov = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((max(8, int(0.03 * hh)), 1), np.uint8))
    k = max(5, int(0.9 * hchar) | 1)                       # bridges rounded corners
    n, lab, st, _ = _cc(cv2.dilate(oh | ov, np.ones((k, k), np.uint8)))
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if w < 0.06 * ww or h < 0.035 * hh or w * h > 0.6 * ww * hh:
            continue
        sub = lab[y:y + h, x:x + w] == i
        b = k + 2
        sides = (sub[:b].any(axis=0).mean(), sub[-b:].any(axis=0).mean(),
                 sub[:, :b].any(axis=1).mean(), sub[:, -b:].any(axis=1).mean())
        inner = sub[b:-b, b:-b]
        if min(sides) > 0.8 and inner.size and inner.mean() < 0.5:
            frame = (lab == i) & cv2.dilate(u, np.ones((3, 3), np.uint8)).astype(bool)
            frame[y + b:y + h - b, x + b:x + w - b] = False
            boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1), frame))
    return boxes


def erase_inside(work, fg, bg, rects, max_w=None):
    """Erase the connected pieces of ink lying completely inside any of the rectangles."""
    if not rects:
        return 0
    n, lab, st, _ = _cc(fg)
    kill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        for (x0, y0, x1, y1) in rects:
            if x >= x0 and y >= y0 and x + w - 1 <= x1 and y + h - 1 <= y1 and (max_w is None or w <= max_w):
                kill[i] = True
                break
    m = kill[lab]
    work[m] = bg
    fg[m] = False
    return int(kill.sum())


def erase_legend_samples(work, fg, bg, labels):
    """Erase the sample drawn left of each legend label (tight boxes of the label texts): the flat pieces
    on one horizontal line (line, dashes, line with a symbol), or, when there is no line, a symbol next to
    the label. Dashes of steep curves that pass left of the legend stay. Returns the number of samples."""
    if not labels:
        return 0
    n, lab, st, cen = _cc(fg)
    kill, found = np.zeros(n, bool), 0
    for ta, tb, tc, td in labels:
        h = td - tb + 1.0
        cy = 0.5 * (tb + td)
        flat, compact = [], []
        for i in range(1, n):
            x, y, w, hc, _ = st[i]
            if x + w - 1 > ta + 1 or x < ta - 6.0 * h or y < tb - 0.15 * h or y + hc - 1 > td + 0.15 * h:
                continue
            if hc <= 0.8 * h and w >= 1.8 * hc and w >= 3:
                flat.append(i)
            elif max(w, hc) <= 1.3 * h and min(w, hc) >= 3 and abs(cen[i][1] - cy) <= 0.3 * h and cen[i][0] >= ta - 3.5 * h:
                compact.append(i)
        if flat:                                           # the pieces on the line of the widest one
            yref = cen[max(flat, key=lambda i: st[i][2])][1]
            pick = [i for i in flat if abs(cen[i][1] - yref) <= max(1.5, 0.12 * h)]
            far = [i for i in range(1, n) if abs(cen[i][1] - yref) <= max(1.5, 0.12 * h) and st[i][3] <= 0.8 * h
                   and st[i][0] < ta - 6.0 * h and st[i][0] + st[i][2] - 1 >= ta - 9.0 * h]
            if far:                                        # the line goes on further left: a data line next to
                continue                                   # its annotation, not a legend sample
        else:
            pick = compact
        if pick:
            kill[pick] = True
            found += 1
    m = kill[lab]
    work[m] = bg
    fg[m] = False
    return found


# ----------------------------------------------------------------------------------------------
# Raster: colour separation
# ----------------------------------------------------------------------------------------------


def _otsu(m):
    hist, edges = np.histogram(m, bins=32, range=(0, float(m.max()) + 1e-6))
    p = hist / hist.sum()
    mid = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * mid)
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    k = int(sb.argmax())
    st = float((p * (mid - mt) ** 2).sum())
    lo, hi = m <= edges[k + 1], m > edges[k + 1]
    if st <= 0 or lo.sum() == 0 or hi.sum() == 0:
        return None
    return edges[k + 1], sb[k] / st, float(m[hi].mean() - m[lo].mean()), min(lo.mean(), hi.mean())


def color_clusters(work, fg, bg):
    """Split the ink into colours. Anti-aliased pixels take the colour of the strongest pixel next to
    them, colours are grouped by the direction of (colour - background), so a faded edge of a line
    stays with its line; a second split by strength separates e.g. light grey from black.
    Returns [(mask, rgb, weight)]."""
    hh, ww = fg.shape
    v = work.astype(np.float32) - bg.astype(np.float32)
    d = np.sqrt((v ** 2).sum(axis=2))
    d[~fg] = 0
    dm, c = np.pad(d, 1), np.pad(v, ((1, 1), (1, 1), (0, 0)))
    d0, v0 = dm.copy(), c.copy()
    for _ in range(3):
        bd, bc = dm.copy(), c.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    sd = np.roll(dm, (dy, dx), (0, 1))
                    sc = np.roll(c, (dy, dx), (0, 1))
                    t = (v0 * sc).sum(axis=2) / np.maximum(sd * sd, 1e-6)
                    perp = np.sqrt(np.maximum(d0 * d0 - (t * sd) ** 2, 0))
                    m = (sd > bd + 1e-3) & (perp < 30) & (t > 0)   # only a faded version of the neighbour's colour,
                    m &= (np.roll(d0, (dy, dx), (0, 1)) > d0 + 1e-3) | (sd <= 1.1 * d0)   # and only downhill in the
                    # picture itself, or with a colour barely stronger than the pixel's own (a band of blended ink)
                    bd[m] = sd[m]
                    bc[m] = sc[m]
        dm, c = bd, bc
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        return []
    cv_, mag = c[ys + 1, xs + 1], dm[ys + 1, xs + 1]
    u = cv_ / np.maximum(mag, 1e-6)[:, None]
    q = np.round(u * 14).astype(np.int32)
    uq, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    un = uq / np.maximum(np.linalg.norm(uq, axis=1), 1e-6)[:, None]
    cos_t, minc = math.cos(math.radians(9)), max(12, int(0.001 * len(ys)))
    centers = []
    for i in np.argsort(-cnt):
        if cnt[i] < minc:
            break
        if not centers or (np.array(centers) @ un[i]).max() < cos_t:
            centers.append(un[i])
    if not centers:
        return []
    centers = np.array(centers)
    for _ in range(2):                       # refine centres, then drop duplicates
        a = (un @ centers.T).argmax(axis=1)
        centers = np.array([(un[a == k] * cnt[a == k, None]).sum(axis=0) if (a == k).any() else centers[k]
                            for k in range(len(centers))])
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        keep = []
        for k in range(len(centers)):
            if all(centers[k] @ centers[j] < cos_t for j in keep):
                keep.append(k)
        centers = centers[keep]
    dots = un @ centers.T
    a = dots.argmax(axis=1)
    a[dots.max(axis=1) < math.cos(math.radians(25))] = -1
    lab = a[inv]
    out = []
    dref = d.max() if d.max() > 0 else 1.0

    def emit(sel):
        if sel.sum() < 12:
            return
        mm = mag[sel]
        strong = mm >= np.percentile(mm, 70)
        rgb = np.clip(np.median(cv_[sel][strong], axis=0) + bg, 0, 255)
        mask = np.zeros((hh, ww), bool)
        mask[ys[sel], xs[sel]] = True
        wgt = np.where(mask, np.clip(d / max(float(np.percentile(mm, 90)), 1.0), 0, 1), 0).astype(np.float32)
        out.append((mask, tuple(int(x) for x in rgb), wgt))

    def split(sel, depth):
        o = _otsu(mag[sel])
        top = dref if depth == 0 else float(mag[sel].max())    # second split: against the strongest ink of this group
        if depth < 2 and o is not None and o[1] > 0.8 and o[2] > 0.35 * top and o[3] > 0.04:
            thr = o[0]
            idx = np.nonzero(sel)[0]
            weak, strong = idx[mag[idx] <= thr], idx[mag[idx] > thr]
            m = np.zeros((hh, ww), np.uint8)
            m[ys[strong], xs[strong]] = 1
            if (cv2.dilate(m, np.ones((5, 5), np.uint8))[ys[weak], xs[weak]] > 0).mean() > 0.5:
                emit(sel)                                  # the weak pixels hug the strong ones: faded parts of the same strokes
                return
            for part in (idx[mag[idx] <= thr], idx[mag[idx] > thr]):
                s2 = np.zeros(len(sel), bool)
                s2[part] = True
                split(s2, depth + 1)
        else:
            emit(sel)
    for k in range(len(centers)):
        sel = lab == k
        if sel.sum() >= 12:
            split(sel, 0)
    return out


def drop_blends(clusters, bg=None):
    """Where curves of two colours touch or cover each other, anti-aliasing leaves pixels of a mixed
    colour. A small cluster whose colour lies between the colours of two larger clusters, and whose
    pixels sit next to their ink, is such a mixture and not a series. Returns (clusters kept, number dropped)."""
    cols = [np.array(c[1], float) for c in clusters]
    size = [int(c[0].sum()) for c in clusters]
    clusters = [list(c) for c in clusters]
    keep, ndrop = [], 0
    near = lambda q: cv2.dilate(clusters[q][0].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    for k in sorted(range(len(clusters)), key=lambda q: -size[q]):
        mix, into = False, None
        for a in range(len(keep)):
            for b in range(a + 1, len(keep)):
                i, j = keep[a], keep[b]
                ab = cols[j] - cols[i]
                l2 = float(ab @ ab)
                if mix or into is not None or l2 < 60.0 ** 2 or size[k] > 0.5 * min(size[i], size[j]):
                    continue
                t = float((cols[k] - cols[i]) @ ab) / l2
                if 0.15 < t < 0.85 and np.linalg.norm(cols[k] - (cols[i] + t * ab)) < 40:
                    mk = clusters[k][0]
                    ni, nj = near(i)[mk], near(j)[mk]
                    if (ni & nj).mean() > 0.5:                 # next to both colours: where two curves touch
                        mix = True
                    elif max(ni.mean(), nj.mean()) > 0.7:      # next to one colour only, and as strong: the
                        q = i if ni.mean() >= nj.mean() else j      # darker centres of that curve's dashes (a
                        strong_ = bg is None or (np.linalg.norm(cols[k] - bg) >= 0.95 * np.linalg.norm(cols[q] - bg))
                        if strong_:                                 # weaker rim is dropped as before)
                            into = q
                        else:
                            mix = True
        if into is not None:
            clusters[into][0] = clusters[into][0] | clusters[k][0]
            clusters[into][2] = np.maximum(clusters[into][2], clusters[k][2])
        elif mix:
            ndrop += 1
        else:
            keep.append(k)
    return [tuple(clusters[k]) for k in sorted(keep)], ndrop


# ----------------------------------------------------------------------------------------------
# Line tracing: thinning, skeleton graph, linking of dashes, sub-pixel centre
# ----------------------------------------------------------------------------------------------


def _thin_luts():
    """Thinning rules as look-up tables on the 8 neighbours (two sub-iterations each): Zhang-Suen and
    Guo-Hall. Zhang-Suen gives the cleaner skeleton at crossings but erases whole anti-aliased bands that
    run at about 45 degrees; Guo-Hall keeps those."""
    z1, z2, g1, g2 = (np.zeros(256, bool) for _ in range(4))
    for code in range(256):
        p = [(code >> k) & 1 for k in range(8)]            # N, NE, E, SE, S, SW, W, NW
        if 2 <= sum(p) <= 6 and sum(p[k] == 0 and p[(k + 1) % 8] == 1 for k in range(8)) == 1:
            n_, e_, s_, w_ = p[0], p[2], p[4], p[6]
            z1[code] = n_ * e_ * s_ == 0 and e_ * s_ * w_ == 0
            z2[code] = n_ * e_ * w_ == 0 and n_ * s_ * w_ == 0
        p2, p3, p4, p5, p6, p7, p8, p9 = p
        c = ((not p2) and (p3 or p4)) + ((not p4) and (p5 or p6)) + ((not p6) and (p7 or p8)) + ((not p8) and (p9 or p2))
        n1 = (p9 or p2) + (p3 or p4) + (p5 or p6) + (p7 or p8)
        n2 = (p2 or p3) + (p4 or p5) + (p6 or p7) + (p8 or p9)
        if c == 1 and 2 <= min(n1, n2) <= 3:
            g1[code] = not ((p6 or p7 or not p9) and p8)
            g2[code] = not ((p2 or p3 or not p5) and p4)
    return (z1, z2), (g1, g2)


_ZS, _GH = _thin_luts()
_ZS_K = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)


def _thin_with(mask, luts):
    m = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for lut in luts:
            code = cv2.filter2D(m, cv2.CV_16S, _ZS_K, borderType=cv2.BORDER_CONSTANT)
            kill = lut[code] & (m > 0)
            if kill.any():
                m[kill] = 0
                changed = True
        if not changed:
            break
    return m[1:-1, 1:-1].astype(bool)


def fill_pins(mask):
    """The mask with its pin holes filled: holes of <= 2 px lying >= 3 px deep in the ink (a light pixel inside
    a symbol). Thinning makes a small loop around such a hole, and the stub beyond it (the start of a curve) is
    pruned like a hair. Shallow holes between two thin strokes (a crossing at a small angle) stay."""
    m = np.asarray(mask) > 0
    n, lab, st, _ = _cc(~m, 4)
    hh, ww = m.shape
    pin = [i for i in range(1, n) if st[i][4] <= 2 and st[i][0] > 0 and st[i][1] > 0
           and st[i][0] + st[i][2] < ww and st[i][1] + st[i][3] < hh]
    if not pin:
        return m
    full = m | np.isin(lab, pin)
    depth = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, 3)
    deep = [i for i in pin if depth[lab == i].max() >= 3.0]
    return m | np.isin(lab, deep) if deep else m


def thin(mask):
    """Zhang-Suen thinning. Stretches of line that it erased altogether (steep anti-aliased bands) are
    taken from the Guo-Hall skeleton; every connected piece keeps at least one pixel."""
    mask = np.asarray(mask) > 0
    sk = _thin_with(mask, _ZS)
    if sk.any():
        far = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 3)       # distance to the skeleton
        half = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)[sk]
        thr = 5.0 * float(np.median(half)) + 3.0
        if (far[mask] > thr).any():                        # ink far from any skeleton pixel: a lost stretch
            cand = _thin_with(mask, _GH) & (far > 1.5)
            n, lab, st, cen = _cc(cand)
            if n > 1:
                top = np.zeros(n)
                np.maximum.at(top, lab[cand], far[cand])
                sk |= np.isin(lab, np.nonzero(top > thr)[0]) & cand
    n, lab, st, cen = _cc(mask)
    if n > 1:
        have = np.zeros(n, bool)
        have[np.unique(lab[sk])] = True
        for i in np.nonzero(~have[1:])[0] + 1:
            x, y, w, h, _ = st[i]
            yy, xx = np.nonzero(lab[y:y + h, x:x + w] == i)
            j = ((yy + y - cen[i][1]) ** 2 + (xx + x - cen[i][0]) ** 2).argmin()
            sk[yy[j] + y, xx[j] + x] = True
    return sk


def skeleton_edges(sk):
    """Cut a skeleton into edges: pixel chains running between end points and junctions.
    Returns [dict(a=node, b=node, pts=(N,2) float xy)]."""
    ys, xs = np.nonzero(sk)
    n = len(ys)
    if n == 0:
        return []
    idx = -np.ones((sk.shape[0] + 2, sk.shape[1] + 2), np.int64)
    Y, X = ys + 1, xs + 1
    idx[Y, X] = np.arange(n)
    nb = [[] for _ in range(n)]
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        j = idx[Y + dy, X + dx]
        for i in np.nonzero(j >= 0)[0]:
            nb[i].append(int(j[i]))
    for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):     # diagonal steps only where no corner pixel links them
        j = idx[Y + dy, X + dx]
        for i in np.nonzero((j >= 0) & (idx[Y + dy, X] < 0) & (idx[Y, X + dx] < 0))[0]:
            nb[i].append(int(j[i]))
    deg = [len(a) for a in nb]
    node = [-1] * n
    nn = 0
    for i in range(n):
        if deg[i] == 2 or node[i] >= 0:
            continue
        node[i] = nn
        if deg[i] >= 3:                                    # neighbouring junction pixels form one node
            stack = [i]
            while stack:
                p = stack.pop()
                for q in nb[p]:
                    if deg[q] >= 3 and node[q] < 0:
                        node[q] = nn
                        stack.append(q)
        nn += 1
    xy = np.c_[xs, ys].astype(float)
    edges, used, seen = [], np.zeros(n, bool), set()
    for p in range(n):
        if node[p] < 0:
            continue
        used[p] = True
        if deg[p] == 0:
            edges.append(dict(a=node[p], b=node[p], pts=xy[[p]]))
        for q in nb[p]:
            if node[q] == node[p] or (p, q) in seen:
                continue
            path, prev, cur = [p, q], p, q
            while node[cur] < 0:
                nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
                path.append(nxt)
                prev, cur = cur, nxt
            seen.add((p, q)); seen.add((cur, prev))
            used[path] = True
            edges.append(dict(a=node[p], b=node[cur], pts=xy[path]))
    for p in range(n):                                     # closed loops without any node
        if used[p]:
            continue
        path, prev, cur = [p], p, nb[p][0]
        while cur != p:
            path.append(cur)
            nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
            prev, cur = cur, nxt
        used[path] = True
        edges.append(dict(a=nn, b=nn, pts=xy[path + [p]]))
        nn += 1
    return edges


def _plen(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def _angle(u, v):
    return math.degrees(math.acos(float(np.clip(u @ v, -1, 1))))


def build_chains(edges, lw, r_min=0.0, diag=1000.0, trim=0.0, wide=False, cover=None, hair=None):
    """Join skeleton edges into curves: straight-through pairing at junctions (so that crossing curves
    stay separate) and linking across gaps (dashes, dots, erased crossings). wide: directions at junctions are taken
    further from the junction (strokes with merged dashes bend the skeleton there). cover: a gap longer than a touch
    is bridged only where this mask (ink) covers it (the solid curve of a solid / dashed pair has no
    gaps of its own). hair: free branches at a junction shorter than this are dropped (dashes merged into a stroke).
    Returns [dict(pts, ink, gaps=[(index, length)], pieces=[ink lengths])]."""
    spur = 1.5 * lw + 2
    for it_ in range(3 if hair else 2):                    # remove skeleton hairs
        degn = {}
        for e in edges:
            degn[e["a"]] = degn.get(e["a"], 0) + 1
            degn[e["b"]] = degn.get(e["b"], 0) + 1
        sp = max(spur, hair) if hair and it_ == 0 else spur
        edges = [e for e in edges if not (e["a"] != e["b"] and _plen(e["pts"]) < sp and
                                          ((degn[e["a"]] == 1 and degn[e["b"]] >= 3) or (degn[e["b"]] == 1 and degn[e["a"]] >= 3)))]
    degn = {}
    for e in edges:
        degn[e["a"]] = degn.get(e["a"], 0) + 1
        degn[e["b"]] = degn.get(e["b"], 0) + 1
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    kept = []
    for e in edges:                                        # merge junctions joined by a very short edge
        if e["a"] != e["b"] and degn[e["a"]] >= 3 and degn[e["b"]] >= 3 and _plen(e["pts"]) < 2 * lw + 2:
            parent[find(e["a"])] = find(e["b"])
        else:
            kept.append(e)
    edges = kept
    if trim > 0:                                           # cut the last pixels of free ends
        degn = {}
        for e in edges:
            for k in ("a", "b"):
                degn[find(e[k])] = degn.get(find(e[k]), 0) + 1
        for e in edges:
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1))]
            if s[-1] < 4 * trim or e["a"] == e["b"]:
                continue
            lo = int(np.searchsorted(s, trim)) if degn[find(e["a"])] == 1 else 0
            hi = int(np.searchsorted(s, s[-1] - trim)) + 1 if degn[find(e["b"])] == 1 else len(s)
            e["pts"] = e["pts"][lo:hi]
    ne = len(edges)
    if ne == 0:
        return []
    partner = {}                                           # (edge, end) -> (edge, end, is_gap)
    span = max(6.0, 3.0 * lw)

    def end_pts(ei, end, need):
        """Points going away from an end into the curve, following existing pairings."""
        pts, total, guard = [], 0.0, 0
        while guard < 50:
            guard += 1
            p = edges[ei]["pts"]
            p = p if end == 0 else p[::-1]
            pts.append(p)
            total += _plen(p)
            if total >= need or (ei, 1 - end) not in partner:
                break
            ei, end, _ = partner[(ei, 1 - end)]
        return np.vstack(pts)

    def tangent(ei, end, need=None, skip=None):
        """Unit vector pointing out of the curve at this end (None if the piece is too short)."""
        need = span if need is None else need
        p = end_pts(ei, end, need + lw)
        if len(p) < 2:
            return None
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if s[-1] < 3.0:
            return None
        i0 = int(np.searchsorted(s, min(lw if skip is None else skip, 0.3 * s[-1])))
        i1 = int(np.searchsorted(s, min(need + lw, s[-1])))
        i1 = min(max(i1, i0 + 1), len(p) - 1)
        v = p[i0] - p[i1]
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    at = {}
    for ei, e in enumerate(edges):
        at.setdefault(find(e["a"]), []).append((ei, 0))
        at.setdefault(find(e["b"]), []).append((ei, 1))
    free, dead = [], set()
    for nd, ends in at.items():
        if len(ends) == 1:
            free.append(ends[0])
            continue
        if len(ends) == 2 and ends[0][0] == ends[1][0]:
            if len(edges[ends[0][0]]["pts"]) == 1:         # an isolated dot: two free ends
                free += ends
            continue                                       # closed loop
        if len(ends) >= 3:                                 # a small closed loop at a junction is the rim of a hole in
            loops = {ei for ei, _ in ends if find(edges[ei]["a"]) == find(edges[ei]["b"])    # thick ink (a dash merged
                     and _plen(edges[ei]["pts"]) <= 4 * lw + 4}                            # into a crossing line), not
            if loops and len(ends) > 2 * len(loops):       # a curve: its ends gave a dashed curve a hook to that line
                dead |= loops
                ends = [e for e in ends if e[0] not in loops]
                if len(ends) == 1:
                    free.append(ends[0])
                    continue
        tg = [tangent(ei, end, max(span, 8.0 * lw), 2.0 * lw) if wide else tangent(ei, end) for ei, end in ends]
        apex = None                                        # the top of a narrow peak (or bottom of a dip): both flanks
        if len(ends) in (2, 3):                            # leave the node on the same side, one going left, one right;
            for i in range(len(ends)):                     # a third edge is the hair to the tip of the stroke (the
                for j in range(i + 1, len(ends)):          # wedge between the outer edges); the flanks' centre lines
                    if apex or tg[i] is None or tg[j] is None or ends[i][0] == ends[j][0]:   # meet at the node
                        continue
                    if _angle(tg[i], tg[j]) > 75 or tg[i][0] * tg[j][0] >= 0:
                        continue
                    rest = [k for k in range(len(ends)) if k not in (i, j)]
                    if rest:
                        ek, ke = ends[rest[0]]
                        far = find(edges[ek]["b"] if ke == 0 else edges[ek]["a"])
                        if len(at.get(far, [])) != 1 or _plen(edges[ek]["pts"]) > 4 * lw + 4:
                            continue
                    apex = (i, j, rest)
        if apex:
            i, j, rest = apex
            partner[ends[i]] = (*ends[j], False)
            partner[ends[j]] = (*ends[i], False)
            dead |= {ends[k][0] for k in rest}
            continue
        cand = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                if tg[i] is None or tg[j] is None:
                    ang = 60.0
                else:
                    ang = _angle(tg[i], -tg[j])
                cand.append((ang, i, j))
        done = set()
        for ang, i, j in sorted(cand):
            if i in done or j in done or ang > (120 if len(ends) == 2 else 55):
                continue
            done |= {i, j}
            partner[ends[i]] = (*ends[j], False)
            partner[ends[j]] = (*ends[i], False)
        free += [ends[i] for i in range(len(ends)) if i not in done]      # a branch that goes nowhere at this junction

    def pos(ei, end):
        return edges[ei]["pts"][0 if end == 0 else -1]

    def hidden(p, q):                                      # the way between two ends lies under other ink
        if cover is None or float(np.linalg.norm(q - p)) <= 2 * lw + 3:
            return True
        t = np.linspace(0, 1, max(3, int(np.linalg.norm(q - p)) + 1))[1:-1, None]
        k = np.round(p + t * (q - p)).astype(int)
        ok = (k[:, 0] >= 0) & (k[:, 0] < cover.shape[1]) & (k[:, 1] >= 0) & (k[:, 1] < cover.shape[0])
        return ok.all() and cover[k[:, 1], k[:, 0]].mean() >= 0.7

    free = [f for f in free if f[0] not in dead]

    def link_pass(radius, max_ang, need):
        fr = [f for f in free if f not in partner]
        if len(fr) < 2:
            return
        P = np.array([pos(*f) for f in fr])
        tg = [tangent(*f, need=need) for f in fr]
        cand = []
        for i, j in cKDTree(P).query_pairs(radius):
            if fr[i][0] == fr[j][0] and len(edges[fr[i][0]]["pts"]) > 1 and _plen(edges[fr[i][0]]["pts"]) < 3 * radius:
                continue
            if fr[i][0] == fr[j][0] and len(edges[fr[i][0]]["pts"]) == 1:
                continue
            v = P[j] - P[i]
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            v /= dist
            a1 = _angle(tg[i], v) if tg[i] is not None else None
            a2 = _angle(tg[j], -v) if tg[j] is not None else None
            if dist <= 2 * lw + 3 and (tg[i] is None or tg[j] is None or _angle(tg[i], -tg[j]) < 45):
                a1 = a2 = 0.0                              # ends almost touching: their offset says nothing
            if (a1 is not None and a1 > max_ang) or (a2 is not None and a2 > max_ang):
                continue
            if need > span and (a1 is None or a2 is None):
                continue
            if not hidden(P[i], P[j]):
                continue
            cand.append((dist * (1 + ((a1 or 0) + (a2 or 0)) / 30.0), i, j))
        for _, i, j in sorted(cand):
            a, b = fr[i], fr[j]
            if a in partner or b in partner:
                continue
            ok = True
            for me, other in ((a, b), (b, a)):             # a dot must be passed straight through
                if len(edges[me[0]]["pts"]) == 1 and (me[0], 1 - me[1]) in partner:
                    o = partner[(me[0], 1 - me[1])]
                    v1, v2 = pos(o[0], o[1]) - pos(*me), pos(*other) - pos(*me)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 0 and n2 > 0 and _angle(v1 / n1, v2 / n2) < 110:
                        ok = False
            if ok:
                partner[a] = (*b, True)
                partner[b] = (*a, True)

    fr = [f for f in free]
    r1 = max(6.0, 3.0 * lw + 3.0, r_min)
    if len(fr) >= 12:
        P = np.array([pos(*f) for f in fr])
        dd, ii = cKDTree(P).query(P, k=min(4, len(fr)))
        nn = []
        for i in range(len(fr)):                           # nearest free end that belongs to another piece
            for d_, j in zip(dd[i][1:], ii[i][1:]):
                if fr[j][0] != fr[i][0]:
                    nn.append(d_)
                    break
        if nn:
            r1 = max(r1, min(2.2 * float(np.median(nn)) + 2.0, 0.08 * diag))
    link_pass(r1, 40, span)
    link_pass(min(5.0 * r1, 0.15 * diag), 22, max(15.0, 5.0 * lw))
    fr = [f for f in free if f not in partner]
    if len(fr) >= 2:
        P = np.array([pos(*f) for f in fr])
        dd, ii = cKDTree(P).query(P, k=min(3, len(fr)))
        for i in range(len(fr)):
            j = int(ii[i][1])
            alone = len(fr) == 2 or (dd[i][2] > 2.0 * dd[i][1] and dd[j][2] > 2.0 * dd[i][1])
            ti, tj = tangent(*fr[i]), tangent(*fr[j])      # two ends leaving the same way, neither pointing at the
            if ti is not None and tj is not None and _angle(ti, -tj) > 120 and dd[i][1] > 0:   # other (two curves
                v_ = (P[j] - P[i]) / dd[i][1]                  # ending at the frame): no U-turn between them
                if min(_angle(ti, v_), _angle(tj, -v_)) > 80:
                    continue
                xm_ = 0.5 * (P[i][0] + P[j][0])                # also when one end is set back: both pieces lie on
                si_ = np.median(end_pts(*fr[i], 3 * span)[:, 0]) - xm_      # the same side in x (a U-turn), not
                sj_ = np.median(end_pts(*fr[j], 3 * span)[:, 0]) - xm_      # on the two sides of a narrow peak
                if si_ * sj_ > 0 and min(abs(si_), abs(sj_)) > 1.0:
                    continue
            if not hidden(P[i], P[j]):
                continue
            if (i < j and int(ii[j][1]) == i and dd[i][1] <= max(15.0, 3.0 * lw + 6.0) and alone and fr[i][0] != fr[j][0]
                    and fr[i] not in partner and fr[j] not in partner):
                partner[fr[i]] = (*fr[j], True)
                partner[fr[j]] = (*fr[i], True)

    chains, visited = [], np.zeros(ne, bool)
    visited[list(dead)] = True
    for s in range(ne):
        if visited[s]:
            continue
        ei, end, guard = s, 0, 0                           # walk back to the start of the curve
        while (ei, end) in partner and guard < ne + 2:
            ei, end, _ = partner[(ei, end)]
            end = 1 - end
            guard += 1
            if ei == s:
                break
        pts, gaps, pieces, ink, npts = [], [], [], 0.0, 0
        while not visited[ei]:
            visited[ei] = True
            p = edges[ei]["pts"] if end == 0 else edges[ei]["pts"][::-1]
            pts.append(p)
            npts += len(p)
            ln = _plen(p)
            pieces.append(ln)
            ink += ln
            nxt = partner.get((ei, 1 - end))
            if nxt is None:
                break
            if nxt[2]:
                gaps.append((npts - 1, float(np.linalg.norm(pos(nxt[0], nxt[1]) - p[-1]))))
            else:
                pieces[-1] += 0.0
            ei, end = nxt[0], nxt[1]
        chains.append(dict(pts=np.vstack(pts), ink=ink, gaps=gaps, pieces=pieces))
    return chains


def extend_chains(chains, diag, lw):
    """A broken curve whose first or last dashes did not link (dots give no direction, the gaps of a
    dash-dot-dot pattern are long) ends early, and those dashes form small chains that are dropped. Append a
    small chain that lies straight ahead of a long broken chain: within 3 of its median gaps, within 25 deg
    of its direction at that end and within 2 line widths of that direction."""
    big = [c for c in chains if len(c["gaps"]) >= 4 and _plen(c["pts"]) >= 0.08 * diag]
    small = [c for c in chains if len(c["pts"]) >= 1 and float(np.linalg.norm(np.ptp(c["pts"], axis=0))) < 0.08 * diag]
    used = set()
    for c in big:
        gap = float(np.median([g[1] for g in c["gaps"]]))
        reach = max(3.0 * gap, 3.0 * lw + 6.0)
        for end in (0, 1):
            grown = True
            while grown:
                grown = False
                p = c["pts"] if end == 1 else c["pts"][::-1]
                s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                k = int(np.searchsorted(s_, s_[-1] - max(20.0, 3.0 * gap)))
                v = p[-1] - p[min(k, len(p) - 2)]
                nv = float(np.linalg.norm(v))
                if nv == 0:
                    break
                v /= nv
                best = None
                for j, o in enumerate(small):
                    if j in used:
                        continue
                    ov = np.ptp(o["pts"], axis=0)
                    if np.linalg.norm(ov) >= 4.0:              # a piece with a direction of its own must follow the curve
                        u = o["pts"][-1] - o["pts"][0]
                        nu = float(np.linalg.norm(u))
                        if nu == 0 or abs(float(u @ v)) / nu < math.cos(math.radians(30)):
                            continue
                    for q in (o["pts"], o["pts"][::-1]):
                        w = q[0] - p[-1]
                        d = float(np.linalg.norm(w))
                        if 0 < d <= reach and float(w @ v) / d >= math.cos(math.radians(25)) \
                                and abs(w[0] * v[1] - w[1] * v[0]) <= 2.0 * lw + 1.0 and (best is None or d < best[0]):
                            best = (d, j, q)
                if best is not None:
                    d, j, q = best
                    used.add(j)
                    o = small[j]
                    if end == 1:
                        c["gaps"] = c["gaps"] + [(len(c["pts"]) - 1, d)]
                        c["pts"] = np.vstack([c["pts"], q])
                    else:
                        n0 = len(q)
                        c["gaps"] = [(n0 - 1, d)] + [(i + n0, g) for i, g in c["gaps"]]
                        c["pts"] = np.vstack([q[::-1], c["pts"]])
                    c["pieces"] = list(c["pieces"]) + list(o["pieces"])
                    c["ink"] += o["ink"]
                    grown = True
    return [c for j, c in enumerate(chains) if not any(c is small[u] for u in used)]


def insert_dots(chains, diag):
    """Dots of a dotted curve where it turns sharply (a peak) are not linked: the turn is larger than the
    linking allows, the dot stays alone and is dropped, and the gap is bridged under the peak. Put such a
    dot (a small chain) into the gap of a broken chain when it lies between the two ends of the gap and the
    way through it is at most 1.6 times the gap."""
    big = [c for c in chains if len(c["gaps"]) >= 4 and _plen(c["pts"]) >= 0.08 * diag]
    small = [c for c in chains if float(np.linalg.norm(np.ptp(c["pts"], axis=0))) < 0.03 * diag]
    if not big or not small:
        return chains
    used = set()
    for c in big:
        ins = {}                                           # gap position -> (detour ratio, small chain index)
        p = c["pts"]
        for j, o in enumerate(small):
            if j in used:
                continue
            q = o["pts"].mean(axis=0)
            best = None
            for gi, (idx, g) in enumerate(c["gaps"]):
                if idx + 1 >= len(p):
                    continue
                a, b = p[idx], p[idx + 1]
                ab = float(np.linalg.norm(b - a))
                if ab < 1e-6:
                    continue
                da, db = float(np.linalg.norm(q - a)), float(np.linalg.norm(q - b))
                ratio = (da + db) / ab
                if da < ab and db < ab and ratio <= 1.6 and (best is None or ratio < best[0]):
                    best = (ratio, gi)
            if best is not None and (best[1] not in ins or best[0] < ins[best[1]][0]):
                ins[best[1]] = (best[0], j)
        if not ins:
            continue
        pts, gaps, shift = list(p), [], 0
        new_pts = []
        last = 0
        for gi, (idx, g) in enumerate(c["gaps"]):
            if gi in ins:
                j = ins[gi][1]
                used.add(j)
                o = small[j]
                q = o["pts"] if np.linalg.norm(o["pts"][0] - p[idx]) <= np.linalg.norm(o["pts"][-1] - p[idx]) else o["pts"][::-1]
                new_pts.extend(p[last:idx + 1])
                gaps.append((len(new_pts) - 1, float(np.linalg.norm(q[0] - p[idx]))))
                new_pts.extend(q)
                gaps.append((len(new_pts) - 1, float(np.linalg.norm(p[idx + 1] - q[-1]))))
                last = idx + 1
                c["pieces"] = list(c["pieces"]) + list(o["pieces"])
                c["ink"] += o["ink"]
            else:
                new_pts.extend(p[last:idx + 1])
                gaps.append((len(new_pts) - 1, g))
                last = idx + 1
        new_pts.extend(p[last:])
        c["pts"], c["gaps"] = np.array(new_pts), gaps
    return [c for c in chains if not any(c is small[u] for u in used)]


def refine_and_fill(pts, weight, lw, gaps):
    """Move each traced point to the ink-weighted centre across the line (sub-pixel), smooth lightly,
    then fill gaps by shape-preserving interpolation along the curve."""
    pts = pts.astype(np.float64)
    n = len(pts)
    if n >= 3 and weight is not None:
        k = 2
        t = np.empty_like(pts)
        t[k:-k] = pts[2 * k:] - pts[:-2 * k]
        t[:k] = pts[k] - pts[0]
        t[-k:] = pts[-1] - pts[-k - 1]
        cut = np.zeros(n, bool)
        for g, _ in gaps:                                   # do not take tangents across a gap
            cut[max(0, g - k + 1):min(n, g + k + 1)] = True
        for i in np.nonzero(cut)[0]:
            lo, hi = i, i
            gl = [g for g, _ in gaps]
            while lo > 0 and i - lo < k and (lo - 1) not in gl:
                lo -= 1
            while hi < n - 1 and hi - i < k and hi not in gl:
                hi += 1
            t[i] = pts[hi] - pts[lo]
        nt = np.linalg.norm(t, axis=1)
        ok = nt > 0
        t[ok] /= nt[ok, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        r = math.ceil(lw / 2.0) + 1.5
        offs = np.arange(-r, r + 0.01, 0.5, dtype=np.float32)
        mx = (pts[:, 0:1] + nrm[:, 0:1] * offs).astype(np.float32)
        my = (pts[:, 1:2] + nrm[:, 1:2] * offs).astype(np.float32)
        w = cv2.remap(weight, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sw = w.sum(axis=1)
        shift = np.where(sw > 0, (w * offs).sum(axis=1) / np.maximum(sw, 1e-6), 0.0)
        shift[~ok] = 0
        pts = pts + nrm * np.clip(shift, -1.5, 1.5)[:, None]
    # light smoothing inside each inked piece
    bounds = [0] + [g + 1 for g, _ in gaps] + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 5 and weight is not None:
            seg = pts[a:b].copy()
            seg[1:-1] = (pts[a:b - 2] + pts[a + 1:b - 1] + pts[a + 2:b]) / 3.0
            pts[a:b] = seg
    if not gaps:
        return _smooth_along(pts, 0.6 * lw) if weight is not None else pts
    if weight is not None and len(gaps) >= 4:              # dots (and very short dashes): each is one measured
        sa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]   # point; take its centre and
        ends = [0] + [g + 1 for g, _ in gaps] + [n]                               # draw a smooth curve through
        piece = [sa[e - 1] - sa[a_] + 1.0 for a_, e in zip(ends[:-1], ends[1:]) if e > a_]   # the centres
        if np.median(piece) <= 2.5 * lw:
            cen = []
            for (a_, e), ln in zip([(a_, e) for a_, e in zip(ends[:-1], ends[1:]) if e > a_], piece):
                if ln <= 1.2 * lw:                         # one dot: its centre
                    cen.append(pts[a_:e].mean(axis=0))
                else:                                      # dots that touch: the line through their centres
                    cen.extend(pts[a_:e])
            if len(cen) >= 3:
                return _through(np.array(cen))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, p = s[keep], pts[keep]
    if len(s) < 3:
        return pts
    fx, fy = PchipInterpolator(s, p[:, 0]), PchipInterpolator(s, p[:, 1])
    out = []
    for i in range(len(s) - 1):
        out.append(p[i])
        step = s[i + 1] - s[i]
        if step > 1.6:
            q = np.linspace(s[i], s[i + 1], int(round(step)) + 1)[1:-1]
            out.extend(np.c_[fx(q), fy(q)])
    out.append(p[-1])
    out = np.array(out)
    return _smooth_along(out, 0.6 * lw) if weight is not None else out





def unkink(px, maxlen):
    """A trace that runs along x and doubles back for a short stretch (an S or a small loop where curves cross
    or where symbols leave holes in the ink): the points in the x range covered twice are dropped and the gap
    is bridged by the curve through the neighbours. Only stretches up to maxlen of arc length; longer turns
    are real (loops, hysteresis)."""
    if len(px) < 5:
        return px
    if px[0, 0] > px[-1, 0]:
        return unkink(px[::-1], maxlen)[::-1]
    x = px[:, 0]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(px, axis=0), axis=1))]
    cm = np.maximum.accumulate(x)
    back = x < cm - 1.0
    if not back.any():
        return px
    drop = np.zeros(len(px), bool)
    i = 0
    while i < len(px):
        if not back[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(px) and back[j + 1]:
            j += 1
        xlo, xhi = x[i:j + 1].min(), cm[i]                 # x range run through twice (or more)
        a = i
        while a > 0 and x[a - 1] >= xlo - 0.5:
            a -= 1
        b = j
        while b + 1 < len(px) and x[b + 1] <= xhi + 0.5:
            b += 1
        if s[b] - s[a] <= maxlen and a > 0 and b < len(px) - 1:
            drop[a:b + 1] = True
        i = j + 1
    if not drop.any():
        return px
    keep = px[~drop]
    sk = np.r_[0, np.cumsum(np.linalg.norm(np.diff(keep, axis=0), axis=1))]
    ok = np.r_[True, np.diff(sk) > 1e-6]
    keep, sk = keep[ok], sk[ok]
    if len(keep) < 3:
        return px
    fx, fy = PchipInterpolator(sk, keep[:, 0]), PchipInterpolator(sk, keep[:, 1])
    out = [keep[:1]]
    for i in range(1, len(keep)):                          # the bridged gaps: one point per pixel
        if sk[i] - sk[i - 1] > 1.6:
            q = np.linspace(sk[i - 1], sk[i], int(round(sk[i] - sk[i - 1])) + 1)[1:-1]
            out.append(np.c_[fx(q), fy(q)])
        out.append(keep[i:i + 1])
    return np.vstack(out)

def through_beads(px, beads, size):
    """A traced line with symbols strung along it: where single symbols can be told apart (peaks, steep parts),
    the curve goes through their centres. The skeleton of the band of merged symbols cuts the corner at a
    narrow peak (the band fills the wedge between the flanks), the symbol centres do not."""
    if len(px) < 3 or not beads:
        return px
    c = np.asarray(beads, float)
    d, j = cKDTree(px).query(c)
    c, j = c[d <= 0.75 * size], j[d <= 0.75 * size]
    if len(c) < 3:
        return px
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(px, axis=0), axis=1))]
    sc = s[j]
    near = np.abs(s[:, None] - sc[None, :]).min(axis=1) < 0.5 * size
    ss = np.sort(sc)                                       # between two symbols that follow closely: their centres only
    for a, b in zip(ss[:-1], ss[1:]):
        if b - a <= 2.5 * size:
            near |= (s > a) & (s < b)
    keep = ~near
    keep[0] = keep[-1] = True                              # the ends of the trace stay where they are
    pts = np.r_[px[keep], c]
    order = np.argsort(np.r_[s[keep], sc], kind="stable")
    return _through(pts[order])

def _through(c):
    """A smooth curve through the points c (in order), about one point per pixel, passing through each."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, c = s[keep], c[keep]
    if len(s) < 3:
        return c
    fx, fy = PchipInterpolator(s, c[:, 0]), PchipInterpolator(s, c[:, 1])
    t = np.unique(np.r_[np.arange(0.0, s[-1], 1.0), s])
    return np.c_[fx(t), fy(t)]


def _smooth_along(p, sig):
    """Gaussian smoothing along the curve (arc length, 1 px steps, sigma in px). The centre of a stroke found in a
    picture scatters by a fraction of a pixel from point to point (more for dots, each a separate estimate);
    nothing narrower than the stroke itself can be resolved, so this removes noise, not data."""
    if sig < 0.75 or len(p) < 5:
        return p
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, p = s[keep], p[keep]
    if len(s) < 5:
        return p
    t = np.arange(0.0, s[-1] + 1e-9, 1.0)
    if t[-1] < s[-1]:
        t = np.r_[t, s[-1]]
    q = np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]
    from scipy.ndimage import gaussian_filter1d
    q = np.c_[gaussian_filter1d(q[:, 0], sig, mode="nearest"), gaussian_filter1d(q[:, 1], sig, mode="nearest")]
    q[0], q[-1] = p[0], p[-1]
    return q


def is_straight(pts):
    d = pts[-1] - pts[0]
    ln = np.linalg.norm(d)
    if ln < 1e-6:
        return False
    dev = np.abs((pts[:, 0] - pts[0, 0]) * d[1] - (pts[:, 1] - pts[0, 1]) * d[0]) / ln
    return dev.max() < max(1.5, 0.01 * ln)


# ----------------------------------------------------------------------------------------------
# Markers
# ----------------------------------------------------------------------------------------------


def classify_marker(m):
    """m: boolean crop of one marker. Returns (shape, open?)."""
    m = np.pad(m, 1)
    hh, ww = m.shape[0] - 2, m.shape[1] - 2
    size = max(hh, ww)
    if size < 5:
        return "dot", False
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    a_f, a_ink = float(filled.sum()), float(m.sum())
    hollow = a_ink < 0.8 * a_f
    pts = np.vstack(cnts)
    hull = cv2.convexHull(pts)
    hm = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(hm, hull, 1)
    if a_f / max(float(hm.sum()), 1.0) < 0.78:             # stroke markers: + x *
        yy, xx = np.nonzero(m)
        dy, dx = yy - 1 - (hh - 1) / 2.0, xx - 1 - (ww - 1) / 2.0
        t = max(1.0, 0.2 * size)
        plus = ((np.abs(dx) <= t) | (np.abs(dy) <= t)).mean()
        cross = ((np.abs(dx - dy) <= 1.4 * t) | (np.abs(dx + dy) <= 1.4 * t)).mean()
        if plus > 0.9 and cross < 0.8:
            return "plus", False
        if cross > 0.9 and plus < 0.8:
            return "cross", False
        return "star", False
    # convex symbols: compare the hull with its tightest (rotated) rectangle and with the upright box
    ha = float(cv2.contourArea(hull))
    rw, rh = cv2.minAreaRect(hull)[1]
    rr = ha / max(rw * rh, 1e-6)
    upright = ha / max((hh - 1) * (ww - 1), 1e-6)
    ys, xs = np.nonzero(filled)
    oy, ox = (ys.mean() - 1 - (hh - 1) / 2.0) / hh, (xs.mean() - 1 - (ww - 1) / 2.0) / ww
    c = max(1, int(round(0.18 * min(hh, ww))))
    f = filled[1:-1, 1:-1]
    corner = float(np.mean([f[:c, :c].mean(), f[:c, -c:].mean(), f[-c:, :c].mean(), f[-c:, -c:].mean()]))
    off = max(abs(oy), abs(ox))
    if rr >= 0.9 and upright < 0.75:
        return "diamond", hollow
    if corner >= (0.6 if size > 6 else 0.99):
        return "square", hollow
    if off > 0.055 or rr < 0.7:
        if abs(oy) >= abs(ox):
            return ("triangle up" if oy > 0 else "triangle down"), hollow
        return ("triangle left" if ox > 0 else "triangle right"), hollow
    return "circle", hollow


def clean_marker(m):
    """Remove thin stubs (pieces of a line attached to a symbol) from a filled marker crop.
    Returns (crop, x offset, y offset); the input itself is returned when nothing sticks out of it
    or when the thin parts belong to the symbol (+, x, star)."""
    u = np.pad(m.astype(np.uint8), 3)
    k = int(round(0.33 * cv2.distanceTransform(u, cv2.DIST_L2, 3).max()))
    if k < 1:
        return m, 0, 0
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)       # a square element keeps the corners of squares
    op = cv2.morphologyEx(u, cv2.MORPH_OPEN, ker)
    if op.sum() < 0.5 * u.sum():
        return m, 0, 0
    n, lab, st, _ = _cc(op)
    j = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, _ = st[j]
    if (w, h) == (m.shape[1], m.shape[0]):
        return m, 0, 0
    nr, _, sr, _ = _cc(u & ~(lab == j).astype(np.uint8))
    if sum(sr[i][4] >= 3 for i in range(1, nr)) >= 3:
        return m, 0, 0
    return lab[y:y + h, x:x + w] == j, x - 3, y - 3


def marker_core(comp, strong):
    """Shape of a marker without its faint anti-aliased rim: the strong ink with its holes filled.
    comp: filled marker crop; strong: crop of the strong-ink mask. Returns (crop, x offset, y offset)."""
    c = (comp & strong).astype(np.uint8)
    ff = np.pad(1 - c, 1, constant_values=1)
    cv2.floodFill(ff, None, (0, 0), 0)
    c |= ff[1:-1, 1:-1]
    n, lab, st, _ = _cc(c)
    if n < 2:
        return comp, 0, 0
    j = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, a = st[j]
    if a < 0.5 * comp.sum():
        return comp, 0, 0
    return lab[y:y + h, x:x + w] == j, int(x), int(y)


def fit_symbols(blob, tmpl, kmax):
    """Place up to kmax copies of the average symbol so that together they cover the ink of a blob
    of touching / overlapping symbols. Returns the centres [(x, y)] in blob coordinates."""
    t = (tmpl > 0.5).astype(np.float32)
    nt = float(t.sum())
    th, tw = t.shape
    pad = max(th, tw)
    b = np.pad(blob.astype(np.float32), pad)
    outside = cv2.matchTemplate(1.0 - b, t, cv2.TM_CCORR)            # symbol pixels falling off the ink

    def covered(pos, skip=-1):
        c = np.zeros_like(b)
        for j, (x, y) in enumerate(pos):
            if j != skip:
                c[y:y + th, x:x + tw] = np.maximum(c[y:y + th, x:x + tw], t)
        return c

    def best(free):
        gain = cv2.matchTemplate(free, t, cv2.TM_CCORR)
        _, _, _, (x, y) = cv2.minMaxLoc(gain - outside)
        return x, y, float(gain[y, x])
    pos = []
    for k in range(kmax):
        x, y, gain = best(b * (1.0 - covered(pos)))
        if gain < (0.6 if k == 0 else 0.25) * nt or outside[y, x] > 0.35 * nt:
            break
        pos.append((x, y))
    for _ in range(2):                                     # settle each copy with the others in place
        for i in range(len(pos)):
            pos[i] = best(b * (1.0 - covered(pos, i)))[:2]
    return [(x + (tw - 1) / 2.0 - pad, y + (th - 1) / 2.0 - pad) for x, y in pos]


def marker_name(shape, hollow):
    return ("open " if hollow else "") + shape


def dedupe_scatter(series, tol):
    """Drop a scatter series whose points coincide with those of another one (edge + fill of a marker)."""
    sc = [s for s in series if s.kind == "scatter"]
    drop = set()
    for a in sc:
        for b in sc:
            if a is b or id(a) in drop or id(b) in drop or len(b.px) > len(a.px):
                continue
            if len(b.px) == len(a.px) and (b.style.startswith("open") < a.style.startswith("open")):
                continue
            d, _ = cKDTree(a.px_alt if a.px_alt is not None else a.px).query(b.px_alt if b.px_alt is not None else b.px)
            if (d < tol).mean() > 0.7:
                drop.add(id(b))
    return [s for s in series if id(s) not in drop]


# ----------------------------------------------------------------------------------------------
# Raster: one colour -> series
# ----------------------------------------------------------------------------------------------


def _along_neighbours(lab, st, members, pieces):
    """Share of the pieces 'members' that are elongated (axis ratio >= 1.2) along the row they lie in (the
    direction between their two nearest neighbours among 'pieces', within 30 degrees). Short dashes of a
    dashed curve: about 0.7; scatter symbols: below 0.15."""
    allp = sorted(set(members) | set(pieces))
    if len(allp) < 3:
        return 0.0
    cen, axis, ratio = {}, {}, {}
    for i in allp:
        x, y, w, h, _ = st[i]
        p = np.argwhere(lab[y:y + h, x:x + w] == i)[:, ::-1].astype(np.float64)
        c = p.mean(axis=0)
        ev, vec = np.linalg.eigh(np.cov((p - c).T) + 1e-9 * np.eye(2)) if len(p) > 2 else (np.ones(2), np.eye(2))
        cen[i], axis[i], ratio[i] = c + (x, y), vec[:, 1], math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
    pts = np.array([cen[i] for i in allp])
    tree, good, cnt = cKDTree(pts), 0, 0
    for i in members:
        _, j = tree.query(cen[i], k=3)
        t = pts[j[2]] - pts[j[1]]
        nt = np.linalg.norm(t)
        if nt == 0:
            continue
        cnt += 1
        good += ratio[i] >= 1.2 and abs(float(t @ axis[i])) / nt >= math.cos(math.radians(30))
    return good / cnt if cnt else 0.0


def straighten_dashes(lm, sk, smax):
    """Replace the skeleton of each short dash by a straight segment along its long axis. A dash a few
    pixels long and nearly as thick thins to a stub whose direction is noise, and the gaps between dashes
    are bridged along those directions. Only for dashes of one length: in a dash-dot pattern the long
    dashes thin well and the dots have no direction."""
    n, lab, st, _ = _cc(lm)
    segs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) > smax or a < 6:
            continue
        p = np.argwhere(lab[y:y + h, x:x + w] == i)[:, ::-1].astype(np.float64)
        c = p.mean(axis=0)
        ev, vec = np.linalg.eigh(np.cov((p - c).T) + 1e-9 * np.eye(2))
        v = vec[:, 1]
        t = (p - c) @ v
        segs.append((i, ev[1] >= 1.44 * ev[0], t.max() - t.min() + 1, a, c + t.min() * v + (x, y), c + t.max() * v + (x, y)))
    if len(segs) < 6:
        return sk
    ln = np.array([s_[2] for s_ in segs])
    stub = np.array([s_[2] ** 2 / s_[3] for s_ in segs])  # length / thickness
    if np.percentile(ln, 80) > 1.5 * np.percentile(ln, 20) or np.median(stub) > 2.5:
        return sk                                          # dash-dot, or dashes long enough to thin well
    out = sk.astype(np.uint8)
    for (i, longish, l_, a, q0, q1), r_ in zip(segs, stub):
        if not longish or r_ > 3.0 or l_ < 5:              # round (a dot), or a dash that thins well
            continue
        x, y, w, h, _ = st[i]
        out[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i] = 0
        cv2.line(out, tuple(int(round(k)) for k in q0), tuple(int(round(k)) for k in q1), 1, 1)
    return out > 0


def erase_insets(img, work, fg, bg, origin, box, texts):
    """A small plot inside the plot (inset): its frame, curves, tick labels and titles are not data of the
    main plot. The region is the inset frame grown by the detected texts that touch it (its labels and axis
    titles); ink lying completely inside it is erased. A main curve that passes through the region goes on
    outside it and stays. Returns the number of insets."""
    X0, Y0, X1, Y1 = box
    ox, oy = origin
    hh, ww = fg.shape
    ins = [f for f in picture_frames(img, insets=True)
           if f[0] > X0 + 3 and f[1] > Y0 + 3 and f[2] < X1 - 3 and f[3] < Y1 - 3]
    if not ins:
        return 0
    n, lab, st, _ = _cc(fg)
    kill = np.zeros(n, bool)
    for fx0, fy0, fx1, fy1 in ins:
        r = [fx0, fy0, fx1, fy1]
        grown = True
        while grown:                                       # labels and titles of the inset
            grown = False
            for t, sc, a, b, c, d, q in texts:
                gap = max(r[0] - c, a - r[2], r[1] - d, b - r[3])
                inside_main = a > X0 and c < X1 and b > Y0 and d < Y1
                if inside_main and gap <= 0.5 * (d - b + c - a) / 2 and not (a >= r[0] and c <= r[2] and b >= r[1] and d <= r[3]):
                    r = [min(r[0], a), min(r[1], b), max(r[2], c), max(r[3], d)]
                    grown = True
        rx0, ry0, rx1, ry1 = (int(r[0] - ox) - 2, int(r[1] - oy) - 2, int(r[2] - ox) + 2, int(r[3] - oy) + 2)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if x >= rx0 and y >= ry0 and x + w - 1 <= rx1 and y + h - 1 <= ry1:
                kill[i] = True
    m = kill[lab]
    work[m] = bg
    fg[m] = False
    return len(ins)


def erase_fills(work, fg, bg):
    """Shaded areas behind the curves (a band marking a range, a filled region): ink of one colour much wider than
    any line (what is left after an opening with a disk 4 % of the plot across), covering at least 1 % of the plot.
    Only the pixels of that fill colour go (within 30 of it, and its blend with the background at the rim); curves
    drawn over the fill keep their own colour and stay. A filled area that stands alone is erase_patches' case.
    Returns the number of fills erased."""
    hh, ww = fg.shape
    mind = min(hh, ww)
    k = max(5, int(round(0.02 * mind)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    wide = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, disk)
    n, lab, st, _ = _cc(wide > 0)
    done = 0
    wf = work.astype(np.float32)
    bgf = np.asarray(bg, np.float32)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.01 * hh * ww or max(w, h) < 0.1 * mind:
            continue
        reg = lab == i
        col = np.median(wf[reg], axis=0)
        if np.abs(wf[reg] - col).max(axis=1).mean() > 20:  # not one flat colour (a mass of curves, a photo)
            continue
        near = cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        dcol = col - bgf                                   # the fill's rim: on the way from the background to it,
        t = ((wf - bgf) @ dcol) / max(float(dcol @ dcol), 1e-6)   # at least 30 % of the way (pale rims of curves
        # near the fill stay)
        kill = near & fg & ((np.abs(wf - col).max(axis=2) < 30) | ((_blend_dist(wf, bgf, col) < 12) & (t >= 0.3)))
        work[kill] = bg
        fg[kill] = False
        done += 1
    return done


def erase_patches(work, fg, bg):
    """Filled boxes and patches (the coloured background of a label, a shaded band): a piece of ink that is
    nearly rectangular once its holes are filled (>= 80 % of its bounding box), at least 5 % of the plot
    thick and longer than any symbol. A curve, even a thick one, fills only a sliver of its box. A patch
    that touches a curve forms one larger piece with it and stays. Returns the number erased."""
    hh, ww = fg.shape
    mind = min(hh, ww)
    n, lab, st, _ = _cc(fg)
    kill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min(w, h) < max(8, 0.05 * mind) or max(w, h) <= 0.09 * mind:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(comp)
        cv2.drawContours(solid, cnts, -1, 1, -1)
        if solid.sum() >= 0.8 * w * h:
            kill[i] = True
            sub = work[y:y + h, x:x + w].astype(np.float32)
            inside = (solid > 0) & fg[y:y + h, x:x + w]
            col = dominant_color(sub[inside])               # the fill colour; ink of other colours inside that runs
            other = inside & (np.abs(sub - col).max(axis=2) > 40)   # across much of the box is a curve drawn over a
            no, lo, so, _ = _cc(other)                     # shaded band (not the text of a label): then only the
            flat = (inside & (np.abs(sub - col).max(axis=2) < 30)).astype(np.uint8)   # a band's fill colour forms
            r_ = max(3, int(round(0.01 * mind)))           # wide areas; an inset's frame (with a plot inside) is a
            wide_ = cv2.morphologyEx(flat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1,) * 2))
            if (no > 1 and wide_.sum() >= max(0.25 * flat.sum(), 0.005 * hh * ww)   # thin line: that goes whole
                    and np.maximum(so[1:, 2], so[1:, 3]).max() >= max(0.5 * min(w, h), 0.15 * mind)):   # fill goes
                bgf = np.asarray(bg, np.float32)
                dcol = col - bgf
                t = ((sub - bgf) @ dcol) / max(float(dcol @ dcol), 1e-6)
                fill = inside & ((np.abs(sub - col).max(axis=2) < 30) | ((_blend_dist(sub, bgf, col) < 12) & (t >= 0.3)))
                work[y:y + h, x:x + w][fill] = bg
                fg[y:y + h, x:x + w][fill] = False
                continue
            work[y:y + h, x:x + w][solid > 0] = bg        # what was inside the box goes too (rests of its text)
            fg[y:y + h, x:x + w][solid > 0] = False
    return int(kill.sum())


def fill_covered(mask, weight, ink):
    """Symbols or a curve of another colour drawn over a line take bites out of its ink, and its traced
    centre moves away from them. Fill the bites: ink of other colours that touches this colour and lies in
    its concave gaps (closing with a disk a little wider than the stroke). Filled pixels get full weight."""
    m8 = mask.astype(np.uint8)
    if m8.sum() < 20:
        return mask, weight
    n0, lab0, st0, _ = _cc(m8)                            # only for colours drawn as lines: a dotted or dashed
    smax = 0.09 * min(mask.shape)                          # curve would take in the line below its gaps
    long_ = [i for i in range(1, n0) if max(st0[i][2], st0[i][3]) > smax]
    if sum(st0[i][4] for i in long_) < 0.6 * m8.sum():
        return mask, weight
    sk = thin(m8) > 0
    if not sk.any():
        return mask, weight
    lw = 2.0 * float(np.median(cv2.distanceTransform(m8, cv2.DIST_L2, 3)[sk]))
    r = int(max(2, min(8, round(1.2 * lw))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    cand = (cv2.morphologyEx(m8, cv2.MORPH_CLOSE, disk) > 0) & ink & ~mask
    if not cand.any():
        return mask, weight
    n, lab, _, _ = _cc(cand)
    touch = np.unique(lab[(cv2.dilate(m8, np.ones((3, 3), np.uint8)) > 0) & cand])
    add = np.isin(lab, touch[touch > 0])
    if not add.any():
        return mask, weight
    w = weight.copy()
    w[add] = 1.0
    return mask | add, w


def dash_ink(lm, lw, smax, sk=None, other=None):
    """A solid and a dashed curve of the same colour: the ink of the short pieces (dashes, dots), to be chained apart
    from the long strokes, so that a dash is never linked into a solid curve or the reverse. A dash merged into a solid
    stroke goes with the stroke; so does a short piece with ink of another colour (or the frame) at both ends: a stretch
    of a solid curve between two crossings (a dash taken by mistake goes back to the dashes when it links to no stroke).
    None when the colour does not hold both, or when they never come close."""
    n, lab, st, cen = _cc(lm)
    if n < 12:
        return None
    ext = np.maximum(st[:, 2], st[:, 3]).astype(float)
    ext[0] = 0
    small = ext[1:][ext[1:] <= smax]
    if len(small) < 10:
        return None
    t_s = max(3.0 * lw + 6.0, 1.3 * float(np.percentile(small, 90)))
    short = (ext <= t_s) & (np.arange(n) > 0)
    hh, ww = lm.shape
    if ext[short].sum() < 0.25 * ww or ext.max() < 0.25 * min(hh, ww) or ext[~short].sum() < 0.3 * ww:
        return None
    ids = [int(i) for i in np.nonzero(short)[0]]
    pieces = [i for i in ids if ext[i] > 4]                # the short pieces must be dashes in a row (tick marks
    edge = (st[:, 0] <= 4) | (st[:, 1] <= 4) | (st[:, 0] + st[:, 2] >= ww - 4) | (st[:, 1] + st[:, 3] >= hh - 4)
    dashes = [i for i in pieces if not edge[i]]            # left at the frame edge are a row too, but not dashes)
    if len(dashes) < 10 or _along_neighbours(lab, st, dashes, ids) < 0.5:
        return None
    longm = (lab > 0) & ~short[lab]                    # the two curves run close: a share of the dashes lies
    near = cv2.dilate(longm.astype(np.uint8), cv2.getStructuringElement(       # next to the long strokes (when they
        cv2.MORPH_ELLIPSE, (2 * int(3 * lw + 3) + 1,) * 2)) > 0                # are far apart, one pass links well)
    hit = np.zeros(n, bool)
    hit[np.unique(lab[near & (lab > 0)])] = True
    if hit[pieces].sum() < max(5, 0.15 * len(pieces)):
        return None
    cs = cen[short]                                    # dash pattern: median distance of neighbouring pieces
    period = float(np.median(cKDTree(cs).query(cs, k=2)[0][:, 1]))
    moved = np.zeros(n, bool)
    if other is not None and sk is not None:           # stretches of a solid curve between two crossings
        block = cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        block[:3] = True; block[-3:] = True; block[:, :3] = True; block[:, -3:] = True
        s8 = sk.astype(np.uint8)
        tips = sk & (cv2.filter2D(s8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s8 == 1)
        p90 = float(np.percentile(small, 90))
        for i in pieces:
            if ext[i] <= p90:                          # longer than the dashes
                continue
            x, y, w, h, _ = st[i]
            ty, tx = np.nonzero(tips[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i))
            if len(ty) >= 2:
                q = np.c_[tx, ty]
                a, b = np.unravel_index(int(np.linalg.norm(q[:, None] - q[None], axis=2).argmax()), (len(q), len(q)))
                if block[ty[a] + y, tx[a] + x] and block[ty[b] + y, tx[b] + x]:
                    short[i] = False
                    moved[i] = True
    return short[lab], period, moved[lab]


def analyse_cluster(mask, weight, color, diag, mind, log, other=None, symbols=True):
    """Find scatter markers, lines and markers sitting on lines in the ink of one colour."""
    out = []
    hh, ww = mask.shape
    smax = max(12.0, 0.09 * mind)
    # hollow markers: holes of similar size are filled so that open and filled symbols look alike
    n, lab, st, _ = _cc(~mask, 4)
    holes = [i for i in range(1, n) if st[i][0] > 0 and st[i][1] > 0 and st[i][0] + st[i][2] < ww and st[i][1] + st[i][3] < hh
             and st[i][4] >= 4 and max(st[i][2], st[i][3]) <= smax and 0.5 <= st[i][2] / st[i][3] <= 2]
    hole_mask = np.zeros_like(mask)
    if len(holes) >= 3:                                    # the most common hole size (fragments are smaller)
        ha = np.array([st[i][4] for i in holes], float)
        cnt = np.array([((ha >= 0.6 * a) & (ha <= 1.6 * a)).sum() for a in ha])
        a = ha[np.argmax(cnt * ha)]
        sel = [i for i, v in zip(holes, ha) if 0.6 * a <= v <= 1.6 * a]
        if len(sel) >= 3:
            hole_mask = np.isin(lab, sel)
    mf = mask | hole_mask
    strong = weight >= 0.4                                # ink without the faint anti-aliased rim
    n, lab, st, cen = _cc(mf)
    kind, hol = [""] * n, np.zeros(n, bool)
    lw0 = None                                             # stroke width of the long pieces (lines) of this colour
    long_ = [i for i in range(1, n) if max(st[i][2], st[i][3]) > smax]
    if long_:
        lm_ = np.isin(lab, long_).astype(np.uint8)
        d_ = cv2.distanceTransform(lm_, cv2.DIST_L2, 3)[thin(lm_) > 0]
        if d_.size:
            lw0 = max(1.0, 2.0 * float(np.percentile(d_, 40)) - 1.0)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        s = max(w, h)
        comp = lab[y:y + h, x:x + w] == i
        hol[i] = hole_mask[y:y + h, x:x + w][comp].mean() > 0.04
        if s <= 3:
            kind[i] = "dot"
        elif s <= smax and min(w, h) >= 3:
            rw, rh = cv2.minAreaRect(np.argwhere(comp).astype(np.float32))[1]
            kind[i] = "mk" if (max(rw, rh) + 1) / (min(rw, rh) + 1) < 1.45 else "dash"
            if kind[i] == "mk" and lw0 is not None and min(rw, rh) + 1 < 1.8 * lw0 + 1:
                kind[i] = "dash"                           # as thick as the line: a piece of a broken or hidden line
        else:
            kind[i] = "line"
    used = np.zeros(n, bool)
    n_dash = sum(k in ("dash", "line") for k in kind)
    compact = [i for i in range(1, n) if kind[i] in ("mk", "dash")]
    dashy = len(compact) >= 6 and _along_neighbours(lab, st, compact, compact) >= 0.5     # a dashed curve of short dashes
    groups, geo, size, area_ = {}, {}, {}, {}
    for i in range(1, n):
        if kind[i] == "mk":
            x, y, w, h, _ = st[i]
            comp = lab[y:y + h, x:x + w] == i
            ink = comp & mask[y:y + h, x:x + w]
            core, sx, sy = marker_core(comp, strong[y:y + h, x:x + w])
            cm, dx, dy = clean_marker(core)
            if cm is comp:                                 # thin strokes (+, x): classify the ink itself
                shape, hollow = classify_marker(ink)
            else:
                shape, hollow = classify_marker(cm)[0], ink.sum() < 0.8 * comp.sum()
            dx, dy = dx + sx, dy + sy
            yy, xx = np.nonzero(cm)
            geo[i] = (x + dx + xx.mean(), y + dy + yy.mean(), x + dx + (cm.shape[1] - 1) / 2.0, y + dy + (cm.shape[0] - 1) / 2.0)
            size[i], area_[i] = max(cm.shape), int(cm.sum())    # of the symbol itself, without attached line stubs
            groups.setdefault((shape, bool(hollow or hol[i])), []).append(i)
    good = {}
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[key]
        ma = np.median([area_[i] for i in g])
        ms = np.median([size[i] for i in g])
        g = [i for i in g if 0.6 * ma - 2 <= area_[i] <= 1.25 * ma + 2 and 0.75 * ms - 1 <= size[i] <= 1.25 * ms + 1]
        if len(g) >= 3:
            c = cen[g]
            d1 = cKDTree(c).query(c, k=2)[0][:, 1] if len(g) >= 4 else np.array([0.0, 1e9])
            even = np.median(np.abs(d1 - np.median(d1))) < 0.25 * np.median(d1)
            if key[0] == "dot":                            # tiny dots: part of a dotted / dash-dot line?
                if n_dash >= len(g) / 3 or (even and np.median(d1) < 6 * ms):
                    continue
            elif ms < 7 and not key[1] and key[0] in ("circle", "square") and even and np.median(d1) < 2.7 * ms:
                continue                                   # evenly spaced small fat dots: a dotted line
            elif not key[1] and _along_neighbours(lab, st, g, compact) >= 0.5:
                continue                                   # short dashes stretched along their row: a dashed line
            good[key] = dict(ids=g, ma=ma, ms=ms)
        else:                                              # a few odd ones: same symbol read differently
            for tk, t in good.items():
                if tk[1] == key[1]:
                    t["ids"] += [i for i in g if 0.6 * t["ma"] <= area_[i] <= 1.25 * t["ma"]]
                    break
    for key, t in good.items():
        used[t["ids"]] = True
    # touching / overlapping symbols: explain the remaining compact blobs with copies of the average symbol
    pts, extra = {key: [geo[i] for i in t["ids"]] for key, t in good.items()}, 0
    for key, t in sorted(good.items(), key=lambda kv: -len(kv[1]["ids"])):
        half = int(math.ceil(t["ms"] / 2.0)) + 2
        mp = np.pad(mask, half).astype(np.float32)
        crops = [mp[int(round(geo[i][3])):int(round(geo[i][3])) + 2 * half + 1,
                    int(round(geo[i][2])):int(round(geo[i][2])) + 2 * half + 1] for i in t["ids"]]
        mean = np.mean(crops, axis=0)
        tm = crops[int(np.argmax([(c * mean).sum() / math.sqrt(max(c.sum(), 1.0)) for c in crops]))]   # most typical one
        t["tm"] = tm
        g = np.array(pts[key])
        off = (g[:, :2] - g[:, 2:]).mean(axis=0)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if used[i] or kind[i] == "dot" or max(w, h) > 6 * t["ms"]:
                continue
            reg = (lab[y:y + h, x:x + w] == i) & mask[y:y + h, x:x + w]
            cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            solid = np.zeros(reg.shape, np.uint8)
            cv2.drawContours(solid, cnts, -1, 1, -1)
            if (reg.sum() < 0.8 * solid.sum()) != key[1]:  # open symbols explain open blobs only
                continue
            kmax = int(math.ceil(reg.sum() / (tm > 0.5).sum() - 0.15))
            found = fit_symbols(reg, tm, min(kmax, 12)) if kmax >= 1 else []
            for mx, my in found:
                pts[key].append((mx + x + off[0], my + y + off[1], mx + x, my + y))
            if found:
                used[i] = True
                extra += len(found)
    if extra:
        log.append(f"{extra} touching or overlapping symbols separated")
    # ---- lines: everything except the symbols found above (pieces of line attached to them stay)
    foot = np.zeros(mask.shape, np.uint8)
    for key, t in good.items():
        for p in pts[key]:
            cv2.circle(foot, (int(round(p[2])), int(round(p[3]))), int(math.ceil(0.75 * t["ms"])) + 1, 1, -1)
    lm = (mf & (foot == 0)).astype(np.uint8)
    if good:                                               # crumbs left at the rim of the removed symbols
        near = cv2.dilate(foot, np.ones((5, 5), np.uint8))
        nc, lc, sc, _ = _cc(lm)
        for i in range(1, nc):
            x, y, w, h, a_ = sc[i]
            if max(w, h) <= 4 and near[y:y + h, x:x + w][lc[y:y + h, x:x + w] == i].any():
                lm[y:y + h, x:x + w][lc[y:y + h, x:x + w] == i] = 0
    lines, owner, on_line = [], {}, {}
    if lm.sum() >= 8:
        lm0 = lm.copy()
        dt = cv2.distanceTransform(lm, cv2.DIST_L2, 3)
        sk = thin(fill_pins(lm))
        lw = max(1.0, 2.0 * float(np.percentile(dt[sk], 40)) - 1.0)
        dash_m = None if good else dash_ink(lm0, lw, smax, sk, other)   # a solid and a dashed curve of this colour
        dash_m, period, moved = dash_m if dash_m is not None else (None, 0.0, None)
        r = max(2, int(math.ceil(lw / 2.0 + 0.5)))
        disk = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        nb, labb, stb, cenb = _cc(cv2.morphologyEx(lm, cv2.MORPH_OPEN, disk(r)))  # thicker than the line: symbols
        thick = [i for i in range(1, nb) if stb[i][4] >= (lw + 2) ** 2 and max(stb[i][2], stb[i][3]) >= 2.2 * lw + 2]
        blobs = []
        for i in thick:                                    # a symbol is compact and clearly thicker than the line;
            if max(stb[i][2], stb[i][3]) <= 1.3 * smax:    # lines that merge or cross at a small angle are not
                x, y, w, h, _ = stb[i]
                rw, rh = cv2.minAreaRect(np.argwhere(labb[y:y + h, x:x + w] == i).astype(np.float32))[1]
                if min(rw, rh) + 1 >= 2.2 * lw + 1 and (max(rw, rh) + 1) / (min(rw, rh) + 1) < 2.0:
                    blobs.append(i)
        if len(blobs) >= 3:                                # thick pieces that stand alone, stretched along their
            nl_, labl, stl, _ = _cc(lm)                    # row: the dashes of a dashed curve, not symbols on a line
            alone = []
            for i in blobs:
                ids, cnt = np.unique(labl[labb == i], return_counts=True)
                j = int(ids[np.argmax(cnt)])
                if j and stl[j][4] <= 1.6 * stb[i][4]:
                    alone.append(j)
            pieces = [j for j in range(1, nl_) if max(stl[j][2], stl[j][3]) <= smax]
            if len(alone) >= 0.5 * len(blobs) and _along_neighbours(labl, stl, alone, pieces) >= 0.5:
                blobs = []
        if len(blobs) >= 3:
            med = np.median([stb[i][4] for i in blobs])
            blobs = [i for i in blobs if 0.6 * med <= stb[i][4] <= 1.35 * med]
        fitted, erase = [], []
        key0 = max((k for k in good if not k[1]), key=lambda k: len(good[k]["ids"]), default=None)
        if key0 is not None:                               # the free-standing symbol of this colour also sits on the line
            blobs, tm, tarea, tsize = [], good[key0]["tm"], float((good[key0]["tm"] > 0.5).sum()), good[key0]["ms"]
        elif len(blobs) >= 3:                              # symbols of one size sit on the line
            tarea = float(np.median([stb[i][4] for i in blobs]))
            tsize = float(np.median([max(stb[i][2], stb[i][3]) for i in blobs]))
            x, y, w, h, _ = stb[min(blobs, key=lambda i: abs(stb[i][4] - tarea))]
            tm = np.pad((labb[y:y + h, x:x + w] > 0).astype(np.float32), 2)
        else:
            blobs = []
        beaded, beads = False, []
        if blobs and key0 is None and len(blobs) >= 6:     # symbols strung along the line close together, most of
            cc_ = np.array([cenb[i] for i in blobs])       # them touching or overlapping: the symbols are the curve
            merged = sum(stb[i][4] for i in thick if i not in blobs)       # thick ink of several symbols in one blob
            beaded = (np.median(cKDTree(cc_).query(cc_, k=2)[0][:, 1]) <= 2.2 * tsize
                      and merged >= 0.5 * sum(stb[i][4] for i in blobs))
        if beaded:                                         # their centres, also of symbols merged into one blob
            beads = [tuple(cenb[i]) for i in blobs]
            for i in thick:
                x, y, w, h, a_ = stb[i]
                if i not in blobs and 1.35 * tarea < a_ <= 14 * tarea and max(w, h) <= 8 * tsize:
                    found = fit_symbols(labb[y:y + h, x:x + w] == i, tm, min(12, int(math.ceil(a_ / tarea - 0.15))))
                    beads += [(mx + x, my + y) for mx, my in found]
            blobs = []                                     # the line: one solid band (ring centres and the gaps
            lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, disk(int(math.ceil(tsize / 2.0))))   # between symbols closed)
            nh, labh, sth, _ = _cc(lm == 0, 4)
            for i in range(1, nh):
                if sth[i][4] <= 2 * tarea and sth[i][0] > 0 and sth[i][1] > 0 and sth[i][0] + sth[i][2] < ww and sth[i][1] + sth[i][3] < hh:
                    lm[labh == i] = 1
            sk = thin(fill_pins(lm))
            log.append("symbols along a line too close to separate: traced as one line through them")
        notsym, quiet = set(), False
        if blobs and not beaded:                           # tips of narrow peaks stay in the line
            notsym = {i for i in blobs if on_line_symbol(lm, labb, stb[i], cenb[i], i, lw) == "apex"}
            blobs = [i for i in blobs if i not in notsym]
            if len(blobs) < 3:
                blobs = []
            else:                                          # the other thick spots are cut out as before, but are
                cc_ = np.array([cenb[i] for i in blobs])   # reported as symbols only if they mark data points: at
                d1_ = cKDTree(cc_).query(cc_, k=2)[0][:, 1]    # least 5, about evenly spaced (some may be hidden);
                quiet = len(blobs) < 5 or np.median(np.abs(d1_ - np.median(d1_))) > 0.35 * np.median(d1_)
                                                           # thick spots where curves cross or part are not symbols
        if blobs and dash_m is not None:                   # thick spots where a dash merges into the solid curve of
            blobs = []                                     # the same colour: kept, the solid goes on through them
        if blobs or key0 is not None:
            for i in thick:                                # touching symbols: several copies fit one blob
                x, y, w, h, a_ = stb[i]
                if (i in blobs or i in notsym or not ((1.35 if blobs else 0.6) * tarea < a_ <= 14 * tarea)
                        or max(w, h) > 8 * tsize):
                    continue
                found = fit_symbols(labb[y:y + h, x:x + w] == i, tm, min(12, int(math.ceil(a_ / tarea - 0.15))))
                fitted += [(mx + x, my + y) for mx, my in found]
                if found:
                    erase.append(i)
        r_min, trim = (max(1.5 * t["ms"] for t in good.values()) + 8.0 if good else 0.0), 0.0
        if blobs or erase:
            lm[cv2.dilate(np.isin(labb, blobs + erase).astype(np.uint8), disk(r + 2)) > 0] = 0
            sk = thin(fill_pins(lm))
            r_min, trim = max(r_min, 1.3 * tsize + 4 * r + 10), r + 1.0
        if dashy and not good:
            sk = straighten_dashes(lm, sk, smax)
        def chain(sk_, r_, wide=False, cover=None, hair=None):
            return insert_dots(extend_chains(build_chains(skeleton_edges(sk_), lw, r_, diag, trim, wide, cover, hair),
                                             diag, lw), diag)
        if dash_m is None or beads:
            chs = chain(sk, r_min)
        else:                                              # solid strokes and dashes of one colour: chained apart;
            chs, back = [], np.zeros_like(sk)              # pieces of the strokes that stay short (two dashes that
            cov = None if other is None else cv2.dilate((other | (lm0 > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            for ch in chain(sk & ~dash_m, r_min, True, cov, 1.2 * period):   # touch), that are mostly pieces taken
                q = np.round(ch["pts"]).astype(int)        # that are a row of dashes go back to the dashes
                if (float(np.linalg.norm(np.ptp(ch["pts"], axis=0))) >= 0.08 * diag and moved[q[:, 1], q[:, 0]].mean() < 0.5
                        and not (len(ch["gaps"]) >= 3 and ch["ink"] < 0.8 * _plen(ch["pts"]))):
                    chs.append(ch)
                else:
                    back[q[:, 1], q[:, 0]] = True
            used_ = np.zeros(sk.shape, np.uint8)            # stroke skeleton no kept chain runs on (branches of
            for ch in chs:                                 # merged dashes dropped at junctions) goes back as well
                cv2.polylines(used_, [np.round(ch["pts"]).astype(np.int32).reshape(-1, 1, 2)], False, 1, 3)
            back |= sk & ~dash_m & (used_ == 0)
            back = (cv2.dilate(back.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & sk & ~dash_m
            chs += chain((sk & dash_m) | back, max(r_min, 2.0 * period + 2.0))
        for ch in chs:
            p = ch["pts"]
            ext = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
            if ext < 0.08 * diag or (ch["ink"] < 0.3 * _plen(p) and len(ch["pieces"]) < 4):
                continue
            if is_straight(p) and ext < 0.25 * diag:       # arrow shaft, legend sample, error bar ...
                continue
            px = refine_and_fill(p, weight, lw, ch["gaps"])
            if beads:
                px = through_beads(px, beads, tsize)
            px = unkink(px, max(6.0 * (tsize if beads else lw), 0.05 * diag))
            lines.append((px[::-1] if px[0, 0] > px[-1, 0] else px, ch))
        # which line does each symbol sit on, and what does it look like without the line
        items = []
        if lines:
            tree = cKDTree(np.vstack([p for p, _ in lines]))
            which = np.concatenate([np.full(len(p), k) for k, (p, _) in enumerate(lines)])
            corr = np.zeros(lm.shape, np.uint8)
            for p, _ in lines:
                cv2.polylines(corr, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], False, 1, int(round(lw + 2)))
        for i in blobs:
            x, y, w, h, _ = stb[i]
            pad = r + 3
            ya, yb_, xa, xb = max(0, y - pad), min(hh, y + h + pad), max(0, x - pad), min(ww, x + w + pad)
            core = (labb[ya:yb_, xa:xb] == i).astype(np.uint8)
            m = (cv2.dilate(core, disk(r + 2)) > 0) & (lm0[ya:yb_, xa:xb] > 0)
            if lines:                                      # cut the line stubs sticking out of the symbol
                m &= ~((corr[ya:yb_, xa:xb] > 0) & ~(cv2.dilate(core, disk(1)) > 0))
            shape, alt = "circle", (x + (w - 1) / 2.0, y + (h - 1) / 2.0)
            if m.any():
                cm, dx, dy = marker_core(m, strong[ya:yb_, xa:xb])
                cm, ex, ey = clean_marker(cm)
                shape = classify_marker(cm)[0]
                alt = (xa + dx + ex + (cm.shape[1] - 1) / 2.0, ya + dy + ey + (cm.shape[0] - 1) / 2.0)
            hollow = bool(hole_mask[y:y + h, x:x + w][labb[y:y + h, x:x + w] == i].mean() > 0.04)
            items.append([(shape, hollow), (cenb[i][0], cenb[i][1], alt[0], alt[1])])
        votes = {}
        for k_, _ in items:
            votes[k_] = votes.get(k_, 0) + 1
        common = max(votes, key=votes.get) if votes else key0
        items += [[common, (fx, fy, fx, fy)] for fx, fy in fitted]
        if quiet:                                          # cut out, but not symbols
            items, fitted = [], []
        for k_, g in items:
            d, j = tree.query(g[2:]) if lines else (1e9, 0)
            owner.setdefault(int(which[j]) if d <= 3.0 * tsize + trim + r else -1, []).append((k_, g))
        if fitted:
            log.append(f"{len(fitted)} symbols touching each other or a line separated")
    for k, (px, ch) in enumerate(lines):
        n_mark = len(owner.get(k, []))
        if n_mark < 3 or np.ptp([g[0] for _, g in owner[k]]) < 0.25 * np.ptp(px[:, 0]):
            n_mark = 0
        ng, gap_len = len(ch["gaps"]), sum(g[1] for g in ch["gaps"])
        pl = np.array(ch["pieces"])
        short = max(3.0, 1.6 * lw)
        srt = np.sort(pl)[::-1]
        half = srt[np.searchsorted(np.cumsum(srt), 0.5 * srt.sum())]      # half of the ink is in pieces at least this long
        if (ng < 3 or gap_len < 0.08 * _plen(ch["pts"]) or (n_mark and ng <= n_mark + 2) or half > max(25.0, 15.0 * lw)
                or ch["ink"] >= 0.8 * _plen(ch["pts"])):
            style = "solid"                                # also a solid line interrupted where other curves cover it
        elif np.percentile(pl, 80) <= short:
            style = "dotted"
        elif np.percentile(pl, 80) > 2.5 * max(np.percentile(pl, 20), 1.0) and np.percentile(pl, 20) <= short:
            style = "dash-dot"
        else:
            style = "dashed"
        note = ("gaps interpolated" if (style != "solid" or ng) else "") + ("; with markers" if n_mark else "")
        on_line[k] = Series("line", px, color, style, note=note.strip("; "))
    # symbols on a line join the free-standing symbols of the same kind; otherwise they follow their line
    tail = []
    for k in sorted(owner, key=lambda v: (v < 0, v)):
        kinds = {}
        for k_, _ in owner[k]:
            kinds[k_] = kinds.get(k_, 0) + 1
        major = max(kinds, key=kinds.get)
        g = [p for _, p in owner[k]]
        if major in pts:
            pts[major] += g
        elif k >= 0 and (len(g) < 3 or np.ptp([q[0] for q in g]) < 0.25 * np.ptp(lines[k][0][:, 0])):
            continue                                       # a few thick spots in one place on a line (where curves
                                                           # cross) are not its symbols: those are spread along it
        elif k >= 0:
            tail.append((k, major, g))
        else:
            same = [t for t in tail if t[1] == major]
            if len(same) == 1:
                same[0][2].extend(g)
            elif len(g) >= 3:
                tail.append((-1, major, g))
    for key in good:
        g = np.array(sorted(pts[key]))
        out.append(Series("scatter", g[:, :2], color, marker_name(*key), px_alt=g[:, 2:]))
    for k in sorted(on_line):
        out.append(on_line[k])
        for kk, major, g in tail:
            if kk == k:
                g = np.array(sorted(g))
                out.append(Series("scatter", g[:, :2], color, marker_name(*major), note="markers of the line above", px_alt=g[:, 2:]))
    for kk, major, g in tail:
        if kk < 0:
            g = np.array(sorted(g))
            out.append(Series("scatter", g[:, :2], color, marker_name(*major), note="markers", px_alt=g[:, 2:]))
    return refine_symbols(out, mask, log, weight, diag, mind, other) if symbols else out



def _symbol_template(mask, pts, hw):
    """Typical symbol: the ink common to the crops around the given centres (line stubs leaving in different
    directions drop out), cut to the piece of ink nearest the centre. Returns the boolean template (2hw+1)^2."""
    mp = np.pad(mask, hw)
    crops = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        crops.append(mp[yi:yi + 2 * hw + 1, xi:xi + 2 * hw + 1])
    # ink present at >= 75 % of the symbols (>= 50 % with few): a line touching every symbol on the same side (a
    # curve running along the tops of its points) is in about half of them and drops out
    med = np.mean(np.array(crops, np.float32), axis=0) >= (0.75 if len(crops) >= 8 else 0.5)
    n, lab, st, _ = _cc(med)
    if n <= 1:
        return None
    yy, xx = np.mgrid[:2 * hw + 1, :2 * hw + 1]
    best, bd = 0, 1e9
    for i in range(1, n):
        d = np.hypot(xx[lab == i] - hw, yy[lab == i] - hw).min()
        if d < bd:
            best, bd = i, d
    t = lab == best
    if bd > 0.6 * hw:                                      # (the ring of an open symbol is a hole radius away)
        return None
    # a line running past the symbols in the same direction at every symbol survives the median: cut it off with
    # the symbol's silhouette (holes filled, then opened with a disc of a quarter of its size)
    filled = np.zeros(t.shape, np.uint8)
    cnts, _ = cv2.findContours(t.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    ys, xs = np.nonzero(t)
    k = max(1, int(round(0.25 * min(np.ptp(xs), np.ptp(ys)))))
    sil = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n2, lab2, _, _ = _cc(sil > 0)
    if n2 > 1:                                             # the silhouette piece at the centre
        cc_ = lab2[hw, hw] if lab2[hw, hw] else int(np.argmax(np.bincount(lab2.ravel())[1:]) + 1)
        sil = (lab2 == cc_).astype(np.uint8)
        r_ = max(1, int(round(0.35 * k)) + 1)             # the opening rounds the corners off: give them back
        tc = t & (cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))) > 0)
        if tc.sum() >= 0.5 * t.sum() * (sil.sum() / max(filled.sum(), 1)) and tc.sum() >= 8:
            t = tc
    return t


def _dense(p):
    """A polyline resampled at about 1 px."""
    seg = np.diff(p, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int))
    return np.vstack([p[k] + np.outer(np.arange(n[k]) / n[k], seg[k]) for k in range(len(seg))] + [p[-1:]])


def _symbol_holes(mask):
    """Centres of the holes of open symbols: enclosed background pieces of about one size (the most common one),
    round enough. Returns an (N, 2) array (empty when fewer than 5)."""
    hh, ww = mask.shape
    n, lab, st, cen = _cc(~mask, 4)
    cen = np.asarray(cen, float)
    ids = [i for i in range(1, n) if st[i][0] > 0 and st[i][1] > 0 and st[i][0] + st[i][2] < ww and st[i][1] + st[i][3] < hh
           and st[i][4] >= 4 and max(st[i][2], st[i][3]) <= 0.08 * max(hh, ww) and 0.5 <= st[i][2] / st[i][3] <= 2.0
           and st[i][4] >= 0.45 * st[i][2] * st[i][3]]
    if len(ids) < 5:
        return np.zeros((0, 2))
    a = np.array([st[i][4] for i in ids], float)
    cnt = np.array([((a >= 0.6 * v) & (a <= 1.6 * v)).sum() for v in a])
    v = a[np.argmax(cnt)]
    sel = [i for i, ai in zip(ids, a) if 0.6 * v <= ai <= 1.6 * v]
    if len(sel) < 5:
        return np.zeros((0, 2))
    # a line through an open symbol cuts its hole in two: halves closer than a hole width are one symbol
    p = cen[sel]
    w = 1.5 * math.sqrt(v)
    grp = list(range(len(p)))
    for i, j in cKDTree(p).query_pairs(w):
        gi, gj = grp[i], grp[j]
        grp = [gi if g == gj else g for g in grp]
    out = np.array([p[[k for k in range(len(p)) if grp[k] == g]].mean(axis=0) for g in sorted(set(grp))])
    return out if len(out) >= 5 else np.zeros((0, 2))


def _match_symbols(mask, pts0, size):
    """Template of the symbol and all its places in the ink (see refine_symbols). Returns (template, hw, centres)
    or None when the symbols found so far do not look alike or the template finds far more than them."""
    hh, ww = mask.shape
    hw = int(min(40, max(6, math.ceil(0.75 * size + 3))))
    t = _symbol_template(mask, pts0, hw)
    if t is None:
        return None
    mf = mask.astype(np.float32)
    cen = np.zeros((0, 2))
    for _ in range(2):                                     # template and centres refined together
        tf = t.astype(np.float32)
        cov = cv2.filter2D(mf, -1, tf / tf.sum(), borderType=cv2.BORDER_CONSTANT)
        filled = np.zeros(t.shape, np.uint8)
        cnts, _ = cv2.findContours(t.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        hole = cv2.erode(((filled > 0) & ~t).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = (cv2.dilate(filled, np.ones((7, 7), np.uint8)) > 0) & ~(cv2.dilate(filled, np.ones((3, 3), np.uint8)) > 0)
        hf = (cv2.filter2D(mf, -1, hole.astype(np.float32) / hole.sum(), borderType=cv2.BORDER_CONSTANT)
              if hole.sum() >= 4 else np.zeros_like(cov))
        of = cv2.filter2D(mf, -1, ring.astype(np.float32) / ring.sum(), borderType=cv2.BORDER_CONSTANT)
        peak = (cov == cv2.dilate(cov, np.ones((3, 3), np.uint8))) & (cov >= 0.5)
        py, px_ = np.nonzero(peak)
        if not len(px_):
            return None
        near = cKDTree(np.c_[px_, py]).query(pts0, k=1)[1]
        ref = float(np.median(cov[py[near], px_[near]]))
        if ref < 0.85:                                     # the symbols found so far do not look alike: leave them
            return None
        ok = (cov[py, px_] >= 0.8 * ref) & (hf[py, px_] <= 0.7) & (of[py, px_] <= 0.5)
        keep = []
        for j in np.argsort(-cov[py, px_]):
            if ok[j] and all(math.hypot(px_[j] - px_[k], py[j] - py[k]) >= 0.6 * size for k in keep):
                keep.append(j)
        c = []
        for j in keep:                                     # sub-pixel: weighted centre of the coverage peak
            x, y = int(px_[j]), int(py[j])
            w_ = cov[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            yy_, xx_ = np.mgrid[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            w_ = np.clip(w_ - 0.9 * w_.min(), 0, None)
            c.append(((xx_ * w_).sum() / w_.sum(), (yy_ * w_).sum() / w_.sum()))
        cen = np.array(c) if c else np.zeros((0, 2))
        if len(cen) > 2 * len(pts0) + 3:                   # a template that matches the line itself (a line through
            return None                                    # open symbols fills their holes in the template)
        if len(cen) >= 3:
            t2 = _symbol_template(mask, cen, hw)
            if t2 is not None:
                t = t2
    if len(cen) < 3:
        return None
    return t, hw, cen


def refine_symbols(out, mask, log, weight=None, diag=1000.0, mind=500.0, other=None):
    """Symbols that touch a line without being its markers (a data set drawn over a curve, as in a measurement
    compared with a calculation). The user's rule: if the line goes through the symbol centres all along, the plot
    is one series, a line with symbols, and is left as traced; if it leaves them, the symbols are a separate data set:
    1. A clean template of the symbol is the ink common to the crops around the symbols found so far.
    2. It is matched over the whole ink of this colour: a symbol is where its ink is covered (>= 80 % of the best
       matches), its hole is not filled (a line may cross it) and the ring around it nearly empty (line stubs
       allowed). The centre is the middle of the symbol, not the centroid of symbol + line stub, and symbols the
       line swallowed are found.
    3. Line or separate? The centres are compared with the lines as first traced: >= 80 % within 0.3 symbol sizes
       of one line: line with symbols, nothing changes. Otherwise the lines along the symbols are traced again on
       the ink without the symbols' own ink (the first tracing erased the symbols with a disc, which also takes away
       a line running along their rims, and short-cut the curve)."""
    # open symbols only: their hole fixes the centre and tells them from the line; a filled symbol on a line is
    # handled by clean_marker. At least 5: too few to learn a symbol from (arrow heads read as triangles).
    scat = [s for s in out if s.kind == "scatter" and len(s.px) >= 5 and s.style.startswith("open")]
    holes = _symbol_holes(mask)                            # open symbols the first tracing took for a line: their holes
    n_sc = sum(len(s.px) for s in scat)
    seeded = None
    if len(holes) >= max(5, n_sc + 3) and any(s.kind == "line" for s in out):
        seeded = holes
    if not scat and seeded is None:
        return out
    hh, ww = mask.shape
    lines = [s for s in out if s.kind == "line"]
    dense = [_dense(s.px) for s in lines if len(s.px) >= 2]
    rest = mask.copy()
    foot_all = np.zeros(mask.shape, np.uint8)
    found_all, n_new, n_moved, n_line = [], 0, 0, 0
    groups = {}                                            # the open symbols of this colour; one data set can be read
    for s in scat:                                         # as two shapes (symbols touched by a line). Symbols that
        groups.setdefault(True, []).append(s)              # differ fail the likeness test below
    gl = []
    for hol, ss in groups.items():
        pts0 = np.vstack([s.px for s in ss])
        d1 = cKDTree(pts0).query(pts0, k=2)[0][:, 1]
        hw0 = int(min(40, max(6, math.ceil(0.75 * np.median(d1)))))
        t0 = _symbol_template(mask, pts0, hw0)
        if t0 is None:
            continue
        ys, xs = np.nonzero(t0)
        gl.append(dict(series=ss, size=float(max(np.ptp(xs), np.ptp(ys)) + 1), hollow=hol))
    groups = gl
    if seeded is not None:                                 # the holes as the symbols found so far
        d1 = cKDTree(seeded).query(seeded, k=2)[0][:, 1]
        hw0 = int(min(40, max(6, math.ceil(0.75 * np.median(d1)))))
        t0 = _symbol_template(mask, seeded, hw0)
        if t0 is not None:
            ys, xs = np.nonzero(t0)
            groups = [dict(series=scat, size=float(max(np.ptp(xs), np.ptp(ys)) + 1), hollow=True, seed=seeded)]
    for g in groups:
        pts0 = g["seed"] if "seed" in g else np.vstack([s.px for s in g["series"]])
        size = g["size"]
        m = _match_symbols(mask, pts0, size)
        if m is None:
            continue
        t, hw, cen = m
        # a "line" that is only the chain of these symbols (touching symbols traced as one curve through them): it
        # lies within the symbols' x range and runs mostly inside them. It goes; the symbols stay.
        ctr = cKDTree(cen)
        chain = []
        for q, dn in zip(lines, dense):
            inside = float((ctr.query(dn)[0] <= 0.3 * size).mean())   # through their middles (a line along
                                                                        # their rims is half a size away)
            if (inside >= 0.6 and dn[:, 0].min() >= cen[:, 0].min() - size and dn[:, 0].max() <= cen[:, 0].max() + size):
                chain.append(id(q))
        if chain:
            out = [q for q in out if id(q) not in chain]
            keep = [(q, dn) for q, dn in zip(lines, dense) if id(q) not in chain]
            lines, dense = [q for q, _ in keep], [dn for _, dn in keep]
            log.append(f"{len(chain)} chain(s) of touching symbols traced as a line: taken as the symbols")
        if dense:                                          # a line through the centres all along: line + symbols
            frac = max(float((cKDTree(dn).query(cen)[0] <= 0.3 * size).mean()) for dn in dense)
            if frac >= 0.8:
                n_line += 1
                continue
        else:
            frac = 0.0
        n_moved += int((cKDTree(cen).query(pts0)[0] > 0.5).sum())
        n_new += int((cKDTree(pts0).query(cen)[0] > 0.5 * size).sum())
        shape, hollow = classify_marker(t)
        tkf = t.astype(np.float32)                         # the symbols' own ink (exactly: a line along their rims
        foot = np.zeros(mask.shape, np.uint8)              # keeps its outer half), placed to the sub-pixel
        for x, y in cen:
            xi, yi = int(round(x)), int(round(y))
            tks = cv2.warpAffine(tkf, np.float32([[1, 0, x - xi], [0, 1, y - yi]]), tkf.shape[::-1]) >= 0.35
            ya, yb, xa, xb = yi - hw, yi + hw + 1, xi - hw, xi + hw + 1
            sa, sb, sc_, sd = max(0, -ya), (2 * hw + 1) - max(0, yb - hh), max(0, -xa), (2 * hw + 1) - max(0, xb - ww)
            foot[max(0, ya):min(hh, yb), max(0, xa):min(ww, xb)] |= tks[sa:sb, sc_:sd].astype(np.uint8)
        rest &= foot == 0
        foot_all |= foot
        nr, lr, sr, _ = _cc(rest)                          # crumbs of the symbols' rims left next to them
        near_f = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0
        for i in range(1, nr):
            x_, y_, w_, h_, a_ = sr[i]
            if max(w_, h_) <= 0.35 * size and near_f[y_:y_ + h_, x_:x_ + w_][lr[y_:y_ + h_, x_:x_ + w_] == i].any():
                rest[y_:y_ + h_, x_:x_ + w_][lr[y_:y_ + h_, x_:x_ + w_] == i] = False
        on = int(round(frac * len(cen)))
        note = ("separate data set (%d of %d symbols on a line)" % (on, len(cen))) if dense else "markers"
        cen = cen[np.argsort(cen[:, 0])]
        found_all.append(dict(cen=cen, size=size, src=g["series"],
                              new=Series("scatter", cen, (g["series"] or out)[0].color, marker_name(shape, hollow), note=note,
                                         px_alt=cen.copy())))
    if n_line:
        log.append(f"{n_line} symbol series on their line all along: kept as line with symbols")
    if not found_all:
        return out
    if n_moved or n_new:
        log.append(f"symbol centres from template matching: {n_moved} moved, {n_new} found on a line")
    # ---- the line along the separate symbols, followed again on the ink without the symbols, column by column
    allc = np.vstack([f["cen"] for f in found_all])
    size_ = max(f["size"] for f in found_all)
    ctree = cKDTree(allc)
    near_sym = lambda q: float(ctree.query(q.px)[0].min()) <= size_
    old_l = [q for q in lines if near_sym(q) and np.ptp(q.px[:, 0]) > 0]
    if old_l:
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        od = np.vstack([_dense(q.px) for q in old_l])
        od = od[np.argsort(od[:, 0])]
        qd = np.clip(np.round(od).astype(int), 0, [ww - 1, hh - 1])
        ux, inv = np.unique(np.round(od[:, 0]), return_inverse=True)        # pieces traced twice: one y per column
        od = np.c_[ux, [np.median(od[inv == k, 1]) for k in range(len(ux))]]
        lw = max(1.0, 2.0 * float(np.median(dt[qd[:, 1], qd[:, 0]])))
        # guide: the first tracing; where it is missing (the first tracing lost the line along the symbols), the
        # course of the symbol centres, shifted to join it
        sc_ = allc[np.argsort(allc[:, 0])]
        def sym_y(x):
            if len(sc_) >= 2 and (x < sc_[0, 0] or x > sc_[-1, 0]):
                e_ = sc_[:3] if x < sc_[0, 0] else sc_[-3:]
                k_, b0 = np.polyfit(e_[:, 0], e_[:, 1], 1)
                return k_ * x + b0
            return float(np.interp(x, sc_[:, 0], sc_[:, 1]))
        xa = int(max(0, math.ceil(min(od[0, 0], sc_[0, 0] - size_))))
        xb = int(min(ww - 1, math.floor(max(od[-1, 0], sc_[-1, 0] + size_))))
        off_l, off_r = od[0, 1] - sym_y(od[0, 0]), od[-1, 1] - sym_y(od[-1, 0])
        def guide(x):
            if x < od[0, 0]:
                return sym_y(x) + off_l
            if x > od[-1, 0]:
                return sym_y(x) + off_r
            return float(np.interp(x, od[:, 0], od[:, 1]))
        R = 1.5 * size_
        ys_, ok_ = np.full(xb - xa + 1, np.nan), np.zeros(xb - xa + 1, bool)
        rest2 = rest & ~(cv2.dilate(foot_all, np.ones((3, 3), np.uint8)) > 0)   # without the symbols' thick rims
        cols = []                                          # candidate line centres per column: the ink runs near the
        for x in range(xa, xb + 1):                        # first tracing (which may cut corners by up to R)
            yo = guide(x)
            lo, hi = int(max(0, yo - R)), int(min(hh, yo + R + 1))
            if hi <= lo:
                continue
            col = rest2[lo:hi, x] if rest2[lo:hi, x].any() else rest[lo:hi, x]
            if not col.any():
                continue
            ed = np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0]))
            cols.append((x, yo, np.array([lo + 0.5 * (a + b - 1) for a, b in zip(ed[::2], ed[1::2])]),
                         np.array([b - a for a, b in zip(ed[::2], ed[1::2])])))
        # the path through the candidates that changes like the first tracing (its shape is right where it cuts a
        # corner, only shifted) and skips as few columns as possible: dynamic programming over the columns
        G, gap_cost, sig = int(max(3, 1.2 * size_)), 0.5, max(1.0, 0.5 * lw)   # sig: jitter of a run centre
        best, back = [], []
        for i_, (x, yo, cc, _) in enumerate(cols):
            b_ = 0.02 * ((cc - yo) / sig) ** 2 + gap_cost * (x - xa)
            bk = np.full(len(cc), -1)
            bj = np.full(len(cc), -1)
            for i2 in range(i_ - 1, -1, -1):
                x2, yo2, cc2, _ = cols[i2]
                if x - x2 > G:
                    break
                d = (cc[:, None] - cc2[None, :]) - (yo - yo2)
                tot = best[i2][None, :] + (d / sig) ** 2 + gap_cost * (x - x2 - 1)
                jm = np.argmin(tot, axis=1)
                v = tot[np.arange(len(cc)), jm]
                better = v < b_
                b_ = np.where(better, v, b_)
                bk = np.where(better, i2, bk)
                bj = np.where(better, jm, bj)
            best.append(b_)
            back.append((bk, bj))
        ys_, ok_ = np.full(xb - xa + 1, np.nan), np.zeros(xb - xa + 1, bool)
        if cols:
            ends = [(best[i_].min() + gap_cost * (xb - cols[i_][0]), i_) for i_ in range(len(cols))]
            _, i_ = min(ends)
            j_ = int(np.argmin(best[i_]))
            long_ = np.zeros(xb - xa + 1, bool)
            while i_ >= 0:
                x = cols[i_][0]
                ys_[x - xa], ok_[x - xa] = cols[i_][2][j_], True
                long_[x - xa] = cols[i_][3][j_] > 3.0 * lw    # a long vertical run: a steep part, or the frame
                i_, j_ = int(back[i_][0][j_]), int(back[i_][1][j_])
            on_ = np.flatnonzero(ok_)                      # a path that ends running down the frame (a vertical
            for _ in range(3):                             # line, a column or three wide): cut there. A steep curve
                if len(on_) > 2 and long_[on_[-1]]:        # has long runs over many columns and stays.
                    ok_[on_[-1]] = False
                    on_ = on_[:-1]
                if len(on_) > 2 and long_[on_[0]]:
                    ok_[on_[0]] = False
                    on_ = on_[1:]
        xs_ = np.arange(xa, xb + 1, dtype=float)
        if ok_.sum() >= 0.5 * len(xs_):
            ys_ = np.interp(xs_, xs_[ok_], ys_[ok_])       # gaps where the line crosses a symbol: bridged
            lo_, hi_ = np.flatnonzero(ok_)[[0, -1]]
            px = np.c_[xs_[lo_:hi_ + 1], ys_[lo_:hi_ + 1]]
            # beyond the symbols the line may go on (to the frame): followed while there is ink in its direction
            def extend(px, step):
                pts = [tuple(p_) for p_ in (px[::-1] if step < 0 else px)][-12:]
                add, x, miss = [], int(pts[-1][0]), 0
                while 0 < x + step < ww - 1 and miss <= G:
                    x += step
                    hx, hy = np.array(pts[-12:]).T
                    k_ = np.polyfit(hx, hy, 1)[0] if len(hx) >= 3 and np.ptp(hx) > 0 else 0.0
                    pred = pts[-1][1] + k_ * (x - pts[-1][0])
                    lo, hi = int(max(0, pred - R)), int(min(hh, pred + R + 1))
                    if hi <= lo:
                        break
                    col = rest[lo:hi, x]
                    ed = np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0]))
                    cc = [lo + 0.5 * (a_ + b_ - 1) for a_, b_ in zip(ed[::2], ed[1::2]) if b_ - a_ <= 3.0 * lw]
                    c = min(cc, key=lambda m: abs(m - pred)) if cc else None
                    if c is None or abs(c - pred) > max(3.0, 1.5 * lw) + abs(k_) * miss:
                        miss += 1
                        continue
                    pts.append((x, c))
                    add.append((x, c))
                    miss = 0
                return np.array(add) if add else np.zeros((0, 2))
            # only where the first tracing stopped at the symbols (it lost the line there); elsewhere it already
            # ended where the line ends
            right = extend(px, 1) if od[-1, 0] <= sc_[-1, 0] + size_ else np.zeros((0, 2))
            left = extend(px, -1) if od[0, 0] >= sc_[0, 0] - size_ else np.zeros((0, 2))
            px = np.vstack([left[::-1], px, right]) if len(left) or len(right) else px
            if len(px) >= 7:
                px[:, 1] = np.convolve(np.pad(px[:, 1], 3, mode="edge"), np.ones(7) / 7, "valid")
            style = max({q.style for q in old_l}, key=lambda st: sum(_plen(q.px) for q in old_l if q.style == st))
            new = Series("line", px, old_l[0].color, "solid" if style == "solid" else style,
                         note="followed on its ink beside the symbols")
            gone_l = {id(q) for q in old_l}
            out = [q for q in out if id(q) not in gone_l] + [new]
            log.append("line along the symbols followed again on its own ink")
    gone = {id(s) for f in found_all for s in f["src"]}
    return [s for s in out if id(s) not in gone] + [f["new"] for f in found_all]


def on_line_symbol(lm, labb, st, cen, i, lw):
    """Is a thick spot on a line the tip of a narrow peak (or the bottom of a narrow dip)? Line stubs leaving it
    are counted on a ring around it (only stubs that cross the whole ring: the corners of a triangle or square
    only poke into it). "apex": both flanks leave on one side, one going left and one right, possibly with the
    tip of the stroke as a third stub on the other side; it stays in the line (cut out, the peak was bridged
    flat). "symbol": anything else (a symbol on the line, a crossing, two lines parting); erased as before."""
    x, y, w, h, _ = st
    cx, cy = cen
    r1 = 0.5 * max(w, h) + 1.5
    r2 = r1 + 2.0 * lw + 3.0
    hh, ww = lm.shape
    x0, x1 = max(0, int(cx - r2) - 1), min(ww, int(cx + r2) + 2)
    y0, y1 = max(0, int(cy - r2) - 1), min(hh, int(cy + r2) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    ring = ((d >= r1) & (d <= r2) & (lm[y0:y1, x0:x1] > 0)).astype(np.uint8)
    n, lab, stc, cc = _cc(ring)
    dirs = []
    for k in range(1, n):
        if stc[k][4] >= 2 and d[lab == k].max() >= r2 - 1.5:
            q = np.array([cc[k][0] + x0 - cx, cc[k][1] + y0 - cy])
            dirs.append(q / max(np.linalg.norm(q), 1e-9))
    if len(dirs) == 2 and _angle(dirs[0], dirs[1]) <= 100 and dirs[0][0] * dirs[1][0] < 0:
        return "apex"
    if len(dirs) == 3:
        for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            m = dirs[a] + dirs[b]
            if (_angle(dirs[a], dirs[b]) <= 75 and dirs[a][0] * dirs[b][0] < 0 and np.linalg.norm(m) > 0
                    and _angle(m / np.linalg.norm(m), dirs[c]) >= 135):
                return "apex"
    return "symbol"


def edge_softness(img, frame):
    """Blur of the picture, measured across the x-axis line: distance in pixels between the 12 % and
    50 % levels of its edges (0.4 to 0.9 for a crisp rendering, more when blurred or enlarged)."""
    xl, yt, xr, yb = frame["box"]
    if frame.get("panel") or xr - xl < 40:
        return 0.0
    a, b = xl + (xr - xl) // 8, xr - (xr - xl) // 8
    band = img[max(0, yb - 25):yb + 26, a:b].astype(np.int16)
    prof = np.median(np.abs(band - dominant_color(band)).max(axis=2), axis=1).astype(float)
    k = int(prof.argmax())
    if prof[k] < 60:
        return 0.0

    def cross(level, seq):                                 # going outwards from the peak
        for i in range(len(seq) - 1):
            if seq[i] >= level > seq[i + 1]:
                return i + (seq[i] - level) / (seq[i] - seq[i + 1])
        return float(len(seq))
    return float(np.mean([cross(0.12 * prof[k], s) - cross(0.5 * prof[k], s) for s in (prof[k:], prof[k::-1])]))


def process_raster(img, progress=_noprog, _depth=0, full=False):
    """Digitize a raster image (RGB uint8 array) of one XY plot. full: no reduction of blurred pictures."""
    log = []
    img = np.ascontiguousarray(img[:, :, :3])
    h, w = img.shape[:2]
    progress(5, "Locating the axes")
    frame = find_frame(img)
    have_frame = frame is not None
    soft = edge_softness(img, frame) if have_frame else 0.0
    f = 1.0
    if _depth == 0 and soft >= 1.0 and not full:           # soft edges: an enlarged or blurred picture
        hc = max(margin_labels(img, frame, "x", False)[2], margin_labels(img, frame, "y", False)[2])
        f = min(hc / 12.0 if hc >= 14 else (soft / 0.5 if soft >= 1.4 else 1.0), min(h, w) / 300.0)
    if f >= 1.4:                                           # work at the resolution the picture really has
        small = cv2.resize(img, (int(round(w / f)), int(round(h / f))), interpolation=cv2.INTER_AREA)
        res = process_raster(small, progress, 1)
        fx, fy = w / small.shape[1], h / small.shape[0]
        up = lambda p: np.c_[(p[:, 0] + 0.5) * fx - 0.5, (p[:, 1] + 0.5) * fy - 0.5]
        for s in res.series:
            s.px = up(s.px)
            s.px_alt = None if s.px_alt is None else up(s.px_alt)
        for ax, k in ((res.xaxis, fx), (res.yaxis, fy), (res.yaxis2, fy)):
            if ax is not None:                             # value = a * p_small + b, p_small = (p + 0.5) / k - 0.5
                ax.b += ax.a * (0.5 / k - 0.5)
                ax.a /= k
        x0, y0, x1, y1 = res.box
        res.box = ((x0 + 0.5) * fx - 0.5, (y0 + 0.5) * fy - 0.5, (x1 + 0.5) * fx - 0.5, (y1 + 0.5) * fy - 0.5)
        res.image = img
        res.log.insert(0, f"The picture is blurred (about {soft:.1f} px): analysed at {small.shape[1]} x {small.shape[0]} px. "
                       "If curves are cut or mixed, try Settings: analyse blurred pictures at full size.")
        return res
    if not have_frame:
        log.append("Axes frame not found: the whole image is used, set the axes by hand.")
        frame = dict(box=(0, 0, w - 1, h - 1), t=(0, 0, 0, 0))
    xl, yt, xr, yb = frame["box"]
    tl, tt, tr, tb = frame["t"]
    x0, y0 = xl + tl + (1 if tl else 0), yt + tt + (1 if tt else 0)
    x1, y1 = xr - tr - (1 if tr else 0), yb - tb - (1 if tb else 0)
    work = img[y0:y1 + 1, x0:x1 + 1].copy()
    hh, ww = work.shape[:2]
    bg = dominant_color(work)
    fg = np.abs(work.astype(np.int16) - bg).max(axis=2) > 30
    fg0 = fg.copy()
    in_bottom, in_left, in_right = remove_stubs(fg, max(4, int(0.05 * min(hh, ww))), max(frame["t"])) if have_frame else ([], [], [])
    if have_frame:
        erase_crumbs(fg, fg0 & ~fg)

    progress(15, "Reading tick labels")
    xax = yax = yax2 = None
    found_text, words = None, []
    hchar, xticks, yticks = 10.0, [], []
    if have_frame:
        found_text = OCR.detect(img)                    # also used below for the text inside the plot
        nlet = lambda t: sum(ch.isalpha() for ch in t)  # words (axis titles), even when misread ('Rence')
        words = [q for t, sc, a, b, c, d, q in found_text if nlet(t) >= 4 or (nlet(t) >= 3 and sc >= 0.8)]
        lx, tx, hx = margin_labels(img, frame, "x", words=words)
        ly, ty, hy = margin_labels(img, frame, "y", words=words)
        hchar = max(hx, hy) or 10.0
        xticks = tx if len(tx) >= 2 else [(p + x0, ln) for p, ln in in_bottom]
        yticks = ty if len(ty) >= 2 else [(p + y0, ln) for p, ln in in_left]
        if xticks and yticks:                              # the frame corners carry labels but no tick stub
            lmx, lmy = max(t[1] for t in xticks), max(t[1] for t in yticks)
            xticks = xticks + [(xl + (tl - 1) / 2.0, lmx), (xr - (tr - 1) / 2.0, lmx)]
            yticks = yticks + [(yt + (tt - 1) / 2.0, lmy), (yb - (tb - 1) / 2.0, lmy)]
        lx = snap_to_ticks(lx, xticks, max(4.0, 0.6 * hchar))
        ly = snap_to_ticks(ly, yticks, max(4.0, 0.6 * hchar))
        if len(lx) >= 2:
            xax = fit_axis([p for p, _ in lx], [v for _, v in lx], xr - xl, True)
        if len(ly) >= 2:
            yax = fit_axis([p for p, _ in ly], [v for _, v in ly], yb - yt, False)
        if tr:                                             # a frame line on the right: it may carry its own scale
            l2, t2, _ = margin_labels(img, frame, "y2", words=words)
            ticks2 = t2 if len(t2) >= 2 else [(p + y0, ln) for p, ln in in_right]
            if ticks2:
                lm2 = max(t[1] for t in ticks2)
                ticks2 = ticks2 + [(yt + (tt - 1) / 2.0, lm2), (yb - (tb - 1) / 2.0, lm2)]
            l2 = snap_to_ticks(l2, ticks2, max(4.0, 0.6 * hchar))
            if len(l2) >= 3:
                yax2 = fit_axis([p for p, _ in l2], [v for _, v in l2], yb - yt, False)
            yax, yax2 = second_axis(yax, yax2, yt, yb, log)
        if OCR.error:
            log.append(f"OCR not available ({OCR.error}): set the axes by hand.")

    progress(35, "Removing grid, legend and text")
    remove_grid(work, fg, bg, [(p - x0, ln) for p, ln in xticks], [(p - y0, ln) for p, ln in yticks], log)
    faint = np.abs(work.astype(np.int16) - bg).max(axis=2) > 10
    faint[:3] = False; faint[-3:] = False; faint[:, :3] = False; faint[:, -3:] = False   # rim of the frame
    for bx0, by0, bx1, by1, m in find_legend_boxes(faint, hchar):
        work[m] = bg
        fg[m] = False
        k = erase_inside(work, fg, bg, [(bx0, by0, bx1, by1)])
        log.append(f"legend / text box removed ({k} items inside)")
    cand = [(is_real_text(t, s), a - x0, b - y0, c - x0, d - y0, s >= 0.95 and len(t.strip()) == 1 and t.strip().isalpha(),
             q - (x0, y0))
            for t, s, a, b, c, d, q in (found_text if found_text is not None else OCR.detect(img))
            if x0 < 0.5 * (a + c) < x1 and y0 < 0.5 * (b + d) < y1]
    cand = [c for c in cand if c[0]]
    leg = legend_like([c[1:5] for c in cand])
    letter = lambda c: c[5] and 0.6 * hchar <= c[4] - c[2] <= 1.8 * hchar      # one letter, as tall as the tick labels
    texts = [c[1:5] + (c[6],) for i, c in enumerate(cand) if c[0] == 2 or (i in leg and len(leg) >= 3) or letter(c)]
    if texts:
        rects = [(a - 2, b - 2, c + 2, d + 2) for a, b, c, d, _ in texts]
        tight, inner = [], np.zeros((hh, ww), np.uint8)   # boxes of the letters themselves (detector boxes are padded)
        for a, b, c, d, q in texts:                        # erase the ink of the text colour inside the text box, which
            box = np.zeros((hh, ww), np.uint8)             # is turned with the text (labels written along a curve)
            cv2.fillPoly(box, [np.round(q).astype(np.int32)], 1)
            box = cv2.dilate(box, np.ones((5, 5), np.uint8)) > 0
            tight.append(None)
            if (fg & box).sum() < 4:
                continue
            px = work[fg & box].astype(np.float32)
            dd = np.abs(px - bg).sum(axis=1)
            col = np.median(px[dd >= np.percentile(dd, 60)], axis=0)
            ink = fg & (_blend_dist(work.astype(np.float32), bg.astype(np.float32), col) < 40)
            n_, lab_, _, _ = _cc(ink)                      # a piece of that colour that goes on well outside the box
            ids = np.unique(lab_[box & ink])               # is a curve passing the label: it stays
            ids = ids[ids > 0]
            inside = np.bincount(lab_[box & ink], minlength=n_)
            total = np.bincount(lab_.ravel(), minlength=n_)
            keep_ = list(ids[inside[ids] < 0.85 * total[ids]])
            qq = np.asarray(q, float)                      # across the text box (turned with the text): pieces that
            vax = qq[3] - qq[0]                            # are flat and lie in its top or bottom margin are a curve
            hq = float(np.linalg.norm(vax))                # passing under or over the word, not letters
            if hq > 0:
                vax /= hq
                for i_ in ids:
                    if i_ in keep_:
                        continue
                    py, px_ = np.nonzero(lab_ == i_)
                    v = ((np.c_[px_, py] - qq[0]) @ vax) / hq
                    if np.ptp(v) <= 0.25 and not 0.18 <= 0.5 * (v.min() + v.max()) <= 0.82:
                        keep_.append(i_)
            kill = box & ink & ~np.isin(lab_, keep_)
            if kill.any():
                ky, kx = np.nonzero(kill)
                tight[-1] = (int(kx.min()), int(ky.min()), int(kx.max()), int(ky.max()))
                inner |= cv2.dilate(kill.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # crumbs next to the letters
            work[kill] = bg
            fg[kill] = False
        n_, lab_, _, _ = _cc(fg)                           # small pieces of any colour left inside a text box
        total = np.bincount(lab_.ravel(), minlength=n_)
        inside = np.bincount(lab_[inner > 0], minlength=n_)
        gone = (inside >= total) & (np.arange(n_) > 0)
        work[gone[lab_]] = bg
        fg[gone[lab_]] = False
        k = erase_legend_samples(work, fg, bg, [tight[i] for i in legend_like(rects) if tight[i] is not None])
        log.append(f"text inside the plot removed ({len(texts)} labels, {k} legend samples)")
    k = erase_patches(work, fg, bg)
    if k:
        log.append(f"{k} filled boxes or patches ignored (label backgrounds, shading)")
    k = erase_fills(work, fg, bg)
    if k:
        log.append(f"{k} shaded area(s) behind the curves ignored")
    if have_frame:
        k = erase_insets(img, work, fg, bg, (x0, y0), frame["box"], found_text or [])
        if k:
            log.append(f"{k} inset plot(s) inside the plot ignored (digitize an inset by selecting it alone)")
    work[~fg] = bg

    progress(50, "Separating colours")
    clusters, nmix = drop_blends(color_clusters(work, fg, bg), np.asarray(bg, float))
    if nmix:
        log.append(f"{nmix} mixed colours ignored (where curves of two colours touch)")
    series = []
    diag, mind = math.hypot(hh, ww), min(hh, ww)
    for k, (mask, rgb, wgt) in enumerate(clusters):
        progress(55 + int(40 * k / max(1, len(clusters))), f"Tracing colour {k + 1} of {len(clusters)}")
        mask, wgt = fill_covered(mask, wgt, fg)
        for s in analyse_cluster(mask, wgt, rgb, diag, mind, log, fg & ~mask):
            s.px = s.px + (x0, y0)
            if s.px_alt is not None:
                s.px_alt = s.px_alt + (x0, y0)
            series.append(s)
    series = dedupe_scatter(series, 2.5)
    if have_frame:                                         # the inner rim of a thick or blurred frame line, traced
        d_rim = max(3.0, 1.5 * max(frame["t"]))            # as a broken line along the frame: not data
        def on_rim(p):
            e = np.minimum.reduce([np.abs(p[:, 0] - x0), np.abs(p[:, 0] - x1), np.abs(p[:, 1] - y0), np.abs(p[:, 1] - y1)])
            return (e <= d_rim).mean() >= 0.85
        nr = sum(1 for q in series if q.kind == "line" and on_rim(q.px))
        series = [q for q in series if not (q.kind == "line" and on_rim(q.px))]
        if nr:
            log.append(f"{nr} traces along the frame ignored")
    nj = join_hidden(series, fg, (x0, y0), diag, bg)
    if nj:
        log.append(f"{nj} curves joined across parts hidden by other curves")
    if have_frame:                                         # a curve that ends on the frame and then follows it: the
        nt = 0                                             # stretch along the frame is the frame, not the curve
        d_on = max(1.5, 0.6 * max(frame["t"]))
        for q in series:
            if q.kind != "line" or len(q.px) < 10:
                continue
            p = q.px
            e = np.minimum.reduce([np.abs(p[:, 0] - x0), np.abs(p[:, 0] - x1), np.abs(p[:, 1] - y0), np.abs(p[:, 1] - y1)])
            on = e <= d_on                                 # on the frame line itself, not just close to it (a steep
            lo, hi = 0, len(p)                             # curve leaving an axis, data hugging the bottom axis)
            while lo < hi and on[lo]:
                lo += 1
            while hi > lo and on[hi - 1]:
                hi -= 1
            lim = max(6.0 * d_rim, 0.03 * diag)
            cut = (lo > 0 and _plen(p[:lo + 1]) > lim) or (hi < len(p) and _plen(p[hi - 1:]) > lim)
            if cut and hi - lo >= 2:
                q.px = p[max(0, lo - 1) if _plen(p[:lo + 1]) <= lim else lo:
                         min(len(p), hi + 1) if _plen(p[hi - 1:]) <= lim else hi]
                nt += 1
        if nt:
            log.append(f"{nt} curve end(s) running along the frame cut off")
    flag_guides(series, (xl, yt, xr, yb))
    name_series(series)
    progress(100, "Done")
    return Result(img, (xl, yt, xr, yb), xax, yax, series, "raster", log, yax2)


def _same_ink(c1, c2, bg):
    """Two traced colours that can belong to one curve: the same colour, or the same hue at a similar
    strength (a curve drawn semi-transparent comes out darker where it lies over another curve)."""
    if c1 == c2:
        return True
    if bg is None:
        return False
    n1, n2 = (float(np.linalg.norm(np.array(c, float) - bg)) for c in (c1, c2))
    if min(n1, n2) < 40 or min(n1, n2) < 0.6 * max(n1, n2):
        return False
    (h1, s1, _), (h2, s2, _) = (colorsys.rgb_to_hsv(*(np.array(c) / 255.0)) for c in (c1, c2))
    if s1 < 0.15 and s2 < 0.15:                            # both grey or black
        return True
    dh = abs(h1 - h2) % 1.0
    return s1 >= 0.15 and s2 >= 0.15 and min(dh, 1 - dh) * 360 < 15


def join_hidden(series, ink, origin, diag, bg=None):
    """Join two pieces of one curve that is interrupted where other curves cover it: same colour (or the
    same hue, see _same_ink), both solid or both broken, the two ends point at each other, and the way
    between them lies on ink (or is very short). The gap is filled with a smooth bridge. Returns the
    number of joins."""
    ox, oy = origin
    bg = None if bg is None else np.asarray(bg, float)
    cover = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hh, ww = cover.shape
    reach = max(12.0, 0.03 * diag)
    tol = max(3.0, 0.007 * diag)                           # "on top of each other", grows with the picture

    def out_dir(px, end):                                  # unit vector leaving the curve at this end
        p = px if end == 1 else px[::-1]
        d = np.linalg.norm(p - p[-1], axis=1)
        far = np.nonzero(d >= reach)[0]
        q = p[far[-1]] if len(far) else p[0]
        v = p[-1] - q
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    def on_ink(pts, mask=None):
        mask = cover if mask is None else mask
        q = np.round(pts - (ox, oy)).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < ww) & (q[:, 1] >= 0) & (q[:, 1] < hh)
        return ok.sum() >= 0.9 * len(q) and mask[q[ok, 1], q[ok, 0]].mean() >= 0.8

    def others(a):                                         # traces of curves of another colour than a
        m = np.zeros((hh, ww), np.uint8)
        for t in series:
            if t.kind == "line" and not _same_ink(a.color, t.color, bg):
                q = np.round(t.px - (ox, oy)).astype(np.int32)
                cv2.polylines(m, [q.reshape(-1, 1, 2)], False, 1, 9)
        return m > 0

    def upright(p):                                        # a straight piece running up or down (x extent under 5 %
        return is_straight(p) and np.ptp(p[:, 0]) < 0.05 * _plen(p)   # of its length): a guide, never part of a curve

    joins = 0
    while True:
        lines = [s for s in series if s.kind == "line" and len(s.px) >= 8 and not upright(s.px)]
        cand = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if not _same_ink(a.color, b.color, bg):
                    continue
                if a.color == b.color and (a.style == "solid") != (b.style == "solid"):   # e.g. n solid, k dashed
                    continue
                lo, hi = max(a.px[:, 0].min(), b.px[:, 0].min()), min(a.px[:, 0].max(), b.px[:, 0].max())
                side = (hi - lo) <= 0.2 * max(1.0, min(np.ptp(a.px[:, 0]), np.ptp(b.px[:, 0])))
                for (p, q, ip, iq) in ((a, b, i, j), (b, a, j, i)):   # overlap: p ends on q, q goes on beyond p
                    tree = cKDTree(p.px)
                    for ep in (0, 1):
                        pe = p.px[-1 if ep else 0]
                        dq = np.linalg.norm(q.px - pe, axis=1)
                        k = int(dq.argmin())
                        if dq[k] > tol:
                            continue
                        sides = (q.px[:k][::-1], q.px[k + 1:])     # both leave point k
                        near = [len(sd) >= 3 and (tree.query(sd)[0] <= tol).mean() >= 0.9 for sd in sides]
                        tp = out_dir(p.px, ep)
                        for sd, other_near in ((sides[0], near[1]), (sides[1], near[0])):
                            if other_near and len(sd) >= 5 and tree.query(sd[-1])[0] > tol + 2 and tp is not None:
                                w = sd[min(len(sd) - 1, int(reach))] - pe          # the curve must go on, not
                                if np.linalg.norm(w) > 0 and _angle(tp, w / np.linalg.norm(w)) < 60:   # turn back
                                    cand.append((0.0, ip, iq, ep, None, "overlap", sd))
                if a.color != b.color and not side:                 # two curves of a similar colour side by side
                    continue
                twin = a.style != "solid" and b.style != "solid"
                for g in series:                       # under: both ends disappear under one curve of another
                    if g.kind != "line" or len(g.px) < 8 or (_same_ink(a.color, g.color, bg) and not (   # colour (or
                            twin and g.style == "solid" and g.color == a.color)):   # the solid twin of a dashed curve)
                        continue                                   # and the hidden part follows that curve
                    gt = cKDTree(g.px)
                    for ea in (0, 1):
                        for eb in (0, 1):
                            pa, pb = a.px[-1 if ea else 0], b.px[-1 if eb else 0]
                            (da, ka), (db, kb) = gt.query(pa), gt.query(pb)
                            tw = g.color == a.color                 # along the solid twin: the dashes merge into it
                            if max(da, db) > (3 if tw else 2) * tol or abs(ka - kb) < 3:   # over a stretch that may
                                continue                                                   # bend (a peak)
                            path = g.px[ka:kb + 1] if ka < kb else g.px[kb:ka + 1][::-1]
                            if tw:                                  # only the stretch between the two ends
                                lo_, hi_ = sorted((pa[0], pb[0]))
                                path = path[(path[:, 0] > lo_) & (path[:, 0] < hi_)]
                                if len(path) < 4:
                                    continue
                            ln = _plen(path)
                            ta, tb = out_dir(a.px, ea), out_dir(b.px, eb)
                            if ln > 0.35 * diag or ta is None or tb is None:
                                continue
                            ga = path[min(len(path) - 1, 3)] - path[0]
                            gb = path[max(0, len(path) - 4)] - path[-1]
                            if min(np.linalg.norm(ga), np.linalg.norm(gb)) < 1e-6:
                                continue
                            amax = 30 if tw else 45
                            if _angle(ta, ga / np.linalg.norm(ga)) > amax or _angle(tb, gb / np.linalg.norm(gb)) > amax:
                                continue
                            ch = pb - pa                                    # and the way as a whole goes on
                            if not tw and (np.linalg.norm(ch) < 1e-6 or _angle(ta, ch / np.linalg.norm(ch)) > 60
                                           or _angle(tb, -ch / np.linalg.norm(ch)) > 60):
                                continue
                            cand.append((ln, i, j, ea, eb, "under", path[1:-1]))
                if twin and a.color == b.color:            # two dashed pieces of one colour, one ending where the
                    for g in series:                       # other starts further on in x, and the solid twin passes
                        if g.kind != "line" or g.style != "solid" or g.color != a.color or len(g.px) < 8:   # close
                            continue                       # to both ends: the dashes ran on top of the solid curve
                        gx = np.argsort(g.px[:, 0], kind="stable")
                        gp = g.px[gx]
                        for (p, q, ip, iq) in ((a, b, i, j), (b, a, j, i)):
                            pe, qs = p.px[-1], q.px[0]     # traces run left to right
                            if not (pe[0] < qs[0] <= pe[0] + 0.12 * diag) or q.px[:, 0].min() < pe[0] - tol:
                                continue
                            if gp[0, 0] > pe[0] or gp[-1, 0] < qs[0]:
                                continue
                            yg = lambda x: np.interp(x, gp[:, 0], gp[:, 1])

                            def meets(tr, at, back):       # offset from the twin at the end; an end further off
                                o = tr[at, 1] - yg(tr[at, 0])   # must be closing in on it (offset smaller there
                                if abs(o) <= 4 * tol:       # than one reach back along the trace)
                                    return o
                                d_ = np.abs(tr[:, 0] - (tr[at, 0] + back))
                                k_ = int(d_.argmin())
                                if (abs(o) > 0.06 * diag or d_[k_] > 2.0 or gp[0, 0] > tr[k_, 0] or gp[-1, 0] < tr[k_, 0]
                                        or abs(tr[k_, 1] - yg(tr[k_, 0])) < abs(o) + tol):
                                    return None
                                return o
                            oa, ob = meets(p.px, -1, -reach), meets(q.px, 0, reach)
                            if oa is None or ob is None:
                                continue
                            path = gp[(gp[:, 0] > pe[0]) & (gp[:, 0] < qs[0])].copy()
                            if len(path) < 2:
                                continue
                            h_ = 0.5 * (qs[0] - pe[0])     # the offsets fade out over half of the gap each
                            path[:, 1] += (oa * np.clip(1 - (path[:, 0] - pe[0]) / h_, 0, 1)
                                           + ob * np.clip(1 - (qs[0] - path[:, 0]) / h_, 0, 1))
                            cand.append((qs[0] - pe[0], ip, iq, 1, 0, "twin", path))
                for ea in (0, 1):
                    for eb in (0, 1):
                        pa, pb = a.px[-1 if ea else 0], b.px[-1 if eb else 0]
                        v = pb - pa
                        d = float(np.linalg.norm(v))
                        if d > 0.35 * diag:
                            continue
                        ta, tb = out_dir(a.px, ea), out_dir(b.px, eb)
                        if ta is None or tb is None:
                            continue
                        if d <= 2.0:                       # the two pieces end in the same point (a curve cut at
                            xm = 0.5 * (pa[0] + pb[0])     # a junction, also at a sharp peak): join when they lie
                            if side and (np.median(a.px[:, 0]) - xm) * (np.median(b.px[:, 0]) - xm) < 0:   # on the
                                cand.append((d, i, j, ea, eb, "touch", np.empty((0, 2))))   # two sides of it in x
                                continue
                        if d < 1e-6:
                            continue
                        a1, a2 = _angle(ta, v / d), _angle(tb, -v / d)
                        near = d <= max(15.0, 0.03 * diag)
                        xa_, xb_ = (a.px[:, 0].max(), b.px[:, 0].min()) if pa[0] < pb[0] else (b.px[:, 0].max(), a.px[:, 0].min())
                        if (a.color == b.color and side and max(a1, a2) > 40.0 and near and abs(v[0]) >= 1.0
                                and abs(pa[0] - xa_ if pa[0] < pb[0] else pb[0] - xa_) <= 1.0
                                and abs(pb[0] - xb_ if pa[0] < pb[0] else pa[0] - xb_) <= 1.0):
                            # facing ends a few pixels apart in x (a turn hidden under other ink, where the end
                            # directions say little): the straight way between them must lie on ink
                            t = np.linspace(0, 1, int(d) + 2)[1:-1, None]
                            if on_ink(pa + t * v):
                                cand.append((1.5 * d, i, j, ea, eb, "xgap", pa + t * v))
                                continue
                        if d <= 2 * tol and side and max(a1, a2) > 40.0:   # ends touching: join as they are
                            if _angle(ta, -tb) < 100:                       # (a bend, not a U-turn)
                                cand.append((2 * d, i, j, ea, eb, "touch", np.empty((0, 2))))
                            continue
                        if max(a1, a2) > (40.0 if near else 30.0):
                            if min(a1, a2) <= 15.0 and d <= 0.05 * diag and side:
                                # one end points straight at the other, which stops just after a dip or
                                # peak hidden under another curve (its own direction is no guide there)
                                t = np.linspace(0, 1, int(d) + 2)[1:-1, None]
                                if on_ink(pa + t * v):
                                    cand.append((1.5 * d, i, j, ea, eb, "turn", pa + t * v))
                                    continue
                            # a corner hidden under other ink: the two ends point at one point close by
                            if d > 0.08 * diag or not side:
                                continue
                            m = np.c_[ta, -tb]
                            if abs(np.linalg.det(m)) < 0.2:
                                continue
                            sa, sb = np.linalg.solve(m, v)
                            if not (0 < sa < 1.5 * d and 0 < sb < 1.5 * d):
                                continue
                            c = pa + sa * ta
                            tt = np.linspace(0, 1, int(sa + sb) + 3)[:, None]
                            curve = (1 - tt) ** 2 * pa + 2 * tt * (1 - tt) * c + tt ** 2 * pb
                            if not on_ink(curve[1:-1], others(a)):     # the corner must lie under a curve of
                                continue                                    # another colour
                            cand.append((1.5 * d, i, j, ea, eb, "corner", curve[1:-1]))
                            continue
                        if not near:                       # a long gap must be hidden under other ink
                            t = np.linspace(0, 1, int(d) + 2)[1:-1, None]
                            if not on_ink(pa + t * v):
                                continue
                        t = np.linspace(0, 1, max(3, int(d) + 1))[1:-1, None]
                        m0, m1 = ta * d, -tb * d           # cubic bridge that leaves and arrives along the curves
                        bridge = ((2 * t ** 3 - 3 * t ** 2 + 1) * pa + (t ** 3 - 2 * t ** 2 + t) * m0
                                  + (-2 * t ** 3 + 3 * t ** 2) * pb + (t ** 3 - t ** 2) * m1)
                        cand.append((d * (1 + (a1 + a2) / 30.0), i, j, ea, eb, "gap", bridge))
        if not cand:
            return joins
        c = min(cand, key=lambda c: c[0])
        a, b = lines[c[1]], lines[c[2]]
        pa_ = a.px if c[3] else a.px[::-1]                 # ... a ends here | b goes on from here ...
        if c[5] == "overlap":
            px = np.vstack([pa_, c[6]])
        else:
            pb_ = b.px[::-1] if c[4] else b.px
            px = np.vstack([pa_, c[6], pb_])
        nb = len(px) - len(pa_)
        a.px = px[::-1].copy() if px[0, 0] > px[-1, 0] else px
        a.style = a.style if len(pa_) >= nb or a.style == "solid" else b.style
        if nb > len(pa_):
            a.color = b.color
        if "joined" not in a.note:
            a.note = (a.note + "; joined across a part hidden by another curve").strip("; ")
        series[:] = [s for s in series if s is not b]
        joins += 1


def flag_guides(series, box=None):
    """Short straight lines in a plot that also has other series are usually guides (tangents, reference
    marks): they stay in the list but are not ticked. A straight line that spans the range of the other
    series is kept as data (an exponential in a semi-log plot, a linear fit), and so is a slanted straight
    line that runs from one edge of the frame to another (a power law cut off by the frame, as the f^-n
    components of a noise spectrum). A vertical line across the frame is a reference mark."""
    def nearly_straight(p):                                # 90 % of the points on one line
        c = p - p.mean(axis=0)
        v = np.linalg.svd(c, full_matrices=False)[2][0]
        dev = np.abs(c[:, 0] * v[1] - c[:, 1] * v[0])
        return np.percentile(dev, 90) < max(2.0, 0.012 * float(np.linalg.norm(p[-1] - p[0])))
    straight = [s for s in series if s.kind == "line" and (is_straight(s.px) or nearly_straight(s.px))]
    others = [s for s in series if not any(s is t for t in straight)]
    if not others:
        return
    span = max(float(np.ptp(s.px[:, 0])) for s in others)

    def edge_to_edge(p):
        if box is None:
            return False
        x0, y0, x1, y1 = box
        tol = max(3.0, 0.02 * max(x1 - x0, y1 - y0))
        on = lambda q: min(abs(q[0] - x0), abs(q[0] - x1), abs(q[1] - y0), abs(q[1] - y1)) <= tol
        return on(p[0]) and on(p[-1]) and np.ptp(p[:, 0]) >= 0.05 * (x1 - x0)
    for s in straight:
        if np.ptp(s.px[:, 0]) < 0.6 * span and not edge_to_edge(s.px):
            s.use = False
            s.note = ("not ticked: short straight line (guide?); " + s.note).strip("; ")
    size = lambda s: float(np.linalg.norm(s.px.max(axis=0) - s.px.min(axis=0)))
    big = sorted(size(s) for s in others if s.kind == "line")
    if len(big) >= 3:                                      # a small piece beside long curves: an ellipse or arrow
        for s in others:                                   # drawn around them, or a crumb of a hidden curve
            if s.kind == "line" and s.use and size(s) < 0.2 * big[-2]:
                s.use = False
                s.note = ("not ticked: small piece (annotation?); " + s.note).strip("; ")
    line = None                                            # markers follow the line they sit on: markers of a
    for s in series:                                       # line that is not data (an arrow) are not data either
        if s.kind == "line":
            line = s
        elif "markers of the line above" in s.note and line is not None and not line.use and s.use:
            s.use = False
            s.note = "not ticked: markers of a line that is not ticked"


def guided_trace(img, box, clicks, style="auto"):
    """Trace one curve through points clicked on it by the user (for crowded plots where the automatic tracing
    mixes curves). Between each two clicks the cheapest path through the picture is taken, with costs per pixel:
    ink of the curve's own colour is cheap, ink of other colours costs more (the curve may pass under it), white
    background costs most. For a broken (dashed, dash-dot) curve the background is cheaper (its gaps) and the solid
    ink of the same colour dearer (its solid twin), the reverse for a solid curve. style: 'auto' (decided from the
    ink under the clicks), 'solid' or 'broken'. Returns (Series or None, message)."""
    clicks = [(float(x), float(y)) for x, y in clicks]
    if len(clicks) < 2:
        return None, "Click at least two points along the curve."
    img = np.ascontiguousarray(img[:, :, :3])
    H, W = img.shape[:2]
    xl, yt, xr, yb = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, xl + 2), max(0, yt + 2), min(W - 1, xr - 2), min(H - 1, yb - 2)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, "The plot frame is too small."
    work = img[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    hh, ww = work.shape[:2]
    bg = dominant_color(work.astype(np.uint8)).astype(np.float32)
    strength = np.abs(work - bg).max(axis=2)
    ink = strength > 40
    pts = [(min(max(x - x0, 0.0), ww - 1.0), min(max(y - y0, 0.0), hh - 1.0)) for x, y in clicks]
    # the curve's colour: the most common ink colour within a few pixels of the clicks (a click may land where
    # another curve crosses it, so all clicks vote)
    cand = []
    for x, y in pts:
        xa, xb = int(max(0, x - 4)), int(min(ww, x + 5))
        ya, yb_ = int(max(0, y - 4)), int(min(hh, y + 5))
        p = work[ya:yb_, xa:xb][strength[ya:yb_, xa:xb] > 0.6 * strength.max()]
        if len(p):
            cand.append(p)
    if not cand:
        return None, "No ink of a curve near the clicked points: click on the curve itself."
    cand = np.vstack(cand)
    q = np.round(cand / 48.0).astype(int)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    col = np.median(cand[inv.ravel() == int(cnt.argmax())], axis=0)
    dist = _blend_dist(work, bg, col)
    t = np.clip(((work - bg) @ (col - bg)) / max(float((col - bg) @ (col - bg)), 1e-6), 0, 1)
    own_w = np.where(dist < 60, t, 0.0).astype(np.float32)        # 0..1: how much of this pixel is the curve
    own = own_w > 0.35                                             # (loose: JPEG noise breaks thin lines up)
    # long (solid) and short (dashes) pieces of the own colour; breaks of 1-2 px (compression) closed first
    ownc = cv2.morphologyEx(own.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    n, lab, st, _ = _cc(ownc)
    ext = np.maximum(st[:, 2], st[:, 3]).astype(float)
    sk = thin(ownc.astype(np.uint8)) > 0
    dt = cv2.distanceTransform(ownc.astype(np.uint8), cv2.DIST_L2, 3)
    lw = max(1.0, 2.0 * float(np.median(dt[sk])) - 1.0) if sk.any() else 2.0
    t_s = max(4.0 * lw + 6.0, 0.06 * min(hh, ww))
    short = (ext <= t_s) & (np.arange(n) > 0)
    shortm = short[lab] & ownc
    longm = ownc & ~shortm
    own = ownc
    other = ink & ~own
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    def costs(kind):
        if kind == "broken":
            c = np.full((hh, ww), 12.0, np.float32)                # background: the gaps of the curve; along the
            c[other] = 9.0                                         # dashes about 4 per pixel, so another curve
            c[longm] = 9.0                                         # or the solid twin costs more to follow
            c[shortm] = 1.0
        else:
            c = np.full((hh, ww), 40.0, np.float32)
            c[other] = 4.0
            c[shortm] = 5.0                                        # dashes of the same colour
            c[longm] = 1.0
        return c

    def route(cost):                                               # cheapest way through all clicked points
        path, pad = [], int(max(10, 0.04 * ww))
        for (xa_, ya_), (xb_, yb2) in zip(pts[:-1], pts[1:]):
            c0, c1 = int(max(0, min(xa_, xb_) - pad)), int(min(ww - 1, max(xa_, xb_) + pad))
            sub = cost[:, c0:c1 + 1]
            h_, w_ = sub.shape
            idx = np.arange(h_ * w_).reshape(h_, w_)
            rows, cols, wts = [], [], []
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if dx >= 0:
                    a, b = idx[0:h_ - dy, 0:w_ - dx], idx[dy:h_, dx:w_]
                else:
                    a, b = idx[0:h_ - dy, 1:w_], idx[dy:h_, 0:w_ - 1]
                wt = 0.5 * (sub.ravel()[a.ravel()] + sub.ravel()[b.ravel()]) * (math.sqrt(2.0) if dx and dy else 1.0)
                rows += [a.ravel(), b.ravel()]
                cols += [b.ravel(), a.ravel()]
                wts += [wt, wt]
            g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(h_ * w_, h_ * w_)).tocsr()
            s_ = int(round(ya_)) * w_ + int(round(xa_)) - c0
            e_ = int(round(yb2)) * w_ + int(round(xb_)) - c0
            _, pred = dijkstra(g, indices=s_, return_predecessors=True)
            if pred[e_] < 0 and e_ != s_:
                return None
            seg, k = [], e_
            while k != s_ and k >= 0:
                seg.append((k % w_ + c0, k // w_))
                k = pred[k]
            seg.append((s_ % w_ + c0, s_ // w_))
            path += seg[::-1] if not path else seg[::-1][1:]
        return path

    if style == "auto":                                            # a solid curve traced as solid runs on long
        path = route(costs("solid"))                               # ink of its colour almost all the way
        if path is not None:
            q = np.array(path)
            style = "solid" if longm[q[:, 1], q[:, 0]].mean() >= 0.7 else "broken"
        if style == "broken":
            path = route(costs("broken"))
    else:
        path = route(costs(style))
    if path is None:
        return None, "No path found between the clicked points."
    p = np.array(path, float)
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
    p = p[keep]
    if len(p) < 3:
        return None, "The clicked points are too close together."
    px = refine_and_fill(p, own_w, lw, [])
    if px[0, 0] > px[-1, 0]:
        px = px[::-1]
    name = "solid"
    if style == "broken":                                          # dots among the dashes on the way: dash-dot
        ids = np.unique(lab[np.round(p[:, 1]).astype(int), np.round(p[:, 0]).astype(int)])
        e_ = ext[[i for i in ids if i > 0 and short[i]]]
        name = "dash-dot" if len(e_) >= 4 and (e_ <= 2 * lw + 1).sum() >= 2 and (e_ >= 3 * lw).sum() >= 2 else "dashed"
    rgb = tuple(int(round(v)) for v in col)
    s = Series("line", px + (x0, y0), rgb, name, note="guided trace (%d points clicked)" % len(clicks))
    return s, "Traced a %s curve of colour rgb%s through %d clicked points." % ("solid" if style == "solid" else "broken", rgb, len(clicks))


def name_series(series):
    for i, s in enumerate(series):
        s.name = f"S{i + 1}"


def _monotone_path(cost, xa, ya, xb, yb):
    """Cheapest 8-connected path from (xa, ya) to (xb, yb) through the cost map that never steps back in x
    (it may run up or down within a column, so steep parts are followed). Returns (N, 2) int x, y."""
    flip = xb < xa
    if flip:
        cost = cost[:, ::-1]
        xa, xb = cost.shape[1] - 1 - xa, cost.shape[1] - 1 - xb
    h = cost.shape[0]
    rows = np.arange(h)
    if xa == xb:
        ys = np.arange(ya, yb + 1) if yb >= ya else np.arange(ya, yb - 1, -1)
        p = np.c_[np.full(len(ys), xa), ys]
    else:
        inf = np.float64(1e18)
        sq2 = math.sqrt(2.0)
        D = np.full(h, inf)
        D[ya] = cost[ya, xa]
        vk_all, ep_all = [], []
        for x in range(xa + 1, xb + 1):
            c = cost[:, x].astype(np.float64)
            cand = np.vstack([np.r_[inf, D[:-1]] + sq2 * c, D + c, np.r_[D[1:], inf] + sq2 * c])
            k = cand.argmin(axis=0)
            E = cand[k, rows]
            ep = rows + (k - 1)                            # row in the previous column
            S = np.cumsum(c)
            m = E - S                                      # runs down the column: D[y] = S[y] + min(E[k] - S[k]), k <= y
            rm = np.minimum.accumulate(m)
            kd = np.maximum.accumulate(np.where(m <= rm, rows, 0))
            Dd = rm + S
            Sr = np.cumsum(c[::-1])[::-1]                   # runs up the column
            mu = (E - Sr)[::-1]
            rmu = np.minimum.accumulate(mu)
            ku = (h - 1 - np.maximum.accumulate(np.where(mu <= rmu, rows, 0)))[::-1]
            Du = rmu[::-1] + Sr
            up = Du < Dd
            D = np.where(up, Du, Dd)
            vk_all.append(np.where(up, ku, kd))
            ep_all.append(ep)
        pts, y = [], yb
        for i in range(len(vk_all) - 1, -1, -1):           # back from the end
            x = xa + 1 + i
            k = int(vk_all[i][y])
            step = 1 if k >= y else -1
            pts.extend((x, yy) for yy in range(y, k + step, step))
            y = int(ep_all[i][k])
        pts.append((xa, y))
        p = np.array(pts[::-1])
    if flip:
        p[:, 0] = cost.shape[1] - 1 - p[:, 0]
    return p


def guided_trace(img, clicks, box=None):
    """Guided trace: one curve through points the user clicked on it, in order along the curve (picture pixels).
    The colour is the one most clicks sit on. Between two clicks the trace is the cheapest path through a cost map
    (cheap on ink of that colour, dearer on ink of other colours where the curve may run under them, dearest on
    the background, more so further from the colour's ink), and it never steps back in x, so it can neither turn
    round nor loop: click at peaks, crossings and where two curves of the colour part. Returns a Series."""
    img = np.ascontiguousarray(img[:, :, :3])
    h, w = img.shape[:2]
    if box is None:
        xl, yt, xr, yb = 0, 0, w - 1, h - 1
    else:
        xl, yt, xr, yb = (int(round(v)) for v in box)
    m = 2 if box is not None else 0                        # stay inside the frame lines
    lo_x, hi_x, lo_y, hi_y = max(0, xl + m), min(w - 1, xr - m), max(0, yt + m), min(h - 1, yb - m)
    pts = []
    for x, y in clicks:
        q = (int(round(min(max(x, lo_x), hi_x))), int(round(min(max(y, lo_y), hi_y))))
        if not pts or abs(q[0] - pts[-1][0]) + abs(q[1] - pts[-1][1]) > 2:
            pts.append(q)
    if len(pts) < 2:
        raise ValueError("Click at least two points along the curve.")
    imf = img.astype(np.float32)
    bg = dominant_color(img[lo_y:hi_y + 1, lo_x:hi_x + 1]).astype(np.float32)
    dist_bg = np.abs(imf - bg).max(axis=2)
    ink = dist_bg > 30
    cores = []                                             # at each click: the strongest ink pixel close by
    for x, y in pts:
        r = 6
        ya, yb_, xa, xb = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        d = dist_bg[ya:yb_, xa:xb]
        if d.max() <= 30:
            continue
        yy, xx = np.nonzero(d >= d.max() - 10)
        k = int(np.argmin((yy + ya - y) ** 2 + (xx + xa - x) ** 2))
        cores.append(imf[yy[k] + ya, xx[k] + xa])
    if not cores:
        raise ValueError("No ink found near the clicked points.")
    cores = np.array(cores)
    votes = [(np.linalg.norm(cores - c, axis=1) < 60).sum() for c in cores]
    col = cores[np.linalg.norm(cores - cores[int(np.argmax(votes))], axis=1) < 60].mean(axis=0)
    t = np.clip(((imf - bg) @ (col - bg)) / max(float((col - bg) @ (col - bg)), 1e-6), 0, 1.2)
    weight = np.where((_blend_dist(imf, bg, col) < 45) & ink, np.clip(t, 0, 1), 0).astype(np.float32)
    own = weight >= 0.5
    snapped = []                                           # each click onto the nearest ink of the colour (6 px)
    for x, y in pts:
        ya, yb_, xa, xb = max(lo_y, y - 6), min(hi_y + 1, y + 7), max(lo_x, x - 6), min(hi_x + 1, x + 7)
        yy, xx = np.nonzero(own[ya:yb_, xa:xb])
        if len(yy):
            k = int(np.argmin((yy + ya - y) ** 2 + (xx + xa - x) ** 2))
            x, y = int(xx[k] + xa), int(yy[k] + ya)
        if not snapped or (x, y) != snapped[-1]:
            snapped.append((x, y))
    pts = snapped if len(snapped) >= 2 else pts
    n, lab, st, _ = _cc(own)                               # long strokes (a solid curve) and short pieces (dashes,
    smax = max(12.0, 0.09 * min(hi_x - lo_x, hi_y - lo_y)) # dots) of the colour; the clicks say which kind the
    longc = np.maximum(st[:, 2], st[:, 3]) > smax          # curve is (most of them sit on that kind)
    longc[0] = False
    stroke = longc[lab] & own
    piece = own & ~stroke
    on_short = [not longc[lab[y, x]] for x, y in pts if own[y, x]]
    dashed = len(on_short) >= 2 and sum(on_short) > 0.5 * len(on_short)
    near = piece if dashed else stroke                     # its own kind of ink, and the background close to it
    dt_near = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 3)
    if dashed:                                             # dashed: the gaps between its dashes are cheap, the solid
        bgc = np.where(dt_near <= 4, 1.5 + 1.0 * dt_near, 5.5 + 3.0 * (dt_near - 4))   # strokes of the colour dearer
        cost = np.where(piece, 1.0, np.where(stroke, 2.5, np.where(weight >= 0.15, 1.5, np.where(ink, 4.0, bgc))))
    else:                                                  # solid: dashes of the colour cost like other ink
        bgc = 2.0 + 1.5 * dt_near
        cost = np.where(stroke, 1.0, np.where(piece, 2.5, np.where(weight >= 0.15, 1.5, np.where(ink, 4.0, bgc))))
    cost = np.minimum(cost, 40.0).astype(np.float32)
    cost[:lo_y] = cost[hi_y + 1:] = 1e6                   # outside the plot area
    path = []
    for (xa, ya), (xb, yb_) in zip(pts[:-1], pts[1:]):
        p = _monotone_path(cost, xa, ya, xb, yb_)
        path.append(p if not path else p[1:])
    path = np.vstack(path).astype(np.float64)
    on = own[path[:, 1].astype(int), path[:, 0].astype(int)]
    d_on = cv2.distanceTransform(own.astype(np.uint8), cv2.DIST_L2, 3)[path[:, 1].astype(int), path[:, 0].astype(int)]
    lw = max(1.0, 2.0 * float(np.median(d_on[on])) - 1.0) if on.any() else 1.0
    px = refine_and_fill(path, weight, lw, [])
    share = float(on.mean())
    style = "solid" if share >= 0.85 else "dashed" if share >= 0.45 else "dotted"
    rgb = tuple(int(round(v)) for v in col)
    return Series("line", px, rgb, style, note="guided trace through %d clicked points (%.0f %% on its own ink)"
                  % (len(pts), 100 * share))


def series_data(res, s, marker_ref="centroid"):
    """Data coordinates (N, 2) of a series, or None while an axis is not calibrated."""
    if res.xaxis is None or res.yaxis is None:
        return None
    px = s.px_alt if (s.kind == "scatter" and marker_ref == "bbox" and s.px_alt is not None) else s.px
    yax = res.yaxis2 if (s.yref == "right" and res.yaxis2 is not None) else res.yaxis
    return np.c_[res.xaxis.to_data(px[:, 0]), yax.to_data(px[:, 1])]


# ----------------------------------------------------------------------------------------------
# Vector graphics: PDF pages (EPS / PS are converted to PDF with Ghostscript first)
# ----------------------------------------------------------------------------------------------


def ghostscript_exe():
    for name in ("gs", "gswin64c", "gswin32c"):
        p = shutil.which(name)
        if p:
            return p
    for pat in (r"C:\Program Files\gs\*\bin\gswin64c.exe", r"C:\Program Files (x86)\gs\*\bin\gswin32c.exe"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def ps_to_pdf(path):
    gs = ghostscript_exe()
    if gs is None:
        raise RuntimeError("Ghostscript is needed to read .eps / .ps files and was not found.\n"
                           "Install it from https://ghostscript.com/releases/gsdnld.html and try again.")
    out = os.path.join(tempfile.mkdtemp(prefix="xydig_"), "converted.pdf")
    args = [gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", f"-sOutputFile={out}"]
    if path.lower().endswith((".eps", ".epsf", ".epsi")):
        args.append("-dEPSCrop")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run(args + [path], capture_output=True, text=True, creationflags=flags)
    if r.returncode or not os.path.exists(out):
        raise RuntimeError("Ghostscript could not convert the file:\n" + (r.stderr or r.stdout)[-600:])
    return out


def _overlaps(r, clip, m=1.0):
    return r.x1 >= clip.x0 - m and r.x0 <= clip.x1 + m and r.y1 >= clip.y0 - m and r.y0 <= clip.y1 + m


def region_info(page, clip, drawings):
    """Is the chosen region drawn with vector paths, or is it an embedded picture?"""
    import pymupdf
    hit = [d for d in drawings if _overlaps(d["rect"], clip)]
    items = sum(len(d["items"]) for d in hit)
    area, ppi = 0.0, 0.0
    for im in page.get_image_info():
        bb = pymupdf.Rect(im["bbox"])
        r = bb & clip
        if not r.is_empty:
            area += r.get_area()
            ppi = max(ppi, 72.0 * im["width"] / max(bb.width, 1e-6))
    frac = area / max(clip.get_area(), 1e-6)
    # a picture under the region (shaded bands, symbols or a photo stored as an image layer of a vector figure)
    # does not make it a picture when many stroked lines are drawn over it: the curves are vector paths then
    stroked = sum(len(d["items"]) for d in hit if d.get("color") is not None)
    return dict(vector=items >= 12 and (frac < 0.5 or stroked >= 60), paths=len(hit), items=items,
                stroked=stroked, image_fraction=frac, image_ppi=ppi)


def _gap(a, b):
    """Distance between two rectangles (0 when they touch or overlap)."""
    return max(a.x0 - b.x1, b.x0 - a.x1, 0.0) + max(a.y0 - b.y1, b.y0 - a.y1, 0.0)


def _edge_gap(l, f):
    """Distance of a label to the outline of a frame (also from inside: the labels of an inset lie inside
    the frame of the plot around it, but close to the inset's own outline)."""
    if l.x0 >= f.x0 and l.x1 <= f.x1 and l.y0 >= f.y0 and l.y1 <= f.y1:
        return min(l.x0 - f.x0, f.x1 - l.x1, l.y0 - f.y0, f.y1 - l.y1)
    return _gap(l, f)


def _page_segments(drawings, region):
    """Solid, dark, stroked axis-aligned segments in a region of the page (page coordinates):
    horizontal [(y, x0, x1)] and vertical [(x, y0, y1)]. Grey or dashed lines (grids) are left out."""
    hs, vs = [], []
    for d in drawings:
        col = d.get("color")
        if col is None or sum(col[:3]) > 1.5 or (d.get("dashes") or "[] 0").strip() not in ("[] 0", "[] 0.0", ""):
            continue
        if max(col[:3]) - min(col[:3]) > 0.25:             # coloured: a curve, not an axis line
            continue
        if not _overlaps(d["rect"], region):
            continue
        for it in d["items"]:
            if it[0] == "l":
                pairs = [(it[1], it[2])]
            elif it[0] == "re":
                r = it[1]
                pairs = [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
            elif it[0] == "qu":
                q = it[1]
                pairs = [(q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)]
            else:
                continue
            for a, b in pairs:
                if abs(a.y - b.y) < 0.3 and abs(a.x - b.x) > 0.5:
                    hs.append((0.5 * (a.y + b.y), min(a.x, b.x), max(a.x, b.x)))
                elif abs(a.x - b.x) < 0.3 and abs(a.y - b.y) > 0.5:
                    vs.append((0.5 * (a.x + b.x), min(a.y, b.y), max(a.y, b.y)))
    return hs, vs


def panel_frames(drawings, region):
    """Axes frames of the plots drawn in a region of a PDF page (page coordinates), for figures made of
    several panels (stacked plots, insets). A frame is a left and a bottom line meeting at a corner,
    each carrying at least two tick marks; its top is the next line above with the same width.
    Returns [Rect]."""
    import pymupdf
    hs, vs = _page_segments(drawings, region)
    H = [h for h in _merge_collinear(hs) if h[2] - h[1] >= 40]
    V = [v for v in _merge_collinear(vs) if v[2] - v[1] >= 30]
    th = [h for h in hs if h[2] - h[1] < 15]               # tick marks: short strokes
    tv = [v for v in vs if v[2] - v[1] < 15]
    tol, frames = 2.0, []
    for y, hx0, hx1 in H:                                  # bottom line
        for x, vy0, vy1 in V:                              # left line through its left end
            if abs(x - hx0) > tol or not (vy0 - tol < y < vy1 + tol) or y - vy0 < 30:
                continue
            above = [h for h in H if h[0] < y - 25 and h[0] > vy0 - tol and abs(h[1] - hx0) < tol and abs(h[2] - hx1) < tol]
            top = max(h[0] for h in above) if above else vy0
            w_, h_ = hx1 - hx0, y - top
            nb = sum(1 for vx, a, b in tv if hx0 - 1 <= vx <= hx1 + 1 and min(abs(a - y), abs(b - y)) < 1.5 and b - a < 0.15 * h_)
            nl = sum(1 for hy, a, b in th if top - 1 <= hy <= y + 1 and min(abs(a - x), abs(b - x)) < 1.5 and b - a < 0.15 * w_)
            if nb >= 2 and nl >= 2:
                f = pymupdf.Rect(hx0, top, hx1, y)
                if not any(abs(g.x0 - f.x0) < tol and abs(g.y0 - f.y0) < tol and abs(g.x1 - f.x1) < tol and abs(g.y1 - f.y1) < tol for g in frames):
                    frames.append(f)
    return frames


def merged_pictures(page):
    """Embedded pictures of a page. A figure is often stored as several strips (or tiles) that touch each
    other; those are joined into one rectangle. Returns [Rect] of at least 50 x 50 pt."""
    import pymupdf
    rects = [pymupdf.Rect(im["bbox"]) for im in page.get_image_info()]
    rects = [r for r in rects if r.width >= 5 and r.height >= 5]
    joined = True
    while joined:
        joined = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                over_y = b.y0 <= a.y1 + 2 and a.y0 <= b.y1 + 2
                over_x = b.x0 <= a.x1 + 2 and a.x0 <= b.x1 + 2
                strips = abs(a.x0 - b.x0) <= 2 and abs(a.x1 - b.x1) <= 2 and over_y
                tiles = abs(a.y0 - b.y0) <= 2 and abs(a.y1 - b.y1) <= 2 and over_x
                if strips or tiles:
                    rects[i] = a | b
                    del rects[j]
                    joined = True
                    break
            if joined:
                break
    return [r for r in rects if r.width >= 50 and r.height >= 50]


def picture_frames(img, insets=False, scan=False):
    """Axes frames in a picture: a dark bottom line and a dark left line meeting at a corner; the top is
    the next line above with the same width. Frames inside another frame (insets, legend boxes) are left
    out, or, with insets=True, only those are returned. Returns [(x0, y0, x1, y1)] in pixels."""
    g = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY)
    dark = (g < 170).astype(np.uint8)                      # thin axis lines are grey after anti-aliasing
    hh, ww = dark.shape
    k = max(25, int((0.06 if scan else 0.12) * min(hh, ww)))   # scanned page: short (small gaps in the lines)
    lim_w, lim_h = (0.15 * min(hh, ww), 0.15 * min(hh, ww)) if scan else (0.15 * ww, 0.15 * hh)   # and small figures
    segs = {}
    for name, ker in (("h", np.ones((1, k), np.uint8)), ("v", np.ones((k, 1), np.uint8))):
        n, lab, st, _ = _cc(cv2.morphologyEx(dark, cv2.MORPH_OPEN, ker))
        out = []
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if name == "h" and h <= 6:
                out.append((y + (h - 1) / 2.0, float(x), float(x + w - 1)))
            elif name == "v" and w <= 6:
                out.append((x + (w - 1) / 2.0, float(y), float(y + h - 1)))
        segs[name] = out
    tol, frames = 4.0, []
    for y, hx0, hx1 in segs["h"]:                          # bottom line
        for x, vy0, vy1 in segs["v"]:                      # left line through its left end (it may stop short of
            if abs(x - hx0) > tol or not (vy0 - tol < y < vy1 + max(tol, 0.12 * (y - vy0))):    # the corner)
                continue
            # top: a line of the same width above that closes the box on the right, or the top of the left line
            # (a zero line across the plot does not close the box)
            above = [h for h in segs["h"] if h[0] < y - 0.1 * (y - vy0) and h[0] > vy0 - tol
                     and abs(h[1] - hx0) < tol and abs(h[2] - hx1) < tol
                     and any(abs(v[0] - hx1) < tol and v[1] < h[0] + tol and v[2] > y - max(tol, 0.12 * (y - h[0])) for v in segs["v"])]
            top = max(h[0] for h in above) if above else vy0
            if hx1 - hx0 >= lim_w and y - top >= lim_h:
                frames.append((hx0, top, hx1, y))
    inside = lambda a, b: a != b and a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol
    smaller = lambda a, b: (a[2] - a[0]) * (a[3] - a[1]) < 0.8 * (b[2] - b[0]) * (b[3] - b[1])
    if insets:
        frames = [f for f in frames if any(inside(f, g_) and smaller(f, g_) for g_ in frames)]
    else:
        frames = [f for f in frames if not any(inside(f, g_) for g_ in frames)]
    keep = []
    for f in frames:                                       # the same frame found twice (thick lines)
        if not any(max(abs(f[i] - g_[i]) for i in range(4)) < 2 * tol for g_ in keep):
            keep.append(f)
    return keep


def picture_panels(img):
    """Plots side by side or stacked in one picture: one cell per axes frame, with its tick labels and
    titles. The cells are divided at the widest empty band between neighbouring frames. Returns
    [(x0, y0, x1, y1)] in pixels, or [] when the picture holds one plot."""
    frames = picture_frames(img)
    if len(frames) < 2:
        return []
    hh, ww = img.shape[:2]
    ink = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY) < 200

    def cut(lo, hi, profile):                              # middle of the widest empty run in profile[lo:hi]
        lo, hi = int(max(0, lo)), int(min(len(profile), hi))
        if hi - lo < 2:
            return 0.5 * (lo + hi)
        empty = profile[lo:hi] == 0
        best, run, start = None, 0, 0
        for i, e in enumerate(empty):
            if e:
                if run == 0:
                    start = i
                run += 1
                if best is None or run > best[0]:
                    best = (run, start)
            else:
                run = 0
        return lo + best[1] + best[0] / 2.0 if best else 0.5 * (lo + hi)

    cells = []
    for f in frames:
        x0, y0, x1, y1 = 0.0, 0.0, float(ww), float(hh)
        for g_ in frames:
            if g_ == f:
                continue
            rows = (min(f[3], g_[3]) - max(f[1], g_[1])) > 0.3 * min(f[3] - f[1], g_[3] - g_[1])     # side by side
            cols = (min(f[2], g_[2]) - max(f[0], g_[0])) > 0.3 * min(f[2] - f[0], g_[2] - g_[0])     # one above the other
            if rows:
                ya, yb = int(min(f[1], g_[1])), int(max(f[3], g_[3])) + 1
                prof = ink[ya:yb].sum(axis=0)
                if g_[0] >= f[2]:
                    x1 = min(x1, cut(f[2] + 2, g_[0] - 2, prof))
                elif g_[2] <= f[0]:
                    x0 = max(x0, cut(g_[2] + 2, f[0] - 2, prof))
            elif cols:
                xa, xb = int(min(f[0], g_[0])), int(max(f[2], g_[2])) + 1
                prof = ink[:, xa:xb].sum(axis=1)
                if g_[1] >= f[3]:
                    y1 = min(y1, cut(f[3] + 2, g_[1] - 2, prof))
                elif g_[3] <= f[1]:
                    y0 = max(y0, cut(g_[3] + 2, f[1] - 2, prof))
        fw, fh = f[2] - f[0], f[3] - f[1]                  # much room on a side without a neighbour (a drawing,
        rows = ink[int(f[1]):int(f[3]) + 1].sum(axis=0)    # a photo next to the plot): cut at the widest gap there
        cols = ink[:, int(f[0]):int(f[2]) + 1].sum(axis=1)
        if f[0] - x0 > 0.6 * fw:
            x0 = cut(x0, f[0] - 2, rows)
        if x1 - f[2] > 0.6 * fw:
            x1 = cut(f[2] + 2, x1, rows)
        if f[1] - y0 > 0.6 * fh:
            y0 = cut(y0, f[1] - 2, cols)
        if y1 - f[3] > 0.6 * fh:
            y1 = cut(f[3] + 2, y1, cols)
        cells.append((x0, y0, x1, y1))
    return cells


def scan_panels(img):
    """Plots on a scanned page (one picture holding text and figures): each axes frame grown outwards over its
    tick labels and axis titles, up to the first empty band wider than 8 % of the frame size (the gap to the
    caption or to the text). Returns [(x0, y0, x1, y1)] in pixels."""
    ink = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY) < 200
    hh, ww = ink.shape
    cells = []
    for f in picture_frames(img, scan=True):
        x0, y0, x1, y1 = (int(round(v)) for v in f)
        fw, fh = x1 - x0, y1 - y0
        gap = max(6, int(0.08 * max(fw, fh)))

        def grow(start, step, limit, line):                # first empty run of `gap` lines going outwards
            run, i = 0, start
            while 0 <= i + step < limit and abs(i - start) < 0.5 * (fw if limit == ww else fh):
                i += step
                run = run + 1 if not line(i) else 0
                if run >= gap:
                    return i - step * (run - 1)
            return i
        a, b = max(0, y0 - gap), min(hh, y1 + gap)
        x0 = grow(x0 - 2, -1, ww, lambda c: ink[a:b, c].any())
        x1 = grow(x1 + 2, +1, ww, lambda c: ink[a:b, c].any())
        y0 = grow(y0 - 2, -1, hh, lambda r: ink[r, x0:x1].any())
        y1 = grow(y1 + 2, +1, hh, lambda r: ink[r, x0:x1].any())
        cells.append((max(0, x0 - 2), max(0, y0 - 2), min(ww, x1 + 2), min(hh, y1 + 2)))
    return cells


def _outline_words(drawings):
    """Words drawn as filled glyph outlines (small filled paths without stroke), joined where the glyphs
    nearly touch. When a figure's drawings form one cluster, these are its tick labels and titles.
    Returns [Rect]."""
    import pymupdf
    gl = [pymupdf.Rect(d["rect"]) for d in drawings
          if d.get("fill") is not None and d.get("color") is None and d["rect"].width < 14 and d["rect"].height < 14]
    joined = True
    while joined:
        joined = False
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                if _gap(gl[i], gl[j]) < 2.0:
                    gl[i] = gl[i] | gl[j]
                    del gl[j]
                    joined = True
                    break
            if joined:
                break
    return gl


def figure_candidates(page, drawings):
    """Likely figures on a page: clusters of vector drawings and embedded pictures. A cluster is grown
    by the short texts next to it (tick labels, axis titles); paragraphs and captions stay out.
    Returns [(Rect, 'vector' | 'raster')]."""
    import pymupdf
    try:
        clusters = [pymupdf.Rect(r) for r in page.cluster_drawings(drawings=drawings, x_tolerance=6, y_tolerance=6, final_filter=False)]
    except Exception:
        clusters = []
    # label-like pieces: short text lines that are not part of a paragraph or a caption, and small
    # drawing clusters (labels of a figure whose text was converted to outlines)
    labels, prose = [], []
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        blocks = []
    for b in blocks:
        lines = [l for l in b.get("lines", []) if "".join(sp["text"] for sp in l["spans"]).strip()]
        text = " ".join("".join(sp["text"] for sp in l["spans"]) for l in lines).strip()
        nw = len(text.split())
        if re.match(r"(?i)(fig\.|fig\s|figure|table|tab\.)", text) or (nw >= 12 and nw >= 4 * len(lines)):
            prose.append(pymupdf.Rect(b["bbox"]))
            continue                                        # caption, or prose: many words in long lines
        for l in lines:
            t = "".join(sp["text"] for sp in l["spans"]).strip()
            if len(t.split()) <= 7 and len(t) <= 60:
                labels.append(pymupdf.Rect(l["bbox"]))
    big = [r for r in clusters if r.width >= 50 and r.height >= 50]
    labels += [r for r in clusters if r.width < 50 or r.height < 50]
    labels += _outline_words(drawings)                    # text converted to outlines inside one big cluster
    out, panels = [], []

    def grow(reg, own=None):
        reg = pymupdf.Rect(reg)
        for _ in range(4):                                  # tick labels first, then the axis titles
            add = [l for l in labels if l not in reg and _gap(l, reg) < 12.0
                   and l.width <= 1.1 * reg.width and l.height <= 1.1 * reg.height and (own is None or own(l))]
            if not add:
                break
            for l in add:
                reg |= l
        return (reg + (-2, -2, 2, 2)) & page.rect

    def grow_all(fr):
        """All panels of one figure grown together: in each round a label goes to the panel whose region,
        as grown so far, is nearest (an axis title sits next to its own tick labels, which may be farther
        from its frame than the frame of the next panel is). Distances above a panel or to its right
        count double: labels are mostly below and left of their frame."""
        regs = [pymupdf.Rect(f) for f in fr]
        taken = set()                                      # a label belongs to one panel only, once given
        for _ in range(6):
            add = []
            for i, l in enumerate(labels):
                if i in taken:
                    continue
                gaps = [_edge_gap(l, r) * (2.0 if l.y1 <= r.y0 + 0.5 or l.x0 >= r.x1 - 0.5 else 1.0) for r in regs]
                k = int(np.argmin(gaps))
                r = regs[k]
                if gaps[k] < 12.0 and l.width <= 1.1 * r.width and l.height <= 1.1 * r.height:
                    add.append((k, i))
            if not add:
                break
            for k, i in add:
                regs[k] |= labels[i]
                taken.add(i)
        return [(r + (-2, -2, 2, 2)) & page.rect for r in regs]

    frames = panel_frames(drawings, page.rect)            # axes frames on the page; a label goes with the nearest
    mine = lambda l, f: _edge_gap(l, f) <= min(_edge_gap(l, g) for g in frames) + 0.5
    inside = lambda r: [f for f in frames if (f & r).get_area() > 0.5 * f.get_area()]
    if any(len(inside(r)) >= 2 for r in big):             # several panels (rows, columns, insets): one region each,
        starts = []                                        # all plots of the page grown together, so that no panel
        for r in big:                                      # takes the labels of a panel in another drawing cluster
            fr = inside(r)
            starts += fr if len(fr) >= 2 else [r] if fr else []
        for g_ in grow_all(list(dict.fromkeys(map(tuple, starts)))):
            out.append((g_, "vector"))
            panels.append(g_)
    for r in big:
        fr = inside(r)
        if fr and any(len(inside(q)) >= 2 for q in big):
            continue
        elif fr:                                           # one plot: it does not take the labels of the panel next to it
            out.append((grow(r, lambda l, f=fr[0]: mine(l, f)), "vector"))
        else:
            out.append((grow(r), "vector"))
    for r in merged_pictures(page):                        # embedded pictures (strips of one figure joined)
        pic, z = render_region(page, r, 2.0)
        scan = (r & page.rect).get_area() >= 0.8 * page.rect.get_area()      # a scanned page: figures within text
        cells = scan_panels(pic) if scan else picture_panels(pic)
        if len(cells) >= 2 or (scan and cells):                              # several plots in one picture: one region each
            for x0, y0, x1, y1 in cells:
                c = pymupdf.Rect(r.x0 + x0 / z, r.y0 + y0 / z, r.x0 + x1 / z, r.y0 + y1 / z)
                if scan:                                   # a caption or paragraph of the OCR text layer ends the cell
                    f = pymupdf.Rect(c.x0 + 0.25 * c.width, c.y0 + 0.25 * c.height, c.x1 - 0.25 * c.width, c.y1 - 0.25 * c.height)
                    for b in prose:
                        if (b & c).is_empty or b.intersects(f):
                            continue
                        if b.y0 >= f.y1:
                            c.y1 = min(c.y1, b.y0 - 1)
                        elif b.y1 <= f.y0:
                            c.y0 = max(c.y0, b.y1 + 1)
                        elif b.x0 >= f.x1:
                            c.x1 = min(c.x1, b.x0 - 1)
                        elif b.x1 <= f.x0:
                            c.x0 = max(c.x0, b.x1 + 1)
                out.append((c, "raster"))
                panels.append(out[-1][0])
        else:
            out.append(((r + (-2, -2, 2, 2)) & page.rect, "raster"))
    vec = [r for r, kind in out if kind == "vector"]          # a picture that spreads over two or more vector plots
    out = [(r, kind) for r, kind in out                         # is an image layer of that figure (symbols, shading,
           if kind == "vector" or any(r == p_ for p_ in panels)  # a schematic next to the plots), not a plot itself
           or sum((r & v).get_area() > 0.15 * v.get_area() for v in vec) < 2]
    keep = []
    for r, kind in sorted(out, key=lambda o: -o[0].get_area()):          # drop regions inside a larger one, except
        if not any(r in o and (o == r or not any(r == p_ for p_ in panels)) for o, _ in keep):   # panels and insets
            keep.append((r, kind))
    return keep


def render_region(page, clip, zoom=None):
    import pymupdf
    z = zoom or float(np.clip(1600.0 / max(clip.width, clip.height), 2.0, 8.0))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=clip, alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy(), z


def _rgb(c):
    if c is None:
        return None
    c = list(c)
    if len(c) == 1:
        c = c * 3
    return tuple(int(round(255 * v)) for v in c[:3])


def vector_paths(drawings, clip, z):
    """Drawing commands inside the region as polylines in image pixel coordinates."""
    out = []
    org = np.array((clip.x0, clip.y0))
    tt = np.linspace(0, 1, 9)[:, None]
    for d in drawings:
        if not _overlaps(d["rect"], clip):
            continue
        subs, cur, curve = [], None, False
        for it in d["items"]:
            if it[0] == "l":
                seg = np.array([(it[1].x, it[1].y), (it[2].x, it[2].y)])
            elif it[0] == "c":
                curve = True
                p = [np.array((q.x, q.y)) for q in it[1:5]]
                seg = (1 - tt) ** 3 * p[0] + 3 * (1 - tt) ** 2 * tt * p[1] + 3 * (1 - tt) * tt ** 2 * p[2] + tt ** 3 * p[3]
            elif it[0] == "re":
                r = it[1]
                subs.append([np.array(q) for q in ((r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0))])
                cur = None
                continue
            elif it[0] == "qu":
                q = it[1]
                subs.append([np.array((v.x, v.y)) for v in (q.ul, q.ur, q.lr, q.ll, q.ul)])
                cur = None
                continue
            else:
                continue
            if cur is not None and np.hypot(*(cur[-1] - seg[0])) < 1e-3:
                cur.extend(seg[1:])
            else:
                cur = list(seg)
                subs.append(cur)
        if d.get("closePath") and cur is not None and np.hypot(*(cur[-1] - cur[0])) > 1e-3:
            cur.append(cur[0])
        polys = [(np.array(s, float) - org) * z - 0.5 for s in subs if len(s) >= 2]
        if not polys:
            continue
        allp = np.vstack(polys)
        dash = [float(v) * z for v in re.findall(r"[-+]?\d*\.?\d+", (d.get("dashes") or "").split("]")[0])]
        kind = d.get("type", "s")
        out.append(dict(idx=len(out), polys=polys, stroke=_rgb(d.get("color")) if "s" in kind else None,
                        fill=_rgb(d.get("fill")) if "f" in kind else None, width=float(d.get("width") or 0) * z,
                        dashes=dash, curve=curve, bbox=(*allp.min(axis=0), *allp.max(axis=0))))
    return out


def vector_words(page, clip, z):
    """Horizontal words of the text layer: text, box (pixels), height, baseline."""
    words = []
    for blk in page.get_text("rawdict", clip=clip)["blocks"]:
        if blk["type"] != 0:
            continue
        for ln in blk["lines"]:
            if ln["dir"][0] < 0.99:                        # rotated text (axis titles)
                continue
            cur = None
            for sp in ln["spans"]:
                for ch in sp["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    h = y1 - y0
                    if ch["c"].isspace() or h <= 0:
                        cur = None
                        continue
                    if (cur is not None and x0 - cur["x1"] < 0.3 * h and abs(h - cur["h"]) < 0.12 * cur["h"]
                            and abs(ch["origin"][1] - cur["base"]) < 0.1 * h):
                        cur["text"] += ch["c"]
                        cur["x1"], cur["y0"], cur["y1"] = max(cur["x1"], x1), min(cur["y0"], y0), max(cur["y1"], y1)
                    else:
                        cur = dict(text=ch["c"], x0=x0, y0=y0, x1=x1, y1=y1, h=h, base=ch["origin"][1])
                        words.append(cur)
    for w in words:
        w["x0"], w["x1"] = (w["x0"] - clip.x0) * z - 0.5, (w["x1"] - clip.x0) * z - 0.5
        w["y0"], w["y1"], w["base"] = (w["y0"] - clip.y0) * z - 0.5, (w["y1"] - clip.y0) * z - 0.5, (w["base"] - clip.y0) * z - 0.5
        w["h"] *= z
    return words


def vector_labels(words):
    """Numbers of the text layer, with superscripts (10^-3) and mantissas (3 x 10^-3) put back together.
    Returns [dict(value, cx, cy, h, x0, x1)]."""
    used, labels = set(), []
    order = sorted(range(len(words)), key=lambda i: words[i]["x0"])

    def left_of(k, texts=None):                            # the word that ends just left of word k, same baseline
        a = words[k]
        for j in order:
            b = words[j]
            if (j != k and -0.1 * a["h"] <= a["x0"] - b["x1"] <= 0.9 * a["h"] and abs(b["base"] - a["base"]) < 0.2 * a["h"]
                    and (texts is None or b["text"] in texts)):
                return j
        return None

    for i in order:
        a = words[i]
        exp, right, left = [], a["x1"], a["x0"]
        for j in order:
            b = words[j]
            small = b["h"] < 0.9 * a["h"] and a["y0"] - 0.6 * a["h"] < b["base"] < a["base"] - 0.12 * a["h"]
            large = b["h"] <= 1.05 * a["h"] and a["base"] - 0.75 * a["h"] < b["base"] < a["base"] - 0.2 * a["h"]
            if j != i and (small or large) and -0.35 * a["h"] <= b["x0"] - right <= 0.6 * a["h"]:
                exp.append(j)
                right = b["x1"]
        if exp and parse_number(a["text"]) is not None:
            used |= set(exp)
            value = power_value(a["text"], "".join(words[j]["text"] for j in exp))
            t = left_of(i, ("\u00d7", "x", "X", "\u00b7", "\u22c5", "*")) if a["text"].strip() == "10" else None
            m = left_of(t) if t is not None else None
            if m is not None and value is not None and parse_number(words[m]["text"]) is not None:
                value *= parse_number(words[m]["text"])
                used |= {t, m}
                left = words[m]["x0"]
        else:
            value = parse_number(a["text"])
        labels.append((i, value, left, right))
    return [dict(value=v, cx=0.5 * (l + r), cy=0.5 * (words[i]["y0"] + words[i]["y1"]), h=words[i]["h"], x0=l, x1=r)
            for i, v, l, r in labels if i not in used and v is not None]


def _axis_segments(paths):
    """Axis-aligned stroked segments: horizontal [(y, x0, x1)], vertical [(x, y0, y1)]."""
    hs, vs = [], []
    for p in paths:
        if p["stroke"] is None:
            continue
        for poly in p["polys"]:
            a, b = poly[:-1], poly[1:]
            d = np.abs(b - a)
            for i in np.nonzero((d[:, 1] < 0.3) & (d[:, 0] > 1.0))[0]:
                hs.append((float(a[i, 1]), float(min(a[i, 0], b[i, 0])), float(max(a[i, 0], b[i, 0]))))
            for i in np.nonzero((d[:, 0] < 0.3) & (d[:, 1] > 1.0))[0]:
                vs.append((float(a[i, 0]), float(min(a[i, 1], b[i, 1])), float(max(a[i, 1], b[i, 1]))))
    return hs, vs


def _merge_collinear(segs):
    """Join segments on one line (position within 0.75) whose spans touch or overlap. Lines of plots
    side by side lie at nearly the same height but far apart: they stay separate."""
    groups = []
    for s in sorted(segs):
        if groups and s[0] - groups[-1][-1][0] < 0.75:
            groups[-1].append(s)
        else:
            groups.append([s])
    out = []
    for g in groups:
        run = []
        for c, lo, hi in sorted(g, key=lambda s: s[1]):
            if run and lo <= run[-1][2] + 1.5:
                run[-1][2] = max(run[-1][2], hi)
            else:
                run.append([c, lo, hi])
        out += run
    return out


def vector_frame(paths, w, h):
    """The axes: the longest horizontal and vertical lines that meet at a corner."""
    hs, vs = _axis_segments(paths)
    hl = [s for s in _merge_collinear(hs) if s[2] - s[1] >= 0.25 * w]
    vl = [s for s in _merge_collinear(vs) if s[2] - s[1] >= 0.25 * h]
    if not hl or not vl:
        return None
    best, tol = None, 3.0
    for y, x0, x1 in hl:
        for x, y0, y1 in vl:
            corner = min(abs(x - x0), abs(x - x1)) < tol and min(abs(y - y0), abs(y - y1)) < tol
            key = (corner, (x1 - x0) * (y1 - y0))
            if best is None or key > best[0]:
                best = (key, (min(x0, x), min(y0, y), max(x1, x), max(y1, y)))
    return best[1]


def clip_polyline(p, box):
    """Parts of a polyline inside a rectangle (Liang-Barsky on every segment)."""
    x0, y0, x1, y1 = box
    out, cur = [], []
    for a, b in zip(p[:-1], p[1:]):
        d = b - a
        t0, t1, ok = 0.0, 1.0, True
        for pp, qq in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if abs(pp) < 1e-12:
                if qq < 0:
                    ok = False
                    break
            else:
                r = qq / pp
                if pp < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if not ok or t0 > t1:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
            continue
        pa, pb = a + t0 * d, a + t1 * d
        if not cur or np.hypot(*(cur[-1] - pa)) > 1e-6:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = [pa]
        cur.append(pb)
        if t1 < 1.0:
            out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def join_polylines(pieces, tol=0.75):
    """Join polylines whose end points coincide (paths that were written in several chunks)."""
    pieces = list(pieces)
    while len(pieces) > 1:
        ends = np.array([q for p in pieces for q in (p[0], p[-1])])
        used, merged = set(), []
        for i, j in sorted(cKDTree(ends).query_pairs(tol)):
            a, b = i // 2, j // 2
            if a == b or a in used or b in used:
                continue
            used |= {a, b}
            pa = pieces[a] if i % 2 == 1 else pieces[a][::-1]       # ... ends at the joint
            pb = pieces[b] if j % 2 == 0 else pieces[b][::-1]       # starts at the joint ...
            merged.append(np.vstack([pa, pb[1:]]))
        if not merged:
            break
        pieces = merged + [p for k, p in enumerate(pieces) if k not in used]
    return pieces


def vector_marker_shape(p):
    """Symbol name from the outline of one marker path. Returns (shape, open?, centroid offset from the box centre)."""
    polys = p["polys"]
    pts = np.vstack(polys)
    x0, y0, x1, y1 = p["bbox"]
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    hollow = p["fill"] is None or min(p["fill"]) >= 250
    if len(polys) == 1 and p["fill"] is not None and len(polys[0]) >= 3 and np.hypot(*(polys[0][0] - polys[0][-1])) > 1e-6:
        polys = [np.vstack([polys[0], polys[0][:1]])]      # a filled outline is closed even if not written so
    closed = len(polys) == 1 and np.hypot(*(polys[0][0] - polys[0][-1])) < 0.02 * max(w, h) and len(polys[0]) >= 4
    if not closed:
        segs = [(q[0], q[-1]) for q in polys if len(q) == 2]
        if len(polys) == 1 and len(polys[0]) == 2:
            return "bar", False, (0.0, 0.0)
        if len(polys) == 1 and p["fill"] is None:          # an open, nearly straight stroke of several points: one
            q = polys[0]                                   # dash of a dashed curve drawn piece by piece
            u = q[-1] - q[0]
            L = np.hypot(*u)
            if L > 1e-6 and np.abs((q[:, 0] - q[0, 0]) * u[1] - (q[:, 1] - q[0, 1]) * u[0]).max() / L < 0.15 * L:
                return "bar", False, (0.0, 0.0)
        if len(segs) == len(polys) == 2:
            d = [np.abs(b - a) / max(np.hypot(*(b - a)), 1e-9) for a, b in segs]
            return ("plus" if all(min(v) < 0.2 for v in d) else "cross"), False, (0.0, 0.0)
        return "star", False, (0.0, 0.0)
    q = polys[0]
    x, y = q[:-1, 0], q[:-1, 1]
    xn, yn = q[1:, 0], q[1:, 1]
    cr = x * yn - xn * y
    area = 0.5 * cr.sum()
    if abs(area) < 1e-9:
        return "bar", False, (0.0, 0.0)
    cx, cy = ((x + xn) * cr).sum() / (6 * area), ((y + yn) * cr).sum() / (6 * area)
    off = (cx - 0.5 * (x0 + x1), cy - 0.5 * (y0 + y1))
    extent = abs(area) / (w * h)
    corners = len(cv2.approxPolyDP(q[:-1].astype(np.float32).reshape(-1, 1, 2), 0.04 * (w + h), True))
    if p["curve"] and extent > 0.7:
        return "circle", hollow, off
    oy, ox = off[1] / h, off[0] / w
    if corners == 3 or (max(abs(ox), abs(oy)) > 0.08 and extent < 0.62):
        if abs(oy) >= abs(ox):
            return ("triangle up" if oy > 0 else "triangle down"), hollow, off
        return ("triangle left" if ox > 0 else "triangle right"), hollow, off
    if corners == 4:
        return ("square" if extent > 0.85 else "diamond"), hollow, off
    if corners == 5:
        return "pentagon", hollow, off
    if corners == 6:
        return "hexagon", hollow, off
    if corners >= 8 and extent < 0.6:
        return "star", hollow, off
    return ("circle" if extent > 0.7 else "polygon"), hollow, off


def shared_x_labels(page, clip, z, box, drawings):
    """x tick labels for a panel without its own: those under the nearest panel below with the same
    left and right edges (a shared x axis). Returns [(pixel x in the region, value)]."""
    import pymupdf
    f = pymupdf.Rect(clip.x0 + (box[0] + 0.5) / z, clip.y0 + (box[1] + 0.5) / z, clip.x0 + (box[2] + 0.5) / z, clip.y0 + (box[3] + 0.5) / z)
    below = sorted((g for g in panel_frames(drawings, page.rect)
                    if abs(g.x0 - f.x0) < 2.5 and abs(g.x1 - f.x1) < 2.5 and g.y0 > f.y1 - 2.5), key=lambda g: g.y0)
    for g in below:
        m = 0.08 * g.width
        strip = pymupdf.Rect(g.x0 - m, g.y1 + 0.5, g.x1 + m, g.y1 + 30) & page.rect
        if strip.is_empty:
            continue
        labs = vector_labels(vector_words(page, strip, z))
        if len(labs) < 3:
            continue
        top = min(l["cy"] for l in labs)
        labs = [l for l in labs if l["cy"] < top + 1.2 * l["h"]]
        if len(labs) >= 3 and np.ptp([l["cx"] for l in labs]) >= 0.5 * g.width * z:   # a row of tick labels
            dx = (strip.x0 - clip.x0) * z
            return [(l["cx"] + dx, l["value"]) for l in labs]
    return []


def process_vector(page, clip, drawings=None, progress=_noprog):
    """Digitize a figure drawn with vector paths: coordinates are read from the paths themselves."""
    log = []
    progress(5, "Reading vector paths")
    image, z = render_region(page, clip)
    hh, ww = image.shape[:2]
    paths = vector_paths(drawings if drawings is not None else page.get_drawings(), clip, z)
    box = vector_frame(paths, ww, hh)
    if box is None:
        raise ValueError("no axes found in the vector paths")
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    diag, tol = math.hypot(bw, bh), 2.0 + 0.5 * z

    # ---- frame, ticks, grid ("furniture")
    xticks, yticks, y2ticks, grid = [], [], [], []
    for p in paths:
        p["furn"] = p["stroke"] is not None
        if p["stroke"] is None:
            continue
        for poly in p["polys"]:
            for a, b in zip(poly[:-1], poly[1:]):
                d = np.abs(b - a)
                ln = float(np.hypot(*d))
                horiz, vert = d[1] < 0.3 and d[0] > 0.3, d[0] < 0.3 and d[1] > 0.3
                on_h = horiz and (abs(a[1] - by0) < tol or abs(a[1] - by1) < tol) and min(a[0], b[0]) > bx0 - tol and max(a[0], b[0]) < bx1 + tol
                on_v = vert and (abs(a[0] - bx0) < tol or abs(a[0] - bx1) < tol) and min(a[1], b[1]) > by0 - tol and max(a[1], b[1]) < by1 + tol
                if on_h or on_v:
                    continue
                if vert and ln <= 0.06 * min(bw, bh) and bx0 - tol < a[0] < bx1 + tol:
                    if min(abs(a[1] - by1), abs(b[1] - by1)) < tol:
                        xticks.append((float(a[0]), ln))
                        continue
                    if min(abs(a[1] - by0), abs(b[1] - by0)) < tol:
                        continue
                if horiz and ln <= 0.06 * min(bw, bh) and by0 - tol < a[1] < by1 + tol:
                    if min(abs(a[0] - bx0), abs(b[0] - bx0)) < tol:
                        yticks.append((float(a[1]), ln))
                        continue
                    if min(abs(a[0] - bx1), abs(b[0] - bx1)) < tol:
                        y2ticks.append((float(a[1]), ln))
                        continue
                if len(poly) == 2 and ((horiz and d[0] >= 0.9 * bw) or (vert and d[1] >= 0.9 * bh)):
                    grid.append((p, "h" if horiz else "v", float(a[1] if horiz else a[0])))
                    continue
                p["furn"] = False
    style = lambda p: (p["stroke"], round(p["width"], 1), tuple(round(v, 1) for v in p["dashes"]))
    n_grid = 0
    for p, o, c in grid:                                   # grid: at a tick, or several lines of one style
        tk = [t[0] for t in (yticks if o == "h" else xticks)]
        if (tk and min(abs(c - t) for t in tk) < 1.0) or sum(style(q) == style(p) for q, _, _ in grid) >= 2:
            n_grid += 1
        else:
            p["furn"] = False
    if n_grid:
        log.append(f"{n_grid} grid lines ignored")

    # ---- axis calibration from the text layer (OCR of the rendered picture as a fallback)
    progress(25, "Reading tick labels")
    words = vector_words(page, clip, z)
    labels = vector_labels(words)
    hchar = float(np.median([l["h"] for l in labels])) if labels else 10.0 * z
    m = 0.08 * max(bw, bh)
    lx = [l for l in labels if l["cy"] > by1 and bx0 - m < l["cx"] < bx1 + m]
    if lx:
        top = min(l["cy"] for l in lx)
        lx = [l for l in lx if l["cy"] < top + 1.2 * l["h"]]
    ly = [l for l in labels if l["x1"] < bx0 + 1 and by0 - m < l["cy"] < by1 + m]
    if ly:
        right = max(l["x1"] for l in ly)
        ly = [l for l in ly if l["x1"] > right - 1.5 * l["h"]]
    px_x, px_y = [(l["cx"], l["value"]) for l in lx], [(l["cy"], l["value"]) for l in ly]
    frame = dict(box=tuple(int(round(v)) for v in box), t=(0, 0, 0, 0))
    if len(px_x) < 2 and drawings is not None:            # stacked panels: the x labels are under the lowest one
        px_x = shared_x_labels(page, clip, z, box, drawings)
        if len(px_x) >= 2:
            log.append("x tick labels taken from the panel below (shared x axis)")
    if len(px_x) < 2:
        px_x = margin_labels(image, frame, "x")[0]
        log.append("x tick labels are not text in this file: read by OCR")
    if len(px_y) < 2:
        px_y = margin_labels(image, frame, "y")[0]
        log.append("y tick labels are not text in this file: read by OCR")
    for tk, lo, hi in ((xticks, bx0, bx1), (yticks, by0, by1), (y2ticks, by0, by1)):
        if tk:                                             # a label at a frame corner has no tick stub of its own
            lm = max(t[1] for t in tk)
            tk += [(float(lo), lm), (float(hi), lm)]
    px_x, px_y = snap_to_ticks(px_x, xticks, 0.6 * hchar), snap_to_ticks(px_y, yticks, 0.6 * hchar)
    xax = fit_axis([p for p, _ in px_x], [v for _, v in px_x], bw, True) if len(px_x) >= 2 else None
    yax = fit_axis([p for p, _ in px_y], [v for _, v in px_y], bh, False) if len(px_y) >= 2 else None
    l2 = [l for l in labels if l["x0"] > bx1 - 1 and by0 - m < l["cy"] < by1 + m]
    if l2:
        left = min(l["x0"] for l in l2)
        l2 = [l for l in l2 if l["x0"] < left + 1.5 * l["h"]]
    px_2 = [(l["cy"], l["value"]) for l in l2]
    if len(px_2) < 3 and y2ticks:                          # right-hand labels drawn as outlines: read by OCR
        px_2 = margin_labels(image, frame, "y2")[0]
    px_2 = snap_to_ticks(px_2, y2ticks, 0.6 * hchar)
    yax2 = fit_axis([p for p, _ in px_2], [v for _, v in px_2], bh, False) if len(px_2) >= 3 else None
    yax, yax2 = second_axis(yax, yax2, by0, by1, log)

    # ---- legend and text boxes
    progress(45, "Removing legend and annotations")
    inside = lambda b, r, mm=0.0: b[0] >= r[0] - mm and b[1] >= r[1] - mm and b[2] <= r[2] + mm and b[3] <= r[3] + mm
    wrects = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in words
              if bx0 < 0.5 * (w["x0"] + w["x1"]) < bx1 and by0 < 0.5 * (w["y0"] + w["y1"]) < by1]
    glyphs = []                                            # text inside the frame drawn as filled outlines (no text
    if not wrects and sum(p["stroke"] is None and inside(p["bbox"], box) for p in paths) >= 8:   # layer): found by OCR;
        glyphs = [(a - 2, b - 2, c + 2, d + 2) for t, sc, a, b, c, d, q in OCR.detect(image)   # its glyphs are not
                  if is_real_text(t, sc) == 2 and bx0 < 0.5 * (a + c) < bx1 and by0 < 0.5 * (b + d) < by1]   # symbols
        wrects = list(glyphs)
        if glyphs:
            log.append(f"text inside the plot drawn as outlines: {len(glyphs)} labels found by OCR and ignored")
    zones = []
    for p in paths:
        b = p["bbox"]
        q = p["polys"][0]
        if (len(p["polys"]) == 1 and len(q) >= 5 and np.hypot(*(q[0] - q[-1])) < 1.0 and inside(b, box, -1.0)
                and b[2] - b[0] >= 0.06 * bw and b[3] - b[1] >= 0.04 * bh and (b[2] - b[0]) * (b[3] - b[1]) <= 0.5 * bw * bh
                and abs(cv2.contourArea(q.astype(np.float32))) > 0.85 * (b[2] - b[0]) * (b[3] - b[1])
                and any(inside(wr, b) for wr in wrects)):
            zones.append((b, p["idx"]))                    # what is painted later inside this box is legend content
    framed = [zn for zn, _ in zones]
    for i in legend_like(wrects):                          # legend without a frame: samples left of stacked labels
        a, b, c, d = wrects[i]
        if not any(inside(wrects[i], zn) for zn in framed):
            zones.append(((a - 5.5 * (d - b), b - 0.3 * (d - b), a, d + 0.3 * (d - b)), -1))
    if zones:
        log.append("legend / text box contents ignored")

    # ---- data paths (not those of an inset plot drawn inside this one)
    progress(60, "Sorting symbols and lines")
    inner = []
    if drawings is not None:
        for f in panel_frames(drawings, clip):
            g = ((f.x0 - clip.x0) * z - 0.5, (f.y0 - clip.y0) * z - 0.5, (f.x1 - clip.x0) * z - 0.5, (f.y1 - clip.y0) * z - 0.5)
            if inside(g, box, 2.0) and (g[2] - g[0]) * (g[3] - g[1]) < 0.8 * bw * bh:
                inner.append(g)
        if inner:
            log.append(f"{len(inner)} inset plot(s) inside the frame ignored (choose the inset itself to read it)")
    # a legend zone without a frame is only a guess (left of stacked labels): a curve drawn as short pieces that
    # passes through it continues outside, and its pieces stay (found by walking from the pieces outside)
    unframed = [zn for zn, first in zones if first == -1]
    passing = set()
    if unframed:
        near = lambda a, b_, r: a[0] - r <= b_[2] and b_[0] - r <= a[2] and a[1] - r <= b_[3] and b_[1] - r <= a[3]
        ins = [p for p in paths if p["stroke"] is not None and not p["furn"] and any(inside(p["bbox"], zn, 1.0) for zn in unframed)]
        reach = [p for p in paths if p["stroke"] is not None and not p["furn"] and not any(inside(p["bbox"], zn, 1.0) for zn in unframed)]
        grown = True
        while grown and ins:
            grown = False
            for p in list(ins):
                r = 3.0 * max(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1], 1.0)
                if any(style(q) == style(p) and near(p["bbox"], q["bbox"], r) for q in reach):
                    passing.add(p["idx"])
                    reach.append(p)
                    ins.remove(p)
                    grown = True
    cand = []
    for p in paths:
        b = p["bbox"]
        if p["furn"] or any(inside(b, zn, 1.0) and p["idx"] >= first and not (first == -1 and p["idx"] in passing)
                            for zn, first in zones) or any(inside(b, g, 1.0) for g in inner):
            continue
        if p["stroke"] is None and any(inside(b, g) for g in glyphs):
            continue
        if b[2] < bx0 - tol or b[0] > bx1 + tol or b[3] < by0 - tol or b[1] > by1 + tol:
            continue
        if p["stroke"] is None and (b[2] - b[0]) * (b[3] - b[1]) > 0.02 * bw * bh:
            continue                                       # background or filled area
        cand.append(p)
    smax = max(6.0, 0.07 * min(bw, bh))
    groups = {}
    for p in cand:
        b = p["bbox"]
        w_, h_ = b[2] - b[0], b[3] - b[1]
        if 1.0 <= max(w_, h_) <= smax and bx0 - tol < 0.5 * (b[0] + b[2]) < bx1 + tol and by0 - tol < 0.5 * (b[1] + b[3]) < by1 + tol:
            key = (sum(len(q) for q in p["polys"]), p["stroke"], p["fill"], round(p["width"], 1))
            for k2 in groups:                              # same outline and (within rounding) the same size
                if k2[:4] == key and abs(k2[4] - w_) < 0.35 + 0.03 * w_ and abs(k2[5] - h_) < 0.35 + 0.03 * h_:
                    groups[k2].append(p)
                    break
            else:
                groups[key + (w_, h_)] = [p]
    series, marker_ids = [], set()
    for key, g in groups.items():
        if len(g) < 3:
            continue
        shape, hollow, off = vector_marker_shape(g[0])
        if shape == "bar":                                 # dashes drawn one by one, or error-bar caps: see lines
            continue
        marker_ids |= {id(p) for p in g}
        c = np.array(sorted((0.5 * (p["bbox"][0] + p["bbox"][2]), 0.5 * (p["bbox"][1] + p["bbox"][3])) for p in g))
        col = g[0]["fill"] if (g[0]["fill"] is not None and not hollow) else (g[0]["stroke"] or g[0]["fill"])
        series.append(Series("scatter", c + off, col, marker_name(shape, hollow), px_alt=c))
    series = dedupe_scatter(series, 0.6)

    lines = {}
    for p in cand:
        if id(p) in marker_ids or p["stroke"] is None:
            continue
        b = p["bbox"]
        if p["fill"] is not None and (b[2] - b[0]) * (b[3] - b[1]) > 0.01 * bw * bh:
            continue                                       # bars, bands
        for poly in p["polys"]:
            lines.setdefault(style(p), []).extend(clip_polyline(poly, box))
    for (col, wd, dash), pieces in lines.items():
        pieces = join_polylines(pieces)
        short = [q for q in pieces if _plen(q) < 0.06 * diag]
        if len(short) >= 8 and not dash:                   # dashes drawn as separate strokes: link them
            edges = []
            for k, q in enumerate(short):
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
                t = np.linspace(0, s[-1], max(2, int(round(s[-1])) + 1))
                edges.append(dict(a=2 * k, b=2 * k + 1, pts=np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]))
            pieces = [q for q in pieces if _plen(q) >= 0.06 * diag]
            for ch in build_chains(edges, max(wd, 1.0), 0.0, diag):
                if len(ch["pieces"]) >= 4:
                    pieces.append(("dashed", refine_and_fill(ch["pts"], None, wd, ch["gaps"])))
        for q in pieces:
            st_ = "solid" if not dash else ("dash-dot" if len(dash) >= 4 else ("dotted" if dash[0] <= 1.6 * max(wd, 1.0) else "dashed"))
            note = ""
            if isinstance(q, tuple):
                st_, q, note = q[0], q[1], "gaps interpolated"
            ext = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))
            if ext < 0.03 * diag or (len(q) < 3 and ext < 0.2 * diag) or (is_straight(q) and ext < 0.25 * diag):
                continue
            if q[0, 0] > q[-1, 0]:
                q = q[::-1]
            if (xax is not None and xax.log) or (yax is not None and yax.log) or (yax2 is not None and yax2.log):
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
                extra = [np.linspace(a, b, int((b - a) // (0.01 * diag)) + 2)[1:-1] for a, b in zip(s[:-1], s[1:])]
                t = np.unique(np.r_[s, np.concatenate(extra)])
                if len(t) > len(s):
                    q = np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]
                    note = (note + "; points added along straight strokes (log axis)").strip("; ")
            series.append(Series("line", q, col, st_, note=note))
    for s in series:                                       # symbols that sit on a line
        if s.kind == "scatter":
            for ln in series:
                if ln.kind == "line" and max(abs(a - b) for a, b in zip(ln.color, s.color)) < 60:
                    d = cKDTree(ln.px).query(s.px_alt)[0]
                    if (d < 1.5).mean() > 0.7:                     # the line's own points give the symbol positions
                        s.note, ln.note, s.px = "markers on a line", (ln.note + "; with markers").strip("; "), s.px_alt.copy()
    series.sort(key=lambda s: s.kind != "line")
    flag_guides(series, tuple(float(v) for v in box))
    name_series(series)
    progress(100, "Done")
    return Result(image, tuple(float(v) for v in box), xax, yax, series, "vector", log, yax2)


# ----------------------------------------------------------------------------------------------
# Sources, processing entry point, export
# ----------------------------------------------------------------------------------------------

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")
PS_EXT = (".eps", ".epsf", ".epsi", ".ps")


def load_image(path):
    """Read an image file as RGB (transparent pixels become white)."""
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError("This file is not an image that can be read.")
    if a.dtype != np.uint8:
        a = cv2.convertScaleAbs(a, alpha=255.0 / float(np.iinfo(a.dtype).max))
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 4:
        alpha = a[:, :, 3:4].astype(np.float32) / 255.0
        a = (a[:, :, :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(a[:, :, 2::-1])


def pdf_source(page, clip, drawings):
    """Source description for a region of a PDF page; includes the check 'vector or picture?'."""
    info = region_info(page, clip, drawings)
    zoom = None if info["vector"] or not info["image_ppi"] else float(np.clip(info["image_ppi"] / 72.0, 2.0, 8.0))
    image, _ = render_region(page, clip, zoom)
    return dict(kind="pdf", page=page, clip=clip, drawings=drawings, info=info, image=image)


def run_source(src, progress=_noprog, full=False):
    """Digitize a source: vector paths when the PDF region has them, image analysis otherwise."""
    if src["kind"] == "pdf" and src["info"]["vector"]:
        try:
            res = process_vector(src["page"], src["clip"], src["drawings"], progress)
            if res.series:
                return res
            why = "no data paths found"
        except ValueError as e:
            why = str(e)
        res = process_raster(src["image"], progress, full=full)
        res.log.insert(0, f"Vector analysis gave nothing ({why}): the rendered picture was analysed instead.")
        return res
    return process_raster(src["image"], progress, full=full)


def _nice_step(raw):
    """Largest of 1, 2, 2.5, 5 x 10^k that is not above raw."""
    e = math.floor(math.log10(raw))
    for m in (5, 2.5, 2, 1):
        if m * 10.0 ** e <= raw * (1 + 1e-9):
            return m * 10.0 ** e
    return 10.0 ** e


def _single_valued(px, tol=4.0):
    """True when a traced line can be written as y(x): where the trace passes the same pixel column more
    than once (a loop, an S-shape, a hook), the passes must lie within tol pixels in y. Small back-steps
    of a trace along a flat part pass; a vertical stretch passes (it is one pass)."""
    if len(px) < 2:
        return False
    col = np.round(px[:, 0]).astype(np.int64)
    o = np.lexsort((np.arange(len(px)), col))
    c, i, y = col[o], o, px[o, 1]
    bad, ncol, k = 0, 0, 0
    while k < len(c):
        e = k
        while e + 1 < len(c) and c[e + 1] == c[k]:
            e += 1
        ncol += 1
        if e > k:
            cut = np.nonzero(np.diff(i[k:e + 1]) > 2)[0]
            if len(cut):
                means = [float(g.mean()) for g in np.split(y[k:e + 1], cut + 1)]
                bad += (max(means) - min(means)) > tol
        k = e + 1
    return bad <= 0.02 * ncol


def common_x_table(res, used, marker_ref="centroid"):
    """One x column shared by the lines, each line interpolated onto it (no extrapolation: blank outside
    its own x range). Step: one value per pixel of the plot, rounded down to 1, 2, 2.5 or 5 x 10^k
    (in decades for a log x axis). Interpolation runs in pixel coordinates, so it is linear in what the
    plot shows (log axes in decades). Symbols and multi-valued lines keep their own x and y columns.
    Returns (header, rows, text describing the grid)."""
    xa = res.xaxis
    lines = [s for s in used if s.kind == "line" and len(s.px) >= 2 and _single_valued(s.px)]
    rest = [s for s in used if not any(s is t for t in lines)]
    if not lines:
        return None
    lo = min(float(s.px[:, 0].min()) for s in lines)
    hi = max(float(s.px[:, 0].max()) for s in lines)
    if hi - lo < 2:
        return None

    def fwd(v):                       # data value -> the quantity that is linear in pixels
        v = np.asarray(v, float)
        return np.log10(v) if xa.log else 1.0 / v if xa.inv else v
    f_lo, f_hi = sorted(float(fwd(xa.to_data(p))) for p in (lo, hi))
    if xa.inv:                        # grid uniform in x itself
        d_lo, d_hi = sorted(float(xa.to_data(p)) for p in (lo, hi))
        step = _nice_step((d_hi - d_lo) / (hi - lo))
        grid = np.arange(math.ceil(d_lo / step - 1e-9), math.floor(d_hi / step + 1e-9) + 1) * step
        gp = (1.0 / grid - xa.b) / xa.a
    else:
        step = _nice_step((f_hi - f_lo) / (hi - lo))
        g = np.arange(math.ceil(f_lo / step - 1e-9), math.floor(f_hi / step + 1e-9) + 1) * step
        grid = 10.0 ** g if xa.log else g
        gp = (g - xa.b) / xa.a
    grid = np.where(np.abs(grid) < 1e-12 * max(1.0, float(np.abs(grid).max())), 0.0, grid)
    cols, header = [grid], ["x"]
    for s in lines:
        o = np.argsort(s.px[:, 0], kind="stable")
        xp, yp = s.px[o, 0], s.px[o, 1]
        y = np.interp(gp, xp, yp)
        yax = res.yaxis2 if (s.yref == "right" and res.yaxis2 is not None) else res.yaxis
        y = yax.to_data(y).astype(float)
        y[(gp < xp[0] - 1e-6) | (gp > xp[-1] + 1e-6)] = np.nan
        cols.append(y)
        header.append(f"{s.name}_y")
    for s in rest:                    # symbols, loops: as traced
        d = series_data(res, s, marker_ref)
        cols += [d[:, 0], d[:, 1]]
        header += [f"{s.name}_x", f"{s.name}_y"]
    n = max(len(c) for c in cols)
    rows = [[(float(c[i]) if i < len(c) and np.isfinite(c[i]) else None) for c in cols] for i in range(n)]
    what = (f"common x for {len(lines)} line(s): {len(grid)} values, step {step:g}" + (" decades" if xa.log else "")
            + " (linear interpolation, blank outside each line's own range)")
    if rest:
        what += "; as traced: " + ", ".join(s.name for s in rest)
    return header, rows, what


def export_result(res, path, marker_ref="centroid", common_x=False):
    """Write the ticked series to .csv or .xlsx: one x and one y column per series, or with common_x one
    shared x column for the lines (see common_x_table). Returns (series written, rows)."""
    used = [s for s in res.series if s.use]
    data = [series_data(res, s, marker_ref) for s in used]
    if not used or any(d is None for d in data):
        raise ValueError("Nothing to export: no series selected, or the axes are not calibrated.")
    tab = common_x_table(res, used, marker_ref) if common_x else None
    if tab is not None:
        header, rows, xinfo = tab
    else:
        header = [f"{s.name}_{c}" for s in used for c in "xy"]
        n = max(len(d) for d in data)
        rows = [[(float(d[i, k]) if i < len(d) else None) for d in data for k in (0, 1)] for i in range(n)]
        xinfo = "as traced, one x column per series"
    if path.lower().endswith(".xlsx"):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "data"
        ws.append(header)
        for r in rows:
            ws.append(r)
        info = wb.create_sheet("info")
        info.append(["series", "type", "style", "points", "colour (R,G,B)", "y axis", "note"])
        for s, d in zip(used, data):
            info.append([s.name, s.kind, s.style, len(d), ",".join(str(c) for c in s.color),
                         "right" if (s.yref == "right" and res.yaxis2 is not None) else "left", s.note])
        x0, y0, x1, y1 = res.box
        info.append([])
        info.append(["x axis", describe_axis(res.xaxis, x0, x1)])
        info.append(["y axis", describe_axis(res.yaxis, y1, y0)])
        if res.yaxis2 is not None:
            info.append(["y axis (right)", describe_axis(res.yaxis2, y1, y0)])
        info.append(["x values", xinfo])
        info.append(["source", "vector paths" if res.source == "vector" else "image analysis"])
        info.append(["marker position", "bounding-box centre" if marker_ref == "bbox" else "centroid"])
        wb.save(path)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join("" if v is None else f"{v:.7g}" for v in r) + "\n")
    return len(used), len(rows)


def overlay_image(res, zoom=2):
    """The imported picture with the ticked series drawn over it (for a visual check without the interface)."""
    im = cv2.resize(res.image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    im = (0.55 * im + 0.45 * 255).astype(np.uint8)
    for s in res.series:
        if not s.use:
            continue
        col = (220, 38, 127) if sum(s.color) < 250 else (0, 0, 0)
        p = np.round((s.px + 0.5) * zoom).astype(np.int32)
        if s.kind == "line":
            cv2.polylines(im, [p.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
        else:
            for q in p:
                cv2.drawMarker(im, (int(q[0]), int(q[1])), col, cv2.MARKER_CROSS, 5 * zoom, 1)
    return im
