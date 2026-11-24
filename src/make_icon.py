"""Draws the program icon: three neon curves on a dark red tile with white axes and arrow heads.
Writes icon/xyd.png (512 px) and icon/xyd.ico (16 to 256 px, each size drawn on its own)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon")
NEON = [(0, 240, 255), (255, 40, 200), (190, 255, 40)]          # cyan, magenta, lime


def curves(n=200):
    t = np.linspace(0, 1, n)
    return [np.c_[t, 0.12 + 0.70 * (1 - np.exp(-4.5 * t))],               # saturating rise
            np.c_[t, 0.20 + 0.55 * np.exp(-((t - 0.55) / 0.20) ** 2)],    # peak
            np.c_[t, 0.08 + 0.80 * t ** 2.2]]                             # power law


def render(size):
    s = 4 * size                                                          # draw large, then reduce
    k = s / 256.0
    small = size <= 32
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(48 * k) if not small else int(30 * k)
    d.rounded_rectangle([0, 0, s - 1, s - 1], r, fill=(96, 0, 10, 255))
    inner = Image.new("RGBA", (s, s), (0, 0, 0, 0))                       # soft light in the middle
    ImageDraw.Draw(inner).ellipse([0.1 * s, 0.05 * s, 0.95 * s, 0.9 * s], fill=(140, 0, 18, 140))
    inner = inner.filter(ImageFilter.GaussianBlur(40 * k))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], r, fill=255)
    img = Image.alpha_composite(img, inner)
    m = (0.17 if not small else 0.13) * s                                 # plot area
    x0, y0, x1, y1 = m, s - m, s - 0.10 * s, 0.10 * s
    lw_ax = max(2, int((10 if small else 7) * k))
    lw_c = max(2, int((13 if small else 8) * k))
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    sharp = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    c = ImageDraw.Draw(sharp)
    for col, p in zip(NEON, curves() if not small else curves()[:3]):
        pts = [(x0 + (x1 - 0.06 * s - x0) * u, y0 + (y1 + 0.06 * s - y0) * v) for u, v in p]
        g.line(pts, fill=col + (230,), width=int(lw_c * 2.6), joint="curve")
        c.line(pts, fill=col + (255,), width=lw_c, joint="curve")
        c.line(pts, fill=(255, 255, 255, 200), width=max(1, lw_c // 3), joint="curve")   # hot core
    glow = glow.filter(ImageFilter.GaussianBlur(7 * k if not small else 4 * k))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, sharp)
    d = ImageDraw.Draw(img)
    ah = (20 if small else 16) * k                                       # arrow heads
    d.line([(x0, y0), (x1 - ah * 0.6, y0)], fill="white", width=lw_ax)
    d.line([(x0, y0), (x0, y1 + ah * 0.6)], fill="white", width=lw_ax)
    d.polygon([(x1 + ah * 0.3, y0), (x1 - ah, y0 - ah * 0.65), (x1 - ah, y0 + ah * 0.65)], fill="white")
    d.polygon([(x0, y1 - ah * 0.3), (x0 - ah * 0.65, y1 + ah), (x0 + ah * 0.65, y1 + ah)], fill="white")
    img.putalpha(Image.fromarray(np.minimum(np.array(img)[:, :, 3], np.array(mask))))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    render(512).save(os.path.join(HERE, "xyd.png"))
    sizes = [256, 128, 64, 48, 32, 24, 16]
    imgs = [render(z) for z in sizes]
    imgs[0].save(os.path.join(HERE, "xyd.ico"), sizes=[(z, z) for z in sizes], append_images=imgs[1:])
    sheet = Image.new("RGBA", (sum(sizes) + 10 * len(sizes), 256), (40, 40, 40, 255))
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 256 - im.size[1]), im)
        x += im.size[0] + 10
    sheet.save(os.path.join(HERE, "sizes_preview.png"))
    print("written", os.listdir(HERE))
