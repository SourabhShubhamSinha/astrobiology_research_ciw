"""Compact method flowchart sized for a Word page.

Canvas is exactly 6.5 in wide (text width of a Letter page with 1-inch margins) and
1 data unit = 1 inch, so 12-pt text here stays 12 pt when inserted at full page width.
Every shape is sized from its measured text, so nothing overflows.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Rectangle, Ellipse

OUT = sys.argv[1] if len(sys.argv) > 1 else "method_flowchart_word"
FONT = "Liberation Sans"          # metrically identical to Arial
FS, FS_SMALL = 12, 11
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#52514e"
LW = 1.0
plt.rcParams.update({"font.family": FONT, "font.size": FS, "svg.fonttype": "none", "pdf.fonttype": 42})

W, H = 6.5, 5.3
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
R = fig.canvas.get_renderer()


def measure(s, **kw):
    t = ax.text(0, 0, s, fontsize=kw.get("fontsize", FS), fontweight=kw.get("fontweight", "normal"),
                linespacing=1.1)
    bb = t.get_window_extent(renderer=R)
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def text(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    kw.setdefault("fontsize", FS); kw.setdefault("color", INK)
    ax.text(x, y, s, linespacing=1.1, zorder=5, **kw)


def _geom(cx, cy, w, h):
    return dict(cx=cx, cy=cy, l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, w=w, h=h)


def box(cx, cy, s, rounded=False, fill="white", edge=EDGE, dashed=False, padx=0.12, pady=0.08, minw=0):
    tw, th = measure(s)
    w, h = max(tw + 2 * padx, minw), th + 2 * pady
    style = "round,pad=0,rounding_size=0.1" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, fc=fill, ec=edge,
                                lw=LW, ls=(0, (3, 2)) if dashed else "-", zorder=3))
    text(cx, cy, s)
    return _geom(cx, cy, w, h)


def para(cx, cy, s, fill="#f1f0ec"):
    tw, th = measure(s)
    h = th + 0.16; skew = 0.18; w = tw + 0.2 + 2 * skew
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2 - skew, cy - h / 2),
           (cx + w / 2, cy + h / 2), (cx - w / 2 + skew, cy + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fill, ec=EDGE, lw=LW, zorder=3))
    text(cx, cy, s)
    g = _geom(cx, cy, w, h); g["r"] -= skew / 2; g["l"] += skew / 2
    return g


def diamond(cx, cy, s):
    tw, th = measure(s)
    # text rectangle (tw x th) fits inside a diamond when tw/w + th/h <= 1; use w = 2*h
    h = (tw / 2 + th) * 1.12
    w = 2 * h
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc="white", ec=EDGE, lw=LW, zorder=3))
    text(cx, cy, s)
    return _geom(cx, cy, w, h)


def cylinder(cx, cy, s, fill="#e9e8e3"):
    tw, th = measure(s)
    w, e = tw + 0.3, 0.12
    h = th + 0.14 + 2 * e
    yb, yt = cy - h / 2 + e / 2, cy + h / 2 - e / 2
    ax.add_patch(Ellipse((cx, yb), w, e, fc=fill, ec=EDGE, lw=LW, zorder=3))
    ax.add_patch(Rectangle((cx - w / 2, yb), w, yt - yb, fc=fill, ec="none", zorder=3.2))
    ax.plot([cx - w / 2] * 2, [yb, yt], color=EDGE, lw=LW, zorder=3.3)
    ax.plot([cx + w / 2] * 2, [yb, yt], color=EDGE, lw=LW, zorder=3.3)
    ax.add_patch(Ellipse((cx, yt), w, e, fc=fill, ec=EDGE, lw=LW, zorder=3.4))
    text(cx, cy - e / 3, s)
    return _geom(cx, cy, w, h)


def line(pts, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=EDGE, lw=LW, ls=(0, (3, 2)) if dashed else "-", zorder=2, solid_capstyle="butt")


def arrow(pts, dashed=False):
    if len(pts) > 2:
        line(pts[:-1], dashed)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2", color=EDGE, lw=LW,
                                shrinkA=0, shrinkB=0, ls=(0, (3, 2)) if dashed else "-"))


def label(x, y, s, ha="left", va="center"):
    text(x, y, s, fontsize=FS_SMALL, color=MUTED, ha=ha, va=va)


def panel(x0, x1, y0, y1, title, color, fill, tcolor, tx):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fill, ec=color, lw=1.2, zorder=1))
    text(tx, y1 - 0.08, title, ha="left", va="top", fontweight="bold", color=tcolor)


# ================= row 1: shared steps =================
y1 = H - 0.4
data  = para(0.78, y1, "Public\napatite data")
clean = box(2.18, y1, "Compile\nand clean")
feat  = box(3.6, y1, "Features:\nREE anomalies;\nSr, Ba, Mg")
isab  = diamond(5.45, y1, "Abiotic?")
for a, b in [(data, clean), (clean, feat), (feat, isab)]:
    arrow([(a["r"], y1), (b["l"], y1)])

# ================= row 2: the two sample sets =================
cA, cB = 0.95, 5.2                       # column of Method 1 entry and Method 2 entry
y2 = y1 - 1.0
ref  = cylinder(cA, y2, "Abiotic\nreference")
test = box(cB, y2, "Test\nsamples", rounded=True, dashed=True)
ybus = (isab["b"] + max(ref["t"], test["t"])) / 2
line([(isab["cx"], isab["b"]), (isab["cx"], ybus), (cA, ybus)])
arrow([(cA, ybus), (cA, ref["t"])]);  label(cA + 0.07, (ybus + ref["t"]) / 2, "yes")
arrow([(cB, ybus), (cB, test["t"])]); label(cB + 0.07, (ybus + test["t"]) / 2, "no")

# ================= panels =================
top = y2 - 0.62
bot = 0.06
xs = 3.98                                # split between the two panels
panel(0.05, xs - 0.06, bot, top, "Method 1:\nGMM scoring", "#2a78d6", "#eef4fc", "#184f95", tx=cA + 0.14)
panel(xs + 0.06, 6.45, bot, top, "Method 2:\nKDE check", "#eb6834", "#fdf1eb", "#a8401a", tx=cB + 0.14)

# both sample sets feed both methods (shared connector above the panels)
ybus2 = top + 0.17
line([(cA, ref["b"]), (cA, ybus2)])
line([(cB, test["b"]), (cB, ybus2)], dashed=True)
line([(cA, ybus2), (cB, ybus2)])

# ================= Method 1 =================
ya = top - 0.88
gmm  = box(cA, ya, "Score vs.\nreference")
arrow([(cA, ybus2), (cA, gmm["t"])])
unus = diamond(2.35, ya, "Unusual?")
arrow([(gmm["r"], ya), (unus["l"], ya)])
yb = ya - 1.0
dirn = diamond(2.35, yb, "Known abiotic\ndirection?")
arrow([(unus["cx"], unus["b"]), (unus["cx"], dirn["t"])])
label(unus["cx"] + 0.07, (unus["b"] + dirn["t"]) / 2, "yes")
typ = box(0.72, yb, "Typical", rounded=True)
yc = bot + 0.3
expl = box(1.15, yc, "Explained", rounded=True)
unex = box(3.12, yc, "Unexplained", rounded=True, fill="#fde7dc", edge="#d95926")


def branch(src, dst, txt, side):
    """Arrow from a point on a diamond edge to the top of a result box, label beside its midpoint."""
    arrow([src, (dst["cx"], dst["t"])])
    mx, my = (src[0] + dst["cx"]) / 2, (src[1] + dst["t"]) / 2
    if side == "above":
        label(mx + 0.1, my - 0.14, txt, ha="left")
    elif side == "left":
        label(mx - 0.08, my, txt, ha="right")
    else:
        label(mx + 0.08, my, txt, ha="left")


branch((unus["cx"] - unus["w"] / 4, unus["cy"] - unus["h"] / 4), typ, "no", "above")
branch((dirn["cx"] - dirn["w"] / 4, dirn["cy"] - dirn["h"] / 4), expl, "yes", "left")
branch((dirn["cx"] + dirn["w"] / 4, dirn["cy"] - dirn["h"] / 4), unex, "no", "right")

# ================= Method 2 =================
kde = box(cB, ya, "Share of density\nfrom reference")
arrow([(cB, ybus2), (cB, kde["t"])])
low = diamond(cB, yb, "Below\nexpected?")
arrow([(cB, kde["b"]), (cB, low["t"])])
own   = box(4.65, yc, "Distinct\ngroup", rounded=True)
mixed = box(5.82, yc, "Mixed with\nreference", rounded=True)
branch((low["cx"] - low["w"] / 4, low["cy"] - low["h"] / 4), own, "yes", "left")
branch((low["cx"] + low["w"] / 4, low["cy"] - low["h"] / 4), mixed, "no", "right")

for ext, kw in [("png", dict(dpi=300)), ("pdf", {}), ("svg", {})]:
    fig.savefig(f"{OUT}.{ext}", **kw)
print("saved", OUT, (W, H))
for n, g in dict(data=data, clean=clean, feat=feat, isab=isab, ref=ref, test=test, gmm=gmm, unus=unus,
                 dirn=dirn, typ=typ, expl=expl, unex=unex, kde=kde, low=low, own=own, mixed=mixed).items():
    print(f"{n:6s} l={g['l']:.2f} r={g['r']:.2f} b={g['b']:.2f} t={g['t']:.2f}")
