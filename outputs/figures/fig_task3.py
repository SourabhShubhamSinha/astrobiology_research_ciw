import sys; S='/tmp/claude-0/-home-claude/e04593b0-15bb-5f01-ba02-58f1666a65ad/scratchpad'; sys.path.insert(0, S+'/proj')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils.directions import fit_baseline_gmm, deviation_vectors, group_directions, cosine_matrix
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False})
INK, MUTED = "#0b0b0b", "#52514e"
Lf = pd.read_pickle(S+'/Lf.pkl'); labels = pd.read_pickle(S+'/lab.pkl')
FE = ["lambda0", "lambda1", "lambda2", "Ce_dev", "Eu_dev", "Y_dev"]
Fd = Lf[FE].replace([np.inf, -np.inf], np.nan).dropna(); lab = labels.loc[Fd.index]
is_base = lab.dataset.eq("GEOROC_apatite") & lab.origin_class.str.startswith("abiotic")
mu, sd = Fd[is_base].mean(), Fd[is_base].std(); Fs = (Fd - mu) / sd
g = pd.Series(pd.NA, index=Fs.index, dtype="object")
g[lab.dataset.eq("OSullivan2019") & lab.origin_class.str.startswith("abiotic")] = "Held-out abiotic"
g[lab.rock_class.isin(["phosphorite"])] = "Phosphorites"
g[lab.dataset.eq("Lumiste2019_recent")] = "Recent marine grains"
g[lab.dataset.eq("Huck2017_fishteeth")] = "Fossil fish teeth"
gmm = fit_baseline_gmm(Fs[is_base]); print("k", gmm.n_components)
z, info = deviation_vectors(Fs, gmm)
keep = g.notna()
dirn, st = group_directions(z[keep], g[keep])
ORDER = ["Held-out abiotic", "Phosphorites", "Recent marine grains", "Fossil fish teeth"]
dirn, st = dirn.loc[ORDER], st.loc[ORDER]
C = cosine_matrix(dirn); print(st.round(2)); print(C.round(2))
COL = {"Held-out abiotic": "#7a7974", "Phosphorites": "#2a78d6", "Recent marine grains": "#1baf7a", "Fossil fish teeth": "#eb6834"}

fig = plt.figure(figsize=(6.5, 2.75))
a = fig.add_axes([0.075, 0.15, 0.40, 0.78]); b = fig.add_axes([0.66, 0.15, 0.30, 0.78])
ic, iy = FE.index("Ce_dev"), FE.index("Y_dev")
B = Fs[is_base]
a.scatter(B.Ce_dev, B.Y_dev, s=2, c="#d4d3cd", lw=0, zorder=1, label="Abiotic ref.")
print("weights", gmm.weights_.round(2))
for k in range(gmm.n_components):
    m = gmm.means_[k][[ic, iy]]
    a.plot(*m, marker="+", color=MUTED, ms=5, mew=1, zorder=3)
    if gmm.weights_[k] < 0.1:
        continue
    S2 = gmm.covariances_[k][np.ix_([ic, iy], [ic, iy])]
    w, v = np.linalg.eigh(S2); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    a.add_patch(Ellipse(m, 2*2.45*np.sqrt(w[1]), 2*2.45*np.sqrt(w[0]), angle=ang, fill=False,
                        ec="#9a9993", lw=0.7, zorder=2))
rng = np.random.default_rng(1)
for grp in ["Phosphorites", "Recent marine grains", "Fossil fish teeth"]:
    idx = g.index[g.eq(grp)]
    idx = rng.choice(idx, size=min(14, len(idx)), replace=False)
    for i in idx:
        k = info.loc[i, "component"]; s0 = gmm.means_[k][[ic, iy]]; s1 = Fs.loc[i, ["Ce_dev", "Y_dev"]].to_numpy()
        a.annotate("", xy=s1, xytext=s0, zorder=4,
                   arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=COL[grp], lw=0.8, alpha=0.75,
                                   shrinkA=0, shrinkB=0))
    a.plot([], [], color=COL[grp], lw=1.5, label={"Recent marine grains": "Marine grains", "Fossil fish teeth": "Fish teeth"}.get(grp, grp))
a.set_xlim(-4.2, 3.2); a.set_ylim(-3.2, 5.2)
a.set_xlabel("Ce anomaly (SD of abiotic reference)"); a.set_ylabel("Y anomaly (SD of abiotic reference)")
a.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.2, borderaxespad=0.1, labelspacing=0.3, markerscale=3)
a.text(-0.17, 1.02, "a", transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom")

# panel b: coherence (diagonal) and cosine similarity (off-diagonal)
n = len(ORDER); M = C.to_numpy().copy()
for i in range(n): M[i, i] = st.coherence.iloc[i]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8312f"])
mask = np.triu(np.ones_like(M, bool), 1)
b.imshow(np.ma.array(M, mask=mask), cmap=cmap, vmin=-1, vmax=1)
for i in range(n):
    for j in range(i + 1):
        v = M[i, j]
        b.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
               color="white" if abs(v) > 0.6 else INK, fontweight="bold" if i == j else "normal")
        if i == j:
            b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
short = ["Held-out\nabiotic", "Phosphorites", "Marine\ngrains", "Fish\nteeth"]
b.set_xticks(range(n)); b.set_xticklabels(["Held-out\nabiotic", "Phos-\nphorites", "Marine\ngrains", "Fish\nteeth"], fontsize=8, rotation=0)
b.set_yticks(range(n)); b.set_yticklabels(short, fontsize=8)
b.tick_params(length=0); [s.set_visible(False) for s in b.spines.values()]
b.set_xlim(-0.5, n - 0.5); b.set_ylim(n - 0.5, -0.5)
b.text(-0.62, 1.02, "b", transform=b.transAxes, fontsize=11, fontweight="bold", va="bottom")
for ext in ["png", "pdf", "svg"]:
    fig.savefig(S+f'/task3/fig4_deviation_vectors.{ext}', dpi=300)
