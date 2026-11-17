"""生成 notebooks/explore.ipynb（改笔记本内容就改这里，再运行一次）。

用法：uv run python scripts/make_notebook.py
"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 看一看：沿轨迹叠加

数据：DEEP 巡天 2019-07-09，DECam 芯片 S6，第 45–74 张（`data/w30`，由 `scripts/prep.py` 生成）。

1. 整块芯片长什么样，已知小行星和候选 A、B 在哪
2. 亮的小行星：单帧里就能看到它在动
3. 暗的候选 A：单帧里几乎看不出，叠加后清楚出现
4. 拖速度滑块：速度对了，点才会浮出来
5. 所有盲搜候选：原数据 vs 取反数据
""")

code(r"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from IPython.display import HTML
import ipywidgets as W

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["animation.embed_limit"] = 50

D = "../data/w30"
m = json.load(open(f"{D}/meta.json"))
N, H, Wd = m["n"], m["h"], m["w"]
t, a = np.array(m["t_hours"]), np.array(m["a"])
cube = np.memmap(f"{D}/cube.f32", dtype="<f4", mode="r", shape=(N, H, Wd))
known = json.load(open(f"{D}/known.json"))
PIX = 0.263  # DECam 像素大小（角秒）

# 参考时刻（第 60 张中点）的位置和速度，单位像素、像素/小时
CANDS = {"A": (1589, 1227, 45, -99), "B": (481, 1590, 45, -93)}
print(f"{N} 帧，跨 {t.max() - t.min():.2f} 小时；每帧 {H}×{Wd} 像素")
""")

md(r"""
## 1. 整块芯片

参考帧的原始图像（zscale 拉伸：天文常用的对比度，把背景附近的灰度拉开）。
彩色线是已知小行星在这 1.2 小时里走过的路径，红圈是候选 A、B。
""")

code(r"""
raw = fits.getdata(f"../data/frames_30/{m['ref']:03d}.fits", 1).astype(np.float32)
lo, hi = ZScaleInterval().get_limits(raw)
fig, ax = plt.subplots(figsize=(6, 11))
ax.imshow(raw, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
for k in known:
    ax.plot(k["x"], k["y"], lw=2)
    ax.annotate(f"{k['name']}  V={k['V']:.1f}", (k["x"][0], k["y"][0]), color="yellow", fontsize=7)
for name, (x0, y0, vx, vy) in CANDS.items():
    ax.add_patch(plt.Circle((x0, y0), 40, fill=False, color="red", lw=1.5))
    ax.text(x0 + 45, y0, name, color="red", fontsize=12)
ax.set_title("芯片 S6，第 60 张")
plt.show()
""")

def blink_cell(title, note, target):
    md(f"## {title}\n\n{note}")
    code(f"""
def blink(x0, y0, vx, vy, R=60, title=""):
    # 不跟踪：固定在参考位置附近，逐帧播放；天体自己走过画面
    frames = [np.asarray(cube[i, y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]) for i in range(N)]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(frames[0], origin="lower", cmap="gray", vmin=-3, vmax=6)
    ax.plot(R + vx * t, R + vy * t, "r:", lw=0.8)  # 预计路径
    ttl = ax.set_title("")
    def upd(i):
        im.set_data(frames[i]); ttl.set_text(f"{{title}}  第 {{i + 1}}/{{N}} 帧  t = {{t[i]:+.2f}} h")
        return im, ttl
    anim = animation.FuncAnimation(fig, upd, frames=N, interval=150)
    plt.close(fig)
    return HTML(anim.to_jshtml())

{target}
""")

blink_cell("2. 亮的小行星：单帧里就能看到",
           "1998 BK21，V = 20.2。画面是去掉静止天空、做过匹配滤波的单帧（单位：噪声标准差）。红色虚线是它的预计路径。",
           'k = next(k for k in known if k["name"] == "1998 BK21")\nblink(int(k["x0"]), int(k["y0"]), k["vx"], k["vy"], title=k["name"])')

blink_cell("3. 候选 A：单帧里几乎看不出",
           "A 估计 22.4 等，平均单帧信噪比约 3：透明度好的几帧里有 6–7σ 的点，但这样的点满屏都是，单看一帧分不出哪个是真的。沿红线找找看。",
           'x0, y0, vx, vy = CANDS["A"]\nblink(x0, y0, vx, vy, title="候选 A")')

md(r"""
### 同样 30 帧，沿 A 的速度叠起来

左边是 30 张单帧小图（已按 A 的速度对齐），右边是它们的加权叠加 $S = \sum a_i \psi_i / \sqrt{\sum a_i^2}$。
""")

code(r"""
def cutouts(x0, y0, vx, vy, R=15):
    out = []
    for i in range(N):
        x, y = int(round(x0 + vx * t[i])), int(round(y0 + vy * t[i]))
        out.append(np.asarray(cube[i, y - R:y + R + 1, x - R:x + R + 1]))
    return np.array(out)

c = cutouts(*CANDS["A"])
stack = np.tensordot(a, c, axes=1) / np.sqrt((a ** 2).sum())
fig = plt.figure(figsize=(13, 5))
for i in range(N):
    ax = fig.add_subplot(5, 12, (i // 6) * 12 + i % 6 + 1)
    ax.imshow(c[i], origin="lower", cmap="gray", vmin=-3, vmax=6); ax.set_axis_off()
ax = fig.add_subplot(1, 2, 2)
ax.imshow(stack, origin="lower", cmap="gray", vmin=-3, vmax=max(8, stack.max() * 0.8))
ax.set_title(f"叠加：中心 S = {stack[15, 15]:.1f}"); ax.set_axis_off()
plt.show()
""")

md(r"""
## 4. 拖速度滑块

固定 A 在参考时刻的位置，按滑块的速度叠加 30 帧。速度偏了，各帧的点叠不到一起，只剩噪声；
拖回 A 的速度（45, −99 像素/时），点就浮出来。也可以换成 B 或任一已知小行星。

（需要在 VS Code 里运行这一格；滑块在导出的静态页面里不能动。）
""")

code(r"""
R, M = 25, 120  # 显示半径；读入区域的额外边距（速度偏差 × 时间跨度）
targets = {**{f"候选 {k}": v for k, v in CANDS.items()},
           **{k["name"]: (int(round(k["x0"])), int(round(k["y0"])), k["vx"], k["vy"]) for k in known if k["frames_inside"] == N}}
cache = {}

def region(name):
    if name not in cache:
        x0, y0, _, _ = targets[name]
        ys, xs = slice(max(0, y0 - R - M), y0 + R + M + 1), slice(max(0, x0 - R - M), x0 + R + M + 1)
        cache[name] = (np.array(cube[:, ys, xs]), ys.start, xs.start)
    return cache[name]

def show(name, dvx, dvy):
    x0, y0, vx0, vy0 = targets[name]
    reg, oy, ox = region(name)
    vx, vy = vx0 + dvx, vy0 + dvy
    acc, w2 = np.zeros((2 * R + 1, 2 * R + 1)), 0.0
    for i in range(N):
        x, y = int(round(x0 + vx * t[i])) - ox, int(round(y0 + vy * t[i])) - oy
        if R <= x < reg.shape[2] - R and R <= y < reg.shape[1] - R:
            acc += a[i] * reg[i, y - R:y + R + 1, x - R:x + R + 1]; w2 += a[i] ** 2
    acc /= np.sqrt(max(w2, 1e-12))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(acc, origin="lower", cmap="gray", vmin=-3, vmax=10)
    sp = np.hypot(vx, vy) * PIX
    ax.set_title(f"{name}  v = ({vx:+.0f}, {vy:+.0f}) px/h = {sp:.1f}″/h\n中心 S = {acc[R, R]:.1f}")
    ax.set_axis_off(); plt.show()

W.interact(show,
           name=W.Dropdown(options=list(targets), value="候选 A", description="目标"),
           dvx=W.IntSlider(value=20, min=-60, max=60, step=1, description="Δvx"),
           dvy=W.IntSlider(value=-25, min=-60, max=60, step=1, description="Δvy"));
""")

md(r"""
## 5. 所有盲搜候选：原数据 vs 取反数据

盲搜在 9,544 个速度上找每个位置的最大 $S$。左：原数据；右：把每帧取反后再搜。
噪声正负对称，所以右边的点全是噪声造成的误报——它给出了"纯噪声最高能到多少"。
左边高出右边最高值的点，才可能是真东西。
""")

code(r"""
cands = json.load(open(f"{D}/cands.json"))
neg = json.load(open(f"{D}/cands_neg.json"))
nmax = max(c["s"] for c in neg)
fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True)
for ax, cs, title in [(axes[0], cands, "原数据"), (axes[1], neg, "取反数据（全为误报）")]:
    s = np.array([c["s"] for c in cs])
    sc = ax.scatter([c["x0"] for c in cs], [c["y0"] for c in cs], c=s, s=4 + 3 * (s - 5) ** 2,
                    cmap="inferno", vmin=5, vmax=18, edgecolor="k", lw=0.3)
    ax.set_xlim(0, Wd); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_facecolor("#ddd")
    ax.set_title(f"{title}：{len(cs)} 个，最大 S = {s.max():.1f}")
for k in known:
    axes[0].plot(k["x"], k["y"], "c-", lw=1)
for n, (x0, y0, *_) in CANDS.items():
    axes[0].annotate(n, (x0, y0), xytext=(x0 + 120, y0), color="red", arrowprops=dict(color="red", arrowstyle="->"))
fig.colorbar(sc, ax=axes, shrink=0.6, label="S")
plt.show()

s = np.array([c["s"] for c in cands]); sn = np.array([c["s"] for c in neg])
bins = np.arange(4, 21, 0.5)
plt.figure(figsize=(7, 3))
plt.hist(sn, bins, alpha=0.6, label="取反（噪声）"); plt.hist(s, bins, alpha=0.6, label="原数据")
plt.axvline(nmax, color="k", ls="--", lw=1); plt.yscale("log"); plt.xlabel("S"); plt.legend(); plt.show()
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/explore.ipynb")
print("写出 notebooks/explore.ipynb")
