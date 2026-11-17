"""检查候选：沿轨迹叠加出小图，并分前后两半分别算 S。

用法：uv run python scripts/stamps.py data/w30 <输出 png> x0,y0,vx,vy[,标签] ...
真天体：叠加小图中心是一个与星像同形的点；前后两半的 S 都显著且相近（各约为总 S 的 1/√2）。
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
m = json.load(open(f"{d}/meta.json"))
cube = np.memmap(f"{d}/cube.f32", dtype="<f4", mode="r", shape=(m["n"], m["h"], m["w"]))
t, a = np.array(m["t_hours"]), np.array(m["a"])
R = 15
specs = [s.split(",") for s in sys.argv[3:]]
fig, axes = plt.subplots(1, len(specs), figsize=(2.2 * len(specs), 2.6))
axes = np.atleast_1d(axes)
for ax, s in zip(axes, specs):
    x0, y0, vx, vy = map(float, s[:4])
    label = s[4] if len(s) > 4 else f"({x0:.0f},{y0:.0f})"
    stamp = np.zeros((2 * R + 1, 2 * R + 1)); wsum = 0.0
    S = {"全部": [0.0, 0.0], "前半": [0.0, 0.0], "后半": [0.0, 0.0]}
    for i in range(m["n"]):
        x, y = int(round(x0 + vx * t[i])), int(round(y0 + vy * t[i]))
        if not (R <= x < m["w"] - R and R <= y < m["h"] - R):
            continue
        cut = np.asarray(cube[i, y - R:y + R + 1, x - R:x + R + 1])
        stamp += a[i] * cut; wsum += a[i] ** 2
        for key in ["全部", "前半" if i < m["n"] // 2 else "后半"]:
            S[key][0] += a[i] * cut[R, R]; S[key][1] += a[i] ** 2
    stamp /= np.sqrt(wsum)
    val = {k: (v[0] / np.sqrt(v[1]) if v[1] > 0 else 0) for k, v in S.items()}
    ax.imshow(stamp, origin="lower", cmap="gray", vmin=-3, vmax=max(8, stamp.max() * 0.8))
    ax.set_title(f"{label}\nS={val['全部']:.1f}  前{val['前半']:.1f} 后{val['后半']:.1f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{label:>14}  S = {val['全部']:6.1f}   前半 {val['前半']:6.1f}   后半 {val['后半']:6.1f}")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
