"""把盲搜候选和已知天体对照，并用取反数据标定误报。

用法：uv run python scripts/evaluate.py data/w30 cands.json cands_neg.json
匹配条件：参考时刻位置相差 ≤ 6 像素，速度相差 ≤ 8 像素/小时。
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1])
    known = json.load(open(d / "known.json"))
    cands = json.load(open(d / sys.argv[2]))
    neg = json.load(open(d / sys.argv[3]))
    neg_max = max((c["s"] for c in neg), default=0.0)
    print(f"候选 {len(cands)} 个；取反数据候选 {len(neg)} 个，其中最大 S = {neg_max:.1f}")
    for tau in [6, 7, 8, 9, 10]:
        print(f"  阈值 {tau:>2}：正向 {sum(c['s'] >= tau for c in cands):4d} 个，取反（全为误报）{sum(c['s'] >= tau for c in neg):4d} 个")

    used = set()
    print("\n已知天体：")
    for k in known:
        best = None
        for j, c in enumerate(cands):
            dp = np.hypot(c["x0"] - k["x0"], c["y0"] - k["y0"])
            dv = np.hypot(c["vx"] - k["vx"], c["vy"] - k["vy"])
            if dp <= 6 and dv <= 8 and (best is None or c["s"] > cands[best]["s"]):
                best = j
        if best is None:
            print(f"  ✗ {k['name']:>12}  V = {k['V']:5.2f}  未找回（帧内 {k['frames_inside']}）")
        else:
            used.add(best)
            c = cands[best]
            print(f"  ✓ {k['name']:>12}  V = {k['V']:5.2f}  S = {c['s']:5.1f}  "
                  f"位置差 {np.hypot(c['x0'] - k['x0'], c['y0'] - k['y0']):.1f} px  速度差 {np.hypot(c['vx'] - k['vx'], c['vy'] - k['vy']):.1f} px/h")

    print(f"\n高于取反最大值（S > {neg_max:.1f}）且不是已知天体的候选：")
    for j, c in enumerate(cands):
        if j not in used and c["s"] > neg_max:
            print(f"  S = {c['s']:5.1f}  (x0, y0) = ({c['x0']:6.0f}, {c['y0']:6.0f})  v = ({c['vx']:+5.0f}, {c['vy']:+5.0f}) px/h")


if __name__ == "__main__":
    main()
