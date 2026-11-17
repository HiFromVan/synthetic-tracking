"""把一段曝光变成可以直接叠加的信噪比图。

  对齐：按各帧 WCS，把每帧双线性重采样到参考帧的像素网格上。
  去静止源：减去天光，再减去所有帧的逐像素中位数（静止天空模板）；模板里的亮源连同周围几格一起遮掉。
  匹配滤波：用该帧的高斯星像 P（按头文件 FWHM，总和归一化为 1）卷积，
      ψᵢ = (Dᵢ ⊗ Pᵢ) / (σᵢ √ΣPᵢ²)，
  使纯噪声下 ψᵢ 的方差约为 1（重采样使像素噪声相关，再按 ψ 的实测离散度归一一次）。
  星等为 m 的点源在 ψᵢ 中的期望正比于 10^(−0.4m)·aᵢ，aᵢ 含星像、噪声和该帧的光度零点（透明度）。
  叠加时取 S = Σ aᵢ ψᵢ / √Σ aᵢ²，是独立高斯噪声下的最优线性组合，纯噪声下 S ~ N(0, 1)。

用法：uv run python scripts/prep.py data/frames_30 <起始帧> <帧数> <参考帧> data/w30
输出：cube.f32（N×H×W，小端 float32），meta.json（尺寸、各帧相对参考时刻的小时数、aᵢ），ref_header.txt
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy import ndimage

STEP = 64  # 稀疏网格间距：在网格点上精确算 WCS 映射，其余插值（畸变很平滑）


def align(data, wcs_from, wcs_ref, shape):
    ny, nx = shape
    gy = np.r_[np.arange(0, ny, STEP), ny - 1]
    gx = np.r_[np.arange(0, nx, STEP), nx - 1]
    GX, GY = np.meshgrid(gx, gy)
    sky = wcs_ref.pixel_to_world(GX.ravel(), GY.ravel())
    fx, fy = wcs_from.world_to_pixel(sky)
    fx = fx.reshape(GX.shape)
    fy = fy.reshape(GX.shape)
    # 稀疏网格 → 全分辨率坐标（双线性）
    yy = np.interp(np.arange(ny), gy, np.arange(len(gy)))
    xx = np.interp(np.arange(nx), gx, np.arange(len(gx)))
    YY, XX = np.meshgrid(yy, xx, indexing="ij")
    mx = ndimage.map_coordinates(fx, [YY, XX], order=1)
    my = ndimage.map_coordinates(fy, [YY, XX], order=1)
    return ndimage.map_coordinates(data, [my, mx], order=1, cval=np.nan)


def main():
    src, first, count, ref, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{src}/*.fits"))
    idx = list(range(first, first + count))
    href = fits.getheader(files[ref], 1)
    wref = WCS(href)
    shape = fits.getdata(files[ref], 1).shape

    def mid_hours(f):
        h0 = fits.getheader(f, 0)
        return Time(h0["MJD-OBS"], format="mjd").mjd * 24 + h0["EXPTIME"] / 7200

    t0 = mid_hours(files[ref])
    frames, times, fwhm, zp = [], [], [], []
    for i in idx:
        d = fits.getdata(files[i], 1).astype(np.float64)
        h = fits.getheader(files[i], 1)
        zp.append(float(fits.getheader(files[i], 0)["MAGZERO"]))
        sat_mask = d >= 0.9 * h.get("SATURATA", np.inf)
        d[sat_mask] = np.nan
        frames.append(align(d, WCS(h), wref, shape).astype(np.float32))
        times.append(mid_hours(files[i]) - t0)
        fwhm.append(float(h["FWHM"]))
        print(f"对齐 {i:03d}  t = {times[-1]:+.3f} h  FWHM = {fwhm[-1]:.2f} px", flush=True)

    cube = np.stack(frames)
    del frames
    for k in range(len(cube)):
        cube[k] -= np.nanmedian(cube[k])
    template = np.nanmedian(cube, axis=0)
    tm = np.nanmedian(template)
    tsig = np.nanmedian(np.abs(template - tm)) * 1.4826
    static = ~np.isfinite(template) | (template - tm > 3 * tsig)
    static = ndimage.binary_dilation(static, iterations=4)
    print(f"静止源遮罩：{static.mean() * 100:.1f}% 的像素", flush=True)

    EDGE = 16  # 芯片边缘和重采样边界附近伪影多，向内遮掉 EDGE 像素
    a = []
    for k in range(len(cube)):
        diff = cube[k] - template
        outside = ~np.isfinite(cube[k])
        outside[:EDGE, :] = outside[-EDGE:, :] = True
        outside[:, :EDGE] = outside[:, -EDGE:] = True
        bad = static | ~np.isfinite(diff) | ndimage.binary_dilation(outside, iterations=EDGE)
        diff[bad] = 0.0
        s = fwhm[k] / 2.3548
        r = int(np.ceil(4 * s))
        g = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        P = np.outer(g, g)
        P /= P.sum()
        conv = ndimage.convolve(diff, P, mode="constant")
        sigma = np.median(np.abs(diff[~bad])) * 1.4826
        norm = np.sqrt((P**2).sum())
        psi = conv / (sigma * norm)
        # 用实测分布再归一一次：卷积后像素相关，但单像素方差仍应为 1
        psi_sig = np.median(np.abs(psi[~bad])) * 1.4826
        psi /= psi_sig
        psi[bad] = 0.0
        cube[k] = psi.astype(np.float32)
        # 点源在 ψ 中的期望 = 计数 · norm / (σ · psi_sig)，所以权重也要除以 psi_sig。
        # 计数 = 10^(0.4 (MAGZEROᵢ − m))：同一天体在有云的帧里计数少，权重按零点缩放（这一晚零点相差近 3 等）。
        a.append(float(norm / (sigma * psi_sig) * 10 ** (0.4 * (zp[k] - 25.0))))
        print(f"滤波 {idx[k]:03d}  σ = {sigma:.2f}  ψ 实测 σ 修正 {psi_sig:.3f}  a = {a[-1]:.5f}", flush=True)

    cube.astype("<f4").tofile(out / "cube.f32")
    meta = {
        "n": len(idx), "h": shape[0], "w": shape[1],
        "frames": idx, "ref": ref, "t_hours": times, "a": a, "fwhm": fwhm, "magzero": zp,
        "static_fraction": float(static.mean()),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    (out / "ref_header.txt").write_text(href.tostring(sep="\n"))
    print("写出", out)


if __name__ == "__main__":
    main()
