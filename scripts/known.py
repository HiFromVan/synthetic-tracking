"""已知天体在参考帧像素网格上的轨迹（SkyBoT，IMCCE）。

在窗口首、中、尾三个时刻查询，对每个天体的 RA、Dec 随时间做二次拟合，求出每帧时刻的位置，
再用参考帧 WCS 换成像素坐标。

SkyBoT 经 astroquery 查询时，观测站参数实测不起作用，返回的是地心位置（807 与 500 只差 0.02″）。
主带小行星在 1.5 AU 左右，地心与 CTIO 的视差可达约 2″（本例约 7 像素），且随高度角变化。
所以这里按地心查询（500），再自己减去 CTIO 的地心位置矢量，得到站心方向。

用法：uv run python scripts/known.py data/w30
输出：known.json，每个天体含名称、类型、星等 V、各帧像素坐标 (x, y)、参考时刻的速度（像素/小时）。
"""

import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.imcce import Skybot


def val(x):
    return float(getattr(x, "value", x))


def main():
    out = Path(sys.argv[1])
    meta = json.load(open(out / "meta.json"))
    href = fits.Header.fromstring((out / "ref_header.txt").read_text(), sep="\n")
    w = WCS(href)
    h, wd = meta["h"], meta["w"]
    center = w.pixel_to_world(wd / 2, h / 2)
    t_ref = Time(fits.getheader(f"data/frames_30/{meta['ref']:03d}.fits", 0)["MJD-OBS"], format="mjd")
    t_ref = t_ref + fits.getheader(f"data/frames_30/{meta['ref']:03d}.fits", 0)["EXPTIME"] / 2 * u.s
    ts = np.array(meta["t_hours"])
    probes = [ts.min(), 0.0, ts.max()]

    ctio = EarthLocation.from_geodetic(-70.80653 * u.deg, -30.16963 * u.deg, 2207 * u.m)  # Blanco 4 m
    tracks = {}
    for tp in probes:
        t = t_ref + tp * u.hour
        res = Skybot.cone_search(center, 0.3 * u.deg, t, location="500")
        obs_pos, _ = ctio.get_gcrs_posvel(t)
        for r in res:
            key = str(r["Name"])
            geo = SkyCoord(val(r["RA"]) * u.deg, val(r["DEC"]) * u.deg, distance=val(r["geodist"]) * u.au).cartesian
            topo = SkyCoord(CartesianRepresentation(geo.xyz - obs_pos.xyz), frame="icrs")
            d = tracks.setdefault(key, {"name": key, "number": str(r["Number"]), "type": str(r["Type"]), "V": val(r["V"]), "obs": []})
            d["obs"].append((tp, topo.ra.deg, topo.dec.deg))

    known = []
    for d in tracks.values():
        if len(d["obs"]) < 3:
            continue
        o = np.array(d["obs"])
        cra = np.polyfit(o[:, 0], o[:, 1], 2)
        cde = np.polyfit(o[:, 0], o[:, 2], 2)
        sky = SkyCoord(np.polyval(cra, ts) * u.deg, np.polyval(cde, ts) * u.deg)
        x, y = w.world_to_pixel(sky)
        inside = (x >= 0) & (x < wd) & (y >= 0) & (y < h)
        if inside.sum() < len(ts) // 2:
            continue
        x0, y0 = w.world_to_pixel(SkyCoord(np.polyval(cra, 0) * u.deg, np.polyval(cde, 0) * u.deg))
        vx, vy = np.polyfit(ts, x, 1)[0], np.polyfit(ts, y, 1)[0]
        known.append({
            "name": d["name"], "number": d["number"], "type": d["type"], "V": d["V"],
            "x0": float(x0), "y0": float(y0), "vx": float(vx), "vy": float(vy),
            "x": x.tolist(), "y": y.tolist(), "frames_inside": int(inside.sum()),
        })
    known.sort(key=lambda k: k["V"])
    json.dump(known, open(out / "known.json", "w"), indent=1)
    for k in known:
        print(f"{k['name']:>14}  {k['type']:<12} V = {k['V']:5.2f}  (x0, y0) = ({k['x0']:7.1f}, {k['y0']:7.1f})  "
              f"v = ({k['vx']:+7.1f}, {k['vy']:+7.1f}) px/h  帧内 {k['frames_inside']}/{len(ts)}")


if __name__ == "__main__":
    main()
