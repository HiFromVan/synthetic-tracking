"""下载一晚某块芯片的全部曝光（NOIRLab Astro Data Archive，公开数据）。

用法：uv run python scripts/fetch.py data/night.json <hdu> data/frames_<hdu>
每张存成单独的 FITS：0 号 HDU 为原主头（含曝光时刻），1 号为该芯片图像（含 WCS）。
"""

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from astropy.io import fits

URL = "https://astroarchive.noirlab.edu/api/retrieve/{md5}/"


def fetch(i: int, row: dict, hdu: int, out: Path) -> str:
    dst = out / f"{i:03d}.fits"
    if dst.exists():
        return f"{i:03d} 已存在"
    for attempt in range(4):
        try:
            r = requests.get(URL.format(md5=row["md5sum"]), params={"hdus": f"0,{hdu}"}, timeout=300)
            r.raise_for_status()
            h = fits.open(io.BytesIO(r.content))
            img = fits.ImageHDU(data=h[1].data.astype("float32"), header=h[1].header)
            tmp = dst.with_suffix(".tmp")
            fits.HDUList([fits.PrimaryHDU(header=h[0].header), img]).writeto(tmp, overwrite=True)
            tmp.rename(dst)
            return f"{i:03d} {h[1].header.get('DETPOS')} {row['dateobs_center']}"
        except Exception as e:  # 网络抖动重试
            err = e
    return f"{i:03d} 失败：{err}"


def main() -> None:
    night = json.load(open(sys.argv[1]))
    hdu = int(sys.argv[2])
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    only = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else range(len(night))
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(lambda i: fetch(i, night[i], hdu, out), only):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
