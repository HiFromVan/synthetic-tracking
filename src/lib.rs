//! 移位叠加的核心。
//!
//! 输入是 N 张已对齐、已减去静止天空、已做匹配滤波的信噪比图 ψᵢ（纯噪声下方差约 1），
//! 以及各帧相对参考时刻的时间 tᵢ（小时）和权重 aᵢ（点源在 ψᵢ 中的期望 = 流量 · aᵢ）。
//! 对候选轨迹 θ = (x₀, y₀, vx, vy)，
//!     S(θ) = Σ aᵢ ψᵢ(x₀ + vx·tᵢ, y₀ + vy·tᵢ) / √Σ aᵢ²，
//! 纯噪声下 S ~ N(0, 1)，真实点源的 S 约为单帧信噪比的 √N 倍。

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
pub struct Meta {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub t_hours: Vec<f64>,
    pub a: Vec<f64>,
}

pub struct Cube {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub t: Vec<f32>,
    pub a: Vec<f32>,
    pub data: Vec<f32>,
}

impl Cube {
    /// 读入 prep.py 的输出。`clip` 把每帧 ψ 截断到 [-clip, clip]，压住单帧的亮点（宇宙线、残余星像）。
    /// `negate` 把 ψ 取反：噪声对称、真实天体只有正信号，取反后搜到的全是误报。
    pub fn load(dir: &Path, clip: Option<f32>, negate: bool) -> Cube {
        let meta: Meta = serde_json::from_reader(std::fs::File::open(dir.join("meta.json")).unwrap()).unwrap();
        let bytes = std::fs::read(dir.join("cube.f32")).unwrap();
        assert_eq!(bytes.len(), meta.n * meta.h * meta.w * 4, "cube.f32 大小不符");
        let mut data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let sign = if negate { -1.0 } else { 1.0 };
        data.par_iter_mut().for_each(|v| {
            let x = *v * sign;
            *v = match clip {
                Some(c) => x.clamp(-c, c),
                None => x,
            };
        });
        Cube {
            n: meta.n,
            h: meta.h,
            w: meta.w,
            t: meta.t_hours.iter().map(|&x| x as f32).collect(),
            a: meta.a.iter().map(|&x| x as f32).collect(),
            data,
        }
    }

    fn frame(&self, i: usize) -> &[f32] {
        &self.data[i * self.h * self.w..(i + 1) * self.h * self.w]
    }

    fn bilinear(&self, i: usize, x: f64, y: f64) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (self.w - 1) as f64 || y > (self.h - 1) as f64 {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let f = self.frame(i);
        let g = |xx: usize, yy: usize| f[yy * self.w + xx];
        Some(g(x0, y0) * (1.0 - fx) * (1.0 - fy) + g(x1, y0) * fx * (1.0 - fy) + g(x0, y1) * (1.0 - fx) * fy + g(x1, y1) * fx * fy)
    }

    /// 沿给定的逐帧位置叠加（位置在图外的帧不计入）。返回 (S, 用到的帧数)。
    pub fn stack_track(&self, xs: &[f64], ys: &[f64]) -> (f32, usize) {
        let (mut num, mut den, mut used) = (0f32, 0f32, 0);
        for i in 0..self.n {
            if let Some(v) = self.bilinear(i, xs[i], ys[i]) {
                num += self.a[i] * v;
                den += self.a[i] * self.a[i];
                used += 1;
            }
        }
        (if den > 0.0 { num / den.sqrt() } else { 0.0 }, used)
    }
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct Candidate {
    pub x0: f32,
    pub y0: f32,
    pub vx: f32,
    pub vy: f32,
    pub s: f32,
}

/// 速度网格：vmin ≤ |v| ≤ vmax，步长 dv（像素/小时），所有方向。
pub fn velocity_grid(vmin: f32, vmax: f32, dv: f32) -> Vec<(f32, f32)> {
    let k = (vmax / dv).ceil() as i32;
    let mut out = Vec::new();
    for i in -k..=k {
        for j in -k..=k {
            let (vx, vy) = (i as f32 * dv, j as f32 * dv);
            let r = (vx * vx + vy * vy).sqrt();
            if r >= vmin && r <= vmax {
                out.push((vx, vy));
            }
        }
    }
    out
}

/// 对每个参考像素 (x₀, y₀) 求 max over v 的 S，以及取到最大值的速度编号。
/// 每帧的位移取整到像素（网格步长保证窗口两端的位置误差不超过约一个像素）。
pub fn search(cube: &Cube, vels: &[(f32, f32)], band: usize) -> (Vec<f32>, Vec<u32>) {
    let (n, h, w) = (cube.n, cube.h, cube.w);
    let norm = cube.a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let wts: Vec<f32> = cube.a.iter().map(|a| a / norm).collect();
    let mut best = vec![f32::NEG_INFINITY; h * w];
    let mut bestv = vec![0u32; h * w];
    best.par_chunks_mut(band * w).zip(bestv.par_chunks_mut(band * w)).enumerate().for_each(|(bi, (bb, bv))| {
        let y_lo = bi * band;
        let rows = bb.len() / w;
        let mut acc = vec![0f32; rows * w];
        for (vi, &(vx, vy)) in vels.iter().enumerate() {
            acc.fill(0.0);
            for i in 0..n {
                let dx = (vx * cube.t[i]).round() as isize;
                let dy = (vy * cube.t[i]).round() as isize;
                let wi = wts[i];
                let xs = (-dx).max(0) as usize;
                let xe = (w as isize - dx).min(w as isize).max(0) as usize;
                if xs >= xe {
                    continue;
                }
                let frame = cube.frame(i);
                for r in 0..rows {
                    let sy = (y_lo + r) as isize + dy;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    let src_row = &frame[sy as usize * w..(sy as usize + 1) * w];
                    let src = &src_row[(xs as isize + dx) as usize..(xe as isize + dx) as usize];
                    let dst = &mut acc[r * w + xs..r * w + xe];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += wi * s;
                    }
                }
            }
            for k in 0..rows * w {
                if acc[k] > bb[k] {
                    bb[k] = acc[k];
                    bv[k] = vi as u32;
                }
            }
        }
    });
    (best, bestv)
}

/// 阈值以上、且在半径 r 的方形邻域内是最大值的像素。
pub fn peaks(best: &[f32], bestv: &[u32], vels: &[(f32, f32)], h: usize, w: usize, tau: f32, r: usize) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..w).filter_map(move |x| {
                let s = best[y * w + x];
                if s < tau {
                    return None;
                }
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                        let o = best[yy * w + xx];
                        if o > s || (o == s && (yy, xx) < (y, x)) {
                            return None;
                        }
                    }
                }
                let (vx, vy) = vels[bestv[y * w + x] as usize];
                Some(Candidate { x0: x as f32, y0: y as f32, vx, vy, s })
            })
        })
        .collect();
    out.sort_by(|a, b| b.s.partial_cmp(&a.s).unwrap());
    out
}
