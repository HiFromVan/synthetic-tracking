//! 合成数据：纯高斯噪声中注入一颗单帧信噪比为 2 的移动点源（单帧不可见），检验盲搜能否找回。

use st::{Cube, peaks, search, velocity_grid};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.next().max(1e-12), self.next());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

const N: usize = 25;
const H: usize = 160;
const W: usize = 128;
const SOURCE: (f32, f32, f32, f32) = (64.0, 80.0, 60.0, -45.0); // x₀, y₀, vx, vy（像素/小时）

fn cube(inject: bool, seed: u64) -> Cube {
    let mut rng = Rng(seed);
    let t: Vec<f32> = (0..N).map(|i| -0.5 + i as f32 / (N - 1) as f32).collect();
    let mut data: Vec<f32> = (0..N * H * W).map(|_| rng.gauss()).collect();
    if inject {
        let (x0, y0, vx, vy) = SOURCE;
        for i in 0..N {
            let (cx, cy) = (x0 + vx * t[i], y0 + vy * t[i]);
            for y in 0..H {
                for x in 0..W {
                    let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    data[i * H * W + y * W + x] += 2.0 * (-r2 / (2.0 * 1.5 * 1.5)).exp();
                }
            }
        }
    }
    Cube { n: N, h: H, w: W, t, a: vec![1.0; N], data }
}

#[test]
fn recovers_injected_source() {
    let vels = velocity_grid(30.0, 100.0, 3.0);
    let c = cube(true, 7);
    let (best, bestv) = search(&c, &vels, 8);
    let top = peaks(&best, &bestv, &vels, H, W, 0.0, 8)[0];
    let (x0, y0, vx, vy) = SOURCE;
    assert!((top.x0 - x0).abs() <= 1.0 && (top.y0 - y0).abs() <= 1.0, "位置 {top:?}");
    assert!((top.vx - vx).abs() <= 3.0 && (top.vy - vy).abs() <= 3.0, "速度 {top:?}");
    // 期望 S ≈ 2√N = 10
    assert!(top.s > 8.0, "{top:?}");

    let c0 = cube(false, 7);
    let (best0, _) = search(&c0, &vels, 8);
    let noise_max = best0.iter().cloned().fold(f32::MIN, f32::max);
    assert!(noise_max < top.s - 3.0, "噪声最大 {noise_max}，注入源 {}", top.s);
}

#[test]
fn stack_track_matches_expectation() {
    let c = cube(true, 11);
    let (x0, y0, vx, vy) = SOURCE;
    let xs: Vec<f64> = c.t.iter().map(|&t| (x0 + vx * t) as f64).collect();
    let ys: Vec<f64> = c.t.iter().map(|&t| (y0 + vy * t) as f64).collect();
    let (s, used) = c.stack_track(&xs, &ys);
    assert_eq!(used, N);
    assert!((s - 2.0 * (N as f32).sqrt()).abs() < 3.0, "S = {s}");
}

#[test]
fn velocity_grid_is_an_annulus() {
    let v = velocity_grid(40.0, 170.0, 3.0);
    assert!(v.iter().all(|&(x, y)| (40.0..=170.0).contains(&(x * x + y * y).sqrt())));
    assert_eq!(v.len(), 9544);
}
