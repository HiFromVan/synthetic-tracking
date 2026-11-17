use serde_json::Value;
use st::{Cube, peaks, search, velocity_grid};
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "\
用法：
  st track  <目录>                          沿 known.json 里已知天体的预测轨迹叠加，报告 S
  st search <目录> [选项]                   盲搜：在速度网格上叠加，输出候选
      --vmin <像素/时> --vmax <像素/时>     速度范围（默认 40、170）
      --dv <像素/时>                        网格步长（默认 3）
      --tau <S>                             候选阈值（默认 6）
      --clip <ψ>                            单帧截断（默认 5）
      --negate                              ψ 取反（误报标定）
      --out <文件>                          候选写成 JSON";

fn opt(args: &[String], name: &str, default: f32) -> f32 {
    args.iter().position(|a| a == name).map(|i| args[i + 1].parse().expect(name)).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("{USAGE}");
        return;
    }
    let dir = Path::new(&args[2]);
    match args[1].as_str() {
        "track" => {
            let cube = Cube::load(dir, None, false);
            let known: Vec<Value> = serde_json::from_reader(std::fs::File::open(dir.join("known.json")).unwrap()).unwrap();
            println!("{:>14} {:>6} {:>8} {:>6} {:>10}", "天体", "V", "S", "帧数", "S/√帧数");
            for k in &known {
                let xs: Vec<f64> = k["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                let ys: Vec<f64> = k["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                let (s, used) = cube.stack_track(&xs, &ys);
                println!("{:>14} {:>6.2} {:>8.1} {:>6} {:>10.2}", k["name"].as_str().unwrap(), k["V"].as_f64().unwrap(), s, used, s / (used as f32).sqrt());
            }
        }
        "search" => {
            let clip = opt(&args, "--clip", 5.0);
            let negate = args.iter().any(|a| a == "--negate");
            let t = Instant::now();
            let cube = Cube::load(dir, Some(clip), negate);
            let vels = velocity_grid(opt(&args, "--vmin", 40.0), opt(&args, "--vmax", 170.0), opt(&args, "--dv", 3.0));
            eprintln!("读入 {} 帧 {}×{}，{} 个速度，用时 {:.1?}", cube.n, cube.w, cube.h, vels.len(), t.elapsed());
            let t = Instant::now();
            let (best, bestv) = search(&cube, &vels, 16);
            let el = t.elapsed();
            let cands = peaks(&best, &bestv, &vels, cube.h, cube.w, opt(&args, "--tau", 6.0), 8);
            let adds = vels.len() as f64 * cube.n as f64 * (cube.h * cube.w) as f64;
            eprintln!("搜索用时 {el:.1?}（{:.1e} 次加法，{:.1} G/秒），候选 {} 个{}", adds, adds / el.as_secs_f64() / 1e9, cands.len(), if negate { "（取反）" } else { "" });
            for c in cands.iter().take(15) {
                eprintln!("  S = {:6.1}  (x0, y0) = ({:6.0}, {:6.0})  v = ({:+6.0}, {:+6.0})", c.s, c.x0, c.y0, c.vx, c.vy);
            }
            if let Some(i) = args.iter().position(|a| a == "--out") {
                serde_json::to_writer(std::fs::File::create(&args[i + 1]).unwrap(), &cands).unwrap();
            }
        }
        _ => eprintln!("{USAGE}"),
    }
}
