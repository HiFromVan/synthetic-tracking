# synthetic-tracking

在公开的天文图像档案中，寻找单帧不可见的太阳系小天体。

## 问题

设同一天区在时刻 $t_1, \dots, t_N$ 拍得图像 $I_1, \dots, I_N$，每帧噪声标准差为 $\sigma$。
一个亮度为 $f$ 的天体沿轨迹 $x(t)$ 运动，在单帧中 $f < k\sigma$，无法检出。

对任一候选轨迹 $\theta$，沿轨迹叠加：

$$
S(\theta) = \frac{1}{\sqrt{N}\,\sigma} \sum_{i=1}^{N} I_i\big(x_\theta(t_i)\big)
$$

若 $\theta$ 与真实轨迹一致，信噪比约为单帧的 $\sqrt{N}$ 倍。

线性情形 $x_\theta(t) = x_0 + v\,(t - t_0)$，$\theta = (x_0, v)$。
对一组 4K×4K 的图像，候选轨迹数量级为 $10^{10}$。

## 要回答的三件事

1. **搜索**：在 $10^{10}$ 量级的 $\theta$ 上计算 $S(\theta)$，单张消费级显卡上完成。
2. **判定**：在这么多次检验下，控制误报率。设定阈值 $\tau$，使纯噪声中 $\max_\theta S(\theta) > \tau$ 的概率足够小。
3. **连接**：把不同夜晚的候选点连成同一天体，拟合出轨道。

轨迹不再是直线时（观测跨度长、天体离得近），把 $x_\theta$ 换成按日心轨道推算的位置，参数空间随之变大。

## 数据

只用公开档案：DECam（NOIRLab Astro Data Archive）、HST（MAST）、TESS。
候选天体经轨道拟合后，提交国际小行星中心（Minor Planet Center）确认。

## 实现

搜索核心用 Rust，显卡部分用 CUDA。

## 进度

- [ ] 在一组 DECam 公开图像中，找回其中已知的小行星
- [ ] 测出阈值 $\tau$ 与误报率的关系
- [ ] 跨夜连接与轨道拟合
- [ ] 第一个未报告的候选天体

## 参考

- Jurić et al. 2019, *Enabling Deep All-Sky Searches of Outer Solar System Objects*, [arXiv:1901.08549](https://arxiv.org/abs/1901.08549)
- *heliostack: A Novel Approach to Minor Planet Discovery*, [arXiv:2604.16731](https://arxiv.org/abs/2604.16731)
